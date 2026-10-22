//! GPU time per stage of the frame, from timestamps written between its
//! passes. Results come back a few frames late, without stalling: a copy
//! is read only once the GPU has mapped it, and a new one starts only
//! after that. Needs the adapter's timestamps inside encoders; without
//! them nothing is measured.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Stage boundaries a frame may mark.
const MARKS: u32 = 16;

pub struct Timer {
	set: wgpu::QuerySet,
	resolve: wgpu::Buffer,
	read: wgpu::Buffer,
	/// Nanoseconds per timestamp tick.
	period: f32,
	labels: Vec<&'static str>,
	/// The labels of the frame whose marks `read` holds.
	resolved: Vec<&'static str>,
	/// The copy in `read` is in flight; set once it maps.
	reading: bool,
	mapped: Arc<AtomicBool>,
	/// Milliseconds per stage, smoothed.
	results: Vec<(&'static str, f32)>,
}

impl Timer {
	/// The device features a timer needs.
	pub const FEATURES: wgpu::Features =
		wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

	pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Timer> {
		if !device.features().contains(Self::FEATURES) {
			return None;
		}
		let set = device.create_query_set(&wgpu::QuerySetDescriptor {
			label: Some("stage times"),
			ty: wgpu::QueryType::Timestamp,
			count: MARKS,
		});
		let size = MARKS as u64 * 8;
		let resolve = super::util::buffer(
			device,
			"stage times",
			size,
			wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
		);
		let read = super::util::buffer(
			device,
			"stage times read",
			size,
			wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
		);
		Some(Timer {
			set,
			resolve,
			read,
			period: queue.get_timestamp_period(),
			labels: Vec::new(),
			resolved: Vec::new(),
			reading: false,
			mapped: Arc::new(AtomicBool::new(false)),
			results: Vec::new(),
		})
	}

	/// Starts the frame's marks.
	pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder) {
		self.labels.clear();
		self.mark(encoder, "");
	}

	/// Ends the stage named `label` here.
	pub fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
		let at = self.labels.len() as u32;
		if at < MARKS {
			encoder.write_timestamp(&self.set, at);
			self.labels.push(label);
		}
	}

	/// Resolves the frame's marks for reading, unless a read is in flight.
	pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder) {
		if self.reading {
			return;
		}
		let n = self.labels.len() as u32;
		encoder.resolve_query_set(&self.set, 0..n, &self.resolve, 0);
		encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, n as u64 * 8);
		self.resolved.clone_from(&self.labels);
	}

	/// After the frame's submit: starts reading what it resolved, or takes
	/// in a read that has arrived.
	pub fn collect(&mut self, device: &wgpu::Device) {
		let _ = device.poll(wgpu::PollType::Poll);
		if !self.reading {
			let mapped = self.mapped.clone();
			self
				.read
				.slice(..)
				.map_async(wgpu::MapMode::Read, move |result| {
					mapped.store(result.is_ok(), Ordering::Release);
				});
			self.reading = true;
			return;
		}
		if !self.mapped.swap(false, Ordering::Acquire) {
			return;
		}
		let ticks: Vec<u64> = {
			let view = self.read.slice(..).get_mapped_range();
			view
				.as_chunks::<8>()
				.0
				.iter()
				.take(self.resolved.len())
				.map(|c| u64::from_le_bytes(*c))
				.collect()
		};
		self.read.unmap();
		self.reading = false;
		for (i, pair) in ticks.windows(2).enumerate() {
			let ms = pair[1].saturating_sub(pair[0]) as f32 * self.period / 1e6;
			let label = self.resolved[i + 1];
			match self.results.iter_mut().find(|(l, _)| *l == label) {
				Some((_, smooth)) => *smooth += (ms - *smooth) * 0.1,
				None => self.results.push((label, ms)),
			}
		}
	}

	/// Milliseconds per stage, smoothed over recent frames.
	pub fn results(&self) -> &[(&'static str, f32)] {
		&self.results
	}
}
