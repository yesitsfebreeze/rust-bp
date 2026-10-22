//! Auto exposure: the eye's adaptation. Each frame one compute workgroup
//! reads a 64×64 grid of the world pass's color and averages its log
//! luminance; the average comes back a frame or two later, and the
//! exposure scale walks toward the one that brings that average to
//! mid grey — opening up in a dark interior, closing down in the sun —
//! at a pace the eye takes, not in one jump.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::util::{self, entry};

const SHADER: &str = r#"
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
// x 1 when the color is sRGB-encoded.
@group(0) @binding(2) var<uniform> encoded: vec4f;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_index) i: u32, @builtin(local_invocation_id) at: vec3u) {
	let size = vec2f(textureDimensions(scene));
	var sum = 0.0;
	for (var y = 0u; y < 4u; y++) {
		for (var x = 0u; x < 4u; x++) {
			let g = (vec2f(f32(at.x * 4u + x), f32(at.y * 4u + y)) + 0.5) / 64.0;
			var c = textureLoad(scene, vec2u(g * size), 0).rgb;
			if (encoded.x > 0.5) {
				c = pow(c, vec3f(2.2));
			}
			sum += log2(max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 1e-4));
		}
	}
	part[i] = sum;
	workgroupBarrier();
	for (var s = 128u; s > 0u; s >>= 1u) {
		if (i < s) {
			part[i] += part[i + s];
		}
		workgroupBarrier();
	}
	if (i == 0u) {
		out[0] = part[0] / 4096.0;
	}
}
"#;

/// The mean displayed luminance the exposure brings a frame to: a touch
/// under mid grey, as a camera's meter aims.
const TARGET: f32 = 0.15;
/// How fast the eye adapts: the rate, per second, at which the remaining
/// step in log space shrinks — half of it gone in about 70 ms.
const RATE: f32 = 10.0;

pub struct Meter {
	pipeline: wgpu::ComputePipeline,
	layout: wgpu::BindGroupLayout,
	encoded: wgpu::Buffer,
	sum: wgpu::Buffer,
	read: wgpu::Buffer,
	/// Whether `read` is mapping or mapped, and whether the map finished.
	reading: bool,
	mapped: Arc<AtomicBool>,
	/// The exposure scale adaptation has reached, and the range it stays
	/// in: darkest to brightest.
	scale: f32,
	range: (f32, f32),
	/// When the last average came back, so each one adapts for the whole
	/// time since.
	last: Instant,
}

impl Meter {
	pub fn new(device: &wgpu::Device, encode: bool) -> Meter {
		let compute = wgpu::ShaderStages::COMPUTE;
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("exposure"),
			entries: &[
				util::raw_texture_entry(0, compute),
				util::buffer_entry(1, compute, util::WRITE),
				util::buffer_entry(2, compute, util::UNIFORM),
			],
		});
		let shader = util::shader(device, "exposure", &[SHADER]);
		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: Some("exposure"),
			layout: Some(&util::layout(device, "exposure", &[&layout])),
			module: &shader,
			entry_point: Some("main"),
			compilation_options: Default::default(),
			cache: None,
		});
		let encoded = util::uniform(
			device,
			"exposure encoded",
			&util::floats(&[encode as u32 as f32, 0.0, 0.0, 0.0]),
		);
		let sum = util::buffer(
			device,
			"exposure sum",
			4,
			wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
		);
		let read = util::buffer(
			device,
			"exposure read",
			4,
			wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
		);
		Meter {
			pipeline,
			layout,
			encoded,
			sum,
			read,
			reading: false,
			mapped: Arc::new(AtomicBool::new(false)),
			scale: 1.0,
			range: (0.25, 8.0),
			last: Instant::now(),
		}
	}

	/// The exposure scale adaptation has reached, 1 for a scene at
	/// [`TARGET`].
	pub fn scale(&self) -> f32 {
		self.scale
	}

	/// The range adaptation stays in: `min` the darkest the exposure
	/// closes down to, `max` the brightest it opens up to.
	pub fn set_range(&mut self, min: f32, max: f32) {
		let min = min.max(0.01);
		self.range = (min, max.max(min));
		self.scale = self.scale.clamp(self.range.0, self.range.1);
	}

	/// Averages `scene`'s log luminance into the meter, unless the last
	/// average is still on its way back.
	pub fn measure(
		&self,
		device: &wgpu::Device,
		encoder: &mut wgpu::CommandEncoder,
		scene: &wgpu::TextureView,
	) {
		if self.reading {
			return;
		}
		let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("exposure"),
			layout: &self.layout,
			entries: &[
				entry(0, wgpu::BindingResource::TextureView(scene)),
				util::bind(1, &self.sum),
				util::bind(2, &self.encoded),
			],
		});
		{
			let mut pass = encoder.begin_compute_pass(&Default::default());
			pass.set_pipeline(&self.pipeline);
			pass.set_bind_group(0, &group, &[]);
			pass.dispatch_workgroups(1, 1, 1);
		}
		encoder.copy_buffer_to_buffer(&self.sum, 0, &self.read, 0, 4);
	}

	/// After the frame's submit: starts reading the average back, or takes
	/// it when it arrived and walks the scale toward it. Without `adapt`
	/// the scale rests at 1.
	pub fn collect(&mut self, device: &wgpu::Device, adapt: bool) {
		if !adapt {
			self.scale = 1.0;
		}
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
		let log_mean = {
			let view = self.read.slice(..).get_mapped_range();
			f32::from_le_bytes([view[0], view[1], view[2], view[3]])
		};
		self.read.unmap();
		self.reading = false;
		let now = Instant::now();
		let dt = now.duration_since(self.last).as_secs_f32().min(0.25);
		self.last = now;
		if adapt && log_mean.is_finite() {
			self.scale = adapted(self.scale, log_mean, dt, self.range);
		}
	}
}

/// `scale` after `dt` seconds of adapting to a frame whose mean log2
/// luminance, drawn at `scale`, was `log_mean`, held within `range`.
fn adapted(scale: f32, log_mean: f32, dt: f32, range: (f32, f32)) -> f32 {
	let step = TARGET.log2() - log_mean;
	let k = 1.0 - (-RATE * dt).exp();
	(scale * (step * k).exp2()).clamp(range.0, range.1)
}

#[cfg(test)]
mod tests {
	use super::*;

	const RANGE: (f32, f32) = (0.25, 8.0);

	#[test]
	fn adaptation_opens_in_the_dark_and_closes_in_the_light() {
		let dark = (TARGET * 0.1).log2();
		let bright = (TARGET * 4.0).log2();
		assert!(adapted(1.0, dark, 0.1, RANGE) > 1.0);
		assert!(adapted(1.0, bright, 0.1, RANGE) < 1.0);
		assert_eq!(adapted(1.0, TARGET.log2(), 0.1, RANGE), 1.0);
		// Held within its range however long the dark lasts.
		let mut s = 1.0;
		for _ in 0..1000 {
			s = adapted(s, dark, 0.1, RANGE);
		}
		assert_eq!(s, RANGE.1);
	}

	#[test]
	fn adaptation_settles_within_a_quarter_second() {
		// A scene a hundred times too dark, drawn at whatever scale the
		// eye has reached, seen at 60 frames a second with each average
		// coming back every other frame.
		let scene = (TARGET * 0.01).log2();
		let mut s = 1.0f32;
		for _ in 0..8 {
			s = adapted(s, scene + s.log2(), 2.0 / 60.0, (0.01, 1000.0));
		}
		assert!(
			(s.log2() - 100f32.log2()).abs() < 0.5,
			"scale {s} after 0.27 s"
		);
	}
}
