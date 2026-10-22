//! Binaural weapon audio with per-voice HRTF convolution and loudness ducking,
//! ported from hiden's fight audio. Positions are world units; `distance_scale`
//! converts them to the metres the gain law falls off over.
//!
//! The mixer runs at 44.1 kHz to match the HRTF dataset; clips resample to it
//! on load. The output device is sokol audio's push queue, drained once per
//! frame by [`Device::update`].
mod clip;
mod device;
mod mhr;
mod mixer;
mod output;
mod room;
mod voice;

pub use clip::Clip;
pub use device::Device;
use mhr::Hrtf;
use voice::Voice;

/// Direction, gain and master update interval: 1/60 s at 44.1 kHz.
pub const BLOCK: usize = 735;
pub const MAX_VOICES: usize = 64;
/// The most frames one mix call renders: the device queue's whole capacity.
pub const MAX_REQUEST_FRAMES: usize = 4096;
/// The mixer and dataset rate, 44.1 kHz.
pub const RATE: u32 = 44100;
/// GoldSrc units per metre (one unit is 0.75 inches); the gain law falls off
/// over metres.
pub const DISTANCE_SCALE: f32 = 1.0 / 52.493;

pub struct Binaural {
	hrtf: Hrtf,
	tail: usize,
	clips: Vec<Clip>,
	voices: Vec<Voice>,
	listener: [[f32; 3]; 3],
	master: f32,
	distance_scale: f32,
	output: output::Output,
}

impl Binaural {
	pub fn new(dataset: &[u8], distance_scale: f32, clips: Vec<Clip>) -> Result<Binaural, String> {
		if !distance_scale.is_finite() || distance_scale <= 0.0 {
			return Err("distance_scale must be finite and positive".into());
		}
		let hrtf = Hrtf::parse(dataset)?;
		if hrtf.rate != RATE {
			return Err(format!("dataset is {} Hz, mixer {} Hz", hrtf.rate, RATE));
		}
		Ok(Binaural {
			tail: hrtf.tail(),
			hrtf,
			clips,
			voices: Vec::new(),
			listener: [[0.0; 3], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
			master: 1.0,
			distance_scale,
			output: output::Output::new(RATE),
		})
	}

	/// The listener's ears: position, forward and up in world units.
	pub fn listener(&mut self, position: [f32; 3], forward: [f32; 3], up: [f32; 3]) {
		if !position
			.iter()
			.chain(&forward)
			.chain(&up)
			.all(|v| v.is_finite())
		{
			return;
		}
		self.listener = [position, forward, up];
	}

	/// Starts a clip at a world position; the oldest voice yields when all are
	/// busy. Returns 1 when started, 0 for an unknown clip or invalid request.
	pub fn play(&mut self, clip: usize, position: [f32; 3], volume: f32) -> u32 {
		if clip >= self.clips.len()
			|| !volume.is_finite()
			|| volume < 0.0
			|| !position.iter().all(|v| v.is_finite())
		{
			return 0;
		}
		if self.voices.len() >= MAX_VOICES {
			self.voices.remove(0);
		}
		self.voices.push(Voice::new(clip, position, volume));
		1
	}

	/// Renders `frames` interleaved stereo frames, the room and limiter applied.
	pub fn mix(&mut self, frames: usize) -> Vec<f32> {
		let frames = frames.min(MAX_REQUEST_FRAMES);
		let mut out = vec![0.0; frames * 2];
		let mut done = 0;
		while done < frames {
			let block = BLOCK.min(frames - done);
			self.block(&mut out[done * 2..(done + block) * 2], block);
			done += block;
		}
		out
	}

	/// One line for the console: voices, master gain and dataset provenance.
	pub fn stats(&self) -> String {
		format!(
			"voices {} master {:.3} rate {} irs {} dataset {:.2} m limiter {:.3}",
			self.voices.len(),
			self.master,
			self.hrtf.rate,
			self.hrtf.count(),
			self.hrtf.distance,
			self.output.limiter_gain
		)
	}

	// Mix every voice at the current master gain, then update it from summed loudness.
	fn block(&mut self, out: &mut [f32], frames: usize) {
		let mut loudness = 0.0;
		for voice in &mut self.voices {
			let (azimuth, elevation, distance) = relative(&self.listener, voice.position);
			let distance = distance * self.distance_scale;
			let clip = &self.clips[voice.clip];
			let gain = mixer::source_gain(distance, voice.volume, self.master);
			loudness += mixer::loudness(clip.loudness_at(voice.cursor), distance, voice.volume);
			voice.render(
				&clip.samples,
				self.hrtf.direction(azimuth, elevation),
				gain,
				out,
				frames,
			);
		}
		let (clips, tail) = (&self.clips, self.tail);
		self
			.voices
			.retain(|v| !v.finished(clips[v.clip].samples.len(), tail));
		self.master = mixer::master(self.master, loudness, frames);
		self.output.process(out);
	}
}

// Listener-space direction of a world point: azimuth clockwise from ahead and
// elevation in degrees, distance in world units.
fn relative(listener: &[[f32; 3]; 3], point: [f32; 3]) -> (f32, f32, f32) {
	let [position, forward, up] = listener;
	let d = [
		point[0] - position[0],
		point[1] - position[1],
		point[2] - position[2],
	];
	let forward = normalize(*forward, [0.0, 0.0, -1.0]);
	let right = normalize(
		cross(forward, *up),
		cross(
			forward,
			if forward[1].abs() > 0.99 {
				[0.0, 0.0, 1.0]
			} else {
				[0.0, 1.0, 0.0]
			},
		),
	);
	let up = cross(right, forward);
	let x = dot(d, right);
	let y = dot(d, up);
	let z = dot(d, forward);
	let distance = dot(d, d).sqrt();
	(
		x.atan2(z).to_degrees(),
		y.atan2((x * x + z * z).sqrt()).to_degrees(),
		distance,
	)
}

fn normalize(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
	let length = dot(v, v).sqrt();
	if length.is_finite() && length > 1e-6 {
		v.map(|c| c / length)
	} else {
		let length = dot(fallback, fallback).sqrt();
		fallback.map(|c| c / length)
	}
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
	[
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
	a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests;
