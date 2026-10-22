//! One playing sound: its mono clip delayed per ear by a fractional interaural delay and
//! convolved with the HRIR for its listener-relative direction. Direction and gain are set
//! per block; each block crossfades from the previous filters and gain to the new ones, so a
//! moving source or listener never clicks.
use crate::clip::sample;
use crate::mhr::Hrir;

pub struct Voice {
	pub clip: usize,
	pub position: [f32; 3],
	pub volume: f32,
	/// Next clip sample the voice outputs.
	pub cursor: usize,
	filters: Option<(Hrir, f32)>,
}

impl Voice {
	pub fn new(clip: usize, position: [f32; 3], volume: f32) -> Voice {
		Voice {
			clip,
			position,
			volume,
			cursor: 0,
			filters: None,
		}
	}

	/// True once the clip and the filter tail have played out.
	pub fn finished(&self, length: usize, tail: usize) -> bool {
		self.cursor > length + tail
	}

	/// Adds `frames` interleaved stereo frames into `out`, crossfading old -> new filters.
	pub fn render(&mut self, samples: &[f32], hrir: Hrir, gain: f32, out: &mut [f32], frames: usize) {
		let previous = self.filters.take().unwrap_or_else(|| (hrir.clone(), gain));
		let unchanged = previous.0 == hrir && previous.1 == gain;
		for i in 0..frames {
			let w = (i + 1) as f32 / frames as f32;
			let n = (self.cursor + i) as f32;
			for side in 0..2 {
				let old = convolve(samples, &previous.0.taps[side], previous.0.delay[side], n) * previous.1;
				out[i * 2 + side] += if unchanged {
					old
				} else {
					let new = convolve(samples, &hrir.taps[side], hrir.delay[side], n) * gain;
					old + (new - old) * w
				};
			}
		}
		self.cursor += frames;
		self.filters = Some((hrir, gain));
	}
}

// Output sample n of one ear: the taps over the clip shifted by a fractional delay.
fn convolve(samples: &[f32], taps: &[f32], delay: f32, n: f32) -> f32 {
	taps
		.iter()
		.enumerate()
		.map(|(k, t)| t * sample(samples, n - k as f32 - delay))
		.sum()
}
