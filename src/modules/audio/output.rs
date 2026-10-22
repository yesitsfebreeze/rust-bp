//! Shared room return, output headroom and a stereo-linked peak limiter. A common gain
//! preserves the HRTF's interaural level difference, including during overlapping shots.
use crate::room::Room;

const HEADROOM: f32 = 0.8;
const CEILING: f32 = 0.95;

pub struct Output {
	room: Room,
	pub limiter_gain: f32,
	release: f32,
}

impl Output {
	pub fn new(rate: u32) -> Self {
		Self {
			room: Room::new(rate),
			limiter_gain: 1.0,
			release: (-1.0 / (rate as f32 * 0.06)).exp(),
		}
	}

	pub fn process(&mut self, samples: &mut [f32]) {
		for frame in samples.as_chunks_mut::<2>().0 {
			let wet = self
				.room
				.process([frame[0], frame[1]])
				.map(|v| v * HEADROOM);
			let peak = wet[0].abs().max(wet[1].abs());
			let target = if peak > CEILING { CEILING / peak } else { 1.0 };
			self.limiter_gain = if target < self.limiter_gain {
				target
			} else {
				target + (self.limiter_gain - target) * self.release
			};
			frame[0] = wet[0] * self.limiter_gain;
			frame[1] = wet[1] * self.limiter_gain;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn limiter_contains_overload_and_preserves_ear_balance() {
		let mut output = Output::new(44100);
		let mut samples = [4.0, 1.0, -2.0, -0.5];
		output.process(&mut samples);
		assert!(samples
			.iter()
			.all(|v| v.is_finite() && v.abs() <= CEILING + 1e-6));
		assert!((samples[0] / samples[1] - 4.0).abs() < 1e-6);
		assert!((samples[2] / samples[3] - 4.0).abs() < 1e-6);
		let mut silence = vec![0.0; 44100 * 2];
		output.process(&mut silence);
		assert!(output.limiter_gain > 0.999);
		assert!(silence
			.iter()
			.all(|v| v.is_finite() && v.abs() <= CEILING + 1e-6));
	}
}
