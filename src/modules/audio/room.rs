//! A quiet, short room around the dry binaural signal. Predelay leaves the attack and its
//! ear-to-ear cues alone; a damped four-line feedback network supplies the diffuse tail.
const DECAY_SECONDS: f32 = 0.28;
const WET: f32 = 0.12;

struct Delay {
	samples: Vec<f32>,
	cursor: usize,
}

impl Delay {
	fn new(seconds: f32, rate: u32) -> Self {
		Self {
			samples: vec![0.0; (seconds * rate as f32).round().max(1.0) as usize],
			cursor: 0,
		}
	}

	fn read(&self) -> f32 {
		self.samples[self.cursor]
	}

	fn write(&mut self, sample: f32) {
		self.samples[self.cursor] = if sample.abs() < 1e-20 { 0.0 } else { sample };
		self.cursor = (self.cursor + 1) % self.samples.len();
	}
}

pub struct Room {
	predelay: Delay,
	lines: [Delay; 4],
	feedback: [f32; 4],
	damped: [f32; 4],
	damping: f32,
	highpass: f32,
	previous_input: f32,
	previous_highpass: f32,
}

impl Room {
	pub fn new(rate: u32) -> Self {
		let lines = [0.0233, 0.0311, 0.0379, 0.0437].map(|time| Delay::new(time, rate));
		let feedback = std::array::from_fn(|i| {
			10.0_f32.powf(-3.0 * lines[i].samples.len() as f32 / (rate as f32 * DECAY_SECONDS))
		});
		Self {
			predelay: Delay::new(0.012, rate),
			lines,
			feedback,
			damped: [0.0; 4],
			damping: 1.0 - (-std::f32::consts::TAU * 4500.0 / rate as f32).exp(),
			highpass: (-std::f32::consts::TAU * 180.0 / rate as f32).exp(),
			previous_input: 0.0,
			previous_highpass: 0.0,
		}
	}

	pub fn process(&mut self, dry: [f32; 2]) -> [f32; 2] {
		let input = (dry[0] + dry[1]) * 0.5;
		let filtered = input - self.previous_input + self.highpass * self.previous_highpass;
		self.previous_input = input;
		self.previous_highpass = filtered;
		let send = self.predelay.read() * 0.5;
		self.predelay.write(filtered);
		let taps: [f32; 4] = std::array::from_fn(|i| self.lines[i].read());
		for (damped, tap) in self.damped.iter_mut().zip(taps) {
			*damped += self.damping * (tap - *damped);
		}
		let [a, b, c, d] = self.damped;
		// Orthonormal Hadamard feedback: no energy amplification inside the network.
		let mixed = [a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d];
		for (i, value) in mixed.into_iter().enumerate() {
			self.lines[i].write(send + value * 0.5 * self.feedback[i]);
		}
		[
			dry[0] + WET * (taps[0] + taps[2]) * 0.5,
			dry[1] + WET * (taps[1] + taps[3]) * 0.5,
		]
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn room_preserves_attack_and_has_a_quiet_decaying_stereo_tail() {
		let mut room = Room::new(44100);
		assert_eq!(room.process([1.0, 0.5]), [1.0, 0.5]);
		let tail: Vec<_> = (1..44100).map(|_| room.process([0.0; 2])).collect();
		assert!(tail[..1000].iter().flatten().all(|&v| v == 0.0));
		let energy =
			|range: std::ops::Range<usize>| tail[range].iter().flatten().map(|v| v * v).sum::<f32>();
		let early = energy(1000..6000);
		assert!(early > 1e-5 && early < 0.02, "wet energy {early}");
		assert!(energy(26460..44099) < early * 1e-6);
		assert!(tail.iter().any(|v| (v[0] - v[1]).abs() > 0.001));
		assert!(tail.iter().flatten().all(|v| v.is_finite()));
	}
}
