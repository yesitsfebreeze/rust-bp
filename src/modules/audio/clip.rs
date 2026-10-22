//! Mono clips for the mixer: 8- or 16-bit PCM WAV at any rate, resampled to the
//! mixer's rate; multi-channel files are averaged to one point source. The
//! loudness envelope (RMS per 0.1 s window) drives the master ducking.
pub struct Clip {
	pub samples: Vec<f32>,
	pub loudness: Vec<f32>,
}

/// Samples per loudness window at 44.1 kHz.
pub const WINDOW: usize = 4410;

impl Clip {
	pub fn load(path: &std::path::Path, rate: u32) -> Result<Clip, String> {
		let wav = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
		Clip::from_wav(&wav, rate)
	}

	/// Decodes a WAV in memory at the mixer's rate.
	pub fn from_wav(wav: &[u8], rate: u32) -> Result<Clip, String> {
		let (samples, file_rate) = decode(wav)?;
		let samples = resample(&samples, file_rate, rate);
		Ok(Clip {
			loudness: windows(&samples),
			samples,
		})
	}

	/// Loudness at a playback position (the last window holds past the end).
	pub fn loudness_at(&self, cursor: usize) -> f32 {
		if self.loudness.is_empty() || cursor >= self.samples.len() {
			return 0.0;
		}
		self.loudness[(cursor / WINDOW).min(self.loudness.len() - 1)]
	}
}

// PCM frames averaged to mono: 8-bit unsigned or 16-bit signed, any rate.
fn decode(wav: &[u8]) -> Result<(Vec<f32>, u32), String> {
	let mut at = 12;
	let (mut channels, mut file_rate, mut bits) = (0usize, 0u32, 0u16);
	while at + 8 <= wav.len() {
		let tag = &wav[at..at + 4];
		let size = u32::from_le_bytes(wav[at + 4..at + 8].try_into().unwrap()) as usize;
		let body = at + 8;
		if tag == b"fmt " {
			channels = u16::from_le_bytes(wav[body + 2..body + 4].try_into().unwrap()) as usize;
			file_rate = u32::from_le_bytes(wav[body + 4..body + 8].try_into().unwrap());
			bits = u16::from_le_bytes(wav[body + 14..body + 16].try_into().unwrap());
		}
		if tag == b"data" {
			if channels == 0 || file_rate == 0 {
				return Err("no fmt chunk before data".into());
			}
			let data = &wav[body..(body + size).min(wav.len())];
			if channels == 0 || file_rate == 0 {
				return Err("no fmt chunk before data".into());
			}
			if bits != 8 && bits != 16 {
				return Err(format!("{bits}-bit PCM is unsupported"));
			}
			let mono = |bytes: &[u8]| {
				if bits == 8 {
					(bytes[0] as f32 - 128.0) / 128.0
				} else {
					i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0
				}
			};
			let width = if bits == 16 { 2 } else { 1 };
			let samples = data
				.chunks_exact(width * channels)
				.map(|frame| frame.chunks(width).map(mono).sum::<f32>() / channels as f32)
				.collect();
			return Ok((samples, file_rate));
		}
		at = body + size + (size & 1);
	}
	Err("no PCM data".into())
}

/// RMS loudness per 0.1 s window, the envelope the master gain ducks to.
fn windows(samples: &[f32]) -> Vec<f32> {
	samples
		.chunks(WINDOW)
		.map(|w| (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt())
		.collect()
}

/// Resamples by the same fractional sampler the voices use, so treble detail
/// survives the rate change.
pub(crate) fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
	if from == to {
		return samples.to_vec();
	}
	(0..(samples.len() as u64 * to as u64 / from as u64))
		.map(|i| sample(samples, i as f32 * from as f32 / to as f32))
		.collect()
}

// Output sample n of one ear: the taps over the clip shifted by a fractional delay.
pub(crate) fn sample(samples: &[f32], at: f32) -> f32 {
	let i = at.floor() as isize;
	let f = at - i as f32;
	let tap = |k: isize| {
		if k < 0 {
			0.0
		} else {
			samples.get(k as usize).copied().unwrap_or(0.0)
		}
	};
	let [a, b, c, d] = [tap(i - 1), tap(i), tap(i + 1), tap(i + 2)];
	-a * f * (f - 1.0) * (f - 2.0) / 6.0 + b * (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0
		- c * (f + 1.0) * f * (f - 2.0) / 2.0
		+ d * (f + 1.0) * f * (f - 1.0) / 6.0
}

#[cfg(test)]
mod tests {
	use super::*;

	// One second of 22.05 kHz 8-bit mono noise, as the bundled 1.6 sounds are.
	fn wav(rate: u32, bits: u16, channels: usize, frames: usize) -> Vec<u8> {
		let width = if bits == 16 { 2 } else { 1 };
		let data_len = frames * width * channels;
		let mut out = Vec::with_capacity(44 + data_len);
		out.extend(b"RIFF");
		out.extend((36 + data_len as u32).to_le_bytes());
		out.extend(b"WAVEfmt ");
		out.extend(16u32.to_le_bytes());
		out.extend(1u16.to_le_bytes()); // PCM
		out.extend((channels as u16).to_le_bytes());
		out.extend(rate.to_le_bytes());
		out.extend((rate * width as u32 * channels as u32).to_le_bytes());
		out.extend(((width * channels) as u16).to_le_bytes());
		out.extend(bits.to_le_bytes());
		out.extend(b"data");
		out.extend((data_len as u32).to_le_bytes());
		for i in 0..frames * channels {
			let s = ((i as f32 * 0.1).sin() * 8000.0) as i16;
			if bits == 16 {
				out.extend(s.to_le_bytes());
			} else {
				out.push((s / 256 + 128) as u8);
			}
		}
		out
	}

	#[test]
	fn decodes_8bit_and_resamples_treble_faithfully() {
		let clip = Clip::from_wav(&wav(22050, 8, 1, 4410), 44100).unwrap();
		assert_eq!(clip.samples.len(), 8820);
		let energy = clip.samples.iter().map(|v| v * v).sum::<f32>();
		assert!(energy > 0.0, "silence decoded");
		assert!(clip.loudness.len() == 2, "one window per 0.1 s");
		assert!(clip.loudness.iter().all(|v| v.is_finite()));
		let passthrough = Clip::from_wav(&wav(44100, 8, 1, 100), 44100).unwrap();
		assert_eq!(passthrough.samples.len(), 100);
	}

	#[test]
	fn averages_stereo_to_one_point_source() {
		let clip = Clip::from_wav(&wav(44100, 16, 2, 100), 44100).unwrap();
		assert_eq!(clip.samples.len(), 100);
		// The sine's left channel carries the whole frame; right is silent.
		assert!(clip.samples.iter().any(|v| *v != 0.0));
	}

	#[test]
	fn fractional_delay_preserves_treble_energy() {
		let signal: Vec<_> = (0..2048)
			.map(|i| (std::f32::consts::TAU * 0.15 * i as f32).sin())
			.collect();
		let input: f32 = signal[16..2016].iter().map(|v| v * v).sum();
		let output: f32 = (16..2016)
			.map(|i| sample(&signal, i as f32 + 0.5).powi(2))
			.sum();
		assert!((0.97..1.01).contains(&(output / input).sqrt()));
		assert_eq!(sample(&signal, -2.0), 0.0);
		assert_eq!(sample(&signal, signal.len() as f32 + 2.0), 0.0);
	}

	#[test]
	fn rejects_wav_without_data() {
		assert!(Clip::from_wav(b"RIFFjunkWAVEfmt ", 44100).is_err());
	}
}
