//! OpenAL Soft HRTF dataset (MinPHR03). Layout: rate
//! u32, channel type u8 (0 = left ear only, right ear from the mirrored azimuth), IR size u8,
//! field count u8; per field distance u16 mm, elevation count u8, azimuth count u8 per
//! elevation; every IR's int24 coefficients per channel; every IR's u8 delay per channel with
//! two fractional bits. Elevation 0 is straight down; azimuth 0 is ahead, increasing clockwise.
const MAGIC: &[u8] = b"MinPHR03";

pub struct Hrtf {
	pub rate: u32,
	pub size: usize,
	pub distance: f32,
	pub stereo: bool,
	pub azimuths: Vec<usize>,
	offsets: Vec<usize>,
	coeffs: Vec<Vec<Vec<f32>>>,
	delays: Vec<Vec<f32>>,
}

/// One direction's filters: per-ear taps and delay in samples.
#[derive(Clone, PartialEq)]
pub struct Hrir {
	pub taps: [Vec<f32>; 2],
	pub delay: [f32; 2],
}

struct Reader<'a> {
	bytes: &'a [u8],
	at: usize,
}

impl Reader<'_> {
	fn take(&mut self, n: usize) -> Result<&[u8], String> {
		let out = self
			.bytes
			.get(self.at..self.at + n)
			.ok_or("truncated dataset")?;
		self.at += n;
		Ok(out)
	}
	fn u8(&mut self) -> Result<u8, String> {
		Ok(self.take(1)?[0])
	}
}

impl Hrtf {
	/// The first dataset in `bytes` (a standalone .mhr or a binary embedding one): code that
	/// merely compares against the magic is skipped by requiring a plausible sample rate.
	pub fn parse(bytes: &[u8]) -> Result<Hrtf, String> {
		let start = (0..bytes.len().saturating_sub(12))
			.find(|&i| {
				&bytes[i..i + 8] == MAGIC
					&& (8000..=192000).contains(&u32::from_le_bytes(
						bytes[i + 8..i + 12].try_into().unwrap(),
					))
			})
			.ok_or("no MinPHR03 dataset")?;
		let mut r = Reader {
			bytes,
			at: start + 8,
		};
		let rate = u32::from_le_bytes(r.take(4)?.try_into().unwrap());
		let stereo = r.u8()? == 1;
		let size = r.u8()? as usize;
		if r.u8()? != 1 {
			return Err("only single-field datasets are supported".into());
		}
		let distance = u16::from_le_bytes(r.take(2)?.try_into().unwrap()) as f32 / 1000.0;
		let rows = r.u8()? as usize;
		let mut azimuths = Vec::with_capacity(rows);
		for _ in 0..rows {
			azimuths.push(r.u8()? as usize);
		}
		let count: usize = azimuths.iter().sum();
		let channels = if stereo { 2 } else { 1 };
		let mut coeffs = Vec::with_capacity(count);
		for _ in 0..count {
			let mut ir = Vec::with_capacity(channels);
			for _ in 0..channels {
				let mut taps = Vec::with_capacity(size);
				for _ in 0..size {
					let b = r.take(3)?;
					let v = (b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i8 as i32) << 16);
					taps.push(v as f32 / 8_388_608.0);
				}
				ir.push(taps);
			}
			coeffs.push(ir);
		}
		let mut delays = Vec::with_capacity(count);
		for _ in 0..count {
			let mut d = Vec::with_capacity(channels);
			for _ in 0..channels {
				d.push(r.u8()? as f32 / 4.0);
			}
			delays.push(d);
		}
		let mut offsets = Vec::with_capacity(rows);
		let mut at = 0;
		for &n in &azimuths {
			offsets.push(at);
			at += n;
		}
		Ok(Hrtf {
			rate,
			size,
			distance,
			stereo,
			azimuths,
			offsets,
			coeffs,
			delays,
		})
	}

	pub fn count(&self) -> usize {
		self.coeffs.len()
	}

	/// Longest delay plus the IR length: how long a voice rings after its clip ends.
	pub fn tail(&self) -> usize {
		let longest = self.delays.iter().flatten().fold(0.0f32, |a, &b| a.max(b));
		self.size + longest.ceil() as usize + 2
	}

	// One ear at an elevation row and azimuth index; a left-only set mirrors for the right.
	fn ear(&self, e: usize, a: usize, right: bool) -> (&[f32], f32) {
		let n = self.azimuths[e];
		let a = if right && !self.stereo {
			(n - a) % n
		} else {
			a
		};
		let index = self.offsets[e] + a;
		let c = usize::from(right && self.stereo);
		(&self.coeffs[index][c], self.delays[index][c])
	}

	/// Bilinear blend of the four nearest IRs and delays (OpenAL Soft's interpolation) for a
	/// direction in degrees: azimuth clockwise from ahead, elevation up from the horizon.
	pub fn direction(&self, azimuth: f32, elevation: f32) -> Hrir {
		let rows = self.azimuths.len();
		let ev = (90.0 + elevation.clamp(-90.0, 90.0)) / 180.0 * (rows - 1) as f32;
		let e0 = (ev.floor() as usize).min(rows - 1);
		let e1 = (e0 + 1).min(rows - 1);
		let ef = ev - e0 as f32;
		let mut out = Hrir {
			taps: [vec![0.0; self.size], vec![0.0; self.size]],
			delay: [0.0; 2],
		};
		for (e, we) in [(e0, 1.0 - ef), (e1, ef)] {
			let n = self.azimuths[e];
			let az = azimuth.rem_euclid(360.0) / 360.0 * n as f32;
			let a0 = az.floor() as usize % n;
			let a1 = (a0 + 1) % n;
			let af = az - az.floor();
			for (a, wa) in [(a0, 1.0 - af), (a1, af)] {
				let w = we * wa;
				if w == 0.0 {
					continue;
				}
				for side in 0..2 {
					let (taps, delay) = self.ear(e, a, side == 1);
					for (o, t) in out.taps[side].iter_mut().zip(taps) {
						*o += t * w;
					}
					out.delay[side] += delay * w;
				}
			}
		}
		out
	}
}
