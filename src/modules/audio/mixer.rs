//! Distance attenuation and envelope-driven master gain: quick recovery, gradual ducking.
//! Source addresses and attribution are retained in hiden's docs/research/reference-evidence.md.
pub const CULL_DISTANCE_SQ: f32 = 4_000_000.0;
pub const LOUDNESS_BASE: f32 = 0.001;
pub const MASTER_FLOOR: f32 = 0.01;
pub const MASTER_RISE: f32 = 0.8;
pub const MASTER_SETTLE: f32 = 0.1;

pub fn falloff(distance: f32) -> f32 {
	1.0 / distance.max(1.0)
}

pub fn source_gain(distance: f32, volume: f32, master: f32) -> f32 {
	if distance * distance > CULL_DISTANCE_SQ {
		return 0.0;
	}
	(falloff(distance) * volume * master).min(1.0)
}

pub fn loudness(envelope: f32, distance: f32, volume: f32) -> f32 {
	envelope * falloff(distance) * volume
}

pub fn master(previous: f32, loudness_sum: f32, frames: usize) -> f32 {
	let total = LOUDNESS_BASE + loudness_sum;
	let target = (1.0 / total.max(1.0)).max(MASTER_FLOOR);
	// Preserve the recovered 60 Hz response regardless of the audio callback size.
	let remaining = if target > previous {
		(1.0 - MASTER_RISE) * (1.0 - MASTER_SETTLE)
	} else {
		1.0 - MASTER_SETTLE
	};
	target + (previous - target) * remaining.powf(frames as f32 / crate::BLOCK as f32)
}
