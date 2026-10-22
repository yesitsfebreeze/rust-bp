//! The game's randomness: one xorshift stream seeded from the clock. Spread,
//! recoil flips and sound and animation picks only need it to look random.

use std::sync::atomic::{AtomicU64, Ordering};

static STATE: AtomicU64 = AtomicU64::new(0);

fn next() -> u64 {
	let mut x = STATE.load(Ordering::Relaxed);
	if x == 0 {
		x = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map_or(0x9E37_79B9_7F4A_7C15, |d| d.as_nanos() as u64)
			| 1;
	}
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	STATE.store(x, Ordering::Relaxed);
	x
}

/// Uniform in `lo..hi`.
pub fn range(lo: f32, hi: f32) -> f32 {
	let unit = (next() >> 40) as f32 / (1u64 << 24) as f32;
	lo + (hi - lo) * unit
}

/// Uniform in `0..n`; `n` must be positive.
pub fn index(n: usize) -> usize {
	(next() % n as u64) as usize
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn draws_stay_in_range_and_vary() {
		let draws: Vec<f32> = (0..1000).map(|_| range(-0.5, 0.5)).collect();
		assert!(draws.iter().all(|d| (-0.5..0.5).contains(d)));
		assert!(draws.iter().any(|d| *d < -0.4) && draws.iter().any(|d| *d > 0.4));
		assert!((0..1000).map(|_| index(3)).all(|i| i < 3));
	}
}
