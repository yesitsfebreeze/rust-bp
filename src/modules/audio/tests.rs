// Checks against the retained HRTF dataset and the bundled 1.6 weapon sounds.
use super::*;

const DATASET: &str = concat!(
	env!("CARGO_MANIFEST_DIR"),
	"/../../../assets/audio/hrtf.mhr"
);
const AK47: &str = concat!(
	env!("CARGO_MANIFEST_DIR"),
	"/../../../assets/sound/weapons/ak47-1.wav"
);

fn dataset() -> Hrtf {
	Hrtf::parse(&std::fs::read(DATASET).expect("HRTF dataset missing from assets/audio")).unwrap()
}

fn energy(v: &[f32]) -> f32 {
	v.iter().map(|x| x * x).sum()
}

#[test]
fn dataset_is_openal_soft_default() {
	let h = dataset();
	assert_eq!(
		(h.rate, h.size, h.count(), h.stereo),
		(44100, 32, 828, false)
	);
	assert_eq!(h.azimuths.len(), 19);
	assert!((h.distance - 1.4).abs() < 1e-6);
}

#[test]
fn right_source_is_louder_and_earlier_in_the_right_ear() {
	let h = dataset();
	let right = h.direction(90.0, 0.0);
	assert!(energy(&right.taps[1]) > 10.0 * energy(&right.taps[0]));
	assert!(right.delay[1] + 30.0 < right.delay[0]);
	let ahead = h.direction(0.0, 0.0);
	assert!((energy(&ahead.taps[0]) - energy(&ahead.taps[1])).abs() < 1e-4);
	assert!((ahead.delay[0] - ahead.delay[1]).abs() < 1e-4);
	let left = h.direction(-90.0, 0.0);
	assert!((energy(&left.taps[0]) - energy(&right.taps[1])).abs() < 1e-4);
}

#[test]
fn listener_space_azimuth_is_clockwise() {
	let listener = [[0.0; 3], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
	let (az, el, d) = relative(&listener, [2.0, 0.0, 0.0]);
	assert!((az - 90.0).abs() < 1e-4 && el.abs() < 1e-4 && (d - 2.0).abs() < 1e-6);
	assert!(relative(&listener, [0.0, 0.0, -3.0]).0.abs() < 1e-4);
	assert!((relative(&listener, [0.0, 1.0, -1.0]).1 - 45.0).abs() < 1e-3);
}

#[test]
fn distance_falloff_and_loudness_ducking() {
	assert_eq!(mixer::source_gain(0.5, 1.0, 1.0), 1.0);
	assert!((mixer::source_gain(4.0, 1.0, 1.0) - 0.25).abs() < 1e-6);
	assert_eq!(mixer::source_gain(2001.0, 1.0, 1.0), 0.0);
	// Quiet: master holds at 1. Loud: ducks 10% toward 1/total per frame, recovers fast.
	assert!((mixer::master(1.0, 0.2, BLOCK) - 1.0).abs() < 1e-6);
	let ducked = mixer::master(1.0, 4.0, BLOCK);
	assert!((ducked - (1.0 + (1.0 / 4.001 - 1.0) * 0.1)).abs() < 1e-5);
	// Recovery after the duck: 80% up at once (0.85), then 10% toward the target (0.865).
	assert!((mixer::master(0.25, 0.0, BLOCK) - 0.865).abs() < 1e-5);
}

fn plain_mixer(clips: Vec<Clip>) -> Binaural {
	Binaural::new(&std::fs::read(DATASET).expect("dataset"), 1.0, clips).unwrap()
}

#[test]
fn voice_from_the_right_renders_right_heavy_and_ends() {
	let mut b = plain_mixer(vec![Clip {
		samples: (0..2000).map(|i| ((i as f32) * 0.3).sin()).collect(),
		loudness: vec![0.5],
	}]);
	b.play(0, [1.0, 0.0, 0.0], 1.0);
	let out = b.mix(BLOCK * 4);
	let left: Vec<f32> = out.iter().step_by(2).copied().collect();
	let right: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
	assert!(
		energy(&right) > 4.0 * energy(&left),
		"right {} left {}",
		energy(&right),
		energy(&left)
	);
	assert!(b.voices.is_empty());
}

#[test]
fn listener_pitch_and_non_unit_up_do_not_distort_angles() {
	let s = 0.5_f32.sqrt();
	let listener = [[0.0; 3], [0.0, -s * 3.0, -s * 3.0], [0.0, 2.0, 0.0]];
	let (az, el, distance) = relative(&listener, [1.0, -s, -s]);
	assert!((az - 45.0).abs() < 1e-4 && el.abs() < 1e-4);
	assert!((distance - 2.0_f32.sqrt()).abs() < 1e-4);
	let (az, el, _) = relative(&listener, [0.0, 0.0, -1.0]);
	assert!(az.abs() < 1e-4 && (el - 45.0).abs() < 1e-4);
	let vertical = [[0.0; 3], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
	let (az, el, d) = relative(&vertical, [1.0, 0.0, 0.0]);
	assert!(az.is_finite() && el.is_finite() && d.is_finite());
}

#[test]
fn master_response_is_independent_of_audio_buffer_size() {
	for (start, loudness) in [(1.0, 4.0), (0.25, 0.0)] {
		let whole = mixer::master(start, loudness, BLOCK);
		let split = [128, 256, 351]
			.into_iter()
			.fold(start, |gain, frames| mixer::master(gain, loudness, frames));
		assert!((whole - split).abs() < 1e-6);
	}
}

fn gunshot_mixer() -> Binaural {
	plain_mixer(vec![Clip::load(std::path::Path::new(AK47), RATE).unwrap()])
}

#[test]
fn real_gunshot_keeps_direction_and_overlapping_shots_do_not_clip() {
	let render = |x: f32, voices: usize| {
		let mut mixer = gunshot_mixer();
		for _ in 0..voices {
			assert_eq!(mixer.play(0, [x, 0.0, -2.0], 1.0), 1);
		}
		mixer.mix(44100 / 2)
	};
	let right = render(1.5, 1);
	let left = render(-1.5, 1);
	let ear_energy = |samples: &[f32], ear| {
		samples
			.iter()
			.skip(ear)
			.step_by(2)
			.map(|v| v * v)
			.sum::<f32>()
	};
	// Gunshots are bass-heavy: test a clear ear preference, not an artificial hard pan.
	assert!(
		ear_energy(&right, 1) > 1.25 * ear_energy(&right, 0),
		"right/left energy {}",
		ear_energy(&right, 1) / ear_energy(&right, 0)
	);
	assert!(
		ear_energy(&left, 0) > 1.25 * ear_energy(&left, 1),
		"left/right energy {}",
		ear_energy(&left, 0) / ear_energy(&left, 1)
	);
	for (a, b) in right
		.as_chunks::<2>()
		.0
		.iter()
		.zip(left.as_chunks::<2>().0.iter())
		.take(1200)
	{
		assert!((a[0] - b[1]).abs() < 1e-5 && (a[1] - b[0]).abs() < 1e-5);
	}
	let burst = render(1.5, MAX_VOICES);
	assert!(burst.iter().all(|v| v.is_finite() && v.abs() <= 0.950001));
	assert!(energy(&burst) > energy(&right));
}

#[test]
fn fixed_source_audio_is_continuous_across_arbitrary_push_sizes() {
	let mut whole = gunshot_mixer();
	let mut split = gunshot_mixer();
	whole.play(0, [1.0, 0.0, -2.0], 0.3);
	split.play(0, [1.0, 0.0, -2.0], 0.3);
	let expected = whole.mix(4096);
	let actual: Vec<_> = [128, 512, 735, 1, 2720]
		.into_iter()
		.flat_map(|n| split.mix(n))
		.collect();
	assert_eq!(expected, actual);
	assert_eq!(whole.play(0, [f32::NAN, 0.0, 0.0], 1.0), 0);
	assert_eq!(whole.play(0, [0.0; 3], f32::INFINITY), 0);
}
