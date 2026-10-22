//! The output device: sokol audio's push queue, drained once per frame so the
//! mixer never blocks the audio thread. When no device answers, mixing simply
//! stops and the game plays on silently.
use crate::{Binaural, MAX_REQUEST_FRAMES};
use sokol::audio as saudio;

pub struct Device {
	enabled: bool,
}

impl Device {
	/// Opens the output stream at the mixer's rate.
	pub fn open() -> Device {
		saudio::setup(&saudio::Desc {
			sample_rate: crate::RATE as i32,
			num_channels: 2,
			buffer_frames: 512,
			packet_frames: 256,
			num_packets: 8,
			logger: saudio::Logger {
				func: Some(sokol::log::slog_func),
				..Default::default()
			},
			..Default::default()
		});
		if !saudio::isvalid() {
			eprintln!("No audio output device; the game continues silently.");
			return Device { enabled: false };
		}
		let desc = saudio::query_desc();
		eprintln!(
			"audio: {} Hz, buffer {}, queue {} frames",
			saudio::sample_rate(),
			saudio::buffer_frames(),
			desc.packet_frames * desc.num_packets
		);
		Device { enabled: true }
	}

	/// Mixes what the queue has room for and pushes it; None mixes nothing.
	pub fn update(&mut self, mixer: Option<&mut Binaural>) {
		if !self.enabled {
			return;
		}
		let Some(mixer) = mixer else {
			return;
		};
		let frames = saudio::expect().clamp(0, MAX_REQUEST_FRAMES as i32);
		if frames <= 0 {
			return;
		}
		let samples = mixer.mix(frames as usize);
		saudio::push(&samples[0], frames);
	}
}

/// Closes the stream, so a hot-reloaded game can open it again in the same
/// process: sokol allows one setup at a time.
impl Drop for Device {
	fn drop(&mut self) {
		saudio::shutdown();
	}
}
