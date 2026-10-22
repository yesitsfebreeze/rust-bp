use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::player::{Input, MovementVars, Player};
use engine::linalg::Vec3;

const HISTORY_MS: u64 = 15_000;
const FUTURE_MS: u64 = 16_000;
const FRAME_SPIKE_MS: f32 = 50.0;
const MAX_HISTORY_SAMPLES: usize = 16_384;
const MAX_CAPTURE_SAMPLES: usize = 32_768;

#[derive(Clone, Copy)]
pub struct Sample {
	elapsed_ms: u64,
	frame_ms: f32,
	frame_work_ms: f32,
	tick_budget_ms: f32,
	tick_total_ms: f32,
	tick_max_ms: f32,
	ticks: u32,
	collision_casts: u32,
	collision_overlaps: u32,
	collision_ms: f32,
	collision_max_ms: f32,
	player_pos: Vec3,
	player_velocity: Vec3,
	on_ground: bool,
	ducked: bool,
	in_duck: bool,
	noclip: bool,
	step_offset: f32,
	fall_velocity: f32,
	jump_penalty: f32,
	input: Input,
	flycam: bool,
	weapon_slot: usize,
	weapon_clip: u32,
	weapon_reserve: u32,
	money: u32,
	gravity: f32,
	max_speed: f32,
	step_size: f32,
	tick_rate: f32,
	accumulator_ms: f32,
}

impl Sample {
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		frame_ms: f32,
		frame_work_ms: f32,
		tick_budget_ms: f32,
		tick_total_ms: f32,
		tick_max_ms: f32,
		ticks: u32,
		collision_casts: u32,
		collision_overlaps: u32,
		collision_ms: f32,
		collision_max_ms: f32,
		player: &Player,
		input: Input,
		flycam: bool,
		weapon_slot: usize,
		weapon_clip: u32,
		weapon_reserve: u32,
		money: u32,
		vars: &MovementVars,
		tick_rate: f32,
		accumulator_ms: f32,
	) -> Sample {
		Sample {
			elapsed_ms: 0,
			frame_ms,
			frame_work_ms,
			tick_budget_ms,
			tick_total_ms,
			tick_max_ms,
			ticks,
			collision_casts,
			collision_overlaps,
			collision_ms,
			collision_max_ms,
			player_pos: player.origin,
			player_velocity: player.velocity,
			on_ground: player.on_ground,
			ducked: player.ducked,
			in_duck: player.in_duck,
			noclip: player.noclip,
			step_offset: player.step_offset,
			fall_velocity: player.fall_velocity,
			jump_penalty: player.jump_penalty,
			input,
			flycam,
			weapon_slot,
			weapon_clip,
			weapon_reserve,
			money,
			gravity: vars.gravity,
			max_speed: vars.maxspeed,
			step_size: vars.stepsize,
			tick_rate,
			accumulator_ms,
		}
	}

	fn spike_reason(self) -> Option<&'static str> {
		if self.frame_ms.max(self.frame_work_ms) >= FRAME_SPIKE_MS
			&& self.tick_max_ms >= self.tick_budget_ms
		{
			Some("frame_and_movement_tick")
		} else if self.frame_ms.max(self.frame_work_ms) >= FRAME_SPIKE_MS {
			Some("frame")
		} else if self.collision_ms >= self.tick_budget_ms {
			// Collision alone ate a whole tick's budget: the rest of the tick
			// and every catch-up tick wait on it, so this stutter is collision's.
			Some("collision_budget")
		} else if self.tick_max_ms >= self.tick_budget_ms {
			Some("movement_tick_budget")
		} else {
			None
		}
	}
}

struct Capture {
	reason: &'static str,
	until_ms: u64,
	samples: VecDeque<Sample>,
	truncated: bool,
}

struct Buffer {
	history: VecDeque<Sample>,
	history_truncated: bool,
	capture: Option<Capture>,
}

impl Buffer {
	fn new() -> Buffer {
		Buffer {
			history: VecDeque::new(),
			history_truncated: false,
			capture: None,
		}
	}

	fn observe(&mut self, mut sample: Sample) -> Option<Incident> {
		if let Some(last) = self.history.back() {
			if sample.elapsed_ms < last.elapsed_ms {
				sample.elapsed_ms = last.elapsed_ms;
			}
		}
		self.history.push_back(sample);
		while self
			.history
			.front()
			.is_some_and(|old| sample.elapsed_ms.saturating_sub(old.elapsed_ms) > HISTORY_MS)
		{
			self.history.pop_front();
		}
		while self.history.len() > MAX_HISTORY_SAMPLES {
			self.history.pop_front();
			self.history_truncated = true;
		}

		if let Some(capture) = &mut self.capture {
			if capture.samples.len() == MAX_CAPTURE_SAMPLES {
				capture.samples.pop_front();
				capture.truncated = true;
			}
			capture.samples.push_back(sample);
			if sample.elapsed_ms >= capture.until_ms {
				let capture = self.capture.take().expect("capture was active");
				return Some(Incident {
					reason: capture.reason,
					truncated: capture.truncated,
					samples: capture.samples,
				});
			}
		} else if let Some(reason) = sample.spike_reason() {
			self.capture = Some(Capture {
				reason,
				until_ms: sample.elapsed_ms.saturating_add(FUTURE_MS),
				samples: std::mem::take(&mut self.history),
				truncated: std::mem::take(&mut self.history_truncated),
			});
		}
		None
	}

	fn capture_remaining_ms(&self) -> Option<u64> {
		let capture = self.capture.as_ref()?;
		Some(
			capture
				.until_ms
				.saturating_sub(capture.samples.back().map_or(0, |s| s.elapsed_ms)),
		)
	}
}

struct Incident {
	reason: &'static str,
	truncated: bool,
	samples: VecDeque<Sample>,
}

pub struct Profiler {
	started: Instant,
	map_name: String,
	output_dir: PathBuf,
	next_id: u64,
	buffer: Buffer,
	sender: Option<SyncSender<(u64, String, Incident)>>,
	writer: Option<thread::JoinHandle<()>>,
}

impl Profiler {
	pub fn new(output_dir: PathBuf, map_name: String) -> io::Result<Profiler> {
		let (sender, receiver) = mpsc::sync_channel::<(u64, String, Incident)>(1);
		let writer_dir = output_dir.clone();
		let writer = thread::Builder::new()
			.name("game-perf-recorder".into())
			.spawn(move || {
				while let Ok((id, map_name, incident)) = receiver.recv() {
					if let Err(error) = write_incident(&writer_dir, id, &map_name, incident) {
						eprintln!("performance capture write failed: {error}");
					}
				}
			})?;
		Ok(Profiler {
			started: Instant::now(),
			map_name,
			output_dir,
			next_id: 0,
			buffer: Buffer::new(),
			sender: Some(sender),
			writer: Some(writer),
		})
	}

	pub fn observe(&mut self, mut sample: Sample) -> bool {
		sample.elapsed_ms = self.started.elapsed().as_millis().min(u64::MAX as u128) as u64;
		let was_capturing = self.buffer.capture_remaining_ms().is_some();
		if let Some(incident) = self.buffer.observe(sample) {
			let id = self.next_id;
			self.next_id = self.next_id.wrapping_add(1);
			match self
				.sender
				.as_ref()
				.expect("writer channel exists while profiler is active")
				.try_send((id, self.map_name.clone(), incident))
			{
				Ok(()) => eprintln!(
					"performance spike recorded; writing capture under {}",
					self.output_dir.display()
				),
				Err(TrySendError::Full(_)) => {
					eprintln!("performance capture dropped: previous capture is still being written")
				}
				Err(TrySendError::Disconnected(_)) => {
					eprintln!("performance capture dropped: writer thread stopped")
				}
			}
		}
		!was_capturing && self.buffer.capture_remaining_ms().is_some()
	}

	pub fn capture_remaining_ms(&self) -> Option<u64> {
		self.buffer.capture_remaining_ms()
	}
}

impl Drop for Profiler {
	fn drop(&mut self) {
		if let Some(capture) = self.buffer.capture.take() {
			let id = self.next_id;
			match self
				.sender
				.as_ref()
				.expect("writer channel exists while profiler is active")
				.send((
					id,
					self.map_name.clone(),
					Incident {
						reason: capture.reason,
						truncated: capture.truncated,
						samples: capture.samples,
					},
				)) {
				Ok(()) => eprintln!("saving partial performance capture before exit"),
				Err(_) => {
					eprintln!("partial performance capture dropped: writer thread stopped")
				}
			}
		}
		drop(self.sender.take());
		if self
			.writer
			.take()
			.is_some_and(|writer| writer.join().is_err())
		{
			eprintln!("performance capture writer thread panicked");
		}
	}
}

fn write_incident(
	output_dir: &Path,
	id: u64,
	map_name: &str,
	incident: Incident,
) -> io::Result<()> {
	fs::create_dir_all(output_dir)?;
	let wall_ms = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or(Duration::ZERO)
		.as_millis();
	let path = output_dir.join(format!("spike-{wall_ms}-{id}.csv"));
	let mut out = BufWriter::new(File::create(&path)?);
	writeln!(
		out,
		"# map={map_name:?},trigger={},pre_ms={HISTORY_MS},post_ms={FUTURE_MS},truncated={}",
		incident.reason, incident.truncated
	)?;
	writeln!(out, "elapsed_ms,frame_ms,frame_work_ms,tick_budget_ms,tick_total_ms,tick_max_ms,ticks,collision_casts,collision_overlaps,collision_ms,collision_max_ms,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,on_ground,ducked,in_duck,noclip,step_offset,fall_velocity,jump_penalty,input_forward,input_side,input_up,yaw,pitch,jump,duck,flycam,weapon_slot,weapon_clip,weapon_reserve,money,gravity,max_speed,step_size,tick_rate,accumulator_ms")?;
	for s in incident.samples {
		let Vec3 {
			x: px,
			y: py,
			z: pz,
		} = s.player_pos;
		let Vec3 {
			x: vx,
			y: vy,
			z: vz,
		} = s.player_velocity;
		writeln!(
			out,
			"{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4},{px:.4},{py:.4},{pz:.4},{vx:.4},{vy:.4},{vz:.4},{},{},{},{},{:.4},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.4}",
			s.elapsed_ms,
			s.frame_ms,
			s.frame_work_ms,
			s.tick_budget_ms,
			s.tick_total_ms,
			s.tick_max_ms,
			s.ticks,
			s.collision_casts,
			s.collision_overlaps,
			s.collision_ms,
			s.collision_max_ms,
			s.on_ground,
			s.ducked,
			s.in_duck,
			s.noclip,
			s.step_offset,
			s.fall_velocity,
			s.jump_penalty,
			s.input.forward,
			s.input.side,
			s.input.up,
			s.input.yaw,
			s.input.pitch,
			s.input.jump,
			s.input.duck,
			s.flycam,
			s.weapon_slot,
			s.weapon_clip,
			s.weapon_reserve,
			s.money,
			s.gravity,
			s.max_speed,
			s.step_size,
			s.tick_rate,
			s.accumulator_ms,
		)?;
	}
	out.flush()?;
	eprintln!("performance capture saved: {}", path.display());
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	fn sample(elapsed_ms: u64, frame_ms: f32, tick_max_ms: f32) -> Sample {
		Sample {
			elapsed_ms,
			frame_ms,
			frame_work_ms: 0.0,
			tick_budget_ms: 8.0,
			tick_total_ms: 0.0,
			tick_max_ms,
			ticks: 1,
			collision_casts: 0,
			collision_overlaps: 0,
			collision_ms: 0.0,
			collision_max_ms: 0.0,
			player_pos: Vec3::ZERO,
			player_velocity: Vec3::ZERO,
			on_ground: false,
			ducked: false,
			in_duck: false,
			noclip: false,
			step_offset: 0.0,
			fall_velocity: 0.0,
			jump_penalty: 0.0,
			input: Input::default(),
			flycam: false,
			weapon_slot: 0,
			weapon_clip: 0,
			weapon_reserve: 0,
			money: 0,
			gravity: 800.0,
			max_speed: 250.0,
			step_size: 18.0,
			tick_rate: 128.0,
			accumulator_ms: 0.0,
		}
	}

	#[test]
	fn capture_contains_fifteen_seconds_before_and_sixteen_after_spike() {
		let mut buffer = Buffer::new();
		for t in (0..=15_000).step_by(1000) {
			buffer.observe(sample(t, 16.0, 1.0));
		}
		buffer.observe(sample(16_000, 50.0, 1.0));
		for t in (17_000..=32_000).step_by(1000) {
			if let Some(incident) = buffer.observe(sample(t, 16.0, 1.0)) {
				assert_eq!(incident.reason, "frame");
				assert_eq!(incident.samples.front().unwrap().elapsed_ms, 1000);
				assert_eq!(incident.samples.back().unwrap().elapsed_ms, 32_000);
				assert_eq!(incident.samples.len(), 32);
				return;
			}
		}
		panic!("capture did not finish at 16 seconds after the spike");
	}

	#[test]
	fn movement_tick_over_budget_triggers_capture_without_frame_spike() {
		let mut buffer = Buffer::new();
		buffer.observe(sample(30_000, 16.0, 1.0));
		buffer.observe(sample(30_001, 16.0, 8.0));
		assert_eq!(buffer.capture_remaining_ms(), Some(FUTURE_MS));
	}

	#[test]
	fn collision_alone_over_a_tick_budget_triggers_collision_capture() {
		let mut buffer = Buffer::new();
		let mut slow_collision = sample(30_000, 16.0, 8.0);
		slow_collision.collision_ms = 8.0;
		slow_collision.collision_max_ms = 8.0;
		buffer.observe(slow_collision);
		assert_eq!(buffer.capture_remaining_ms(), Some(FUTURE_MS));
		for t in (31_000..=46_000).step_by(1000) {
			if let Some(incident) = buffer.observe(sample(t, 16.0, 1.0)) {
				assert_eq!(incident.reason, "collision_budget");
				assert_eq!(incident.samples.front().unwrap().collision_ms, 8.0);
				return;
			}
		}
		panic!("capture did not finish at 16 seconds after the spike");
	}

	#[test]
	fn writes_csv_state_and_collision_columns() {
		let dir = std::env::temp_dir().join(format!(
			"game-perf-test-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(UNIX_EPOCH)
				.unwrap()
				.as_nanos()
		));
		let incident = Incident {
			reason: "frame",
			truncated: false,
			samples: VecDeque::from([sample(100, 50.0, 1.0)]),
		};
		write_incident(&dir, 0, "test-map", incident).unwrap();

		let entry = fs::read_dir(&dir).unwrap().next().unwrap().unwrap();
		let csv = fs::read_to_string(entry.path()).unwrap();
		let mut lines = csv.lines();
		assert!(lines.next().unwrap().contains("trigger=frame"));
		let columns = lines.next().unwrap();
		assert!(columns.contains("collision_casts,collision_overlaps,collision_ms,collision_max_ms"));
		assert!(columns.contains("pos_x,pos_y,pos_z"));
		assert_eq!(
			lines.next().unwrap().split(',').count(),
			columns.split(',').count()
		);

		fs::remove_file(entry.path()).unwrap();
		fs::remove_dir(dir).unwrap();
	}
}
