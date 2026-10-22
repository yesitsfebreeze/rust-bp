//! Bounded headless workload for the profiling tools: scripted movement
//! ticks and bullet traces against Dust2, with channel timers and counters
//! the `benchmark`, `valgrind_run` and `channels` tools read. No window is
//! opened and the process exits when the passes are done, so wall-time and
//! instruction counts are well-defined.

use engine::linalg::vec3;
use gameplay::bullets;
use gameplay::cvars::{movement_vars, Cvars};
use gameplay::player::{Input, Player};
use gameplay::weapons;

/// Movement ticks simulated, 160 simulated seconds at the default rate.
const TICKS: usize = 20480;
/// Bullet traces fired, sweeping a full circle from the spawn.
const SHOTS: usize = 4096;

fn main() {
	// The daemon harness reads live channel data from this socket; serving it
	// once at startup is the framework contract, and the workload ends by
	// saving the final snapshot for the benchmark tool.
	if let Err(error) = daemon::channels::serve(&daemon::app_socket()) {
		eprintln!("app: {error}");
	}
	let map_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/maps/de_dust2.bsp");
	let map = match engine::bsp::load(&map_path) {
		Ok(map) => map,
		Err(e) => {
			eprintln!("{}: {e}", map_path.display());
			std::process::exit(1);
		}
	};
	let mut play = engine::physics::World::new(map.world);
	// GoldSrc gravity, the same constant the game loop sets.
	play.gravity = vec3(0.0, -800.0, 0.0);
	let cvars = Cvars::default();
	let vars = movement_vars(&cvars);
	let dt = 1.0 / cvars.sv_tickrate;

	// One pass of movement: run forward, jump every half second, duck every
	// other half, turning steadily. The player respawns on falling out.
	let mut player = Player::spawn(map.spawn);
	let phase = std::time::Instant::now();
	for i in 0..TICKS {
		let input = Input {
			forward: 250.0,
			side: (i / 64) as f32 * 6.0 % 50.0 - 25.0,
			up: 0.0,
			yaw: i as f32 * 0.7,
			pitch: -20.0 + (i / 32) as f32 * 3.0 % 40.0,
			jump: i % 64 == 0,
			duck: i % 128 >= 64,
		};
		let tick = daemon::channels::timer("bench.movement.tick");
		player.tick(input, &vars, &play.mesh, dt);
		tick.done();
		if player.origin.y < map.kill_y {
			player = Player::spawn(map.spawn);
		}
	}
	let movement_ms = phase.elapsed().as_secs_f64() * 1000.0;
	daemon::channels::set("bench.movement.ms", movement_ms);
	daemon::channels::set("bench.movement.ticks", TICKS as f64);

	// One pass of bullet traces: AK-47 penetration fire sweeping a circle
	// around the spawn eye, the same cast the game makes per shot.
	let mut noted = Vec::new();
	let table = weapons::load(&mut noted);
	let weapon = table
		.weapons
		.iter()
		.find(|w| w.name == "ak47")
		.expect("ak47 in the weapon table");
	let eye = Player::spawn(map.spawn).eye();
	let phase = std::time::Instant::now();
	let mut walls = 0u64;
	for i in 0..SHOTS {
		let yaw = i as f32 * 360.0 / SHOTS as f32;
		let (forward, _, _) = gameplay::player::movement::angle_vectors(yaw, -10.0);
		let shot = daemon::channels::timer("bench.bullets.fire");
		let fired = bullets::fire(
			&play.mesh,
			&map.surfaces,
			eye + vec3(0.0, 8.0, 0.0),
			forward,
			weapon,
			true,
		);
		shot.done();
		walls += fired.walls.len() as u64;
	}
	daemon::channels::set("bench.bullets.ms", phase.elapsed().as_secs_f64() * 1000.0);
	daemon::channels::set("bench.bullets.shots", SHOTS as f64);
	daemon::channels::set("bench.bullets.walls", walls as f64);

	eprintln!(
		"bench: {TICKS} movement ticks {movement_ms:.1}ms, {SHOTS} traces {:.1}ms, {walls} wall hits",
		phase.elapsed().as_secs_f64() * 1000.0
	);
	let _ = daemon::channels::save_requested_snapshot();
}
