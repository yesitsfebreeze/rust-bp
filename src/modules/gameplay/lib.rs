//! Counter-Strike gameplay: movement, weapons, combat, and arena state.
//!
//! WASD move - mouse look - Space or mouse wheel jump - Ctrl or C duck - Shift walk
//! F flycam - G throws a physics sphere, H clears them - ` or F1 console - Esc
//! opens 1.6's menu and frees the mouse, Resume Game grabs it again; alt-tab
//! frees it too, and focusing the window re-grabs it
//!
//! The map draws through `engine::gpu`: traced per pixel as voxels.
//!
//! Movement ticks at `sv_tickrate`; the view turns every frame from raw mouse counts
//! and the player is drawn between the last two ticks. The first argument names the
//! TOML config, by default `player.toml` beside this file; a missing one is created.
//! Dust2 loads by default; pass a `.bsp` path to load another map.

#![deny(warnings)]

// The profiling workload in `src/bin/engine_bench.rs` drives movement, bullet
// traces and the weapon table headless, so these stay public.
pub mod bullets;
pub mod cvars;
pub mod materials;
mod menu;
pub mod player;
mod profiler;
mod rng;
mod state;
mod viewmodel;
pub mod weapons;

fn root() -> &'static std::path::Path {
	std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
		.parent()
		.and_then(std::path::Path::parent)
		.and_then(std::path::Path::parent)
		.expect("gameplay crate must be under src/modules")
}

fn selected_map(args: &[String]) -> std::path::PathBuf {
	args
		.iter()
		.find(|arg| arg.to_ascii_lowercase().ends_with(".bsp"))
		.map(std::path::PathBuf::from)
		.unwrap_or_else(|| root().join("assets").join("maps").join("de_dust2.bsp"))
}

use engine::console::{Action, Console};
use engine::cvar::Registry;
use engine::gpu::{self, Color, Gfx, Key, MouseButton, View, Window};
use engine::linalg::{vec3, Quat, Vec3};
use engine::{bsp, lightmap, physics};
use player::movement::{angle_vectors, clamp_wish};
use player::{Input, Player};

use crate::bullets::Shot;
use crate::cvars::{movement_vars, Cvars};
use crate::materials::SurfaceMaterial;
use crate::weapons::{load, load_clip, Armed, Table};
use audio::{Binaural, Clip};

/// How long a fired bullet's tracer and holes stay drawn, seconds.
const SHOT_LIFE: f32 = 2.0;
/// How long a shot's muzzle flash lights the world, seconds, and its
/// light: warm, reaching this far, world units.
const FLASH_LIFE: f32 = 0.05;
const FLASH_COLOR: Vec3 = Vec3 {
	x: 6.0,
	y: 4.2,
	z: 2.0,
};
const FLASH_REACH: f32 = 400.0;
/// The player's shape for shadows: a capsule slimmer than the hull, as a
/// body is narrower than its collision box.
const BODY_RADIUS: f32 = 11.0;
/// Spheres in the physics playground at most; past it, the oldest goes.
const BODY_CAP: usize = 64;

/// Longest frame time simulated; a longer stall drops time instead of piling up ticks.
const MAX_FRAME: f32 = 0.25;
const TEXT: Color = Color::new(0.95, 0.96, 0.97, 1.0);
const MUTED: Color = Color::new(0.75, 0.78, 0.8, 1.0);

/// A horizontal `fov` at 4:3 as the vertical angle a projection takes:
/// GoldSrc's widescreen keeps the vertical extent of the 4:3 frame.
pub fn vertical_fov(fov: f32) -> f32 {
	2.0 * ((fov.clamp(10.0, 170.0).to_radians() * 0.5).tan() * 0.75).atan()
}

/// Sets `cvars` from the config, writing the defaults there first if it is missing.
pub fn load_config(console: &mut Console, cvars: &mut Cvars) {
	let path = console.config().to_owned();
	if !path.exists() {
		match std::fs::write(&path, cvars.to_toml()) {
			Ok(()) => console.print(format!("wrote defaults to {}", path.display())),
			Err(e) => console.print(format!("{}: {e}", path.display())),
		}
	}
	console.exec(&path, cvars);
}

/// Edits the console line from this frame's keys; returns whether it was submitted.
pub fn type_into(console: &mut Console, keys: &gpu::Input) -> bool {
	for &c in keys.chars() {
		if !c.is_control() && c != '`' && c != '^' {
			console.input.push(c);
		}
	}
	if keys.pressed(Key::Backspace) {
		console.input.pop();
	}
	if keys.pressed(Key::Tab) {
		console.complete::<Cvars>();
	}
	if keys.pressed(Key::ArrowUp) {
		console.recall(true);
	}
	if keys.pressed(Key::ArrowDown) {
		console.recall(false);
	}
	keys.pressed(Key::Enter) || keys.pressed(Key::NumpadEnter)
}

pub fn draw_console(gfx: &mut Gfx, console: &Console) {
	let (width, height) = gfx.size();
	let h = height * 0.45;
	gfx.rect(0.0, 0.0, width, h, Color::new(0.05, 0.07, 0.09, 0.88));
	let size = 20.0;
	let mut y = h - 12.0;
	gfx.text(&format!("] {}_", console.input), 12.0, y, size, TEXT);
	for line in console.lines().iter().rev() {
		y -= size;
		if y < size {
			break;
		}
		gfx.text(line, 12.0, y, size, MUTED);
	}
}

/// The heads-up display reads like 1.6's: the weapon bottom-left, the money and
/// ammo top-right. Every argument is one HUD line's data.
#[allow(clippy::too_many_arguments)]
fn hud(
	gfx: &mut Gfx,
	fps: u32,
	cvars: &Cvars,
	player: &Player,
	fly: bool,
	ticks: u32,
	weapon: &crate::weapons::Weapon,
	armed: &Armed,
	money: u32,
	report: &str,
) {
	let (width, height) = gfx.size();
	let mut y = 28.0;
	let mut line = |gfx: &mut Gfx, text: String| {
		gfx.text(&text, 16.0, y, 24.0, TEXT);
		y += 24.0;
	};
	if cvars.cl_showfps != 0.0 {
		line(gfx, format!("{fps} fps  {ticks} ticks/s"));
		// GPU milliseconds per stage, where the adapter can time them.
		let stages: Vec<String> = gfx
			.timings()
			.iter()
			.map(|(stage, ms)| format!("{stage} {ms:.2}"))
			.collect();
		if !stages.is_empty() {
			let total: f32 = gfx.timings().iter().map(|(_, ms)| ms).sum();
			line(gfx, format!("gpu {total:.2} ms: {}", stages.join("  ")));
		}
	}
	if gfx.view_mode() != "lit" {
		line(gfx, format!("view: {} (F5)", gfx.view_mode()));
	}
	if cvars.cl_showspeed != 0.0 {
		let state = if fly {
			"flycam"
		} else {
			match (player.noclip, player.on_ground, player.ducked) {
				(true, _, _) => "noclip",
				(_, true, true) => "ducked",
				(_, true, false) => "ground",
				(_, false, true) => "air, ducked",
				(_, false, false) => "air",
			}
		};
		line(gfx, format!("{:.1} u/s  {state}", player.speed()));
	}
	gfx.text(
		"WASD move  Space/wheel jump  Ctrl duck  Shift walk  mouse 1 fire  mouse 2 zoom/burst  1-8 weapon  B buy  R reload  F flycam  G throw ball  H clear balls  ` console",
		16.0,
		height - 16.0,
		20.0,
		MUTED,
	);
	line(
		gfx,
		format!(
			"{}  (1-8: knife, glock, usp, deagle, mp5, ak47, m4a1, awp)",
			weapon.name
		),
	);
	if !report.is_empty() {
		line(gfx, report.to_string());
	}
	// The economy and ammo read from the top-right, as in 1.6's HUD.
	gfx.text(&format!("${money}"), width - 90.0, 28.0, 24.0, TEXT);
	let ammo = if weapon.melee {
		"-".to_string()
	} else {
		format!("{} / {}", armed.clip, armed.reserve)
	};
	gfx.text(&ammo, width - 90.0, 52.0, 24.0, TEXT);
	if cvars.cl_crosshair != 0.0 {
		let (cx, cy) = (width / 2.0, height / 2.0);
		for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
			gfx.line(
				cx + dx * 4.0,
				cy + dy * 4.0,
				cx + dx * 12.0,
				cy + dy * 12.0,
				2.0,
				Color::GREEN,
			);
		}
	}
}

/// One weapon's sounds, loaded once from the 1.6 assets as mixer clips; a
/// synthesized report stands in when the files are missing. Each field is the
/// clip's index in the audio engine.
struct Gun {
	fire: Vec<usize>,
	dry: Option<usize>,
	deploy: Option<usize>,
	/// Reload sounds at their fraction of the reload, as `weapons.toml`
	/// stages them (`file@fraction`).
	reload: Vec<(f32, usize)>,
	zoom: Option<usize>,
	/// The knife's stab swing and its sound when it meets a wall.
	stab: Option<usize>,
	wall: Option<usize>,
}

/// The gun in hand sits low-right ahead of the eye, as the drawn viewmodel.
fn muzzle(eye: Vec3, forward: Vec3, right: Vec3, up: Vec3) -> Vec3 {
	eye + forward * 8.0 - right * 3.0 - up * 2.5
}

/// Spatial weapon audio: the frame's sounds queue with world positions, then
/// play through the HRTF mixer at the listener's ears and out the device.
struct GameAudio {
	engine: Option<Binaural>,
	device: audio::Device,
	queue: Vec<(usize, [f32; 3], f32)>,
}

impl GameAudio {
	/// Loads the HRTF dataset beside the clips and opens the output device; a
	/// missing or mismatched dataset leaves the game running silently.
	fn new(root: &std::path::Path, clips: Vec<Clip>) -> GameAudio {
		let engine = match std::fs::read(root.join("assets").join("audio").join("hrtf.mhr")) {
			Ok(bytes) => match Binaural::new(&bytes, audio::DISTANCE_SCALE, clips) {
				Ok(engine) => Some(engine),
				Err(e) => {
					eprintln!("HRTF audio unavailable: {e}");
					None
				}
			},
			Err(e) => {
				eprintln!("assets/audio/hrtf.mhr: {e}");
				None
			}
		};
		GameAudio {
			engine,
			device: audio::Device::open(),
			queue: Vec::new(),
		}
	}

	fn play(&mut self, clip: usize, position: Vec3, volume: f32) {
		self
			.queue
			.push((clip, [position.x, position.y, position.z], volume));
	}

	/// One frame: place the listener at the view, flush the queue, push the mix.
	fn frame(&mut self, eye: Vec3, forward: Vec3, up: Vec3) {
		let GameAudio {
			engine,
			device,
			queue,
		} = self;
		if let Some(mixer) = engine {
			mixer.listener(
				[eye.x, eye.y, eye.z],
				[forward.x, forward.y, forward.z],
				[up.x, up.y, up.z],
			);
			for (clip, position, volume) in std::mem::take(queue) {
				mixer.play(clip, position, volume);
			}
		} else {
			queue.clear();
		}
		device.update(engine.as_mut());
	}
}

/// Draws the weapon in `to`: the old magazine goes back to the store, the new
/// one comes out of it, the draw animates and sounds, and 1.6's speed cap
/// follows the weapon.
#[allow(clippy::too_many_arguments)]
fn holster(
	slot: &mut usize,
	armed: &mut Armed,
	mags: &mut [(u32, u32)],
	table: &Table,
	viewmodels: &mut [Option<viewmodel::Viewmodel>],
	guns: &[Gun],
	to: usize,
	now: f32,
	console: &mut Console,
	audio: &mut GameAudio,
	at: Vec3,
) {
	mags[*slot] = (armed.clip, armed.reserve);
	let w = &table.weapons[to];
	armed.switch(w, mags[to].0, mags[to].1, now);
	if let Some(vm) = &mut viewmodels[to] {
		vm.start("draw", now, false);
	}
	if let Some(s) = guns[to].deploy {
		audio.play(s, at, 0.7);
	}
	console.print(format!("{} ({} of 8)", w.name, to + 1));
	*slot = to;
}

/// Opens the window and plays until the game quits or the window closes.
pub fn run() {
	let mut window = match Window::open("player", 1600, 900) {
		Ok(window) => window,
		Err(e) => {
			eprintln!("{e}");
			std::process::exit(1);
		}
	};
	play(&mut window, None, 0, &mut || false);
}

/// One generation of the game in an open window. `reload` is asked once per
/// frame; when it answers true the game returns its [`state::Carried`] bytes
/// for the next generation, which passes them back as `carried`. `generation`
/// counts hot reloads, 0 when there are none. None when the game quits or
/// the window closes.
pub fn play(
	window: &mut Window,
	carried: Option<&[u8]>,
	generation: u32,
	reload: &mut dyn FnMut() -> bool,
) -> Option<Vec<u8>> {
	let args: Vec<String> = std::env::args().skip(1).collect();
	if args.iter().any(|arg| arg == "--arena") {
		eprintln!("--arena is no longer supported; the game now loads Dust2 by default");
		std::process::exit(2);
	}
	let map_path = selected_map(&args);
	let config = args
		.iter()
		.find(|a| !a.to_ascii_lowercase().ends_with(".bsp"))
		.map(std::path::PathBuf::from)
		.unwrap_or_else(|| root().join("player.toml"));
	let mut console = Console::new(config);
	let mut cvars = Cvars::default();
	load_config(&mut console, &mut cvars);
	console.print("type help for commands");

	let map = match bsp::load(&map_path) {
		Ok(map) => map,
		Err(e) => {
			eprintln!("{e}");
			std::process::exit(1);
		}
	};
	console.print(format!(
		"{}: {} faces, {} triangles, spawn {:?}",
		map_path.display(),
		map.render.faces.len(),
		map.world.tris().len(),
		map.spawn
	));
	// The sun and sky the map's entities describe; the raycast voxel field
	// stores each voxel's light from them as the view draws it.
	let light = lightmap::config(&map_path).unwrap_or_else(|e| {
		console.print(format!("map light unreadable, using defaults: {e}"));
		engine::lightbake::Config::default()
	});
	// The map's own baked light: direct sun, sky and bounces, 2 units per
	// texel, cached beside the map after the first bake. The surface pass
	// samples it as lightmaps, so its shadows and ambient light are the
	// bake's, sharp.
	let (baked, from_cache) = match lightmap::bake_or_load(&map, &map_path) {
		Ok(baked) => baked,
		Err(e) => {
			eprintln!("{e}");
			std::process::exit(1);
		}
	};
	console.print(if from_cache {
		format!(
			"light bake: loaded from {}",
			map_path.with_extension("lightbake").display()
		)
	} else {
		"light bake: baked fresh".into()
	});
	let traced = window.gfx().load_world(&map, &light, &baked);
	console.print(format!("traced world: {} triangles", traced.triangles));
	// The map's collision mesh, with the physics playground's spheres in it.
	let (mut play, surfaces, spawn, spawn_yaw, spawn_pitch, kill_y) = (
		physics::World::new(map.world),
		map.surfaces,
		map.spawn,
		map.spawn_yaw,
		map.spawn_pitch,
		map.kill_y,
	);
	// GoldSrc gravity, so playground spheres fall like the player does.
	play.gravity = vec3(0.0, -800.0, 0.0);
	let mut player = Player::spawn(spawn);
	let mut previous = player;
	// The free camera on F: where it floats, or None while the player plays.
	let (mut fly, mut fly_eye) = (false, player.eye());
	let (mut yaw, mut pitch) = (spawn_yaw, spawn_pitch);
	// Esc's menu; while it is open the cursor is free and the game takes no input.
	let mut menu = menu::Menu::new();
	// The 1.6 weapon table, its sounds and viewmodels, and the gun in hand.
	let table = {
		let mut noted = Vec::new();
		let t = load(&mut noted);
		for line in noted {
			console.print(line);
		}
		t
	};
	let cstrike = viewmodel::cstrike().filter(|p| p.is_dir());
	let sound_dir = cstrike.as_ref().map(|p| p.join("sound").join("weapons"));
	let mut viewmodels: Vec<Option<viewmodel::Viewmodel>> = table
		.weapons
		.iter()
		.map(|w| {
			cstrike
				.as_ref()
				.and_then(|dir| viewmodel::Viewmodel::load(window.gfx(), dir, w))
		})
		.collect();
	// The reload takes at least as long as its animation, the deploy as the
	// draw's.
	let mut table = table;
	for (i, w) in table.weapons.iter_mut().enumerate() {
		if let Some(vm) = viewmodels[i].as_ref() {
			w.deploy = w.deploy.max(vm.draw_duration());
			if w.reload > 0.0 {
				w.reload = w.reload.max(vm.reload_duration());
			}
		}
	}
	let mut guns = Vec::with_capacity(table.weapons.len());
	// The clips all load into the one list the mixer owns; each gun keeps indices.
	let mut clips: Vec<Clip> = Vec::new();
	for w in &table.weapons {
		let dir = sound_dir.as_deref();
		let mut at = |file: &str| -> usize {
			clips.push(load_clip(dir, file, w));
			clips.len() - 1
		};
		let fire: Vec<usize> = w.sounds.fire.iter().map(|n| at(n)).collect();
		let mut reload = Vec::new();
		for s in &w.sounds.reload {
			let (file, frac) = s.split_once('@').unwrap_or((s.as_str(), "0"));
			reload.push((frac.parse::<f32>().unwrap_or(0.0), at(file)));
		}
		guns.push(Gun {
			fire,
			dry: w.sounds.dry.as_deref().map(&mut at),
			deploy: w.sounds.deploy.as_deref().map(&mut at),
			reload,
			zoom: w
				.zoom
				.as_ref()
				.and_then(|z| z.sound.as_deref())
				.map(&mut at),
			stab: w.sounds.stab.as_deref().map(&mut at),
			wall: w.sounds.wall.as_deref().map(at),
		});
	}
	// Binaural HRTF audio: the clips above, convolved per voice against the
	// listener's ears.
	let mut audio = GameAudio::new(root(), clips);
	if let Some(engine) = &audio.engine {
		console.print(format!("binaural HRTF audio: {}", engine.stats()));
	}
	if cstrike.is_some() {
		console.print("1.6 models and sounds loaded".to_string());
	} else {
		console.print("no 1.6 assets; drawing boxes with synthesized sound".to_string());
	}
	// The economy: 1.6's starting money from the weapon table.
	let mut money = table.money.0;
	// The player owns the knife and the glock, as 1.6 spawns; B buys the rest.
	let mut owned = [true, true, false, false, false, false, false, false];
	let mut buy_menu = false;
	// Each weapon keeps its own magazine in the store; the gun in hand holds
	// its loaded state.
	let mut mags: Vec<(u32, u32)> = table.weapons.iter().map(|w| (w.clip, w.reserve)).collect();
	let mut slot = 1;
	let mut armed = Armed::new(&table.weapons[slot], 0.0);
	if let Some(vm) = &mut viewmodels[slot] {
		vm.start("draw", 0.0, false);
	}
	cvars.player_maxspeed = table.weapons[slot].maxspeed;
	let (mut now, mut recoil) = (0.0f32, 0.0f32);
	// A hot reload picks up where the previous generation left off.
	if let Some(bytes) = carried {
		match state::Carried::restore(bytes) {
			Ok(c) => {
				(player, previous, yaw, pitch) = (c.player, c.previous, c.yaw, c.pitch);
				(fly, fly_eye, money, now) = (c.fly, c.fly_eye, c.money, c.now);
				console.open = c.console_open;
				for (owned, carried) in owned.iter_mut().zip(&c.owned) {
					*owned = *carried;
				}
				for (mag, carried) in mags.iter_mut().zip(&c.mags) {
					*mag = *carried;
				}
				for (name, value) in &c.cvars {
					let _ = cvars.set(name, *value);
				}
				if c.slot < table.weapons.len() {
					slot = c.slot;
				}
				armed = Armed::new(&table.weapons[slot], now);
				if let Err(e) = armed.restore(&c.armed) {
					console.print(format!(
						"hot reload: fresh {}: {e}",
						table.weapons[slot].name
					));
				}
				if let Some(vm) = &mut viewmodels[slot] {
					vm.start("draw", now, false);
				}
				cvars.player_maxspeed = table.weapons[slot].maxspeed;
			}
			Err(e) => console.print(format!("hot reload: starting fresh: {e}")),
		}
	}
	let mut shots: Vec<(Shot, f32)> = Vec::new();
	// The muzzle flash's time left, and where it is.
	let mut flash: (f32, Vec3) = (0.0, Vec3::ZERO);
	let mut report = String::new();
	window.grab_cursor(true);
	let mut accumulator = 0.0;
	// Presses seen between ticks, so a tap shorter than a tick still counts.
	let mut jump_pressed = false;
	let (mut tick_count, mut ticks_per_second, mut second) = (0, 0, 0.0);
	let map_name = map_path
		.file_name()
		.unwrap_or_default()
		.to_string_lossy()
		.into_owned();
	let mut profiler =
		match profiler::Profiler::new(root().join("logs").join("performance"), map_name) {
			Ok(profiler) => {
				console.print("automatic performance spike capture armed".to_string());
				Some(profiler)
			}
			Err(error) => {
				eprintln!("automatic performance capture unavailable: {error}");
				None
			}
		};

	loop {
		// Between frames, a hot reload hands the state to the next generation.
		if reload() {
			return Some(
				state::Carried {
					player,
					previous,
					yaw,
					pitch,
					fly,
					fly_eye,
					slot,
					money,
					owned: owned.to_vec(),
					mags,
					armed: armed.save(),
					now,
					console_open: console.open,
					cvars: Cvars::NAMES
						.iter()
						.filter_map(|name| Some((name.to_string(), cvars.get(name)?)))
						.collect(),
				}
				.save(),
			);
		}
		let frame_time = window.frame_time();
		let keys = window.input().clone();
		let frame_interval_ms = frame_time * 1000.0;
		let frame_started = std::time::Instant::now();
		if keys.pressed(Key::Backquote) || keys.pressed(Key::F1) {
			console.open = !console.open;
		}
		// F5 steps through what the trace shows: lit, then its debug views.
		if keys.pressed(Key::F5) {
			let gfx = window.gfx();
			gfx.cycle_view_mode();
			console.print(format!("view {}", gfx.view_mode()));
		}
		let typing = console.open;
		if typing {
			if type_into(&mut console, &keys) {
				for action in console.submit(&mut cvars) {
					match action {
						Action::Quit => return None,
						Action::Respawn => {
							player = Player::spawn(spawn);
							previous = player;
						}
						Action::Noclip => {
							player.noclip = !player.noclip;
							console.print(format!(
								"noclip {}",
								if player.noclip { "on" } else { "off" }
							));
						}
					}
				}
			}
			if keys.pressed(Key::Escape) {
				console.open = false;
			}
		} else if keys.pressed(Key::Escape) && !buy_menu {
			menu.escape(&mut cvars);
		} else if !menu.open && keys.pressed(Key::KeyF) {
			fly = !fly;
			if fly {
				fly_eye = player.eye();
			}
			console.print(format!("flycam {}", if fly { "on" } else { "off" }));
		}
		let typing = typing || menu.open;
		// B is the buy menu; there the number keys buy, elsewhere they switch.
		if !typing && keys.pressed(Key::KeyB) {
			buy_menu = !buy_menu;
		}
		if buy_menu && (keys.pressed(Key::Escape) || typing) {
			buy_menu = false;
		}
		// The drawn weapon's report comes from where it sits ahead of the eye.
		let (draw_forward, draw_right, draw_up) = angle_vectors(yaw, pitch);
		let draw_at = muzzle(player.eye(), draw_forward, draw_right, draw_up);
		for (i, key) in [
			Key::Digit1,
			Key::Digit2,
			Key::Digit3,
			Key::Digit4,
			Key::Digit5,
			Key::Digit6,
			Key::Digit7,
			Key::Digit8,
		]
		.iter()
		.enumerate()
		{
			if !keys.pressed(*key) {
				continue;
			}
			if buy_menu {
				let w = &table.weapons[i];
				let rich = cvars.sv_infinite_money != 0.0;
				if owned[i] {
					// Buying a gun already owned fills its ammo, as 1.6's
					// buyammo does.
					if w.ammo_price > 0 && (rich || money >= w.ammo_price) && mags[i].1 < w.reserve {
						if !rich {
							money -= w.ammo_price;
						}
						mags[i].1 = w.reserve;
						console.print(format!("{} ammo full (${})", w.ammo_name, w.ammo_price));
					}
				} else if rich || money >= w.price {
					if !rich {
						money -= w.price;
					}
					owned[i] = true;
					mags[i] = (w.clip, w.reserve);
					console.print(format!("bought {} for ${}", w.name, w.price));
					buy_menu = false;
					holster(
						&mut slot,
						&mut armed,
						&mut mags,
						&table,
						&mut viewmodels,
						&guns,
						i,
						now,
						&mut console,
						&mut audio,
						draw_at,
					);
				} else {
					console.print(format!("${} needed for {}", w.price, w.name));
				}
			} else if i != slot && owned[i] {
				holster(
					&mut slot,
					&mut armed,
					&mut mags,
					&table,
					&mut viewmodels,
					&guns,
					i,
					now,
					&mut console,
					&mut audio,
					draw_at,
				);
			} else if i != slot {
				console.print(format!("{} not owned; B to buy", table.weapons[i].name));
			}
		}
		let looking = !typing && keys.focused();
		window.grab_cursor(looking);

		// Look: raw counts every frame, never waiting for a tick.
		let (dx, dy) = keys.mouse_delta();
		if looking {
			let scale = cvars.sensitivity;
			yaw = (yaw + dx * scale * cvars.m_yaw).rem_euclid(360.0);
			pitch = (pitch + dy * scale * cvars.m_pitch).clamp(-89.0, 89.0);
		}

		let down = |k: Key| !typing && keys.down(k);
		jump_pressed |= !typing && (keys.pressed(Key::Space) || keys.wheel() != 0.0);
		let axis = |pos: Key, neg: Key, p: f32, n: f32| {
			(if down(pos) { p } else { 0.0 }) - (if down(neg) { n } else { 0.0 })
		};
		let duck = down(Key::ControlLeft) || down(Key::KeyC);
		let vars = movement_vars(&cvars);
		let (forward, side, up) = clamp_wish(
			axis(
				Key::KeyW,
				Key::KeyS,
				cvars.cl_forwardspeed,
				cvars.cl_backspeed,
			),
			axis(Key::KeyD, Key::KeyA, cvars.cl_sidespeed, cvars.cl_sidespeed),
			if player.noclip {
				(if down(Key::Space) {
					cvars.cl_upspeed
				} else {
					0.0
				}) - (if duck { cvars.cl_upspeed } else { 0.0 })
			} else {
				0.0
			},
			down(Key::ShiftLeft),
			vars.maxspeed,
			cvars.cl_movespeedkey,
		);
		let input = Input {
			forward,
			side,
			up,
			yaw,
			pitch,
			jump: down(Key::Space) || jump_pressed,
			duck,
		};

		let dt = 1.0 / cvars.sv_tickrate.clamp(10.0, 1000.0);
		// The held weapon caps the run speed, the scope further still, as 1.6.
		let weapon = &table.weapons[slot];
		cvars.player_maxspeed = if armed.zoom > 0 {
			weapon
				.zoom
				.as_ref()
				.map_or(weapon.maxspeed, |z| z.scoped_speed)
		} else {
			weapon.maxspeed
		};
		accumulator += frame_time.min(MAX_FRAME);
		// The flycam moves per frame along the look direction; the player stands
		// still and is drawn where it was.
		if fly {
			let (forward, right, _) = angle_vectors(yaw, pitch);
			let mut toward = Vec3::ZERO;
			if down(Key::KeyW) {
				toward += forward;
			}
			if down(Key::KeyS) {
				toward -= forward;
			}
			if down(Key::KeyD) {
				toward += right;
			}
			if down(Key::KeyA) {
				toward -= right;
			}
			if down(Key::Space) {
				toward.y += 1.0;
			}
			if duck {
				toward.y -= 1.0;
			}
			if toward.length2() > 0.0 {
				let speed = cvars.cl_forwardspeed * if down(Key::ShiftLeft) { 2.0 } else { 1.0 };
				fly_eye += toward.normalize() * speed * frame_time.min(MAX_FRAME);
			}
		}
		while accumulator >= dt {
			accumulator -= dt;
			previous = player;
			if !fly {
				if let Some(profiler) = &mut profiler {
					player::movement::begin_collision_query_count();
					let tick_started = std::time::Instant::now();
					player.tick(input, &vars, &play.mesh, dt);
					let elapsed_ms = tick_started.elapsed().as_secs_f32() * 1000.0;
					let queries = player::movement::take_collision_query_count();
					if profiler.observe(profiler::Sample::new(
						0.0,
						frame_started.elapsed().as_secs_f32() * 1000.0,
						dt * 1000.0,
						elapsed_ms,
						elapsed_ms,
						1,
						queries.casts,
						queries.overlaps,
						queries.nanos as f32 / 1e6,
						queries.longest_nanos as f32 / 1e6,
						&player,
						input,
						fly,
						slot,
						armed.clip,
						armed.reserve,
						money,
						&vars,
						cvars.sv_tickrate,
						accumulator * 1000.0,
					)) {
						console
							.print("movement tick spike detected; recording 16 seconds of follow-up".to_string());
					}
				} else {
					player.tick(input, &vars, &play.mesh, dt);
				}
			}
			jump_pressed = false;
			tick_count += 1;
		}
		second += frame_time;
		if second >= 1.0 {
			(ticks_per_second, tick_count, second) = (tick_count, 0, second - 1.0);
			// The stage times on the console too, for profiling runs.
			let stages: Vec<String> = window
				.gfx()
				.timings()
				.iter()
				.map(|(stage, ms)| format!("{stage} {ms:.2}"))
				.collect();
			eprintln!("{ticks_per_second} ticks/s  gpu: {}", stages.join("  "));
		}
		if player.origin.y < kill_y {
			player = Player::spawn(spawn);
			previous = player;
		}

		let alpha = if cvars.cl_interp != 0.0 {
			accumulator / dt
		} else {
			1.0
		};
		let shown = previous.interpolate(&player, alpha);
		// The view kicks with the punch and bullets leave along it, as in 1.6.
		// The landing roll turns right and up together about forward, so the
		// basis stays orthonormal.
		let (forward, right, up) = angle_vectors(yaw + armed.punch.1, pitch + armed.punch.0);
		let roll = Quat::from_axis_angle(forward, shown.punch_roll.to_radians());
		let (right, up) = (roll.rotate(right), roll.rotate(up));
		let eye = if fly { fly_eye } else { shown.eye() };
		// `fov` is horizontal at 4:3, as in GoldSrc; the scope narrows it.
		let fov = if armed.zoom > 0 {
			armed.fov(weapon)
		} else {
			cvars.fov
		};
		let fovy = vertical_fov(fov);

		// Fire: the trigger within the cycle, from the eye, along the punched
		// view through the heat's cone; the reload stages its sounds, and the
		// glock's burst rounds fire on their own schedule.
		let frame = frame_time.min(MAX_FRAME);
		now += frame;
		recoil *= (-10.0 * frame).exp();
		let ease = (-8.0 * frame).exp();
		armed.punch.0 *= ease;
		armed.punch.1 *= ease;
		let tick = armed.tick(weapon, now);
		for (frac, s) in &guns[slot].reload {
			if tick.reload_sounds.contains(frac) {
				audio.play(*s, muzzle(eye, forward, right, up), 0.7);
			}
		}
		// R reloads: staged sounds above, the animation here.
		if looking && keys.pressed(Key::KeyR) {
			let was = armed.reload_at.is_none();
			armed.start_reload(weapon, now);
			if was && armed.reload_at.is_some() {
				if let Some(vm) = &mut viewmodels[slot] {
					vm.start("reload", now, false);
				}
			}
		}
		// Mouse 2 is the weapon's second use: the AWP's scope, the glock's
		// burst mode.
		if looking && keys.clicked(MouseButton::Right) {
			if weapon.zoom.is_some() {
				if let Some(fov) = armed.toggle_zoom(weapon, now) {
					if let Some(s) = guns[slot].zoom {
						audio.play(s, muzzle(eye, forward, right, up), 0.5);
					}
					console.print(if fov > 0.0 {
						format!("zoom {fov}")
					} else {
						"zoom off".into()
					});
				}
			} else if weapon.burst.is_some() && armed.toggle_burst(weapon, now) {
				console.print(if armed.burst_mode {
					"burst on".to_string()
				} else {
					"burst off".to_string()
				});
			} else if weapon.melee {
				// The knife's right-click arms a stab, which the next fire
				// below swings on its own cycle.
				armed.want_stab(weapon, now);
			}
		}
		let pulled = looking
			&& !buy_menu
			&& (keys.clicked(MouseButton::Left) || (weapon.auto && keys.button_down(MouseButton::Left)));
		let will_dry = pulled && armed.can_fire(now) && armed.clip == 0;
		if tick.burst || pulled {
			let grounded = !fly && player.on_ground;
			// An armed stab is the same swing with the stab's damage and cycle.
			let stab = armed.stab;
			let stab_w;
			let used = if stab {
				stab_w = crate::weapons::Weapon {
					damage: weapon.stab_damage,
					cycle: weapon.stab_cycle,
					..weapon.clone()
				};
				&stab_w
			} else {
				weapon
			};
			match armed.fire(used, now, grounded, player.ducked, player.speed()) {
				Some(spread) => {
					recoil = 1.0;
					armed.stab = false;
					if let Some(vm) = &mut viewmodels[slot] {
						vm.start(if stab { "stab" } else { "shoot" }, now, armed.clip == 0);
					}
					let dir = used.dir(forward, right, up, spread);
					let shot = bullets::fire(
						&play.mesh,
						&surfaces,
						eye,
						dir,
						used,
						cvars.sv_penetration != 0.0,
					);
					let distance = (shot.end - shot.origin).length();
					let through: Vec<String> = shot
						.walls
						.iter()
						.map(|w| {
							format!(
								"{} {}u",
								w.surface.material().name,
								(w.exit - w.entry).length().round()
							)
						})
						.collect();
					report = if through.is_empty() {
						format!("{}: {:.0} dmg at {:.0}u", used.name, shot.damage, distance)
					} else {
						format!(
							"{}: through {}, {:.0} dmg after {:.0}u",
							used.name,
							through.join(", "),
							shot.damage,
							distance
						)
					};
					if used.melee && distance < used.range - 1.0 {
						if let Some(s) = guns[slot].wall {
							// The knife met the wall at the hit, not in the hand.
							audio.play(s, shot.end, 0.9);
						}
					}
					let g = &guns[slot];
					if stab {
						if let Some(s) = g.stab {
							audio.play(s, muzzle(eye, forward, right, up), 1.0);
						}
					} else if !g.fire.is_empty() {
						let pick = rng::index(g.fire.len());
						audio.play(g.fire[pick], muzzle(eye, forward, right, up), 1.0);
					}
					if !stab {
						flash = (FLASH_LIFE, muzzle(eye, forward, right, up));
					}
					shots.push((shot, 0.0));
				}
				None if will_dry => {
					if let Some(s) = guns[slot].dry {
						audio.play(s, muzzle(eye, forward, right, up), 0.6);
					}
				}
				None => {}
			}
		}
		for (_, age) in &mut shots {
			*age += frame;
		}
		shots.retain(|(_, age)| *age < SHOT_LIFE);

		// The playground: G throws a sphere ahead of the eye, H clears them.
		// They collide with the map and each other and rest on it asleep.
		if looking && keys.pressed(Key::KeyG) {
			let mut body = physics::Body::default();
			body.pos = eye + forward * 24.0;
			body.vel = player.velocity + forward * 700.0;
			body.radius = 8.0;
			body.rotates = true;
			body.friction = 0.5;
			if play.bodies.len() >= BODY_CAP {
				play.bodies.remove(0);
			}
			play.bodies.push(body);
		}
		if looking && keys.pressed(Key::KeyH) {
			play.bodies.clear();
		}
		play.step(frame, true);
		play.bodies.retain(|b| b.pos.y > kill_y);
		let spheres: Vec<(Vec3, f32)> = play.bodies.iter().map(|b| (b.pos, b.radius)).collect();

		let fps = window.fps();
		let gfx = window.gfx();
		gfx.set_exposure(cvars.r_exposure);
		gfx.set_bodies(&spheres);
		gfx.set_voxel(cvars.r_voxel);
		gfx.set_shell(cvars.r_shell != 0.0);
		gfx.set_surfaces(cvars.r_surface != 0.0);
		gfx.set_wire(cvars.r_wire != 0.0);
		gfx.set_lod(cvars.r_lod_near, cvars.r_lod_curve);
		gfx.set_rays(cvars.r_aa);
		gfx.set_smooth(cvars.r_fxaa != 0.0);
		gfx.set_indirect(cvars.r_gi, cvars.r_ao, cvars.r_ao_radius);
		gfx.set_traced_shadows(cvars.r_shadows != 0.0);
		gfx.set_fog(
			cvars.r_fog,
			cvars.r_fog_height,
			vec3(cvars.r_fog_r, cvars.r_fog_g, cvars.r_fog_b),
		);
		gfx.set_auto_exposure(cvars.r_autoexposure != 0.0);
		gfx.set_exposure_range(cvars.r_exposure_min, cvars.r_exposure_max);
		gfx.set_vsync(cvars.r_vsync != 0.0);
		gfx.set_pbr(cvars.r_pbr != 0.0);
		gfx.set_dynamic_lights(cvars.r_dlights != 0.0);
		gfx.set_dynamic_shadows(cvars.r_dynshadows != 0.0);
		// The player's body casts its shadow, and the playground's spheres
		// theirs.
		let hull = shown.hull();
		let reach = Vec3::Y * (hull.half_height - BODY_RADIUS);
		gfx.occluder(shown.origin - reach, shown.origin + reach, BODY_RADIUS);
		for &(center, radius) in &spheres {
			gfx.occluder(center, center, radius);
		}
		if flash.0 > 0.0 {
			gfx.light(flash.1, FLASH_COLOR * (flash.0 / FLASH_LIFE), FLASH_REACH);
			flash.0 -= frame;
		}
		gfx.set_view(View {
			eye,
			forward,
			right,
			up,
			fovy,
		});
		for (shot, age) in &shots {
			let fade = 1.0 - age / SHOT_LIFE;
			gfx.line_3d(shot.origin, shot.end, Color::new(1.0, 0.85, 0.4, fade));
			for wall in &shot.walls {
				gfx.line_3d(wall.entry, wall.exit, Color::new(0.35, 1.0, 0.45, fade));
				gfx.cube(
					wall.entry,
					vec3(1.0, 1.0, 1.0),
					Color::new(0.02, 0.02, 0.02, fade),
				);
				gfx.cube(
					wall.exit,
					vec3(1.0, 1.0, 1.0),
					Color::new(0.35, 1.0, 0.45, fade),
				);
			}
			gfx.cube(
				shot.end,
				vec3(0.8, 0.8, 0.8),
				Color::new(0.9, 0.25, 0.2, fade),
			);
		}
		if !fly && armed.zoom == 0 {
			if let Some(vm) = &mut viewmodels[slot] {
				vm.tick(now);
				vm.draw(gfx, eye, forward, right, up, recoil, now);
			} else {
				crate::weapons::draw_viewmodel(gfx, weapon, eye, forward, right, up, recoil);
			}
		}
		hud(
			gfx,
			fps,
			&cvars,
			&player,
			fly,
			ticks_per_second,
			weapon,
			&armed,
			money,
			&report,
		);
		let (width, height) = gfx.size();
		// The version in red at the lower right, with the hot reload count.
		let version = match generation {
			0 => format!("v{}", env!("CARGO_PKG_VERSION")),
			n => format!("v{} #{n}", env!("CARGO_PKG_VERSION")),
		};
		let version_width = gfx.text_width(&version, 20.0);
		gfx.text(
			&version,
			width - version_width - 16.0,
			height - 16.0,
			20.0,
			Color::new(1.0, 0.0, 0.0, 1.0),
		);
		// The scope: crosshair lines across the screen, as 1.6 draws them.
		if armed.zoom > 0 {
			let (cx, cy) = (width / 2.0, height / 2.0);
			gfx.line(cx - 9999.0, cy, cx + 9999.0, cy, 1.5, Color::BLACK);
			gfx.line(cx, cy - 9999.0, cx, cy + 9999.0, 1.5, Color::BLACK);
		}
		// The buy menu: 1.6's list, keyed by number.
		if buy_menu {
			gfx.rect(
				40.0,
				60.0,
				460.0,
				60.0 + 30.0 * table.weapons.len() as f32,
				Color::new(0.05, 0.07, 0.09, 0.85),
			);
			let mut y = 92.0;
			gfx.text("BUY  (1-8, B closes)", 56.0, y, 22.0, TEXT);
			y += 28.0;
			for (i, w) in table.weapons.iter().enumerate() {
				let (color, tag) = if owned[i] {
					(Color::GREEN, "owned".to_string())
				} else if cvars.sv_infinite_money != 0.0 || money >= w.price {
					(TEXT, format!("${}", w.price))
				} else {
					(MUTED, format!("${}", w.price))
				};
				gfx.text(
					&format!("{}. {:<8} {}", i + 1, w.name, tag),
					56.0,
					y,
					22.0,
					color,
				);
				y += 26.0;
			}
		}
		match menu.frame(gfx, &keys, &mut cvars) {
			Some(menu::Choice::NewGame) => {
				player = Player::spawn(spawn);
				previous = player;
			}
			Some(menu::Choice::Console) => console.open = true,
			Some(menu::Choice::Save) => {
				console.execute("writeconfig", &mut cvars);
			}
			Some(menu::Choice::Quit) => return None,
			Some(menu::Choice::Resume) | None => {}
		}
		if console.open {
			draw_console(gfx, &console);
		}
		if let Some(profiler) = &mut profiler {
			if profiler.observe(profiler::Sample::new(
				frame_interval_ms,
				frame_started.elapsed().as_secs_f32() * 1000.0,
				dt * 1000.0,
				0.0,
				0.0,
				0,
				0,
				0,
				0.0,
				0.0,
				&player,
				input,
				fly,
				slot,
				armed.clip,
				armed.reserve,
				money,
				&vars,
				cvars.sv_tickrate,
				accumulator * 1000.0,
			)) {
				console.print("performance spike detected; recording 16 seconds of follow-up".to_string());
			}
			if let Some(remaining_ms) = profiler.capture_remaining_ms() {
				gfx.text(
					&format!("PERFORMANCE CAPTURE  {:.1}s", remaining_ms as f32 / 1000.0),
					16.0,
					height - 42.0,
					18.0,
					Color::new(1.0, 0.72, 0.2, 1.0),
				);
			}
		}
		// The ears sit at the view; this frame's queued sounds mix and push out.
		audio.frame(eye, forward, up);
		if !window.next_frame() {
			return None;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use engine::linalg::Camera;

	#[test]
	fn defaults_to_dust2_with_its_baked_lightmap() {
		let path = selected_map(&[]);
		assert_eq!(
			path,
			root().join("assets").join("maps").join("de_dust2.bsp")
		);
		let map = bsp::load(&path).expect("load bundled Dust2 BSP");
		assert!(!map.render.faces.is_empty());
		assert_eq!(map.spawn_yaw, 105.0);
		assert!(
			map
				.render
				.lightmap
				.rgba
				.as_chunks::<4>()
				.0
				.iter()
				.any(|rgba| rgba[..3] != [128, 128, 128]),
			"Dust2 BSP has no baked lightmap samples"
		);
		let has_albedo = map.render.faces.iter().any(|face| {
			let x = face.texture.x + face.texture.w / 2;
			let y = face.texture.y + face.texture.h / 2;
			let at = ((y * map.render.albedo.side + x) * 4) as usize;
			let pixel = &map.render.albedo.rgba[at..at + 4];
			pixel[3] != 0 && pixel[..3].iter().any(|&channel| channel != 0)
		});
		assert!(has_albedo, "Dust2 has no visible albedo texels");
		let eye = Player::spawn(map.spawn).eye();
		let (forward, right, _) = angle_vectors(map.spawn_yaw, map.spawn_pitch);
		let camera = Camera {
			eye,
			target: eye + forward,
			up: right.cross(forward),
			fovy: vertical_fov(90.0),
			near: 1.0,
			far: 8192.0,
		};
		let view = camera.view_projection(1600.0 / 900.0);
		let visible = map
			.render
			.faces
			.iter()
			.filter(|face| {
				face.verts.iter().any(|v| {
					let p = view.project_point(v.pos);
					p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && (-1.0..=1.0).contains(&p.z)
				})
			})
			.count();
		assert!(visible > 0, "spawn camera sees no Dust2 face from {eye:?}");
	}
}
