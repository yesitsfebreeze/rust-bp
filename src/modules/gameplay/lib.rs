//! Counter-Strike gameplay: movement, weapons, combat, and arena state.
//!
//! WASD move - mouse look - Space or mouse wheel jump - Ctrl or C duck - Shift walk
//! F flycam - F3 wireframe - ` or F1 console - Esc frees the mouse, click to grab it again
//!
//! Movement ticks at `sv_tickrate`; the view turns every frame from raw mouse counts
//! and the player is drawn between the last two ticks. The first argument names the
//! TOML config, by default `player.toml` beside this file; a missing one is created.
//! Dust2 loads by default; pass a `.bsp` path to load another map.

#![deny(warnings)]

mod bullets;
mod cvars;
mod materials;
mod player;
#[allow(dead_code)]
mod state;
mod viewmodel;
mod weapons;

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
use engine::linalg::{vec3, Camera, Quat, Vec3};
use engine::macroquad;
use engine::macroquad::prelude::*;
use engine::{bsp, lightmap, map_render};
use player::movement::angle_vectors;
use player::{Input, Player};

use crate::bullets::Shot;
use crate::cvars::{movement_vars, Cvars};
use crate::materials::SurfaceMaterial;
use crate::weapons::{load, load_sound, Armed, Table};
use engine::map_render::MapRenderer;

/// How long a fired bullet's tracer and holes stay drawn, seconds.
const SHOT_LIFE: f32 = 2.0;

/// Longest frame time simulated; a longer stall drops time instead of piling up ticks.
const MAX_FRAME: f32 = 0.25;
const TEXT: Color = Color::new(0.95, 0.96, 0.97, 1.0);
const MUTED: Color = Color::new(0.75, 0.78, 0.8, 1.0);

/// Sets `cvars` from the config, writing the defaults there first if it is missing.
fn load_config(console: &mut Console, cvars: &mut Cvars) {
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
fn type_into(console: &mut Console) -> bool {
	while let Some(c) = get_char_pressed() {
		if !c.is_control() && c != '`' && c != '^' {
			console.input.push(c);
		}
	}
	if is_key_pressed(KeyCode::Backspace) {
		console.input.pop();
	}
	if is_key_pressed(KeyCode::Tab) {
		console.complete::<Cvars>();
	}
	if is_key_pressed(KeyCode::Up) {
		console.recall(true);
	}
	if is_key_pressed(KeyCode::Down) {
		console.recall(false);
	}
	is_key_pressed(KeyCode::Enter) || is_key_pressed(KeyCode::KpEnter)
}

fn draw_console(console: &Console) {
	let h = screen_height() * 0.45;
	draw_rectangle(
		0.0,
		0.0,
		screen_width(),
		h,
		Color::new(0.05, 0.07, 0.09, 0.88),
	);
	let size = 20.0;
	let mut y = h - 12.0;
	draw_text(format!("] {}_", console.input), 12.0, y, size, TEXT);
	for line in console.lines().iter().rev() {
		y -= size;
		if y < size {
			break;
		}
		draw_text(line, 12.0, y, size, MUTED);
	}
}

/// The heads-up display reads like 1.6's: the weapon bottom-left, the money and
/// ammo top-right. Every argument is one HUD line's data.
#[allow(clippy::too_many_arguments)]
fn hud(
	cvars: &Cvars,
	player: &Player,
	fly: bool,
	ticks: u32,
	weapon: &crate::weapons::Weapon,
	armed: &Armed,
	money: u32,
	report: &str,
) {
	let mut y = 28.0;
	let mut line = |text: String| {
		draw_text(&text, 16.0, y, 24.0, TEXT);
		y += 24.0;
	};
	if cvars.cl_showfps != 0.0 {
		line(format!("{} fps  {} ticks/s", get_fps(), ticks));
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
		line(format!("{:.1} u/s  {state}", player.speed()));
	}
	draw_text(
		"WASD move  Space/wheel jump  Ctrl duck  Shift walk  mouse 1 fire  mouse 2 zoom/burst  1-8 weapon  B buy  R reload  F flycam  F3 wireframe  ` console",
		16.0,
		screen_height() - 16.0,
		20.0,
		MUTED,
	);
	line(format!(
		"{}  (1-8: knife, glock, usp, deagle, mp5, ak47, m4a1, awp)",
		weapon.name
	));
	if !report.is_empty() {
		line(report.to_string());
	}
	// The economy and ammo read from the top-right, as in 1.6's HUD.
	draw_text(format!("${money}"), screen_width() - 90.0, 28.0, 24.0, TEXT);
	let ammo = if weapon.melee {
		"-".to_string()
	} else {
		format!("{} / {}", armed.clip, armed.reserve)
	};
	draw_text(ammo, screen_width() - 90.0, 52.0, 24.0, TEXT);
	if cvars.cl_crosshair != 0.0 {
		let (cx, cy) = (screen_width() / 2.0, screen_height() / 2.0);
		for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
			draw_line(
				cx + dx * 4.0,
				cy + dy * 4.0,
				cx + dx * 12.0,
				cy + dy * 12.0,
				2.0,
				GREEN,
			);
		}
	}
}

/// One weapon's sounds, loaded once from the 1.6 assets; a synthesized report
/// stands in when the files are missing.
struct Gun {
	fire: Vec<macroquad::audio::Sound>,
	dry: Option<macroquad::audio::Sound>,
	deploy: Option<macroquad::audio::Sound>,
	/// Reload sounds at their fraction of the reload, as `weapons.toml`
	/// stages them (`file@fraction`).
	reload: Vec<(f32, macroquad::audio::Sound)>,
	zoom: Option<macroquad::audio::Sound>,
	/// The knife's stab swing and its sound when it meets a wall.
	stab: Option<macroquad::audio::Sound>,
	wall: Option<macroquad::audio::Sound>,
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
) {
	mags[*slot] = (armed.clip, armed.reserve);
	let w = &table.weapons[to];
	armed.switch(w, mags[to].0, mags[to].1, now);
	if let Some(vm) = &mut viewmodels[to] {
		vm.start("draw", now, false);
	}
	if let Some(s) = guns[to].deploy.as_ref() {
		macroquad::audio::play_sound_once(s);
	}
	console.print(format!("{} ({} of 8)", w.name, to + 1));
	*slot = to;
}

pub async fn run() {
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
	let renderer = match MapRenderer::new(&map.render) {
		Ok(renderer) => renderer,
		Err(e) => {
			eprintln!("{e}");
			std::process::exit(1);
		}
	};
	// The BSP's embedded GoldSrc lightdata is the default baked lightmap.
	// This optional bake remains available for comparison and rebaking.
	if cvars.r_lightbake != 0.0 {
		let start = std::time::Instant::now();
		match lightmap::bake(&map, &map_path) {
			Ok(baked) => console.print(format!(
				"lightmap: {} charts in a {}x{} atlas, baked in {:?}",
				baked.charts(),
				baked.atlas().0,
				baked.atlas().0,
				start.elapsed()
			)),
			Err(e) => console.print(format!("lightmap bake failed: {e}")),
		}
	}
	let (world, surfaces, spawn, spawn_yaw, spawn_pitch, kill_y) = (
		map.world,
		map.surfaces,
		map.spawn,
		map.spawn_yaw,
		map.spawn_pitch,
		map.kill_y,
	);
	let mut player = Player::spawn(spawn);
	let mut previous = player;
	// The free camera on F: where it floats, or None while the player plays.
	let (mut fly, mut fly_eye) = (false, player.eye());
	let (mut yaw, mut pitch) = (spawn_yaw, spawn_pitch);
	let mut wireframe = false;
	let mut grabbed = true;
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
				.and_then(|dir| viewmodel::Viewmodel::load(dir, w))
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
	for w in &table.weapons {
		let dir = sound_dir.as_deref();
		let mut fire = Vec::new();
		for name in &w.sounds.fire {
			fire.push(load_sound(dir, name, w).await);
		}
		let mut reload = Vec::new();
		for s in &w.sounds.reload {
			let (file, frac) = s.split_once('@').unwrap_or((s.as_str(), "0"));
			reload.push((
				frac.parse::<f32>().unwrap_or(0.0),
				load_sound(dir, file, w).await,
			));
		}
		guns.push(Gun {
			fire,
			dry: match w.sounds.dry.as_deref() {
				Some(f) => Some(load_sound(dir, f, w).await),
				None => None,
			},
			deploy: match w.sounds.deploy.as_deref() {
				Some(f) => Some(load_sound(dir, f, w).await),
				None => None,
			},
			reload,
			zoom: match w.zoom.as_ref().and_then(|z| z.sound.as_deref()) {
				Some(f) => Some(load_sound(dir, f, w).await),
				None => None,
			},
			stab: match w.sounds.stab.as_deref() {
				Some(f) => Some(load_sound(dir, f, w).await),
				None => None,
			},
			wall: match w.sounds.wall.as_deref() {
				Some(f) => Some(load_sound(dir, f, w).await),
				None => None,
			},
		});
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
	let mut shots: Vec<(Shot, f32)> = Vec::new();
	let mut report = String::new();
	set_cursor_grab(true);
	show_mouse(false);
	let mut last_mouse = mouse_position();
	let mut accumulator = 0.0;
	// Presses seen between ticks, so a tap shorter than a tick still counts.
	let mut jump_pressed = false;
	let (mut tick_count, mut ticks_per_second, mut second) = (0, 0, 0.0);

	loop {
		if is_key_pressed(KeyCode::GraveAccent) || is_key_pressed(KeyCode::F1) {
			console.open = !console.open;
			while get_char_pressed().is_some() {}
		}
		let typing = console.open;
		if typing {
			if type_into(&mut console) {
				for action in console.submit(&mut cvars) {
					match action {
						Action::Quit => return,
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
			if is_key_pressed(KeyCode::Escape) {
				console.open = false;
			}
		} else if is_key_pressed(KeyCode::Escape) {
			grabbed = false;
		} else if !grabbed && is_mouse_button_pressed(MouseButton::Left) {
			grabbed = true;
		} else if !typing && is_key_pressed(KeyCode::F) {
			fly = !fly;
			if fly {
				fly_eye = player.eye();
			}
			console.print(format!("flycam {}", if fly { "on" } else { "off" }));
		}
		if !typing && is_key_pressed(KeyCode::F3) {
			wireframe = !wireframe;
			console.print(format!(
				"wireframe {}",
				if wireframe { "on" } else { "off" }
			));
		}
		// B is the buy menu; there the number keys buy, elsewhere they switch.
		if !typing && is_key_pressed(KeyCode::B) {
			buy_menu = !buy_menu;
		}
		if buy_menu && (is_key_pressed(KeyCode::Escape) || typing) {
			buy_menu = false;
		}
		for (i, key) in [
			KeyCode::Key1,
			KeyCode::Key2,
			KeyCode::Key3,
			KeyCode::Key4,
			KeyCode::Key5,
			KeyCode::Key6,
			KeyCode::Key7,
			KeyCode::Key8,
		]
		.iter()
		.enumerate()
		{
			if !is_key_pressed(*key) {
				continue;
			}
			if buy_menu {
				let w = &table.weapons[i];
				if owned[i] {
					// Buying a gun already owned fills its ammo, as 1.6's
					// buyammo does.
					if w.ammo_price > 0 && money >= w.ammo_price && mags[i].1 < w.reserve {
						money -= w.ammo_price;
						mags[i].1 = w.reserve;
						console.print(format!("{} ammo full (${})", w.ammo_name, w.ammo_price));
					}
				} else if money >= w.price {
					money -= w.price;
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
				);
			} else if i != slot {
				console.print(format!("{} not owned; B to buy", table.weapons[i].name));
			}
		}
		let looking = grabbed && !typing;
		set_cursor_grab(looking);
		show_mouse(!looking);

		// Look: raw counts every frame, never waiting for a tick.
		let mouse = mouse_position();
		if looking {
			let scale = cvars.sensitivity;
			yaw = (yaw + (mouse.0 - last_mouse.0) * scale * cvars.m_yaw).rem_euclid(360.0);
			pitch = (pitch + (mouse.1 - last_mouse.1) * scale * cvars.m_pitch).clamp(-89.0, 89.0);
		}
		last_mouse = mouse;

		let down = |k: KeyCode| !typing && is_key_down(k);
		jump_pressed |= !typing && (is_key_pressed(KeyCode::Space) || mouse_wheel().1 != 0.0);
		let walk = if down(KeyCode::LeftShift) {
			cvars.cl_movespeedkey
		} else {
			1.0
		};
		let axis = |pos: KeyCode, neg: KeyCode, p: f32, n: f32| {
			(if down(pos) { p } else { 0.0 }) - (if down(neg) { n } else { 0.0 })
		};
		let duck = down(KeyCode::LeftControl) || down(KeyCode::C);
		let input = Input {
			forward: axis(
				KeyCode::W,
				KeyCode::S,
				cvars.cl_forwardspeed,
				cvars.cl_backspeed,
			) * walk,
			side: axis(
				KeyCode::D,
				KeyCode::A,
				cvars.cl_sidespeed,
				cvars.cl_sidespeed,
			) * walk,
			up: if player.noclip {
				(if down(KeyCode::Space) {
					cvars.cl_upspeed
				} else {
					0.0
				}) - (if duck { cvars.cl_upspeed } else { 0.0 })
			} else {
				0.0
			},
			yaw,
			pitch,
			jump: down(KeyCode::Space) || jump_pressed,
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
		accumulator += get_frame_time().min(MAX_FRAME);
		let vars = movement_vars(&cvars);
		// The flycam moves per frame along the look direction; the player stands
		// still and is drawn where it was.
		if fly {
			let (forward, right, _) = angle_vectors(yaw, pitch);
			let mut toward = Vec3::ZERO;
			if down(KeyCode::W) {
				toward += forward;
			}
			if down(KeyCode::S) {
				toward -= forward;
			}
			if down(KeyCode::D) {
				toward += right;
			}
			if down(KeyCode::A) {
				toward -= right;
			}
			if down(KeyCode::Space) {
				toward.y += 1.0;
			}
			if duck {
				toward.y -= 1.0;
			}
			if toward.length2() > 0.0 {
				let speed = cvars.cl_forwardspeed * if down(KeyCode::LeftShift) { 2.0 } else { 1.0 };
				fly_eye += toward.normalize() * speed * get_frame_time().min(MAX_FRAME);
			}
		}
		while accumulator >= dt {
			accumulator -= dt;
			previous = player;
			if !fly {
				player.tick(input, &vars, &world, dt);
			}
			jump_pressed = false;
			tick_count += 1;
		}
		second += get_frame_time();
		if second >= 1.0 {
			(ticks_per_second, tick_count, second) = (tick_count, 0, second - 1.0);
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
		let (forward, _, up) = angle_vectors(yaw + armed.punch.1, pitch + armed.punch.0);
		let up = Quat::from_axis_angle(forward, shown.punch_roll.to_radians()).rotate(up);
		let eye = if fly { fly_eye } else { shown.eye() };
		// `fov` is horizontal at 4:3, as in GoldSrc; the scope narrows it.
		let fov = if armed.zoom > 0 {
			armed.fov(weapon)
		} else {
			cvars.fov
		};
		let fovy = 2.0 * ((fov.clamp(10.0, 170.0).to_radians() * 0.5).tan() * 0.75).atan();
		let (_, right, _) = angle_vectors(yaw + armed.punch.1, pitch + armed.punch.0);

		// Fire: the trigger within the cycle, from the eye, along the punched
		// view through the heat's cone; the reload stages its sounds, and the
		// glock's burst rounds fire on their own schedule.
		let frame = get_frame_time().min(MAX_FRAME);
		now += frame;
		recoil *= (-10.0 * frame).exp();
		let ease = (-8.0 * frame).exp();
		armed.punch.0 *= ease;
		armed.punch.1 *= ease;
		let tick = armed.tick(weapon, now);
		for (frac, s) in &guns[slot].reload {
			if tick.reload_sounds.contains(frac) {
				macroquad::audio::play_sound_once(s);
			}
		}
		// R reloads: staged sounds above, the animation here.
		if looking && is_key_pressed(KeyCode::R) {
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
		if looking && is_mouse_button_pressed(MouseButton::Right) {
			if weapon.zoom.is_some() {
				if let Some(fov) = armed.toggle_zoom(weapon, now) {
					if let Some(s) = guns[slot].zoom.as_ref() {
						macroquad::audio::play_sound_once(s);
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
			&& (is_mouse_button_pressed(MouseButton::Left)
				|| (weapon.auto && is_mouse_button_down(MouseButton::Left)));
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
						&world,
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
						if let Some(s) = guns[slot].wall.as_ref() {
							macroquad::audio::play_sound_once(s);
						}
					}
					let g = &guns[slot];
					if stab {
						if let Some(s) = g.stab.as_ref() {
							macroquad::audio::play_sound_once(s);
						}
					} else if !g.fire.is_empty() {
						let pick = macroquad::rand::gen_range(0usize, g.fire.len());
						macroquad::audio::play_sound_once(&g.fire[pick]);
					}
					shots.push((shot, 0.0));
				}
				None if will_dry => {
					if let Some(s) = guns[slot].dry.as_ref() {
						macroquad::audio::play_sound_once(s);
					}
				}
				None => {}
			}
		}
		for (_, age) in &mut shots {
			*age += frame;
		}
		shots.retain(|(_, age)| *age < SHOT_LIFE);

		clear_background(map_render::SKY);
		set_camera(&Camera {
			eye,
			target: eye + forward,
			up,
			fovy,
			near: 1.0,
			far: 8192.0,
		});
		renderer.draw(cvars.r_exposure);
		if wireframe {
			renderer.draw_wireframe();
		}
		for (shot, age) in &shots {
			let fade = 1.0 - age / SHOT_LIFE;
			// macroquad draws in its own Vec3; the game keeps linalg's.
			let q = |v: Vec3| macroquad::math::Vec3::from(v);
			draw_line_3d(
				q(shot.origin),
				q(shot.end),
				Color::new(1.0, 0.85, 0.4, fade),
			);
			for wall in &shot.walls {
				draw_line_3d(
					q(wall.entry),
					q(wall.exit),
					Color::new(0.35, 1.0, 0.45, fade),
				);
				draw_cube(
					q(wall.entry),
					q(vec3(1.0, 1.0, 1.0)),
					None,
					Color::new(0.02, 0.02, 0.02, fade),
				);
				draw_cube(
					q(wall.exit),
					q(vec3(1.0, 1.0, 1.0)),
					None,
					Color::new(0.35, 1.0, 0.45, fade),
				);
			}
			draw_cube(
				q(shot.end),
				q(vec3(0.8, 0.8, 0.8)),
				None,
				Color::new(0.9, 0.25, 0.2, fade),
			);
		}
		if !fly && armed.zoom == 0 {
			if let Some(vm) = &mut viewmodels[slot] {
				vm.tick(now);
				vm.draw(eye, forward, right, up, recoil, now);
			} else {
				crate::weapons::draw_viewmodel(weapon, eye, forward, right, up, recoil);
			}
		}
		set_default_camera();
		hud(
			&cvars,
			&player,
			fly,
			ticks_per_second,
			weapon,
			&armed,
			money,
			&report,
		);
		// The scope: crosshair lines across the screen, as 1.6 draws them.
		if armed.zoom > 0 {
			let (cx, cy) = (screen_width() / 2.0, screen_height() / 2.0);
			draw_line(cx - 9999.0, cy, cx + 9999.0, cy, 1.5, BLACK);
			draw_line(cx, cy - 9999.0, cx, cy + 9999.0, 1.5, BLACK);
		}
		// The buy menu: 1.6's list, keyed by number.
		if buy_menu {
			draw_rectangle(
				40.0,
				60.0,
				460.0,
				60.0 + 30.0 * table.weapons.len() as f32,
				Color::new(0.05, 0.07, 0.09, 0.85),
			);
			let mut y = 92.0;
			draw_text("BUY  (1-8, B closes)", 56.0, y, 22.0, TEXT);
			y += 28.0;
			for (i, w) in table.weapons.iter().enumerate() {
				let (color, tag) = if owned[i] {
					(GREEN, "owned".to_string())
				} else if money >= w.price {
					(TEXT, format!("${}", w.price))
				} else {
					(MUTED, format!("${}", w.price))
				};
				draw_text(
					format!("{}. {:<8} {}", i + 1, w.name, tag),
					56.0,
					y,
					22.0,
					color,
				);
				y += 26.0;
			}
		}
		if console.open {
			draw_console(&console);
		}
		next_frame().await;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

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
	}
}
