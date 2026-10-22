//! The playground: the engine without a map file. The arena is generated in
//! code — boxes and a ramp — and the same geometry feeds collision, the
//! raycast renderer and the light sources, so movement, physics and lighting
//! can be tried where every input is known. No BSP, no WADs, no weapons.
//!
//! WASD move - mouse look - Space or mouse wheel jump - Ctrl or C duck -
//! Shift walk - F flycam - G throws a physics sphere, H clears them and the
//! dents they struck into the map - ` or
//! F1 console - Esc frees the mouse, click to grab it again.
//!
//! Cvars load from `playground.toml` beside the workspace root; a missing one
//! is created with the defaults.

use engine::bsp::{Atlas, Face, Map, MapRender, Rect, Vert};
use engine::console::{Action, Console};
use engine::gpu::{Color, Gfx, Key, MouseButton, View, Window, MAX_DENTS};
use engine::lightbake;
use engine::linalg::{vec3, Vec3};
use engine::physics;
use engine::surface::Surface;
use gameplay::cvars::{movement_vars, Cvars};
use gameplay::player::movement::angle_vectors;
use gameplay::player::{Input, Player};
use gameplay::{draw_console, load_config, type_into, vertical_fov};

/// Spheres in the playground at most; past it, the oldest goes. The
/// renderer's body buffer is this big.
const BODY_CAP: usize = 64;
/// Speed into the map, units per second, a sphere strikes it at before it
/// dents; and the speed past that at which the dent is deepest.
const DENT_SPEED: f32 = 250.0;
const DENT_FULL: f32 = 600.0;
/// The deepest dent, in sphere radii.
const DENT_DEPTH: f32 = 0.6;
/// Longest frame time simulated; a longer stall drops time instead of piling up ticks.
const MAX_FRAME: f32 = 0.25;
/// World units one albedo tile repeats over; the atlas tiles are 1024 texels.
const TILE: f32 = 128.0;

const TEXT: Color = Color::new(0.95, 0.96, 0.97, 1.0);
const MUTED: Color = Color::new(0.75, 0.78, 0.8, 1.0);

/// The albedo tiles: floor, wall, steel and accent, each 1024 texels in one
/// 2048-texel atlas. Every tile layers a 64-texel checker of two near tones
/// with two octaves of value noise, so tiling reads as a worn surface
/// instead of a flat color, and voxel sampling walks real texel detail —
/// a stress texture, sixteen times finer per axis than the first draft.
fn albedo_atlas() -> Atlas {
	let side = 2048;
	let tile = 1024;
	let mut atlas = Atlas {
		side,
		rgba: vec![255; (side * side * 4) as usize],
	};
	// One hashed lattice value per point, [0, 1).
	let lattice = |x: u32, y: u32| {
		let mut h = x.wrapping_mul(0x9E3779B1) ^ y.wrapping_mul(0x85EBCA77);
		h ^= h >> 13;
		h = h.wrapping_mul(0xC2B2AE3D);
		(h >> 8) as f32 / (1 << 24) as f32
	};
	// Smooth value noise on a `step`-texel lattice, [0, 1).
	let noise = |x: u32, y: u32, step: u32| {
		let (fx, fy) = (x as f32 / step as f32, y as f32 / step as f32);
		let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
		let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
		let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
		let v = |dx: u32, dy: u32| lattice(x0 + dx, y0 + dy);
		(v(0, 0) * (1.0 - sx) + v(1, 0) * sx) * (1.0 - sy) + (v(0, 1) * (1.0 - sx) + v(1, 1) * sx) * sy
	};
	let mut paint = |x0: u32, y0: u32, base: [u8; 3], alt: [u8; 3]| {
		for y in 0..tile {
			for x in 0..tile {
				// The coarse checker sets the tile's tone; the fine one and
				// two noise octaves wear it.
				let even = (x / 64 + y / 64) % 2 == 0;
				let mut c = if even { base } else { alt };
				if (x / 8 + y / 8) % 2 == 0 {
					for channel in &mut c {
						*channel = channel.saturating_add(4);
					}
				}
				let worn = noise(x, y, 128) * 0.6 + noise(x, y, 32) * 0.4;
				for channel in &mut c {
					let lift = ((worn - 0.5) * 18.0) as i8;
					*channel = channel.saturating_add_signed(lift);
				}
				let at = (((y0 + y) * side + x0 + x) * 4) as usize;
				atlas.rgba[at..at + 3].copy_from_slice(&c);
			}
		}
	};
	paint(0, 0, [150, 148, 145], [138, 136, 134]); // floor
	paint(1024, 0, [186, 168, 132], [176, 158, 122]); // wall
	paint(0, 1024, [104, 112, 122], [94, 102, 112]); // steel
	paint(1024, 1024, [156, 118, 84], [146, 108, 76]); // accent
	atlas
}

fn floor_tex() -> Rect {
	Rect {
		x: 0,
		y: 0,
		w: 1024,
		h: 1024,
	}
}
fn wall_tex() -> Rect {
	Rect {
		x: 1024,
		y: 0,
		w: 1024,
		h: 1024,
	}
}
fn steel_tex() -> Rect {
	Rect {
		x: 0,
		y: 1024,
		w: 1024,
		h: 1024,
	}
}
fn accent_tex() -> Rect {
	Rect {
		x: 1024,
		y: 1024,
		w: 1024,
		h: 1024,
	}
}

/// One quad, wound so its fan's normal points along `outward`, with `uv`
/// mapping the face's plane at one tile per `TILE` units — the same dominant
/// axis choice the renderer makes for tiling textures.
fn quad(tex: Rect, a: Vec3, b: Vec3, c: Vec3, d: Vec3, outward: Vec3) -> Face {
	let wind = if (b - a).cross(c - a).dot(outward) >= 0.0 {
		[a, b, c, d]
	} else {
		[a, d, c, b]
	};
	let n = (wind[1] - wind[0]).cross(wind[2] - wind[0]);
	let uv = |p: Vec3| -> (f32, f32) {
		if n.y.abs() >= n.x.abs() && n.y.abs() >= n.z.abs() {
			(p.x / TILE, p.z / TILE)
		} else if n.x.abs() >= n.z.abs() {
			(p.z / TILE, p.y / TILE)
		} else {
			(p.x / TILE, p.y / TILE)
		}
	};
	Face {
		texture: tex,
		lightmap: Rect {
			x: 0,
			y: 0,
			w: 0,
			h: 0,
		},
		solid: true,
		verts: wind
			.into_iter()
			.map(|p| Vert {
				pos: p,
				uv: uv(p),
				lm: (0.0, 0.0),
			})
			.collect(),
	}
}

/// The generated arena: the same boxes and ramps in the collision mesh and in
/// the rendered faces, one tile per element kind.
struct Arena {
	mesh: physics::Mesh,
	faces: Vec<Face>,
}

impl Arena {
	fn new() -> Arena {
		Arena {
			mesh: physics::Mesh::default(),
			faces: Vec::new(),
		}
	}

	fn box_(&mut self, min: Vec3, max: Vec3, tex: Rect) {
		self.mesh.push_box(min, max);
		for (a, b, c, d, n) in box_quads(min, max) {
			self.faces.push(quad(tex, a, b, c, d, n));
		}
	}

	/// A sloped quad: a ramp up to the edge it meets at its high end.
	fn ramp(&mut self, corners: [Vec3; 4], tex: Rect) {
		let n = (corners[1] - corners[0])
			.cross(corners[2] - corners[0])
			.normalize();
		self
			.mesh
			.push_quad(corners[0], corners[1], corners[2], corners[3], n);
		self
			.faces
			.push(quad(tex, corners[0], corners[1], corners[2], corners[3], n));
	}

	/// The arena as a `Map`, with spawn and lights, ready for `load_world`.
	fn map(self) -> Map {
		let Arena { mesh, faces } = self;
		let tris = mesh.tris().len();
		Map {
			world: mesh,
			surfaces: vec![Surface::Concrete; tris],
			render: MapRender {
				albedo: albedo_atlas(),
				// The renderer lights every hit itself from the sun and sky;
				// no baked lightmap exists to sample.
				lightmap: Atlas {
					side: 1,
					rgba: vec![0, 0, 0, 0],
				},
				faces,
			},
			spawn: vec3(-320.0, 1.0, -320.0),
			spawn_yaw: 45.0,
			spawn_pitch: 0.0,
			kill_y: -400.0,
		}
	}
}

/// An axis-aligned box's six quads, each with its outward normal.
fn box_quads(min: Vec3, max: Vec3) -> [(Vec3, Vec3, Vec3, Vec3, Vec3); 6] {
	let (x0, y0, z0) = (min.x, min.y, min.z);
	let (x1, y1, z1) = (max.x, max.y, max.z);
	let p = vec3;
	[
		// top, bottom, +x, -x, +z, -z, wound outward
		(
			p(x0, y1, z0),
			p(x0, y1, z1),
			p(x1, y1, z1),
			p(x1, y1, z0),
			Vec3::Y,
		),
		(
			p(x0, y0, z0),
			p(x1, y0, z0),
			p(x1, y0, z1),
			p(x0, y0, z1),
			-Vec3::Y,
		),
		(
			p(x1, y0, z0),
			p(x1, y1, z0),
			p(x1, y1, z1),
			p(x1, y0, z1),
			Vec3::X,
		),
		(
			p(x0, y0, z1),
			p(x0, y1, z1),
			p(x0, y1, z0),
			p(x0, y0, z0),
			-Vec3::X,
		),
		(
			p(x0, y0, z1),
			p(x1, y0, z1),
			p(x1, y1, z1),
			p(x0, y1, z1),
			Vec3::Z,
		),
		(
			p(x1, y0, z0),
			p(x0, y0, z0),
			p(x0, y1, z0),
			p(x1, y1, z0),
			-Vec3::Z,
		),
	]
}

/// The arena's contents: a floor slab inside boundary walls, a central
/// plinth with a step and a ramp up, and two pillars.
fn arena() -> Map {
	let mut a = Arena::new();
	// Floor and boundary walls.
	a.box_(
		vec3(-512.0, -64.0, -512.0),
		vec3(512.0, 0.0, 512.0),
		floor_tex(),
	);
	a.box_(
		vec3(-512.0, 0.0, 480.0),
		vec3(512.0, 192.0, 512.0),
		wall_tex(),
	);
	a.box_(
		vec3(-512.0, 0.0, -512.0),
		vec3(512.0, 192.0, -480.0),
		wall_tex(),
	);
	a.box_(
		vec3(480.0, 0.0, -480.0),
		vec3(512.0, 192.0, 480.0),
		wall_tex(),
	);
	a.box_(
		vec3(-512.0, 0.0, -480.0),
		vec3(-480.0, 192.0, 480.0),
		wall_tex(),
	);
	// The central plinth, a step up to it, and a ramp on the far side.
	a.box_(
		vec3(-160.0, 0.0, -160.0),
		vec3(160.0, 64.0, 160.0),
		accent_tex(),
	);
	a.box_(
		vec3(-256.0, 0.0, -48.0),
		vec3(-192.0, 32.0, 48.0),
		steel_tex(),
	);
	a.ramp(
		[
			vec3(-96.0, 0.0, 256.0),
			vec3(96.0, 0.0, 256.0),
			vec3(96.0, 32.0, 192.0),
			vec3(-96.0, 32.0, 192.0),
		],
		steel_tex(),
	);
	// Two pillars in the open corners.
	a.box_(
		vec3(-416.0, 0.0, 256.0),
		vec3(-352.0, 128.0, 320.0),
		wall_tex(),
	);
	a.box_(
		vec3(256.0, 0.0, -320.0),
		vec3(320.0, 128.0, -256.0),
		wall_tex(),
	);
	a.map()
}

/// The heads-up display: frame rate, movement state and the controls.
fn hud(gfx: &mut Gfx, fps: u32, player: &Player, fly: bool, bodies: usize, ticks: u32) {
	let (width, height) = gfx.size();
	let mut y = 28.0;
	let mut line = |gfx: &mut Gfx, text: String| {
		gfx.text(&text, 16.0, y, 24.0, TEXT);
		y += 24.0;
	};
	line(gfx, format!("{fps} fps  {ticks} ticks/s  {bodies} spheres"));
	if gfx.view_mode() != "lit" {
		line(gfx, format!("view: {} (F5)", gfx.view_mode()));
	}
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
	gfx.text(
		"WASD move  Space/wheel jump  Ctrl duck  Shift walk  F flycam  G throw ball  H clear  ` console",
		16.0,
		height - 16.0,
		20.0,
		MUTED,
	);
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

/// The dent a sphere moving at `before` leaves, if its velocity lost
/// enough into a triangle of `mesh` this step: the sphere's own shape, its
/// center pushed into the surface by the depth the strike earns.
fn dent(mesh: &physics::Mesh, body: &physics::Body, before: Vec3) -> Option<(Vec3, f32)> {
	let lost = body.vel - before;
	if lost.length() < DENT_SPEED {
		return None;
	}
	// The surface the push came from: back along it from the center.
	let away = lost.normalize();
	let (p, q) = (body.pos, body.pos - away * (body.radius * 1.5));
	let (t, tri) = mesh
		.ray_tris(p, -away)
		.filter_map(|(_, tri)| {
			let t = physics::intersect_segment_triangle(p, q, tri)
				.or_else(|| physics::intersect_segment_triangle(p, q, &[tri[0], tri[2], tri[1]]))?;
			Some((t, tri))
		})
		.min_by(|a, b| a.0.total_cmp(&b.0))?;
	let mut n = physics::triangle_normal(tri);
	if n.dot(away) < 0.0 {
		n = -n;
	}
	let speed = -before.dot(n);
	if speed < DENT_SPEED {
		return None;
	}
	let depth = body.radius * DENT_DEPTH * ((speed - DENT_SPEED) / DENT_FULL).min(1.0);
	let at = p + (q - p) * t;
	Some((at + n * (body.radius - depth), body.radius))
}

fn main() {
	let config = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("playground.toml");
	let mut console = Console::new(config);
	let mut cvars = Cvars::default();
	load_config(&mut console, &mut cvars);
	console.print("playground: generated arena, no map file".to_string());
	console.print("type help for commands".to_string());

	let mut window = match Window::open("playground", 1600, 900) {
		Ok(window) => window,
		Err(e) => {
			eprintln!("{e}");
			std::process::exit(1);
		}
	};
	let map = arena();
	let light = lightbake::Config::default();
	let traced = window.gfx().load_world(&map, &light);
	console.print(format!("traced world: {} triangles", traced.triangles));
	// The arena's collision mesh, with the playground's spheres in it.
	// GoldSrc gravity, so spheres fall like the player does.
	let mut play = physics::World::new(map.world);
	// Dents the spheres struck into the map, oldest first.
	let mut dents: Vec<(Vec3, f32)> = Vec::new();
	play.gravity = vec3(0.0, -800.0, 0.0);
	let (spawn, spawn_yaw, spawn_pitch, kill_y) =
		(map.spawn, map.spawn_yaw, map.spawn_pitch, map.kill_y);

	let mut player = Player::spawn(spawn);
	let mut previous = player;
	// The free camera on F: where it floats, or None while the player plays.
	let (mut fly, mut fly_eye) = (false, player.eye());
	let (mut yaw, mut pitch) = (spawn_yaw, spawn_pitch);
	let mut grabbed = true;
	let mut accumulator = 0.0;
	// Presses seen between ticks, so a tap shorter than a tick still counts.
	let mut jump_pressed = false;
	let (mut tick_count, mut ticks_per_second, mut second) = (0, 0, 0.0f32);

	window.grab_cursor(true);
	loop {
		let frame_time = window.frame_time();
		let keys = window.input().clone();
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
			if keys.pressed(Key::Escape) {
				console.open = false;
			}
		} else if keys.pressed(Key::Escape) {
			grabbed = false;
		} else if !grabbed && keys.clicked(MouseButton::Left) {
			grabbed = true;
		} else if keys.pressed(Key::KeyF) {
			fly = !fly;
			if fly {
				fly_eye = player.eye();
			}
			console.print(format!("flycam {}", if fly { "on" } else { "off" }));
		}
		let looking = grabbed && !typing && keys.focused();
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
		let walk = if down(Key::ShiftLeft) {
			cvars.cl_movespeedkey
		} else {
			1.0
		};
		let axis = |pos: Key, neg: Key, p: f32, n: f32| {
			(if down(pos) { p } else { 0.0 }) - (if down(neg) { n } else { 0.0 })
		};
		let duck = down(Key::ControlLeft) || down(Key::KeyC);
		let input = Input {
			forward: axis(
				Key::KeyW,
				Key::KeyS,
				cvars.cl_forwardspeed,
				cvars.cl_backspeed,
			) * walk,
			side: axis(Key::KeyD, Key::KeyA, cvars.cl_sidespeed, cvars.cl_sidespeed) * walk,
			up: if player.noclip {
				(if down(Key::Space) {
					cvars.cl_upspeed
				} else {
					0.0
				}) - (if duck { cvars.cl_upspeed } else { 0.0 })
			} else {
				0.0
			},
			yaw,
			pitch,
			jump: down(Key::Space) || jump_pressed,
			duck,
		};

		let dt = 1.0 / cvars.sv_tickrate.clamp(10.0, 1000.0);
		let vars = movement_vars(&cvars);
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
				player.tick(input, &vars, &play.mesh, dt);
			}
			jump_pressed = false;
			tick_count += 1;
		}
		second += frame_time;
		if second >= 1.0 {
			(ticks_per_second, tick_count, second) = (tick_count, 0, second - 1.0);
		}
		if player.origin.y < kill_y {
			player = Player::spawn(spawn);
			previous = player;
		}

		// G throws a sphere ahead of the eye, H clears them.
		let (forward, right, up) = angle_vectors(yaw, pitch);
		let alpha = if cvars.cl_interp != 0.0 {
			accumulator / dt
		} else {
			1.0
		};
		let shown = previous.interpolate(&player, alpha);
		let eye = if fly { fly_eye } else { shown.eye() };
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
			dents.clear();
		}
		let before: Vec<Vec3> = play.bodies.iter().map(|b| b.vel).collect();
		play.step(frame_time.min(MAX_FRAME), true);
		let struck = dents.len();
		for (body, &vel) in play.bodies.iter().zip(&before) {
			dents.extend(dent(&play.mesh, body, vel));
		}
		if dents.len() > MAX_DENTS {
			dents.drain(..dents.len() - MAX_DENTS);
		}
		play.bodies.retain(|b| b.pos.y > kill_y);
		let spheres: Vec<(Vec3, f32)> = play.bodies.iter().map(|b| (b.pos, b.radius)).collect();

		let fps = window.fps();
		let gfx = window.gfx();
		gfx.set_exposure(cvars.r_exposure);
		gfx.set_bodies(&spheres);
		if dents.len() != struck || keys.pressed(Key::KeyH) {
			gfx.set_dents(&dents);
		}
		gfx.set_voxel(cvars.r_voxel);
		gfx.set_shell(cvars.r_shell != 0.0);
		gfx.set_view(View {
			eye,
			forward,
			right,
			up,
			fovy: vertical_fov(cvars.fov),
		});
		hud(gfx, fps, &shown, fly, spheres.len(), ticks_per_second);
		if console.open {
			draw_console(gfx, &console);
		}
		if !window.next_frame() {
			return;
		}
	}
}
