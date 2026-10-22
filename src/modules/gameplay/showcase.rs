//! The physics half of the arena: loose bodies by the spawn the player can
//! shove, and a fenced pen whose chute drops bodies onto a swinging deck and a
//! lift. The player stays outside the pen — its walls are in [`crate::map`]'s
//! mesh, so the player's collision keeps them out while the bodies see the same
//! mesh through [`physics::World`] — and watches `physics` work.
//!
//! Stepping a stair snaps the player's collision at once (see `movement`); the
//! drawn position, which this module draws in the flycam, rides the virtual
//! plane of `Player::step_offset` instead, so the body never bobs.

use crate::player::Player;
use engine::linalg::{vec3, Mat4, Quat, Vec3};
use engine::macroquad;
use engine::macroquad::math as mq;
use engine::macroquad::prelude::*;
use engine::physics::{self, Body, Mesh, Platform, World};

use crate::map::{CHUTE_TOP, PEN};

/// How many of the world's bodies are the loose ones by the spawn; the rest
/// belong to the pen and recycle through its chute.
pub const YARD: usize = 9;

/// The swinging deck's boxes in its own frame: the floor slab and two rails
/// across z, open along x so bodies roll off its ends.
const DECK: [(Vec3, Vec3); 3] = [
	(vec3(-96.0, -8.0, -96.0), vec3(96.0, 8.0, 96.0)),
	(vec3(-96.0, 8.0, -96.0), vec3(96.0, 24.0, -80.0)),
	(vec3(-96.0, 8.0, 80.0), vec3(96.0, 24.0, 96.0)),
];
/// The lift's box in its own frame, its top at y = 0.
const LIFT: [(Vec3, Vec3); 1] = [(vec3(-48.0, -64.0, -48.0), vec3(48.0, 0.0, 48.0))];
/// Each platform's boxes, in the order [`world`] adds the platforms.
const PLATFORMS: [&[(Vec3, Vec3)]; 2] = [&DECK, &LIFT];

const BALL: Color = Color::new(0.86, 0.62, 0.16, 1.0);
const LOG: Color = Color::new(0.20, 0.62, 0.60, 1.0);
const PEN_BODY: Color = Color::new(0.72, 0.22, 0.28, 1.0);
const STEEL_PLAT: Color = Color::new(0.55, 0.57, 0.60, 1.0);
const PLAYER_BODY: Color = Color::new(0.35, 0.45, 0.72, 1.0);

fn boxes_mesh(boxes: &[(Vec3, Vec3)]) -> Mesh {
	let mut m = Mesh::default();
	for &(min, max) in boxes {
		m.push_box(min, max);
	}
	m
}

/// A capsule lying across the slope: its axis along z, so it rolls down x.
fn lying() -> Quat {
	Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2)
}

/// Bodies and moving platforms in the arena. `g` is gravity, units/s², positive
/// down.
pub fn world(arena: &Mesh, g: f32) -> World {
	let ball = Body {
		radius: 16.0,
		friction: 0.7,
		rotates: true,
		..Body::default()
	};
	let log = Body {
		radius: 10.0,
		half_height: 14.0,
		friction: 0.7,
		rotates: true,
		orient: lying(),
		..Body::default()
	};
	let mut yard = Vec::new();
	for i in 0..6 {
		yard.push(Body {
			pos: vec3(
				-224.0 + (i % 3) as f32 * 72.0,
				16.0,
				-48.0 + (i / 3) as f32 * 96.0,
			),
			..ball
		});
	}
	for (i, z) in [176.0, 224.0, 176.0].into_iter().enumerate() {
		yard.push(Body {
			pos: vec3(-200.0 + i as f32 * 104.0, 10.0, z),
			..log
		});
	}
	// The pen's bodies start rolling down the chute, a few units apart.
	let mut pen = Vec::new();
	for i in 0..4 {
		pen.push(Body {
			pos: CHUTE_TOP + vec3(0.0, i as f32 * 36.0, 0.0),
			vel: vec3(64.0, 0.0, 0.0),
			..if i % 2 == 0 {
				Body {
					radius: 12.0,
					friction: 0.6,
					rotates: true,
					..ball
				}
			} else {
				Body {
					radius: 8.0,
					half_height: 12.0,
					friction: 0.5,
					..log
				}
			}
		});
	}
	let mut w = World::new(arena.clone());
	w.gravity = vec3(0.0, -g, 0.0);
	w.platforms = vec![
		Platform::new(
			boxes_mesh(&DECK),
			vec3(832.0, 104.0, 672.0),
			vec3(48.0, 0.0, 0.0),
			5.0,
		),
		Platform::new(
			boxes_mesh(&LIFT),
			vec3(688.0, 192.0, 768.0),
			vec3(0.0, 80.0, 0.0),
			4.0,
		),
	];
	w.bodies = [yard, pen].concat();
	w.prepare();
	w
}

/// The player's hull pushes bodies out of its way and shoves them along — the
/// demo's stand-in for character-vs-physics collision.
pub fn shove(bodies: &mut [Body], p: &Player) {
	let hull = p.hull();
	for b in bodies {
		let d = b.pos - p.origin;
		let flat = vec3(d.x, 0.0, d.z);
		let dist = flat.length();
		let reach = hull.radius + b.radius;
		if dist >= reach || d.y.abs() >= hull.half_height + b.half_height + b.radius {
			continue;
		}
		if dist == 0.0 {
			b.pos.x += 0.5;
			continue;
		}
		let dir = flat / dist;
		b.pos += dir * (reach - dist + 0.5);
		let push = p.velocity.dot(dir).max(0.0);
		b.vel += dir * (push * 1.5 + 40.0);
	}
}

/// Returns a pen body that has come to rest on the pen floor to the top of the
/// chute, so the pen never runs down.
pub fn recycle(bodies: &mut [Body]) {
	let (lo, hi) = PEN;
	for (i, b) in bodies.iter_mut().enumerate() {
		let rest = b.radius + b.half_height;
		let in_pen = b.pos.x > lo.x && b.pos.x < hi.x && b.pos.z > lo.z && b.pos.z < hi.z;
		if in_pen && b.pos.y < rest + 4.0 && b.vel.x.hypot(b.vel.z) < 10.0 {
			b.pos = CHUTE_TOP + vec3(0.0, i as f32 * 36.0, 0.0);
			b.vel = vec3(64.0, 0.0, 0.0);
			b.spin = Vec3::ZERO;
			if b.half_height > 0.0 {
				b.orient = lying();
			}
		}
	}
}

/// Draws the bodies and platforms between the last tick (`prev`) and this one
/// (`now`), `alpha` of the way across; `player` is drawn too when given, as the
/// flycam sees it, on the drawn plane of `Player::step_offset`.
pub fn draw(prev: &World, now: &World, alpha: f32, player: Option<&Player>) {
	for (i, (was, is)) in prev.bodies.iter().zip(&now.bodies).enumerate() {
		let b = was.interpolate(is, alpha);
		let color = if i < YARD {
			if b.half_height > 0.0 {
				LOG
			} else {
				BALL
			}
		} else {
			PEN_BODY
		};
		body(&b, color);
	}
	for (i, (was, is)) in prev.platforms.iter().zip(&now.platforms).enumerate() {
		let at = was.offset(prev.time).lerp(is.offset(now.time), alpha);
		for &(min, max) in PLATFORMS[i] {
			let (center, size) = ((min + max) * 0.5, max - min);
			draw_cube((at + center).into(), size.into(), None, STEEL_PLAT);
		}
	}
	if let Some(p) = player {
		let hull = p.hull();
		let drawn = p.origin + vec3(0.0, p.step_offset, 0.0);
		let half = vec3(0.0, hull.half_height, 0.0);
		draw_cylinder(
			(drawn - half).into(),
			hull.radius,
			hull.radius,
			hull.half_height * 2.0,
			None,
			PLAYER_BODY,
		);
		draw_sphere((drawn + half).into(), hull.radius, None, PLAYER_BODY);
		draw_sphere((drawn - half).into(), hull.radius, None, PLAYER_BODY);
	}
}

/// One body: a sphere, or a capsule wherever its orientation leaves it.
fn body(b: &Body, color: Color) {
	if b.half_height == 0.0 {
		draw_sphere(b.pos.into(), b.radius, None, color);
		return;
	}
	let gl = unsafe { get_internal_gl() };
	gl.quad_gl
		.push_model_matrix(Mat4::from_rotation_translation(b.orient, b.pos).into());
	let ends = [
		mq::vec3(0.0, b.half_height, 0.0),
		mq::vec3(0.0, -b.half_height, 0.0),
	];
	draw_cylinder(
		mq::vec3(0.0, -b.half_height, 0.0),
		b.radius,
		b.radius,
		b.half_height * 2.0,
		None,
		color,
	);
	for e in ends {
		draw_sphere(e, b.radius, None, color);
	}
	gl.quad_gl.pop_model_matrix();
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_pen_holds_its_bodies() {
		let (arena, _) = crate::map::surfaced_arena();
		let mut w = world(&arena, 800.0);
		let dt = 1.0 / 128.0;
		for _ in 0..640 {
			recycle(&mut w.bodies[YARD..]);
			w.step(dt, true);
		}
		let (lo, hi) = PEN;
		for (i, b) in w.bodies[YARD..].iter().enumerate() {
			let r = b.radius + b.half_height;
			assert!(
				b.pos.x > lo.x - r
					&& b.pos.x < hi.x + r
					&& b.pos.z > lo.z - r
					&& b.pos.z < hi.z + r
					&& b.pos.y > -1.0,
				"body {i} escaped the pen at {:?}",
				b.pos
			);
			assert!(b.penetration(&arena) < 2.0, "body {i} sunk at {:?}", b.pos);
		}
	}

	#[test]
	fn the_player_shoves_balls_aside() {
		let (arena, _) = crate::map::surfaced_arena();
		let p = Player::spawn(Vec3::ZERO);
		let mut bodies = [Body {
			pos: vec3(24.0, 16.0, 0.0),
			radius: 16.0,
			..Body::default()
		}];
		shove(&mut bodies, &p);
		let reach = p.hull().radius + bodies[0].radius;
		let gap = bodies[0].pos.x - p.origin.x;
		assert!(gap > reach, "ball still overlaps at {gap} of {reach}");
		assert!(
			bodies[0].vel.x >= 40.0,
			"ball was not shoved: {:?}",
			bodies[0].vel
		);
		assert!(bodies[0].penetration(&arena) < 2.0);
	}

	#[test]
	fn yard_bodies_start_clear_of_each_other() {
		let (arena, _) = crate::map::surfaced_arena();
		let w = world(&arena, 800.0);
		for i in 0..w.bodies.len() {
			for j in i + 1..w.bodies.len() {
				let (a, b) = (&w.bodies[i], &w.bodies[j]);
				let gap = (a.pos - b.pos).length() - a.radius - b.radius - a.half_height - b.half_height;
				assert!(gap > 0.0, "bodies {i} and {j} overlap by {}", -gap);
			}
		}
	}
}
