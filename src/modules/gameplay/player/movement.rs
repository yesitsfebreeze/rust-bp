//! Counter-Strike 1.6 player movement: `PM_PlayerMove` for a walking player, ported
//! from the decompiled `cs.so` and cross-checked against ReGameDLL_CS `pm_shared.cpp`.
//!
//! Differences from the original, all deliberate:
//! - The player is a vertical cylinder swept through a triangle mesh
//!   ([`physics::Mesh::cast_cylinder`]) instead of a box through BSP clip hulls. Its
//!   flat ends land on floors and step tops as the box does, `DIST_EPSILON` away;
//!   its round side makes movement the same in every direction instead of
//!   catching on the box's corners.
//! - Y is up (GoldSrc uses Z) and sizes stay in GoldSrc units (1 unit ≈ 1 inch).
//! - Ticks run at a fixed rate with fractional milliseconds instead of one tick per
//!   client frame with whole milliseconds, so movement no longer depends on frame rate.
//! - Stuck players try a short list of nudges instead of the original's table of
//!   random offsets.
//! - Water, ladders, moving platforms, sounds and spectator modes are not ported.
//! - With `sv_stepsmooth` (the default), a walking player snaps down onto steps up to
//!   `sv_stepsize` below instead of falling off each one, and the drawn height rides a
//!   virtual plane at `sv_stepangle` over the steps the collision snapped through
//!   ([`Player::step_offset`]): it rises and falls with the distance walked, like a
//!   ramp, so the character never bobs through stairs. Collision still steps
//!   instantly, as in Quake and Source — the flat hull cannot slide up a step face.
//!   Off is CS 1.6 exactly.

use engine::linalg::{lerp, vec3, Vec3};
use engine::physics::{Cylinder, Mesh};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollisionQueries {
	pub casts: u32,
	pub overlaps: u32,
	/// Time spent inside the mesh queries, nanoseconds.
	pub nanos: u64,
	/// The longest single mesh query, nanoseconds.
	pub longest_nanos: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct CollisionQueryCounter {
	active: bool,
	counts: CollisionQueries,
}

thread_local! {
	static COLLISION_QUERIES: Cell<CollisionQueryCounter> = const { Cell::new(CollisionQueryCounter {
		active: false,
		counts: CollisionQueries {
			casts: 0,
			overlaps: 0,
			nanos: 0,
			longest_nanos: 0,
		},
	}) };
}

pub fn begin_collision_query_count() {
	COLLISION_QUERIES.with(|counter| {
		counter.set(CollisionQueryCounter {
			active: true,
			counts: CollisionQueries::default(),
		})
	});
}

pub fn take_collision_query_count() -> CollisionQueries {
	COLLISION_QUERIES.with(|counter| {
		let previous = counter.replace(CollisionQueryCounter::default());
		previous.counts
	})
}

/// A mesh query of kind `overlap` that took `elapsed`: counted, and its time
/// added to the tick's collision total and worst-query record.
fn count_collision_query(overlap: bool, elapsed: std::time::Duration) {
	COLLISION_QUERIES.with(|counter| {
		let mut state = counter.get();
		if !state.active {
			return;
		}
		let nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
		if overlap {
			state.counts.overlaps = state.counts.overlaps.saturating_add(1);
		} else {
			state.counts.casts = state.counts.casts.saturating_add(1);
		}
		state.counts.nanos = state.counts.nanos.saturating_add(nanos);
		state.counts.longest_nanos = state.counts.longest_nanos.max(nanos);
		counter.set(state);
	});
}

pub const UP: Vec3 = vec3(0.0, 1.0, 0.0);

/// Standing hull: 72 tall and 32 across, as GoldSrc's 32×32×72 box.
pub const STANDING: Cylinder = Cylinder {
	radius: 16.0,
	half_height: 36.0,
};
/// Ducked hull: 36 tall.
pub const DUCKED: Cylinder = Cylinder {
	radius: 16.0,
	half_height: 18.0,
};

/// Gap traces keep from surfaces, GoldSrc's `DIST_EPSILON`.
pub const SKIN: f32 = 1.0 / 32.0;
pub const VIEW_STANDING: f32 = 17.0;
pub const VIEW_DUCKED: f32 = 12.0;
/// How far the origin drops when a duck completes on the ground: the difference of
/// the hulls' bottoms, so the feet stay put.
const DUCK_DROP: f32 = STANDING.half_height - DUCKED.half_height;
/// sqrt(2 · 800 · 45): a 45 unit jump at default gravity.
const JUMP_SPEED: f32 = 268.328_16;
/// Milliseconds of slowdown after a jump (`fuser2`); it scales the next jump and
/// ground speed by up to 25%.
const JUMP_PENALTY_MS: f32 = 1_315.789_4;
const DUCK_MOVE_SCALE: f32 = 0.333;
const TIME_TO_DUCK: f32 = 0.4;
/// Wish speed cap while airborne: what makes air strafing work.
const AIR_WISH_CAP: f32 = 30.0;
/// Surfaces with a normal this upright or more are floors.
const FLOOR: f32 = 0.7;
/// Velocity above this upward is never on the ground.
const LAUNCH_SPEED: f32 = 180.0;
/// Jumping faster than this multiple of max speed scales velocity by `BUNNYHOP_SCALE`
/// of the limit (`PM_PreventMegaBunnyJumping`).
const BUNNYHOP_LIMIT: f32 = 1.2;
const BUNNYHOP_SCALE: f32 = 0.8;
const FALL_PUNCH_THRESHOLD: f32 = 250.0;
const MAX_CLIP_PLANES: usize = 5;
const MAX_BUMPS: usize = 4;
/// `STOP_EPSILON`: clipped velocity components smaller than this become zero.
const STOP_EPSILON: f32 = 0.1;
/// Slowest the drawn height catches up with a step, u/s: Quake's stair smoothing rate,
/// so a step taken while nearly still does not leave the view behind for long.
const STEP_SMOOTH_MIN_RATE: f32 = 80.0;

/// The movement cvars a tick reads, gathered once per tick by the game from its
/// config. Defaults are Counter-Strike 1.6's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovementVars {
	pub gravity: f32,
	pub accelerate: f32,
	pub airaccelerate: f32,
	pub friction: f32,
	pub edgefriction: f32,
	pub stopspeed: f32,
	/// Effective max speed: the weapon's, capped by the server's.
	pub maxspeed: f32,
	pub maxvelocity: f32,
	pub stepsize: f32,
	/// Step down onto stairs and ease the drawn height through steps.
	pub stepsmooth: bool,
	/// Slope in degrees the drawn height follows through a step at walking speed.
	pub stepangle: f32,
}

/// One tick of player input (`usercmd_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Input {
	/// Wanted speed forward, to the right and (noclip only) up, u/s.
	pub forward: f32,
	pub side: f32,
	pub up: f32,
	/// View angles in degrees: yaw turns right, pitch looks down, as in GoldSrc.
	pub yaw: f32,
	pub pitch: f32,
	pub jump: bool,
	pub duck: bool,
}

/// Forward, right and up of a view, GoldSrc's `AngleVectors` without roll.
pub fn angle_vectors(yaw: f32, pitch: f32) -> (Vec3, Vec3, Vec3) {
	let (sy, cy) = yaw.to_radians().sin_cos();
	let (sp, cp) = pitch.to_radians().sin_cos();
	let forward = vec3(cp * cy, -sp, cp * sy);
	let right = vec3(-sy, 0.0, cy);
	(forward, right, right.cross(forward))
}

/// Result of sweeping the player's hull.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trace {
	/// Fraction of the way travelled; 1 when nothing was hit.
	pub fraction: f32,
	pub end: Vec3,
	/// Plane that stopped the hull: GoldSrc's `trace.plane.normal`.
	pub normal: Option<Vec3>,
}

/// `PM_PlayerTrace`: sweeps `hull` from `start` toward `end`.
pub fn trace(world: &Mesh, hull: Cylinder, start: Vec3, end: Vec3) -> Trace {
	let motion = end - start;
	let started = std::time::Instant::now();
	let hit = world.cast_cylinder(start, hull, motion, SKIN);
	count_collision_query(false, started.elapsed());
	match hit {
		Some(c) => Trace {
			fraction: c.fraction,
			end: start + motion * c.fraction,
			normal: Some(c.normal),
		},
		None => Trace {
			fraction: 1.0,
			end,
			normal: None,
		},
	}
}

/// `PM_TestPlayerPosition`: whether `hull` at `origin` overlaps nothing.
pub fn fits(world: &Mesh, hull: Cylinder, origin: Vec3) -> bool {
	let started = std::time::Instant::now();
	let overlaps = world.cylinder_overlaps(origin, hull);
	count_collision_query(true, started.elapsed());
	!overlaps
}

/// `PM_ClipVelocity` with an overbounce of 1: removes the motion into `normal`.
fn clip_velocity(v: Vec3, normal: Vec3) -> Vec3 {
	let out = v - normal * v.dot(normal);
	let settle = |c: f32| if c.abs() < STOP_EPSILON { 0.0 } else { c };
	out.map(settle)
}

fn horizontal(v: Vec3) -> Vec3 {
	vec3(v.x, 0.0, v.z)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Player {
	/// Centre of the hull.
	pub origin: Vec3,
	pub velocity: Vec3,
	pub on_ground: bool,
	/// Fully ducked: the small hull is in use (`FL_DUCKING`).
	pub ducked: bool,
	/// Going down but not yet ducked (`bInDuck`); the standing hull is still in use.
	pub in_duck: bool,
	/// Milliseconds left of the duck transition, counting down from 1000.
	pub duck_time: f32,
	/// Milliseconds of post-jump slowdown left (`fuser2`).
	pub jump_penalty: f32,
	/// Eye height above the origin.
	pub view_offset: f32,
	/// Downward speed while airborne, for the landing punch.
	pub fall_velocity: f32,
	/// View roll in degrees from a hard landing, decaying each tick.
	pub punch_roll: f32,
	/// Drawn height minus the origin's after a step snap: the negated snap, shrinking
	/// to zero as the player moves on, so the view climbs or descends steps along
	/// `stepangle` instead of jumping. Collision never sees it.
	pub step_offset: f32,
	pub noclip: bool,
	jump_held: bool,
	duck_held: bool,
}

impl Player {
	/// Standing with its feet at `feet`.
	pub fn spawn(feet: Vec3) -> Player {
		Player {
			origin: feet + UP * (STANDING.half_height + SKIN),
			velocity: Vec3::ZERO,
			on_ground: false,
			ducked: false,
			in_duck: false,
			duck_time: 0.0,
			jump_penalty: 0.0,
			view_offset: VIEW_STANDING,
			fall_velocity: 0.0,
			punch_roll: 0.0,
			step_offset: 0.0,
			noclip: false,
			jump_held: false,
			duck_held: false,
		}
	}

	pub fn hull(&self) -> Cylinder {
		if self.ducked {
			DUCKED
		} else {
			STANDING
		}
	}

	pub fn eye(&self) -> Vec3 {
		self.origin + UP * (self.view_offset + self.step_offset)
	}

	pub fn speed(&self) -> f32 {
		self.velocity.x.hypot(self.velocity.z)
	}

	/// This tick's state moved `alpha` of the way toward `next`, for drawing between
	/// ticks. Only what the camera shows is blended.
	pub fn interpolate(&self, next: &Player, alpha: f32) -> Player {
		Player {
			origin: self.origin.lerp(next.origin, alpha),
			view_offset: lerp(self.view_offset, next.view_offset, alpha),
			punch_roll: lerp(self.punch_roll, next.punch_roll, alpha),
			step_offset: lerp(self.step_offset, next.step_offset, alpha),
			..*next
		}
	}

	/// `PM_PlayerMove` for one tick of `dt` seconds.
	pub fn tick(&mut self, input: Input, vars: &MovementVars, world: &Mesh, dt: f32) {
		let mut cmd = input;
		// PM_CheckParameters: the wanted speed never exceeds max speed.
		let wanted = (cmd.forward * cmd.forward + cmd.side * cmd.side + cmd.up * cmd.up).sqrt();
		if wanted > vars.maxspeed && wanted > 0.0 {
			let ratio = vars.maxspeed / wanted;
			(cmd.forward, cmd.side, cmd.up) = (cmd.forward * ratio, cmd.side * ratio, cmd.up * ratio);
		}
		self.drop_punch(dt);

		// PM_ReduceTimers.
		let ms = dt * 1000.0;
		self.duck_time = (self.duck_time - ms).max(0.0);
		self.jump_penalty = (self.jump_penalty - ms).max(0.0);

		if self.noclip {
			self.fly(&cmd, dt);
			self.step_offset = 0.0;
			return;
		}

		self.unstick(world);
		self.categorize(world);
		if !self.on_ground {
			self.fall_velocity = -self.velocity.y;
		}

		let scale = self.duck(cmd.duck, world);
		(cmd.forward, cmd.side) = (cmd.forward * scale, cmd.side * scale);

		// Half the gravity before moving and half after: jump height then barely
		// depends on the tick rate.
		self.add_half_gravity(vars, dt);
		if cmd.jump {
			self.jump(vars, dt);
		} else {
			self.jump_held = false;
		}
		if self.on_ground {
			self.velocity.y = 0.0;
			self.friction(vars, world, dt);
		}
		self.check_velocity(vars);

		let (forward, right, _) = angle_vectors(cmd.yaw, cmd.pitch);
		let wish =
			horizontal(forward).normalize() * cmd.forward + horizontal(right).normalize() * cmd.side;
		let mut wishspeed = wish.length();
		let wishdir = if wishspeed > 0.0 {
			wish * (1.0 / wishspeed)
		} else {
			Vec3::ZERO
		};
		wishspeed = wishspeed.min(vars.maxspeed);
		// How far this tick moved, for the drawn height's ramp along `stepangle`.
		let start = self.origin;
		if self.on_ground {
			self.walk_move(wishdir, wishspeed, vars, world, dt);
			self.stick_to_floor(vars, world);
		} else {
			self.air_accelerate(wishdir, wishspeed, vars.airaccelerate, dt);
			self.fly_move(world, dt);
		}

		self.categorize(world);
		self.check_velocity(vars);
		self.add_half_gravity(vars, dt);
		if self.on_ground {
			self.velocity.y = 0.0;
		}
		self.check_falling();
		self.settle_step(vars, dt, self.origin - start);
	}

	/// Eases the drawn height along a virtual plane at `stepangle` over the steps the
	/// collision snapped through: `step_offset` shrinks by the distance just walked
	/// times `tan(stepangle)`, so the view climbs or descends a step like a ramp while
	/// the player walks, and standing still it catches up at `STEP_SMOOTH_MIN_RATE`.
	/// Collision still snaps at once — the flat-bottomed hull cannot slide up a step
	/// face — so the plane is the drawn side of the step, never the collision's.
	fn settle_step(&mut self, vars: &MovementVars, dt: f32, moved: Vec3) {
		if !vars.stepsmooth {
			self.step_offset = 0.0;
			return;
		}
		let rise = horizontal(moved).length() * vars.stepangle.to_radians().tan();
		let left = (self.step_offset.abs() - rise.max(STEP_SMOOTH_MIN_RATE * dt))
			.clamp(0.0, vars.stepsize.max(0.0));
		self.step_offset = self.step_offset.signum() * left;
	}

	/// Keeps a walking player on stairs going down: when the move left no floor within
	/// `categorize`'s 2 units, snaps onto one within `stepsize` below. CS 1.6 has no
	/// such pass; there the player falls off every step.
	fn stick_to_floor(&mut self, vars: &MovementVars, world: &Mesh) {
		if !vars.stepsmooth || self.velocity.y > 0.0 {
			return;
		}
		let hull = self.hull();
		let floor = |tr: &Trace| tr.normal.is_some_and(|n| n.y >= FLOOR);
		if floor(&trace(world, hull, self.origin, self.origin - UP * 2.0)) {
			return;
		}
		let tr = trace(world, hull, self.origin, self.origin - UP * vars.stepsize);
		if floor(&tr) {
			self.step_offset += self.origin.y - tr.end.y;
			self.origin = tr.end;
		}
	}

	fn add_half_gravity(&mut self, vars: &MovementVars, dt: f32) {
		self.velocity.y -= vars.gravity * 0.5 * dt;
		self.check_velocity(vars);
	}

	fn check_velocity(&mut self, vars: &MovementVars) {
		let bound = |c: f32| {
			if c.is_nan() {
				0.0
			} else {
				c.clamp(-vars.maxvelocity, vars.maxvelocity)
			}
		};
		self.velocity = self.velocity.map(bound);
	}

	/// `PM_CategorizePosition`: on the ground when a floor is within 2 units below,
	/// and then snapped onto it.
	fn categorize(&mut self, world: &Mesh) {
		if self.velocity.y > LAUNCH_SPEED {
			self.on_ground = false;
			return;
		}
		let tr = trace(world, self.hull(), self.origin, self.origin - UP * 2.0);
		self.on_ground = tr.normal.is_some_and(|n| n.y >= FLOOR);
		if self.on_ground {
			self.origin = tr.end;
		}
	}

	/// `PM_Duck`; returns the scale for this tick's movement keys.
	fn duck(&mut self, down: bool, world: &Mesh) -> f32 {
		let pressed = down && !self.duck_held;
		self.duck_held = down;
		if !down && !self.in_duck && !self.ducked {
			return 1.0;
		}
		if down {
			if pressed && !self.ducked {
				self.duck_time = 1000.0;
				self.in_duck = true;
			}
			if self.in_duck {
				let done = self.duck_time / 1000.0;
				if done <= 1.0 - TIME_TO_DUCK || !self.on_ground {
					self.ducked = true;
					self.in_duck = false;
					self.view_offset = VIEW_DUCKED;
					if self.on_ground {
						self.origin.y -= DUCK_DROP;
						self.fix_crouch_stuck(world);
						self.categorize(world);
					}
				} else {
					// Eases the eye from standing to its ducked height measured from
					// the undropped origin.
					let t = ((1.0 - done) / TIME_TO_DUCK).max(0.0);
					let f = 3.0 * t * t - 2.0 * t * t * t;
					self.view_offset = (VIEW_DUCKED - DUCK_DROP) * f + VIEW_STANDING * (1.0 - f);
				}
			}
		} else {
			self.unduck(world);
		}
		DUCK_MOVE_SCALE
	}

	/// `PM_UnDuck`. Releasing duck before it completes still raises the origin as if
	/// it had: the 18 unit hop of the "duck tap".
	fn unduck(&mut self, world: &Mesh) {
		let origin = if self.on_ground {
			self.origin + UP * DUCK_DROP
		} else {
			self.origin
		};
		if !fits(world, self.hull(), origin) || !fits(world, STANDING, origin) {
			return;
		}
		self.ducked = false;
		self.in_duck = false;
		self.view_offset = VIEW_STANDING;
		self.duck_time = 0.0;
		self.origin = origin;
		self.categorize(world);
	}

	/// `PM_FixPlayerCrouchStuck` upward: tries up to 36 one-unit raises.
	fn fix_crouch_stuck(&mut self, world: &Mesh) {
		let start = self.origin;
		for _ in 0..=36 {
			if fits(world, self.hull(), self.origin) {
				return;
			}
			self.origin.y += 1.0;
		}
		self.origin = start;
	}

	/// Replaces `PM_CheckStuck`: pushes an overlapping player out along its deepest
	/// overlap, repeating for corners where pushing out of one wall leaves it in
	/// another. Should that not free it, the nearest of a few nudges, up first, that
	/// fits.
	fn unstick(&mut self, world: &Mesh) {
		let hull = self.hull();
		for _ in 0..8 {
			let Some((out, depth)) = world.cylinder_penetration(self.origin, hull) else {
				return;
			};
			self.origin += out * (depth + SKIN);
		}
		if fits(world, hull, self.origin) {
			return;
		}
		let directions = [
			UP,
			vec3(1.0, 0.0, 0.0),
			vec3(-1.0, 0.0, 0.0),
			vec3(0.0, 0.0, 1.0),
			vec3(0.0, 0.0, -1.0),
			-UP,
		];
		for step in 1..=36 {
			for d in directions {
				let at = self.origin + d * step as f32;
				if fits(world, hull, at) {
					self.origin = at;
					return;
				}
			}
		}
	}

	/// `PM_Jump` for a player out of water.
	fn jump(&mut self, vars: &MovementVars, dt: f32) {
		if !self.on_ground {
			self.jump_held = true;
			return;
		}
		// No jump until the key is released: no auto bunny hop.
		if self.jump_held || (self.in_duck && self.ducked) {
			return;
		}
		self.on_ground = false;
		let limit = vars.maxspeed * BUNNYHOP_LIMIT;
		let speed = self.velocity.length();
		if limit > 0.0 && speed > limit {
			self.velocity *= limit / speed * BUNNYHOP_SCALE;
		}
		self.velocity.y = JUMP_SPEED * self.penalty_ratio();
		self.jump_penalty = JUMP_PENALTY_MS;
		self.add_half_gravity(vars, dt);
		self.jump_held = true;
	}

	/// Speed kept while the jump penalty lasts.
	fn penalty_ratio(&self) -> f32 {
		if self.jump_penalty > 0.0 {
			(100.0 - self.jump_penalty * 0.001 * 19.0) * 0.01
		} else {
			1.0
		}
	}

	/// `PM_Friction` on the ground; doubled near a drop-off by `edgefriction`.
	fn friction(&mut self, vars: &MovementVars, world: &Mesh, dt: f32) {
		let speed = self.velocity.length();
		if speed < 0.1 {
			return;
		}
		// Is there floor 16 units ahead within 34 below the feet?
		let hull = self.hull();
		let ahead = horizontal(self.velocity) * (16.0 / speed);
		let start = self.origin + ahead - UP * hull.half_height;
		let edge = trace(world, hull, start, start - UP * 34.0).fraction == 1.0;
		let friction = vars.friction * if edge { vars.edgefriction } else { 1.0 };
		let drop = speed.max(vars.stopspeed) * friction * dt;
		self.velocity *= (speed - drop).max(0.0) / speed;
	}

	fn accelerate(&mut self, wishdir: Vec3, wishspeed: f32, accel: f32, dt: f32) {
		let add = wishspeed - self.velocity.dot(wishdir);
		if add > 0.0 {
			self.velocity += wishdir * add.min(accel * dt * wishspeed);
		}
	}

	/// Like `accelerate`, but only up to `AIR_WISH_CAP` along the wish direction, with
	/// the full wish speed's acceleration: turning while strafing gains speed.
	fn air_accelerate(&mut self, wishdir: Vec3, wishspeed: f32, accel: f32, dt: f32) {
		let add = wishspeed.min(AIR_WISH_CAP) - self.velocity.dot(wishdir);
		if add > 0.0 {
			self.velocity += wishdir * add.min(accel * dt * wishspeed);
		}
	}

	/// `PM_WalkMove`: accelerate along the ground, then move, stepping up stairs.
	fn walk_move(
		&mut self,
		wishdir: Vec3,
		wishspeed: f32,
		vars: &MovementVars,
		world: &Mesh,
		dt: f32,
	) {
		let ratio = self.penalty_ratio();
		self.velocity.x *= ratio;
		self.velocity.z *= ratio;

		self.velocity.y = 0.0;
		self.accelerate(wishdir, wishspeed, vars.accelerate, dt);
		self.velocity.y = 0.0;
		if self.velocity.length() < 1.0 {
			self.velocity = Vec3::ZERO;
			return;
		}

		let hull = self.hull();
		let tr = trace(
			world,
			hull,
			self.origin,
			self.origin + horizontal(self.velocity) * dt,
		);
		if tr.fraction == 1.0 {
			self.origin = tr.end;
			return;
		}

		// Blocked: slide along the ground, and separately step up, slide, and step
		// back down. Keep whichever got farther.
		let (original, original_velocity) = (self.origin, self.velocity);
		self.fly_move_with_first_trace(world, dt, Some(tr));
		let (down, down_velocity) = (self.origin, self.velocity);

		(self.origin, self.velocity) = (original, original_velocity);
		self.origin = trace(world, hull, self.origin, self.origin + UP * vars.stepsize).end;
		self.fly_move(world, dt);
		// `SKIN` past `stepsize`: the cast stops within `SKIN` of a surface, and a
		// floor at the step's own height — the top of a low ridge, flat ground past
		// a lip — sits just inside that margin. An exact `stepsize` trace's last
		// sweep step overshoots it, finds no floor, and rejects the step, leaving
		// the clipped slide to snag on the lip.
		let tr = trace(
			world,
			hull,
			self.origin,
			self.origin - UP * (vars.stepsize + SKIN),
		);
		if !tr.normal.is_some_and(|n| n.y >= FLOOR) {
			(self.origin, self.velocity) = (down, down_velocity);
			return;
		}
		self.origin = tr.end;
		let flat = |p: Vec3| horizontal(p - original).length2();
		if flat(down) > flat(self.origin) {
			(self.origin, self.velocity) = (down, down_velocity);
		} else {
			self.velocity.y = down_velocity.y;
			if vars.stepsmooth {
				self.step_offset -= self.origin.y - original.y;
			}
		}
	}

	/// `PM_FlyMove`: moves for `dt`, sliding along up to five surfaces.
	fn fly_move(&mut self, world: &Mesh, dt: f32) {
		self.fly_move_with_first_trace(world, dt, None);
	}

	fn fly_move_with_first_trace(&mut self, world: &Mesh, dt: f32, mut first: Option<Trace>) {
		let hull = self.hull();
		let primal = self.velocity;
		let mut original = self.velocity;
		let mut planes = [Vec3::ZERO; MAX_CLIP_PLANES];
		let mut count = 0;
		let mut left = dt;
		let mut travelled = 0.0;
		for _ in 0..MAX_BUMPS {
			if self.velocity == Vec3::ZERO {
				break;
			}
			let tr = first
				.take()
				.unwrap_or_else(|| trace(world, hull, self.origin, self.origin + self.velocity * left));
			travelled += tr.fraction;
			if tr.fraction > 0.0 {
				self.origin = tr.end;
				original = self.velocity;
				count = 0;
			}
			let Some(normal) = tr.normal else {
				break;
			};
			left -= left * tr.fraction;
			if count >= MAX_CLIP_PLANES {
				self.velocity = Vec3::ZERO;
				break;
			}
			planes[count] = normal;
			count += 1;

			if count == 1 && !self.on_ground {
				self.velocity = clip_velocity(original, normal);
				original = self.velocity;
				continue;
			}
			// Find a plane whose clip keeps clear of all the others.
			let planes = &planes[..count];
			let clear = planes.iter().find_map(|&p| {
				let v = clip_velocity(original, p);
				planes
					.iter()
					.all(|&q| q == p || v.dot(q) >= 0.0)
					.then_some(v)
			});
			match clear {
				Some(v) => self.velocity = v,
				// In a crease of two planes: slide along it.
				None if count == 2 => {
					let dir = planes[0].cross(planes[1]);
					self.velocity = dir * dir.dot(self.velocity);
				}
				None => {
					self.velocity = Vec3::ZERO;
					break;
				}
			}
			// Never turn back against the original motion: that bounces in corners.
			if self.velocity.dot(primal) <= 0.0 {
				self.velocity = Vec3::ZERO;
				break;
			}
		}
		if travelled == 0.0 {
			self.velocity = Vec3::ZERO;
		}
	}

	/// `PM_CheckFalling`: a hard landing rolls the view.
	fn check_falling(&mut self) {
		if !self.on_ground {
			return;
		}
		if self.fall_velocity >= FALL_PUNCH_THRESHOLD {
			self.punch_roll = self.fall_velocity * 0.013;
		}
		self.fall_velocity = 0.0;
	}

	/// `PM_DropPunchAngle`.
	fn drop_punch(&mut self, dt: f32) {
		let len = self.punch_roll.abs();
		let left = (len - (10.0 + len * 0.5) * dt).max(0.0);
		self.punch_roll = self.punch_roll.signum() * left;
	}

	/// `PM_NoClip`: flies where the view points, through everything.
	fn fly(&mut self, cmd: &Input, dt: f32) {
		let (forward, right, _) = angle_vectors(cmd.yaw, cmd.pitch);
		let wish = forward * cmd.forward + right * cmd.side + UP * cmd.up;
		self.origin += wish * dt;
		self.velocity = Vec3::ZERO;
		self.on_ground = false;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const RATE: f32 = 128.0;

	#[test]
	fn collision_query_counts_are_scoped_to_a_movement_tick() {
		let world = Mesh::default();
		begin_collision_query_count();
		trace(&world, STANDING, Vec3::ZERO, Vec3::ZERO);
		fits(&world, STANDING, Vec3::ZERO);
		let counted = take_collision_query_count();
		assert_eq!((counted.casts, counted.overlaps), (1, 1));
		assert!(
			counted.nanos > 0 && counted.longest_nanos > 0,
			"{counted:?}"
		);
		trace(&world, STANDING, Vec3::ZERO, Vec3::ZERO);
		assert_eq!(take_collision_query_count(), CollisionQueries::default());
	}

	fn floor() -> Mesh {
		let mut m = Mesh::default();
		for i in -16..16 {
			for k in -16..16 {
				let (x, z) = (i as f32 * 64.0, k as f32 * 64.0);
				m.push_quad(
					vec3(x, 0.0, z),
					vec3(x + 64.0, 0.0, z),
					vec3(x + 64.0, 0.0, z + 64.0),
					vec3(x, 0.0, z + 64.0),
					UP,
				);
			}
		}
		m
	}

	/// The movement variables of a default CS 1.6 server holding a knife.
	fn vars() -> MovementVars {
		MovementVars {
			gravity: 800.0,
			accelerate: 5.0,
			airaccelerate: 10.0,
			friction: 4.0,
			edgefriction: 2.0,
			stopspeed: 75.0,
			maxspeed: 250.0,
			maxvelocity: 2000.0,
			stepsize: 18.0,
			stepsmooth: true,
			stepangle: 45.0,
		}
	}

	/// Runs `secs` of ticks at `rate`, returning the state after each.
	fn run(p: &mut Player, world: &Mesh, input: Input, secs: f32, rate: f32) -> Vec<Player> {
		run_with(p, world, input, secs, rate, &vars())
	}

	fn run_with(
		p: &mut Player,
		world: &Mesh,
		input: Input,
		secs: f32,
		rate: f32,
		vars: &MovementVars,
	) -> Vec<Player> {
		let ticks = (secs * rate).round() as usize;
		(0..ticks)
			.map(|_| {
				p.tick(input, vars, world, 1.0 / rate);
				*p
			})
			.collect()
	}

	/// Six 16 unit steps, 32 deep, from x = 64 up to a platform at 96 from x = 256.
	fn stairs() -> Mesh {
		let mut world = floor();
		for s in 0..6 {
			let x = 64.0 + s as f32 * 32.0;
			world.push_box(
				vec3(x, 0.0, -64.0),
				vec3(x + 32.0, 16.0 * (s + 1) as f32, 64.0),
			);
		}
		world.push_box(vec3(256.0, 0.0, -64.0), vec3(512.0, 96.0, 64.0));
		world
	}

	/// Largest tick-to-tick change of the drawn height beyond what the virtual plane
	/// allows: the distance walked times `tan(stepangle)`, or the minimum rate.
	fn worst_view_jump(states: &[Player]) -> f32 {
		let tan = vars().stepangle.to_radians().tan();
		states
			.windows(2)
			.map(|w| {
				let rise = (w[1].eye().y - w[0].eye().y).abs();
				let walked = horizontal(w[1].origin - w[0].origin).length();
				rise - (walked * tan + STEP_SMOOTH_MIN_RATE / RATE)
			})
			.fold(f32::MIN, f32::max)
	}

	fn grounded(world: &Mesh, feet: Vec3) -> Player {
		let mut p = Player::spawn(feet);
		run(&mut p, world, Input::default(), 0.1, RATE);
		assert!(p.on_ground, "did not land: {p:?}");
		p
	}

	fn forward() -> Input {
		Input {
			forward: 400.0,
			..Input::default()
		}
	}

	#[test]
	fn runs_at_weapon_speed_and_no_faster() {
		let world = floor();
		let mut p = grounded(&world, vec3(-900.0, 0.0, 0.0));
		let states = run(&mut p, &world, forward(), 2.0, RATE);
		let top = states.iter().map(Player::speed).fold(0.0, f32::max);
		assert!(
			(p.speed() - 250.0).abs() < 0.5 && top < 250.01,
			"speed {} top {top}",
			p.speed()
		);
		assert!(states
			.iter()
			.all(|s| s.on_ground && (s.origin.y - p.origin.y).abs() < 1e-3));
		assert!(fits(&world, p.hull(), p.origin));
	}

	#[test]
	fn friction_stops_in_about_half_a_second() {
		let world = floor();
		let mut p = grounded(&world, vec3(-900.0, 0.0, 0.0));
		run(&mut p, &world, forward(), 2.0, RATE);
		let states = run(&mut p, &world, Input::default(), 1.0, RATE);
		let stop = states
			.iter()
			.position(|s| s.speed() == 0.0)
			.expect("never stopped") as f32
			/ RATE;
		// ln(250/75)/4 s of proportional friction, then 75/(4·75) s at stopspeed.
		assert!((stop - 0.55).abs() < 0.03, "stopped after {stop} s");
	}

	#[test]
	fn jump_is_45_units_at_any_tick_rate() {
		let world = floor();
		for rate in [64.0, 100.0, 128.0, 333.0] {
			let mut p = grounded(&world, Vec3::ZERO);
			let rest = p.origin.y;
			let jump = Input {
				jump: true,
				..Input::default()
			};
			let states = run(&mut p, &world, jump, 1.0, rate);
			let apex = states.iter().map(|s| s.origin.y).fold(f32::MIN, f32::max) - rest;
			assert!((apex - 45.0).abs() < 1.0, "apex {apex} at {rate} Hz");
			// Held through the landing: no second jump.
			assert!(p.on_ground && (p.origin.y - rest).abs() < 1e-3, "{p:?}");
		}
	}

	#[test]
	fn bunny_hop_speed_is_capped_on_takeoff() {
		let world = floor();
		let mut p = grounded(&world, Vec3::ZERO);
		p.velocity = vec3(400.0, 0.0, 0.0);
		let jump = Input {
			jump: true,
			..Input::default()
		};
		p.tick(jump, &vars(), &world, 1.0 / RATE);
		// The cap, 1.2 · 250 · 0.8, applies before friction and acceleration.
		assert!(!p.on_ground);
		assert!((p.speed() - 240.0).abs() < 0.05, "speed {}", p.speed());
	}

	#[test]
	fn strafing_while_turning_gains_air_speed() {
		let world = Mesh::default();
		let mut p = Player::spawn(Vec3::ZERO);
		p.velocity = vec3(250.0, 0.0, 0.0);
		let dt = 1.0 / RATE;
		let gain = vars().airaccelerate * dt * 250.0;
		for _ in 0..RATE as usize {
			let speed = p.speed();
			let along = ((AIR_WISH_CAP - gain) / speed).clamp(-1.0, 1.0).acos();
			let heading = p.velocity.z.atan2(p.velocity.x);
			// Strafing right: the wish direction is 90° right of the view.
			let yaw = (heading + along).to_degrees() - 90.0;
			let input = Input {
				side: 400.0,
				yaw,
				..Input::default()
			};
			p.tick(input, &vars(), &world, dt);
		}
		assert!(p.speed() > 400.0, "speed {}", p.speed());
		let mut q = Player::spawn(Vec3::ZERO);
		q.velocity = vec3(250.0, 0.0, 0.0);
		run(&mut q, &world, forward(), 1.0, RATE);
		assert!(q.speed() < 251.0, "holding forward gained {}", q.speed());
	}

	/// The arena's 30° ramp: prism rising along +x, slope and end faces and sides.
	fn ramp(world: &mut Mesh, at: Vec3, length: f32, height: f32, width: f32) {
		let (a, b, c) = (
			at,
			at + vec3(length, height, 0.0),
			at + vec3(length, 0.0, 0.0),
		);
		let w = vec3(0.0, 0.0, width);
		world.push_quad(a, b, b + w, a + w, vec3(-height, length, 0.0));
		world.push_quad(b, c, c + w, b + w, vec3(1.0, 0.0, 0.0));
		world.push_triangle([a, c, b]);
		world.push_triangle([a + w, b + w, c + w]);
	}

	#[test]
	fn walks_up_the_arena_ramp() {
		let mut world = floor();
		ramp(&mut world, vec3(256.0, 0.0, 256.0), 222.0, 128.0, 128.0);
		let mut p = grounded(&world, vec3(0.0, 0.0, 320.0));
		let states = run(&mut p, &world, forward(), 4.0, RATE);
		let stuck = states
			.iter()
			.find(|s| s.speed() < 10.0 && s.origin.x > 100.0);
		assert!(
			stuck.is_none(),
			"stuck at {:?}: {stuck:?}",
			stuck.map(|s| s.origin)
		);
		let peak = states
			.iter()
			.map(|s| s.origin.y - s.hull().half_height)
			.fold(0.0, f32::max);
		assert!(peak > 100.0, "never climbed: peak feet {peak}");
	}

	#[test]
	fn lands_on_a_ramp_at_its_seams() {
		let mut world = floor();
		ramp(&mut world, vec3(256.0, 0.0, 256.0), 222.0, 128.0, 128.0);
		// Mid-slope, the base seam with the floor, and the top edge over its drop.
		for (x, feet) in [(350.0, 54.0), (256.0, 36.0), (466.0, 122.0)] {
			let mut p = Player::spawn(vec3(x, feet, 320.0));
			let states = run(&mut p, &world, Input::default(), 1.5, RATE);
			assert!(p.on_ground, "never landed at {x}: {p:?}");
			let rest = p.origin.y - p.hull().half_height;
			assert!(
				states.iter().all(|s| fits(&world, s.hull(), s.origin)),
				"sunk into the seam at {x}: {p:?}"
			);
			// A flat-bottomed hull rests on its rim on a slope and slides while
			// settling, so only require it to be on or above the surface.
			let surface = (x - 256.0) * 128.0 / 222.0;
			assert!(rest > surface - 2.0, "below the slope at {x}: rest {rest}");
			// Walk on from where it landed without getting stuck.
			let before = p.origin.x;
			let walked = run(&mut p, &world, forward(), 1.0, RATE);
			let moved = walked.last().unwrap().origin.x - before;
			assert!(moved > 24.0, "stuck after landing at {x}: moved {moved}");
		}
	}

	#[test]
	fn walks_up_stairs_but_not_onto_tall_steps() {
		let mut world = floor();
		for s in 0..6 {
			let x = 64.0 + s as f32 * 32.0;
			world.push_box(
				vec3(x, 0.0, -64.0),
				vec3(x + 32.0, 16.0 * (s + 1) as f32, 64.0),
			);
		}
		world.push_box(vec3(256.0, 0.0, -64.0), vec3(512.0, 96.0, 64.0));
		let mut p = grounded(&world, Vec3::ZERO);
		run(&mut p, &world, forward(), 1.5, RATE);
		let feet = p.origin.y - p.hull().half_height;
		assert!(p.on_ground && (feet - 96.0).abs() < 0.1, "feet at {feet}");

		let mut world = floor();
		world.push_box(vec3(64.0, 0.0, -64.0), vec3(128.0, 24.0, 64.0));
		let mut p = grounded(&world, Vec3::ZERO);
		run(&mut p, &world, forward(), 1.0, RATE);
		assert!(p.origin.x < 64.0 - 16.0 + 0.1 && p.speed() < 1.0, "{p:?}");
	}

	#[test]
	fn view_climbs_stairs_along_the_step_angle() {
		let world = stairs();
		let mut p = grounded(&world, Vec3::ZERO);
		let states = run(&mut p, &world, forward(), 1.5, RATE);
		let snap = states
			.windows(2)
			.map(|w| w[1].origin.y - w[0].origin.y)
			.fold(0.0, f32::max);
		assert!(
			snap > 15.0,
			"collision should still step at once, largest rise {snap}"
		);
		let jump = worst_view_jump(&states);
		assert!(jump < 1e-3, "view jumped {jump} past the step angle");
		assert!(
			p.step_offset == 0.0 && (p.origin.y - p.hull().half_height - 96.0).abs() < 0.1,
			"{p:?}"
		);
	}

	#[test]
	fn stays_on_the_ground_down_stairs() {
		let world = stairs();
		let back = Input {
			yaw: 180.0,
			..forward()
		};
		let mut p = grounded(&world, vec3(400.0, 96.0, 0.0));
		let states = run(&mut p, &world, back, 2.0, RATE);
		assert!(
			states.iter().all(|s| s.on_ground),
			"left the ground going down"
		);
		let jump = worst_view_jump(&states);
		assert!(jump < 1e-3, "view jumped {jump} past the step angle");
		assert!(
			p.step_offset == 0.0 && (p.origin.y - p.hull().half_height).abs() < 0.1,
			"{p:?}"
		);

		// CS 1.6 falls off each step instead.
		let cs = MovementVars {
			stepsmooth: false,
			..vars()
		};
		let mut p = grounded(&world, vec3(400.0, 96.0, 0.0));
		let states = run_with(&mut p, &world, back, 2.0, RATE, &cs);
		assert!(states.iter().any(|s| !s.on_ground));
		assert!(states.iter().all(|s| s.step_offset == 0.0));
	}

	#[test]
	fn slides_along_walls() {
		let mut world = floor();
		world.push_box(vec3(64.0, 0.0, -512.0), vec3(96.0, 128.0, 512.0));
		let mut p = grounded(&world, Vec3::ZERO);
		let diagonal = Input {
			forward: 400.0,
			yaw: 45.0,
			..Input::default()
		};
		let states = run(&mut p, &world, diagonal, 1.0, RATE);
		assert!(states
			.iter()
			.all(|s| s.origin.x <= 64.0 - 16.0 && fits(&world, s.hull(), s.origin)));
		// Along the wall, acceleration toward the 45° wish balances friction near
		// 221 u/s: faster than 250 · cos 45°, as in GoldSrc.
		assert!(
			p.velocity.x == 0.0 && p.velocity.z > 210.0 && p.velocity.z < 222.0,
			"{:?}",
			p.velocity
		);
	}

	#[test]
	fn ducks_through_a_low_tunnel_and_stands_after() {
		let mut world = floor();
		world.push_box(vec3(64.0, 48.0, -128.0), vec3(256.0, 96.0, 128.0));
		let mut p = grounded(&world, Vec3::ZERO);
		let duck = Input {
			duck: true,
			..Input::default()
		};
		run(&mut p, &world, duck, 0.5, RATE);
		assert!(p.ducked && p.view_offset == VIEW_DUCKED);
		// Ducked, friction at stopspeed outweighs most of the acceleration below 75 u/s,
		// so reaching the 83 u/s ducked speed takes most of a second.
		run(
			&mut p,
			&world,
			Input {
				forward: 400.0,
				..duck
			},
			2.0,
			RATE,
		);
		assert!(p.origin.x > 64.0 && p.origin.x < 256.0, "{p:?}");
		// Releasing under the ceiling keeps the player ducked.
		run(&mut p, &world, Input::default(), 0.2, RATE);
		assert!(p.ducked);
		// Still ducked until clear of the ceiling, so at ducked speed from rest.
		run(&mut p, &world, forward(), 3.0, RATE);
		assert!(
			p.origin.x > 256.0 + 16.0 && !p.ducked && p.view_offset == VIEW_STANDING,
			"{p:?}"
		);
		assert!(p.speed() > 249.0);
	}

	#[test]
	fn duck_tap_hops_18_units() {
		let world = floor();
		let mut p = grounded(&world, Vec3::ZERO);
		let rest = p.origin.y;
		let duck = Input {
			duck: true,
			..Input::default()
		};
		p.tick(duck, &vars(), &world, 1.0 / RATE);
		assert!(p.in_duck && !p.ducked);
		p.tick(Input::default(), &vars(), &world, 1.0 / RATE);
		assert!(
			!p.on_ground && (p.origin.y - rest - DUCK_DROP).abs() < 0.5,
			"{p:?}"
		);
		run(&mut p, &world, Input::default(), 1.0, RATE);
		assert!(p.on_ground && (p.origin.y - rest).abs() < 1e-3);
	}

	#[test]
	fn steep_slopes_are_not_ground() {
		let mut world = floor();
		// 60° slope facing -x.
		let (a, b) = (vec3(0.0, 0.0, -256.0), vec3(148.0, 256.0, -256.0));
		let w = vec3(0.0, 0.0, 512.0);
		world.push_quad(a, b, b + w, a + w, vec3(-1.0, 0.5, 0.0));
		let mut p = Player::spawn(vec3(74.0 - 40.0, 140.0, 0.0));
		let states = run(&mut p, &world, Input::default(), 2.0, RATE);
		// Only the floor, reached in the corner at the slope's foot, holds the player.
		assert!(states
			.iter()
			.filter(|s| s.origin.y - s.hull().half_height > 1.0)
			.all(|s| !s.on_ground));
		assert!(p.on_ground && p.origin.x < 0.0, "{p:?}");
	}

	#[test]
	fn surfs_along_a_steep_ramp() {
		let mut world = floor();
		// 60° ramp rising toward +x, 2048 long along z.
		let (a, b) = (vec3(0.0, 0.0, -1024.0), vec3(296.0, 512.0, -1024.0));
		let w = vec3(0.0, 0.0, 2048.0);
		world.push_quad(a, b, b + w, a + w, vec3(-0.866, 0.5, 0.0));
		// Against the ramp, 300 up, gliding along it at 600 u/s.
		let n = vec3(-0.866_025_4, 0.5, 0.0);
		let on = vec3(148.0, 256.0, -900.0);
		let cyl = STANDING;
		let lift = cyl.radius * n.x.abs() + cyl.half_height * n.y;
		let mut p = Player::spawn(Vec3::ZERO);
		p.origin = on + n * (lift + 0.5 * SKIN + 0.01);
		p.velocity = vec3(0.0, 0.0, 600.0);
		// Facing along the ramp, strafing into it.
		let strafe = Input {
			side: -400.0,
			yaw: 90.0,
			..Input::default()
		};
		let states = run(&mut p, &world, strafe, 1.0, RATE);
		assert!(states.iter().all(|s| !s.on_ground), "stuck to the ramp");
		assert!(states.iter().all(|s| fits(&world, s.hull(), s.origin)));
		let travelled = p.origin.z - (-900.0);
		assert!(p.velocity.z > 590.0 && travelled > 590.0, "{p:?}");
		// Still on the ramp, sliding down it, not launched off.
		let gap = n.dot(p.origin - on) - lift;
		assert!(gap < 1.0, "left the ramp: gap {gap}");
	}

	#[test]
	fn fast_hits_into_a_corner_never_trap_the_player() {
		let mut world = floor();
		// A right-angled inside corner of two walls, and an acute one of the first
		// wall and a diagonal one.
		world.push_box(vec3(128.0, 0.0, -512.0), vec3(160.0, 256.0, 512.0));
		world.push_box(vec3(-512.0, 0.0, 128.0), vec3(160.0, 256.0, 160.0));
		world.push_quad(
			vec3(-512.0, 0.0, -512.0),
			vec3(128.0, 0.0, -128.0),
			vec3(128.0, 256.0, -128.0),
			vec3(-512.0, 256.0, -512.0),
			vec3(-0.5, 0.0, 0.8),
		);
		let dt = 1.0 / RATE;
		for (velocity, out) in [
			(vec3(2000.0, -300.0, 2000.0), vec3(-1.0, 0.0, -1.0)),
			(vec3(2000.0, 600.0, -1500.0), vec3(-1.0, 0.0, 0.5)),
			(vec3(2000.0, -2000.0, 2000.0), vec3(-1.0, 0.0, -1.0)),
		] {
			let mut p = grounded(&world, Vec3::ZERO);
			p.velocity = velocity;
			for _ in 0..32 {
				p.tick(Input::default(), &vars(), &world, dt);
				assert!(fits(&world, p.hull(), p.origin), "inside at {:?}", p.origin);
			}
			// Then walk out of the corner.
			let start = p.origin;
			let yaw = out.z.atan2(out.x).to_degrees();
			run(
				&mut p,
				&world,
				Input {
					forward: 400.0,
					yaw,
					..Input::default()
				},
				0.5,
				RATE,
			);
			assert!(
				(p.origin - start).dot(out.normalize()) > 50.0,
				"stuck at {:?} from {velocity:?}",
				p.origin
			);
		}
	}

	#[test]
	fn jumping_again_on_landing_costs_stamina() {
		let world = floor();
		let mut p = grounded(&world, Vec3::ZERO);
		let rest = p.origin.y;
		let jump = Input {
			jump: true,
			..Input::default()
		};
		p.tick(jump, &vars(), &world, 1.0 / RATE);
		while !p.on_ground {
			p.tick(Input::default(), &vars(), &world, 1.0 / RATE);
		}
		// CS 1.6 PM_Jump: the second jump's speed is scaled by
		// (100 - fuser2 / 1000 · 19) / 100, fuser2 counting down from 1315.79 ms.
		let left = p.jump_penalty - 1000.0 / RATE;
		let ratio = (100.0 - left * 0.001 * 19.0) * 0.01;
		let states = run(&mut p, &world, jump, 1.0, RATE);
		let apex = states.iter().map(|s| s.origin.y).fold(f32::MIN, f32::max) - rest;
		let expected = 45.0 * ratio * ratio;
		assert!(
			ratio < 0.9 && (apex - expected).abs() < 1.0,
			"apex {apex}, expected {expected}"
		);
	}

	#[test]
	fn one_air_tick_adds_exactly_cs_air_acceleration() {
		let world = Mesh::default();
		let mut p = Player::spawn(Vec3::ZERO);
		p.velocity = vec3(250.0, 0.0, 0.0);
		let dt = 1.0 / RATE;
		// Facing +x, strafing right (+z): PM_AirAccelerate adds
		// min(30 - v·wishdir, airaccelerate · dt · wishspeed) along the wish direction.
		p.tick(
			Input {
				side: 400.0,
				..Input::default()
			},
			&vars(),
			&world,
			dt,
		);
		let add = (30.0f32).min(vars().airaccelerate * dt * 250.0);
		assert!(
			(p.velocity.z - add).abs() < 1e-3 && (p.velocity.x - 250.0).abs() < 1e-3,
			"{:?}",
			p.velocity
		);
	}

	#[test]
	fn hard_landings_roll_the_view() {
		let world = floor();
		let mut p = Player::spawn(vec3(0.0, 200.0, 0.0));
		let states = run(&mut p, &world, Input::default(), 2.0, RATE);
		let roll = states.iter().map(|s| s.punch_roll).fold(0.0, f32::max);
		// Falling 200 units lands at sqrt(2 · 800 · 200) = 566 u/s.
		assert!((roll - 566.0 * 0.013).abs() < 0.3, "roll {roll}");
		assert_eq!(p.punch_roll, 0.0);
	}

	/// A run at full speed, and its worst per-tick speed loss and turn over the
	/// states within `x` in `near`: what a snag on a seam or lip shows up as.
	fn worst_snag(states: &[Player], near: std::ops::Range<f32>) -> (f32, f32, f32) {
		let window: Vec<&Player> = states
			.iter()
			.filter(|s| near.contains(&s.origin.x))
			.collect();
		let min_speed = window.iter().map(|s| s.speed()).fold(f32::MAX, f32::min);
		let drop = window
			.windows(2)
			.map(|w| w[0].speed() - w[1].speed())
			.fold(0.0, f32::max);
		let turn = window
			.windows(2)
			.filter(|w| w[0].speed() > 100.0 && w[1].speed() > 100.0)
			.map(|w| {
				(w[1].velocity.z.atan2(w[1].velocity.x) - w[0].velocity.z.atan2(w[0].velocity.x))
					.abs()
					.to_degrees()
			})
			.fold(0.0, f32::max);
		(min_speed, drop, turn)
	}

	/// Two floor faces a hair apart, as adjacent world brushes in a map are: the
	/// seam between them must not snag a player walking over it.
	#[test]
	fn walks_over_seam_deltas_smoothly() {
		for delta in [0.0, 1.0 / 32.0, 0.25, 0.5, 1.0] {
			let mut world = Mesh::default();
			world.push_quad(
				vec3(-512.0, 0.0, -64.0),
				vec3(128.0, 0.0, -64.0),
				vec3(128.0, 0.0, 64.0),
				vec3(-512.0, 0.0, 64.0),
				UP,
			);
			world.push_quad(
				vec3(128.0, delta, -64.0),
				vec3(2000.0, delta, -64.0),
				vec3(2000.0, delta, 64.0),
				vec3(128.0, delta, 64.0),
				UP,
			);
			let mut p = grounded(&world, vec3(-400.0, 0.0, 0.0));
			run(&mut p, &world, forward(), 1.0, RATE);
			let states = run(&mut p, &world, forward(), 6.0, RATE);
			assert!(states.first().unwrap().speed() > 249.0);
			let (min_speed, worst, turn) = worst_snag(&states, 0.0..500.0);
			assert!(p.origin.x > 700.0, "stuck at {:?} delta {delta}", p.origin);
			assert!(
				min_speed > 249.0 && worst < 0.5 && turn < 0.5,
				"delta {delta}: snag min {min_speed} drop {worst} turn {turn}"
			);
		}
	}

	/// 64-unit floor cells alternating 0 and `delta`: a bumpy road of face seams.
	#[test]
	fn walks_a_bumpy_road_smoothly() {
		for delta in [1.0 / 32.0, 0.25, 0.5] {
			let mut world = Mesh::default();
			for i in 0..24 {
				let y = if i % 2 == 0 { 0.0 } else { delta };
				let x = -512.0 + i as f32 * 64.0;
				world.push_quad(
					vec3(x, y, -64.0),
					vec3(x + 64.0, y, -64.0),
					vec3(x + 64.0, y, 64.0),
					vec3(x, y, 64.0),
					UP,
				);
			}
			let mut p = grounded(&world, vec3(-500.0, 0.0, 0.0));
			run(&mut p, &world, forward(), 1.0, RATE);
			let states = run(&mut p, &world, forward(), 6.0, RATE);
			assert!(states.first().unwrap().speed() > 249.0);
			let (min_speed, worst, turn) = worst_snag(&states, -400.0..900.0);
			assert!(p.origin.x > 900.0, "stuck at {:?} delta {delta}", p.origin);
			assert!(
				min_speed > 249.0 && worst < 0.5 && turn < 0.5,
				"delta {delta}: snag min {min_speed} drop {worst} turn {turn}"
			);
		}
	}

	/// Flat floor strip `x` in [x0, x1], `z` in [-2048, 2048], at `y`.
	fn slab(world: &mut Mesh, x0: f32, x1: f32, y: f32) {
		world.push_quad(
			vec3(x0, y, -2048.0),
			vec3(x1, y, -2048.0),
			vec3(x1, y, 2048.0),
			vec3(x0, y, 2048.0),
			UP,
		);
	}

	/// One bump case: its name, the world it builds, the speed it costs.
	type Bump = (&'static str, Box<dyn Fn(&mut Mesh)>, f32);

	/// The shapes flat ground in a map snags on, crossed at full speed: lips and
	/// crests at angles to the path, kinked and faceted slopes, a ridge narrower
	/// than the hull. None may cost speed or turn the player.
	#[test]
	fn walks_smoothly_over_bumps_and_seams() {
		let cases: [Bump; 5] = [
			// 1-unit lip crossed at 45°.
			(
				"diagonal lip 1u",
				Box::new(|w: &mut Mesh| {
					slab(w, -1024.0, 128.0, 0.0);
					slab(w, 128.0, 2048.0, 1.0);
				}),
				45.0,
			),
			// Crest line at 30° to the path, 0.5 units tall.
			(
				"crest 30deg 0.5u",
				Box::new(|w: &mut Mesh| {
					slab(w, -1024.0, 0.0, 0.0);
					let a = vec3(0.0, 0.0, -2048.0);
					let b = vec3(768.0, 0.0, -2048.0);
					let c = vec3(768.0, 0.5, 2048.0);
					let d = vec3(0.0, 0.5, 2048.0);
					w.push_quad(a, b, c, d, UP);
					slab(w, 768.0, 2048.0, 0.5);
				}),
				0.0,
			),
			// 3° kink up and back down: a road bump 128 long.
			(
				"3deg bump",
				Box::new(|w: &mut Mesh| {
					slab(w, -1024.0, 0.0, 0.0);
					let tan = 3.0f32.to_radians().tan();
					w.push_quad(
						vec3(0.0, 0.0, -512.0),
						vec3(128.0, 128.0 * tan, -512.0),
						vec3(128.0, 128.0 * tan, 512.0),
						vec3(0.0, 0.0, 512.0),
						UP,
					);
					slab(w, 128.0, 2048.0, 128.0 * tan);
				}),
				0.0,
			),
			// Narrow ridge, 4 wide, 1 tall.
			(
				"ridge 4x1",
				Box::new(|w: &mut Mesh| {
					slab(w, -1024.0, 2048.0, 0.0);
					w.push_box(vec3(128.0, 0.0, -2048.0), vec3(132.0, 1.0, 2048.0));
				}),
				45.0,
			),
			// Faceted ramp: six 5° facets.
			(
				"facets 5deg",
				Box::new(|w: &mut Mesh| {
					slab(w, -1024.0, 0.0, 0.0);
					let tan = 5.0f32.to_radians().tan();
					let mut y = 0.0;
					for i in 0..6 {
						let x = i as f32 * 64.0;
						let y2 = y + 64.0 * tan;
						w.push_quad(
							vec3(x, y, -512.0),
							vec3(x + 64.0, y2, -512.0),
							vec3(x + 64.0, y2, 512.0),
							vec3(x, y, 512.0),
							UP,
						);
						y = y2;
					}
					slab(w, 384.0, 2048.0, y);
				}),
				0.0,
			),
		];
		for (name, build, yaw) in cases {
			let mut world = Mesh::default();
			build(&mut world);
			let input = Input {
				forward: 400.0,
				yaw,
				..Input::default()
			};
			let mut p = grounded(&world, vec3(-900.0, 0.0, 0.0));
			run(&mut p, &world, input, 1.0, RATE);
			let states = run(&mut p, &world, input, 6.0, RATE);
			assert!(states.first().unwrap().speed() > 249.0);
			let (min_speed, worst, turn) = worst_snag(&states, -300.0..500.0);
			assert!(
				min_speed > 249.0 && worst < 0.5 && turn < 0.5,
				"{name}: snag min {min_speed} drop {worst} turn {turn}"
			);
		}
	}

	#[test]
	fn movement_barely_depends_on_tick_rate() {
		let world = floor();
		let travel = |rate: f32| {
			let mut p = grounded(&world, vec3(-900.0, 0.0, 0.0));
			let start = p.origin.x;
			run(&mut p, &world, forward(), 1.0, rate);
			p.origin.x - start
		};
		let (a, b) = (travel(64.0), travel(256.0));
		assert!((a - b).abs() / b < 0.02, "{a} vs {b}");
	}
}
