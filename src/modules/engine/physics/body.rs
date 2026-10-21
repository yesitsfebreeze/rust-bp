//! Bodies, moving platforms, and the world that steps them together.
//!
//! A body is a sphere or capsule. It collides with the static mesh and platforms
//! through swept motion and contact projection (see [`Body::step`]), and with other
//! bodies through pairwise contacts in [`World::step`]. Contact velocities are solved
//! as impulses at the contact point, so friction makes a rotating body roll or tip;
//! a body that does not rotate keeps its orientation, like an upright character.

use crate::{closest_point_between_segments, triangle_normal, Mesh, EPS};
use linalg::{vec3, Quat, Vec3};
use std::f32::consts::TAU;

/// Capsule along its local y axis; a sphere when `half_height` is zero.
#[derive(Clone, Copy, Debug)]
pub struct Body {
	pub pos: Vec3,
	pub vel: Vec3,
	pub orient: Quat,
	/// Angular velocity, world space, radians per second.
	pub spin: Vec3,
	pub half_height: f32,
	pub radius: f32,
	/// Only matters against other bodies; meshes and platforms do not yield.
	pub mass: f32,
	/// Coulomb coefficient: a contact's tangential impulse is at most `friction`
	/// times its normal impulse.
	pub friction: f32,
	/// Whether contacts turn the body. When false it keeps `orient`, like an upright
	/// character capsule, and friction only slows its sliding.
	pub rotates: bool,
	pub grounded: bool,
}

impl Default for Body {
	fn default() -> Body {
		Body {
			pos: Vec3::ZERO,
			vel: Vec3::ZERO,
			orient: Quat::IDENTITY,
			spin: Vec3::ZERO,
			half_height: 0.0,
			radius: 0.5,
			mass: 1.0,
			friction: 0.0,
			rotates: false,
			grounded: false,
		}
	}
}

/// A mesh as one step sees it: at `offset` when the step starts, moving at `vel`.
#[derive(Clone, Copy)]
struct Placed<'a> {
	mesh: &'a Mesh,
	offset: Vec3,
	vel: Vec3,
}

const SOLVER_ITERATIONS: usize = 4;
/// Sweeps per step: each one moves to the first blocking triangle, resolves the
/// contacts there, and continues with the rest of the step along the new velocity.
/// Sliding along a faceted wall at speed takes one sweep per facet crossed.
const MAX_SWEEPS: usize = 16;
/// A contact through a face this fraction of the radius from the center is deeper than any
/// sweep leaves a body (see `SWEEP_TARGET`), so the body started there or was put
/// there. Its side is then taken from the velocity: the side it came from.
const DEEP: f32 = 0.2;

impl Body {
	pub fn segment(&self) -> (Vec3, Vec3) {
		let h = self.orient.rotate(vec3(0.0, self.half_height, 0.0));
		(self.pos - h, self.pos + h)
	}

	/// This body moved `alpha` of the way toward `next`, the state one step later, for
	/// drawing between fixed steps. Position and orientation are blended.
	pub fn interpolate(&self, next: &Body, alpha: f32) -> Body {
		Body {
			pos: self.pos.lerp(next.pos, alpha),
			orient: self.orient.nlerp(next.orient, alpha),
			..*next
		}
	}

	/// Inverse inertia applied to `v` in world space; zero when the body does not rotate.
	/// Solid capsule: a cylinder plus two hemispheres of the same density.
	fn inv_inertia(&self, v: Vec3) -> Vec3 {
		if !self.rotates {
			return Vec3::ZERO;
		}
		let (r, l) = (self.radius, 2.0 * self.half_height);
		let (cyl, ball) = (l, 4.0 / 3.0 * r);
		let mc = self.mass * cyl / (cyl + ball);
		let ms = self.mass - mc;
		let axis = mc * r * r / 2.0 + ms * 2.0 * r * r / 5.0;
		let perp = mc * (l * l / 12.0 + r * r / 4.0)
			+ ms * (2.0 * r * r / 5.0 + l * l / 4.0 + 3.0 * l * r / 8.0);
		let local = self.orient.conjugate().rotate(v);
		self.orient.rotate(local / vec3(perp, axis, perp))
	}

	/// Velocity of the body's material at offset `r` from its center.
	fn point_vel(&self, r: Vec3) -> Vec3 {
		self.vel + self.spin.cross(r)
	}

	fn apply_impulse(&mut self, r: Vec3, p: Vec3) {
		self.vel += p * (1.0 / self.mass);
		self.spin += self.inv_inertia(r.cross(p));
	}

	/// Inverse of the mass an impulse along `n` at offset `r` meets.
	fn inv_mass_along(&self, r: Vec3, n: Vec3) -> f32 {
		1.0 / self.mass + n.dot(self.inv_inertia(r.cross(n)).cross(r))
	}

	/// Removes the approach along `n` at offset `r` against a surface moving at
	/// `surface`, then applies friction against the remaining sliding.
	fn hit_surface(&mut self, r: Vec3, n: Vec3, surface: Vec3) {
		let vn = (self.point_vel(r) - surface).dot(n);
		if vn >= 0.0 {
			return;
		}
		let jn = -vn / self.inv_mass_along(r, n);
		self.apply_impulse(r, n * jn);
		let rel = self.point_vel(r) - surface;
		let slide = rel - n * rel.dot(n);
		let speed = slide.length();
		if speed <= EPS {
			return;
		}
		let dir = slide * (1.0 / speed);
		let jt = (speed / self.inv_mass_along(r, dir)).min(self.friction * jn);
		self.apply_impulse(r, dir * -jt);
	}

	/// Symplectic Euler with swept motion, so a fast body stops at thin geometry
	/// instead of passing it between steps. After each sweep the body is projected out
	/// of the mesh and its inward velocity removed, less friction along the surface.
	/// `filter` toggles internal-edge removal for comparison.
	pub fn step(&mut self, mesh: &Mesh, dt: f32, gravity: Vec3, filter: bool) {
		self.advance(
			&[Placed {
				mesh,
				offset: Vec3::ZERO,
				vel: Vec3::ZERO,
			}],
			dt,
			gravity,
			filter,
		);
	}

	/// [`Body::step`] against meshes that may be moving. Rotation is applied before the
	/// sweeps and is not swept itself; a step turns a body far less than its radius.
	fn advance(&mut self, colliders: &[Placed], dt: f32, gravity: Vec3, filter: bool) {
		self.vel += gravity * dt;
		if self.rotates {
			self.orient = self.orient.integrate(self.spin, dt);
		}
		self.grounded = false;
		let mut left = dt;
		for _ in 0..MAX_SWEEPS {
			let motion = self.vel * left;
			let (p0, p1) = self.segment();
			let elapsed = dt - left;
			let mut t = 1.0f32;
			for c in colliders {
				// In the collider's frame: placed where it is now, moving relative to it.
				let at = c.offset + c.vel * elapsed;
				if let Some(f) = c
					.mesh
					.sweep(p0 - at, p1 - at, self.radius, motion - c.vel * left)
				{
					t = t.min(f);
				}
			}
			self.pos += motion * t;
			left *= 1.0 - t;
			self.resolve(colliders, dt - left, filter);
			if left <= 0.0 {
				break;
			}
		}
	}

	/// Projects the body out of `colliders`, placed as they are `elapsed` into the step,
	/// and solves the contact velocities.
	fn resolve(&mut self, colliders: &[Placed], elapsed: f32, filter: bool) {
		for _ in 0..SOLVER_ITERATIONS {
			let mut moved = Vec3::ZERO;
			let mut touched = false;
			for c in colliders {
				let at = c.offset + c.vel * elapsed;
				let (p0, p1) = self.segment();
				let (p0, p1) = (p0 - at, p1 - at);
				let mut contacts = c.mesh.contacts(p0, p1, self.radius, filter);
				if self.rotates && self.half_height > 0.0 {
					// The axis query gives one point per triangle; a capsule lying on a
					// face needs both ends supported or it rocks.
					contacts.extend(c.mesh.contacts(p0, p0, self.radius, filter));
					contacts.extend(c.mesh.contacts(p1, p1, self.radius, filter));
				}
				touched |= !contacts.is_empty();
				// Contacts are found once per pass, so judge them against the state they
				// were found in: the body's center then, and its velocity then.
				let center = self.pos;
				let approach = self.vel - c.vel;
				for ct in &contacts {
					let mut n = ct.hit.normal;
					let mut depth = ct.depth;
					// Only straight through the face does the other side lie as close.
					let across = triangle_normal(&c.mesh.tris()[ct.tri]).dot(n).abs() > 0.999;
					if across && (0.0..self.radius * DEEP).contains(&ct.hit.dist) && approach.dot(n) > 0.0 {
						(n, depth) = (-n, self.radius + ct.hit.dist);
					}
					let depth = depth - moved.dot(n);
					if depth > 0.0 {
						self.pos += n * depth;
						moved += n * depth;
					}
					// Offset of the body's surface point that touches, unchanged by the push.
					let r = ct.hit.seg_point + at - center - n * self.radius;
					self.hit_surface(r, n, c.vel);
					self.grounded |= n.y > 0.7;
				}
			}
			if !touched {
				break;
			}
		}
	}

	/// Deepest remaining overlap with the mesh.
	pub fn penetration(&self, mesh: &Mesh) -> f32 {
		let (p0, p1) = self.segment();
		mesh
			.contacts(p0, p1, self.radius, false)
			.first()
			.map_or(0.0, |c| c.depth)
	}
}

/// Pushes two overlapping bodies apart by inverse mass and solves their contact
/// velocity, with friction from the geometric mean of their coefficients.
fn collide_pair(a: &mut Body, b: &mut Body) {
	let (a0, a1) = a.segment();
	let (b0, b1) = b.segment();
	let (_, [pa, pb]) = closest_point_between_segments(a0, a1, b0, b1);
	let d = (pa - pb).length();
	let reach = a.radius + b.radius;
	if d >= reach {
		return;
	}
	let n = if d > EPS {
		(pa - pb) * (1.0 / d)
	} else {
		vec3(0.0, 1.0, 0.0)
	};
	let (wa, wb) = (1.0 / a.mass, 1.0 / b.mass);
	let share = (reach - d) / (wa + wb);
	a.pos += n * (share * wa);
	b.pos -= n * (share * wb);
	a.grounded |= n.y > 0.7;
	b.grounded |= -n.y > 0.7;

	let point = (pa - n * a.radius + pb + n * b.radius) * 0.5;
	let (ra, rb) = (point - a.pos, point - b.pos);
	let vn = (a.point_vel(ra) - b.point_vel(rb)).dot(n);
	if vn >= 0.0 {
		return;
	}
	let jn = -vn / (a.inv_mass_along(ra, n) + b.inv_mass_along(rb, n));
	a.apply_impulse(ra, n * jn);
	b.apply_impulse(rb, n * -jn);
	let rel = a.point_vel(ra) - b.point_vel(rb);
	let slide = rel - n * rel.dot(n);
	let speed = slide.length();
	if speed <= EPS {
		return;
	}
	let dir = slide * (1.0 / speed);
	let k = a.inv_mass_along(ra, dir) + b.inv_mass_along(rb, dir);
	let jt = (speed / k).min((a.friction * b.friction).sqrt() * jn);
	a.apply_impulse(ra, dir * -jt);
	b.apply_impulse(rb, dir * jt);
}

/// A mesh that moves without rotating, swinging along `swing` about `origin` once
/// every `period` seconds.
#[derive(Clone, Debug)]
pub struct Platform {
	pub mesh: Mesh,
	pub origin: Vec3,
	pub swing: Vec3,
	pub period: f32,
}

impl Platform {
	/// Where the platform's mesh sits at `time`.
	pub fn offset(&self, time: f32) -> Vec3 {
		self.origin + self.swing * (TAU * time / self.period).sin()
	}
}

/// A static mesh, moving platforms and bodies, stepped together.
#[derive(Clone, Debug)]
pub struct World {
	pub mesh: Mesh,
	pub platforms: Vec<Platform>,
	pub bodies: Vec<Body>,
	pub gravity: Vec3,
	pub time: f32,
}

impl World {
	pub fn new(mesh: Mesh) -> World {
		World {
			mesh,
			platforms: Vec::new(),
			bodies: Vec::new(),
			gravity: vec3(0.0, -9.81, 0.0),
			time: 0.0,
		}
	}

	/// Moves every body against the meshes, then alternates resolving body pairs with
	/// projecting bodies the pairs pushed into a mesh back out, so a stack settles
	/// against the floor. Platforms move at constant velocity within a step, from
	/// where they are now to where they are next. Bodies are not swept against each
	/// other.
	pub fn step(&mut self, dt: f32, filter: bool) {
		let next = self.time + dt;
		let mut colliders = vec![Placed {
			mesh: &self.mesh,
			offset: Vec3::ZERO,
			vel: Vec3::ZERO,
		}];
		colliders.extend(self.platforms.iter().map(|p| Placed {
			mesh: &p.mesh,
			offset: p.offset(self.time),
			vel: (p.offset(next) - p.offset(self.time)) * (1.0 / dt),
		}));
		for b in &mut self.bodies {
			b.advance(&colliders, dt, self.gravity, filter);
		}
		if self.bodies.len() > 1 {
			for _ in 0..SOLVER_ITERATIONS {
				for i in 1..self.bodies.len() {
					let (head, tail) = self.bodies.split_at_mut(i);
					for a in head {
						collide_pair(a, &mut tail[0]);
					}
				}
				for b in &mut self.bodies {
					b.resolve(&colliders, dt, filter);
				}
			}
		}
		self.time = next;
	}
}
