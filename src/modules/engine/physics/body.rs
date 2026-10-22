//! Bodies, moving platforms, and the world that steps them together.
//!
//! A body is a sphere, capsule, cylinder, the convex hull of arbitrary mesh
//! geometry, or a compound of convex parts with their concavity between them
//! ([`Shape`]). It collides with the static mesh and platforms through
//! swept motion and contact projection (see [`Body::step`]), and with other
//! bodies through pairwise contacts in [`World::step`]. Contact velocities are
//! solved as impulses at the contact point, so friction makes a rotating body
//! roll or tip; a body that does not rotate keeps its orientation, like an
//! upright character.
//!
//! A body with zero mass is static: it never moves and never sleeps, and other
//! bodies collide with it as with terrain. A body at rest sleeps after
//! [`SLEEP_AFTER`] seconds below the sleep speed and is skipped until a moving
//! platform comes under it or an awake body touches it, so a settled level
//! costs nothing to keep stepping.

use crate::convex_bounds;
use crate::{closest_point_between_segments, triangle_normal, Convex, Mesh, EPS};
use linalg::{vec3, Quat, Vec3};
use std::f32::consts::TAU;
use std::sync::Arc;

/// What a body is. The round shapes share `Body::radius` and `Body::half_height`
/// (a sphere is a capsule with zero half height); the convex shapes answer GJK
/// supports through [`crate::Convex`].
#[derive(Clone, Debug)]
pub enum Shape {
	Sphere,
	Capsule,
	/// A flat-bottomed cylinder along the local y axis.
	Cylinder,
	/// The convex hull of arbitrary mesh geometry, shared between bodies.
	Hull(Arc<crate::Hull>),
	/// A compound of convex parts with their concavity between them, shared
	/// between bodies. The parent hull answers the whole-body questions.
	Parts(Arc<crate::Parts>),
}

/// Capsule along its local y axis; a sphere when `half_height` is zero.
#[derive(Clone, Debug)]
pub struct Body {
	pub pos: Vec3,
	pub vel: Vec3,
	pub orient: Quat,
	/// Angular velocity, world space, radians per second.
	pub spin: Vec3,
	pub shape: Shape,
	/// Sphere, capsule, and cylinder radius; a hull ignores it.
	pub radius: f32,
	/// Capsule and cylinder half height; a sphere is a capsule with zero.
	pub half_height: f32,
	/// Zero for a static body: infinite mass, never moves. Only matters against
	/// other bodies; meshes and platforms do not yield.
	pub mass: f32,
	/// Coulomb coefficient: a contact's tangential impulse is at most `friction`
	/// times its normal impulse.
	pub friction: f32,
	/// Whether contacts turn the body. When false it keeps `orient`, like an upright
	/// character capsule, and friction only slows its sliding.
	pub rotates: bool,
	pub grounded: bool,
	/// Seconds since a contact held the body up. `grounded` flickers — a body
	/// rests at its contact skin, and one step's fall can stay inside it — so
	/// sleep waits on this instead.
	pub since_grounded: f32,
	/// Skipped by [`World::step`] until something wakes it: a moving platform
	/// under it or a contact with an awake body.
	pub sleeping: bool,
	/// Seconds spent grounded below the sleep speed.
	pub(crate) sleep_timer: f32,
}

impl Default for Body {
	fn default() -> Body {
		Body {
			pos: Vec3::ZERO,
			vel: Vec3::ZERO,
			orient: Quat::IDENTITY,
			spin: Vec3::ZERO,
			shape: Shape::Sphere,
			half_height: 0.0,
			radius: 0.5,
			mass: 1.0,
			friction: 0.0,
			rotates: false,
			grounded: false,
			since_grounded: f32::INFINITY,
			sleeping: false,
			sleep_timer: 0.0,
		}
	}
}

/// Seconds grounded below the sleep speed before a body sleeps.
const SLEEP_AFTER: f32 = 0.5;

/// The convex shapes as `dyn Convex`: a cylinder from the body's fields, a hull
/// from its shape, a compound's parent hull over its parts. Cylinder bodies
/// need this to avoid handing out a reference to a temporary.
enum ConvexBody<'a> {
	Cylinder(crate::Cylinder),
	Hull(&'a crate::Hull),
	Parts(&'a crate::Parts),
}

impl ConvexBody<'_> {
	fn get(&self) -> &dyn crate::Convex {
		match self {
			ConvexBody::Cylinder(c) => c,
			ConvexBody::Hull(h) => *h,
			// The parent hull: whole-body questions — the sweep's box, the
			// broadphase bounds, the sleep extents — are answered loosely by
			// the parent, and the contacts answer per part.
			ConvexBody::Parts(p) => *p as &dyn crate::Convex,
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
/// Angular velocity lost per second: rotation has no positional correction, so
/// without a loss term a rocking body never settles and a rolling one never
/// stops. Everything real loses spin.
const ANGULAR_DAMPING: f32 = 6.0;
/// Sweeps per step: each one moves to the first blocking triangle, resolves the
/// contacts there, and continues with the rest of the step along the new velocity.
/// Sliding along a faceted wall at speed takes one sweep per facet crossed.
const MAX_SWEEPS: usize = 16;
/// A contact through a face this fraction of the radius from the center is deeper than any
/// sweep leaves a body (see `SWEEP_TARGET`), so the body started there or was put
/// there. Its side is then taken from the velocity: the side it came from.
const DEEP: f32 = 0.2;

impl Body {
	/// A body built around arbitrary mesh geometry.
	pub fn from_hull(hull: crate::Hull) -> Body {
		Body {
			shape: Shape::Hull(Arc::new(hull)),
			..Body::default()
		}
	}

	/// A compound body built around convex parts filling a concave shape.
	pub fn from_parts(parts: crate::Parts) -> Body {
		Body {
			shape: Shape::Parts(Arc::new(parts)),
			..Body::default()
		}
	}

	/// Marks the body awake and resets its sleep timer. Static bodies ignore it.
	pub fn wake(&mut self) {
		if self.mass > 0.0 {
			self.sleeping = false;
			self.sleep_timer = 0.0;
		}
	}

	pub fn segment(&self) -> (Vec3, Vec3) {
		let h = self.orient.rotate(vec3(0.0, self.half_height, 0.0));
		(self.pos - h, self.pos + h)
	}

	/// This body moved `alpha` of the way toward `next`, the state one step later, for
	/// drawing between fixed steps. Position and orientation are blended.
	pub fn interpolate(&self, next: &Body, alpha: f32) -> Body {
		let mut out = next.clone();
		out.pos = self.pos.lerp(next.pos, alpha);
		out.orient = self.orient.nlerp(next.orient, alpha);
		out
	}

	/// The convex shape when the body is a cylinder, hull, or compound; the
	/// compound's is its parent hull, loose over the concavity.
	fn convex(&self) -> Option<ConvexBody<'_>> {
		match &self.shape {
			Shape::Cylinder => Some(ConvexBody::Cylinder(crate::Cylinder {
				radius: self.radius,
				half_height: self.half_height,
			})),
			Shape::Hull(h) => Some(ConvexBody::Hull(h)),
			Shape::Parts(p) => Some(ConvexBody::Parts(p)),
			_ => None,
		}
	}

	fn is_round(&self) -> bool {
		matches!(self.shape, Shape::Sphere | Shape::Capsule)
	}

	/// The body's contact skin: the resting margin a cylinder or hull brings to a
	/// mesh query or a pair, zero for the round shapes, whose radius is their
	/// margin against a mesh and whose pairs run the segment path.
	pub(crate) fn skin(&self) -> f32 {
		match self.convex() {
			Some(c) => c.get().margin(),
			None => 0.0,
		}
	}

	/// Smallest half dimension, the scale of sleep and deflation thresholds.
	fn extent(&self) -> f32 {
		match self.convex() {
			Some(c) => c.get().min_half(),
			None => self.radius,
		}
	}

	/// Largest uniform deflation with any volume left, for pair penetration.
	fn max_shrink(&self) -> f32 {
		match self.convex() {
			Some(c) => c.get().max_shrink(),
			None if self.half_height > 0.0 => self.radius.min(self.half_height),
			None => self.radius,
		}
	}

	/// The body's support point `shrink` inside its shape along `d`, world space.
	fn support_world(&self, d: Vec3, shrink: f32) -> Vec3 {
		let len = d.length();
		if len <= EPS {
			return self.pos;
		}
		let dir = d * (1.0 / len);
		match &self.shape {
			Shape::Sphere => self.pos + dir * (self.radius - shrink),
			Shape::Capsule => {
				let h = self
					.orient
					.rotate(vec3(0.0, (self.half_height - shrink).max(0.0), 0.0));
				let (p0, p1) = (self.pos - h, self.pos + h);
				let axis = p1 - p0;
				let t = (d.dot(axis) / axis.length2().max(EPS * EPS)).clamp(0.0, 1.0);
				p0 + axis * t + dir * (self.radius - shrink)
			}
			Shape::Cylinder => {
				let cyl = crate::Cylinder {
					radius: self.radius,
					half_height: self.half_height,
				};
				let local = self.orient.conjugate().rotate(d);
				self.pos
					+ self
						.orient
						.rotate(crate::Convex::support(&cyl, local, shrink))
			}
			Shape::Hull(h) => {
				let local = self.orient.conjugate().rotate(d);
				self.pos
					+ self
						.orient
						.rotate(crate::Convex::support(&**h, local, shrink))
			}
			// The parent hull: this answers the whole-body questions, never a
			// contact — contacts go per part.
			Shape::Parts(p) => {
				let local = self.orient.conjugate().rotate(d);
				self.pos
					+ self
						.orient
						.rotate(crate::Convex::support(&**p, local, shrink))
			}
		}
	}

	/// Inverse of the mass an impulse meets: zero for a static body.
	fn inv_mass(&self) -> f32 {
		if self.mass == 0.0 {
			0.0
		} else {
			1.0 / self.mass
		}
	}

	/// Inverse inertia applied to `v` in world space; zero when the body does not
	/// rotate or is static.
	fn inv_inertia(&self, v: Vec3) -> Vec3 {
		if !self.rotates || self.mass == 0.0 {
			return Vec3::ZERO;
		}
		let diag = match &self.shape {
			Shape::Sphere => {
				let i = 2.0 / 5.0 * self.mass * self.radius * self.radius;
				vec3(i, i, i)
			}
			// Solid capsule: a cylinder plus two hemispheres of the same density.
			Shape::Capsule => {
				let (r, l) = (self.radius, 2.0 * self.half_height);
				let (cyl, ball) = (l, 4.0 / 3.0 * r);
				let mc = self.mass * cyl / (cyl + ball);
				let ms = self.mass - mc;
				let axis = mc * r * r / 2.0 + ms * 2.0 * r * r / 5.0;
				let perp = mc * (l * l / 12.0 + r * r / 4.0)
					+ ms * (2.0 * r * r / 5.0 + l * l / 4.0 + 3.0 * l * r / 8.0);
				vec3(perp, axis, perp)
			}
			Shape::Cylinder => crate::Cylinder {
				radius: self.radius,
				half_height: self.half_height,
			}
			.inertia(self.mass),
			Shape::Hull(h) => h.inertia(self.mass),
			// The parts' inertia by volume share, each spinning about the
			// body's center through the parallel axis.
			Shape::Parts(p) => crate::Convex::inertia(&**p, self.mass),
		};
		let local = self.orient.conjugate().rotate(v);
		self.orient.rotate(local / diag)
	}

	/// Velocity of the body's material at offset `r` from its center.
	fn point_vel(&self, r: Vec3) -> Vec3 {
		self.vel + self.spin.cross(r)
	}

	fn apply_impulse(&mut self, r: Vec3, p: Vec3) {
		self.vel += p * self.inv_mass();
		self.spin += self.inv_inertia(r.cross(p));
	}

	/// Inverse of the mass an impulse along `n` at offset `r` meets.
	fn inv_mass_along(&self, r: Vec3, n: Vec3) -> f32 {
		self.inv_mass() + n.dot(self.inv_inertia(r.cross(n)).cross(r))
	}

	/// Removes the approach along `n` at offset `r` against a surface moving at
	/// `surface`, then applies friction against the remaining sliding.
	fn hit_surface(&mut self, r: Vec3, n: Vec3, surface: Vec3) {
		let vn = (self.point_vel(r) - surface).dot(n);
		if vn >= 0.0 {
			return;
		}
		let k = self.inv_mass_along(r, n);
		if k <= EPS {
			return;
		}
		let jn = -vn / k;
		self.apply_impulse(r, n * jn);
		let rel = self.point_vel(r) - surface;
		let slide = rel - n * rel.dot(n);
		let speed = slide.length();
		if speed <= EPS {
			return;
		}
		let dir = slide * (1.0 / speed);
		let kt = self.inv_mass_along(r, dir);
		let jt = if kt > EPS {
			(speed / kt).min(self.friction * jn)
		} else {
			0.0
		};
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
	/// Static and sleeping bodies stay put.
	fn advance(&mut self, colliders: &[Placed], dt: f32, gravity: Vec3, filter: bool) {
		if self.mass == 0.0 || self.sleeping {
			return;
		}
		self.vel += gravity * dt;
		if self.rotates {
			// Angular damping near rest: rotation has no positional
			// correction, so a rocking body never gives the energy back and a
			// slowly rolling one never stops. While moving it rolls as
			// friction dictates — untouched; near rest, where the spin is
			// jitter standing between the body and sleep, it is spent.
			if self.vel.length() < gravity.length() * dt * 2.0 {
				self.spin *= 1.0 / (1.0 + ANGULAR_DAMPING * dt);
			}
			self.orient = self.orient.integrate(self.spin, dt);
		}
		self.grounded = false;
		self.since_grounded += dt;
		let mut left = dt;
		for _ in 0..MAX_SWEEPS {
			let motion = self.vel * left;
			let elapsed = dt - left;
			let mut t = 1.0f32;
			for c in colliders {
				// In the collider's frame: placed where it is now, moving relative to it.
				let at = c.offset + c.vel * elapsed;
				match self.convex() {
					Some(cv) => {
						if let Some(f) = c.mesh.cast_convex(
							self.pos - at,
							self.orient,
							cv.get(),
							motion - c.vel * left,
							self.skin(),
						) {
							t = t.min(f.fraction);
						}
					}
					None => {
						let (p0, p1) = self.segment();
						if let Some(f) = c
							.mesh
							.sweep(p0 - at, p1 - at, self.radius, motion - c.vel * left)
						{
							t = t.min(f);
						}
					}
				}
			}
			self.pos += motion * t;
			left *= 1.0 - t;
			let touched = self.resolve(colliders, dt - left, filter);
			// Blocked from the start with nothing to push out of is jammed for
			// this step — a grazing block — and re-sweeping the same pose costs
			// a full query batch each time for nothing.
			if left <= 0.0 || (t == 0.0 && !touched) {
				break;
			}
		}
	}

	/// Projects the body out of `colliders`, placed as they are `elapsed` into the
	/// step, and solves the contact velocities. Returns whether any contact was
	/// found.
	fn resolve(&mut self, colliders: &[Placed], elapsed: f32, filter: bool) -> bool {
		if self.mass == 0.0 || self.sleeping {
			return false;
		}
		let mut contacts = Vec::new();
		let mut touched = false;
		for _ in 0..SOLVER_ITERATIONS {
			let mut moved = Vec3::ZERO;
			let mut any = false;
			for c in colliders {
				let at = c.offset + c.vel * elapsed;
				contacts.clear();
				match &self.shape {
					// Each part brings its own contacts against the mesh, at
					// its own offset: the concavity between parts is real, and
					// the parent hull's blanket answer would fill it in.
					Shape::Parts(p) => {
						for part in &p.parts {
							c.mesh.convex_contacts_into(
								self.pos - at + self.orient.rotate(part.offset),
								self.orient,
								part.shape.get(),
								part.shape.get().margin(),
								filter,
								&mut contacts,
							);
						}
					}
					_ => match self.convex() {
						Some(cv) => c.mesh.convex_contacts_into(
							self.pos - at,
							self.orient,
							cv.get(),
							self.skin(),
							filter,
							&mut contacts,
						),
						None => {
							let (p0, p1) = self.segment();
							let (p0, p1) = (p0 - at, p1 - at);
							c.mesh
								.contacts_into(p0, p1, self.radius, filter, &mut contacts);
							if self.rotates && self.half_height > 0.0 {
								// The axis query gives one point per triangle; a capsule lying on a
								// face needs both ends supported or it rocks.
								c.mesh
									.contacts_into(p0, p0, self.radius, filter, &mut contacts);
								c.mesh
									.contacts_into(p1, p1, self.radius, filter, &mut contacts);
							}
						}
					},
				}
				any |= !contacts.is_empty();
				// Contacts are found once per pass, so judge them against the state they
				// were found in: the body's center then, and its velocity then.
				let center = self.pos;
				let approach = self.vel - c.vel;
				for ct in &contacts {
					let mut n = ct.hit.normal;
					let mut depth = ct.depth;
					if self.is_round() {
						// Only straight through the face does the other side lie as close.
						let across = triangle_normal(&c.mesh.tris()[ct.tri]).dot(n).abs() > 0.999;
						if across && (0.0..self.radius * DEEP).contains(&ct.hit.dist) && approach.dot(n) > 0.0 {
							(n, depth) = (-n, self.radius + ct.hit.dist);
						}
					}
					let depth = depth - moved.dot(n);
					if depth > 0.0 {
						self.pos += n * depth;
						moved += n * depth;
					}
					// Offset of the body's material at the contact point, unchanged by
					// the push: the segment axis pulled out by the radius, or, for a
					// convex shape, the GJK witness, already on its surface.
					let r = match self.is_round() {
						true => ct.hit.seg_point + at - center - n * self.radius,
						false => ct.hit.seg_point + at - center,
					};
					self.hit_surface(r, n, c.vel);
					self.grounded |= n.y > 0.7;
					self.since_grounded = 0.0;
				}
			}
			touched |= any;
			if !any {
				break;
			}
		}
		touched
	}

	/// Deepest remaining overlap with the mesh.
	pub fn penetration(&self, mesh: &Mesh) -> f32 {
		match &self.shape {
			// The deepest part: the parent hull fills the concavity in, and
			// the body resting wedged in its own fit would read as through it.
			Shape::Parts(p) => p
				.parts
				.iter()
				.map(|part| {
					mesh
						.convex_penetration(
							self.pos + self.orient.rotate(part.offset),
							self.orient,
							part.shape.get(),
						)
						.map_or(0.0, |(_, d)| d)
				})
				.fold(0.0_f32, f32::max),
			_ => match self.convex() {
				Some(cv) => mesh
					.convex_penetration(self.pos, self.orient, cv.get())
					.map_or(0.0, |(_, d)| d),
				None => {
					let (p0, p1) = self.segment();
					mesh
						.contacts(p0, p1, self.radius, false)
						.first()
						.map_or(0.0, |c| c.depth)
				}
			},
		}
	}
}

/// Pushes two overlapping or nearly touching round bodies apart by inverse mass
/// and solves their contact velocity, with friction from the geometric mean of
/// their coefficients. Exact and cheap, from the segment math.
fn collide_round(a: &mut Body, b: &mut Body) {
	let (a0, a1) = a.segment();
	let (b0, b1) = b.segment();
	let (_, [pa, pb]) = closest_point_between_segments(a0, a1, b0, b1);
	let d = (pa - pb).length();
	let reach = a.radius + b.radius;
	if d >= reach {
		return;
	}
	a.wake();
	b.wake();
	let n = if d > EPS {
		(pa - pb) * (1.0 / d)
	} else {
		vec3(0.0, 1.0, 0.0)
	};
	let (wa, wb) = (a.inv_mass(), b.inv_mass());
	if wa + wb <= 0.0 {
		return;
	}
	let share = (reach - d) / (wa + wb);
	a.pos += n * (share * wa);
	b.pos -= n * (share * wb);
	a.grounded |= n.y > 0.7;
	b.grounded |= -n.y > 0.7;
	a.since_grounded = 0.0;
	b.since_grounded = 0.0;

	let point = (pa - n * a.radius + pb + n * b.radius) * 0.5;
	solve_pair(a, b, point, n);
}

/// The contact between two convex bodies from their GJK support pair `pair`:
/// the normal from B toward A, the depth to resolve, and a point between the
/// witnesses. `skin` is how close they rest, `most` the largest deflation with
/// any volume left, for measuring an overlap.
fn convex_contact(
	hint: Vec3,
	mid: Vec3,
	skin: f32,
	most: f32,
	mut pair: impl FnMut(f32) -> Option<(Vec3, Vec3, Vec3)>,
) -> Option<(Vec3, f32, Vec3)> {
	match pair(0.0) {
		Some((v, pa, pb)) => {
			let dist = v.length();
			if dist >= skin {
				return None;
			}
			Some((v * (1.0 / dist), skin - dist, (pa + pb) * 0.5))
		}
		None => {
			// Overlapping. Eroding both bodies by `s` separates them once the
			// overlap `2 * s` is gone, so the largest still-overlapping `s`
			// measures half of it.
			let (mut inside, mut clear) = (0.0, most);
			for _ in 0..24 {
				let mid = (inside + clear) * 0.5;
				match pair(mid) {
					None => inside = mid,
					Some(_) => clear = mid,
				}
			}
			match pair(clear) {
				Some((v, pa, pb)) => {
					let dist = v.length();
					let n = v * (1.0 / dist.max(EPS));
					Some((n, (2.0 * clear - dist).max(0.0), (pa + pb) * 0.5))
				}
				None => {
					// Core inside core: apart along the line of centers.
					let n = if hint.length2() > EPS {
						hint.normalize()
					} else {
						vec3(0.0, 1.0, 0.0)
					};
					Some((n, 2.0 * most, mid))
				}
			}
		}
	}
}

/// Pushes two bodies apart by a contact and solves it: by inverse mass along
/// `n` (from B toward A, `depth` deep at `point`), then the contact velocity.
fn push_solve(a: &mut Body, b: &mut Body, n: Vec3, depth: f32, point: Vec3) {
	a.wake();
	b.wake();
	let (wa, wb) = (a.inv_mass(), b.inv_mass());
	if wa + wb <= 0.0 {
		return;
	}
	let share = depth / (wa + wb);
	a.pos += n * (share * wa);
	b.pos -= n * (share * wb);
	a.grounded |= n.y > 0.7;
	b.grounded |= -n.y > 0.7;
	a.since_grounded = 0.0;
	b.since_grounded = 0.0;
	solve_pair(a, b, point, n);
}

/// A part's support point in world space: the part's shape, at its offset,
/// `shrink` inside it along `d`.
fn support_part(body: &Body, part: &crate::Part, d: Vec3, shrink: f32) -> Vec3 {
	let len = d.length();
	if len <= EPS {
		return body.pos;
	}
	let local = body.orient.conjugate().rotate(d);
	body.pos
		+ body
			.orient
			.rotate(part.offset + crate::Convex::support(part.shape.get(), local, shrink))
}

/// Any other pair of shapes, through GJK: separation from the support points,
/// and overlap from the largest erosion, by distance, that still overlaps.
/// Compound bodies run their parts, part against part, so their concavity
/// holds and is held: the parent hulls would fill it in.
fn collide_convex(a: &mut Body, b: &mut Body) {
	// The parts are Arc'd off first: the loops below borrow the bodies
	// mutably while the parts' shapes are read.
	let pa = match &a.shape {
		Shape::Parts(p) => Some(Arc::clone(p)),
		_ => None,
	};
	let pb = match &b.shape {
		Shape::Parts(p) => Some(Arc::clone(p)),
		_ => None,
	};
	let hint = a.pos - b.pos;
	let mid = (a.pos + b.pos) * 0.5;
	// The contact between a part (or the whole body) of A and a part (or the
	// whole body) of B, solved as it is found. Returns the normal and point
	// of a contact that was solved, for the flat-contact spread after.
	let once = |a: &mut Body,
	            b: &mut Body,
	            part_a: Option<&crate::Part>,
	            part_b: Option<&crate::Part>|
	 -> Option<(Vec3, Vec3)> {
		let support_a = |d: Vec3, shrink: f32| match part_a {
			Some(p) => support_part(a, p, d, shrink),
			None => a.support_world(d, shrink),
		};
		let support_b = |d: Vec3, shrink: f32| match part_b {
			Some(p) => support_part(b, p, d, shrink),
			None => b.support_world(d, shrink),
		};
		// Each part brings its own margin and deflation; the whole body its own.
		let (skin_a, most_a) = match part_a {
			Some(p) => (p.shape.get().margin(), p.shape.get().max_shrink()),
			None => (a.skin(), a.max_shrink()),
		};
		let (skin_b, most_b) = match part_b {
			Some(p) => (p.shape.get().margin(), p.shape.get().max_shrink()),
			None => (b.skin(), b.max_shrink()),
		};
		let contact = convex_contact(
			hint,
			mid,
			skin_a.max(skin_b),
			most_a.min(most_b),
			|shrink| crate::gjk_pair(hint, |d| support_a(d, shrink), |d| support_b(d, shrink)),
		);
		contact.map(|(n, depth, point)| {
			push_solve(a, b, n, depth, point);
			(n, point)
		})
	};
	let flat = match (&pa, &pb) {
		(Some(pa), Some(pb)) => {
			for part_a in &pa.parts {
				for part_b in &pb.parts {
					once(a, b, Some(part_a), Some(part_b));
				}
			}
			None
		}
		(Some(pa), None) => {
			for part_a in &pa.parts {
				once(a, b, Some(part_a), None);
			}
			None
		}
		(None, Some(pb)) => {
			for part_b in &pb.parts {
				once(a, b, None, Some(part_b));
			}
			None
		}
		(None, None) => once(a, b, None, None),
	};
	// A flat contact between two whole hulls is a region, and one point lets a
	// box rock on a box forever: the same spread of tangent extremes that holds
	// a hull still on the mesh, extended by the narrower body's reach along
	// each tangent, with the same normal and depth. Curved sides roll on a
	// point, as they should, so only hulls spread; a compound's parts are
	// small enough to rock on their own single contacts.
	if let (Shape::Hull(_), Shape::Hull(_), Some((n, point))) = (&a.shape, &b.shape, flat) {
		let helper = if n.x.abs() < 0.9 {
			vec3(1.0, 0.0, 0.0)
		} else {
			vec3(0.0, 1.0, 0.0)
		};
		let t1 = n.cross(helper).normalize();
		let t2 = n.cross(t1);
		for t in [t1, -t1, t2, -t2] {
			let reach = (t.dot(a.support_world(t, 0.0) - a.pos))
				.min(t.dot(b.support_world(t, 0.0) - b.pos))
				.max(EPS);
			solve_pair(a, b, point + t * reach, n);
		}
	}
}

/// Solves the contact velocity at `point` between bodies `a` and `b` along `n`
/// (from `b` toward `a`): the approach along `n` is removed, then friction
/// against the remaining sliding, with friction from the geometric mean of the
/// coefficients.
fn solve_pair(a: &mut Body, b: &mut Body, point: Vec3, n: Vec3) {
	let (ra, rb) = (point - a.pos, point - b.pos);
	let vn = (a.point_vel(ra) - b.point_vel(rb)).dot(n);
	if vn >= 0.0 {
		return;
	}
	let k = a.inv_mass_along(ra, n) + b.inv_mass_along(rb, n);
	if k <= EPS {
		return;
	}
	let jn = -vn / k;
	a.apply_impulse(ra, n * jn);
	b.apply_impulse(rb, n * -jn);
	let rel = a.point_vel(ra) - b.point_vel(rb);
	let slide = rel - n * rel.dot(n);
	let speed = slide.length();
	if speed <= EPS {
		return;
	}
	let dir = slide * (1.0 / speed);
	let kt = a.inv_mass_along(ra, dir) + b.inv_mass_along(rb, dir);
	if kt <= EPS {
		return;
	}
	let jt = (speed / kt).min((a.friction * b.friction).sqrt() * jn);
	a.apply_impulse(ra, dir * -jt);
	b.apply_impulse(rb, dir * jt);
}

fn collide_pair(a: &mut Body, b: &mut Body) {
	if a.is_round() && b.is_round() {
		collide_round(a, b);
	} else {
		collide_convex(a, b);
	}
}

/// A body's world bounds.
fn body_bounds(b: &Body) -> (Vec3, Vec3) {
	match b.convex() {
		Some(cv) => convex_bounds(b.pos, b.orient, cv.get()),
		None => {
			let (p0, p1) = b.segment();
			let r = vec3(b.radius, b.radius, b.radius);
			(p0.min(p1) - r, p0.max(p1) + r)
		}
	}
}

fn component(v: Vec3, k: usize) -> f32 {
	match k {
		0 => v.x,
		1 => v.y,
		_ => v.z,
	}
}

/// Candidate body pairs whose bounds overlap, from sweep and prune along the
/// axis with the widest spread, so bodies clustered on a plane still separate.
/// Pairs nothing can move — two static bodies, or two sleepers — are left out.
fn broadphase(bodies: &[Body]) -> Vec<(usize, usize)> {
	let n = bodies.len();
	if n < 2 {
		return Vec::new();
	}
	let boxes: Vec<(Vec3, Vec3)> = bodies.iter().map(body_bounds).collect();
	let spread = |k: usize| {
		let (mut lo, mut hi) = (f32::MAX, f32::MIN);
		for b in &boxes {
			lo = lo.min(component(b.0, k));
			hi = hi.max(component(b.1, k));
		}
		hi - lo
	};
	let axis = [0, 1, 2]
		.into_iter()
		.max_by(|a, b| spread(*a).total_cmp(&spread(*b)))
		.unwrap();
	let mut order: Vec<usize> = (0..n).collect();
	order.sort_by(|&i, &j| component(boxes[i].0, axis).total_cmp(&component(boxes[j].0, axis)));
	let active = |b: &Body| b.mass > 0.0 && !b.sleeping;
	let mut pairs = Vec::new();
	for si in 0..n - 1 {
		let i = order[si];
		for &j in &order[si + 1..] {
			if component(boxes[j].0, axis) > component(boxes[i].1, axis) {
				break;
			}
			if crate::overlaps(boxes[i], boxes[j].0, boxes[j].1)
				&& (active(&bodies[i]) || active(&bodies[j]))
			{
				pairs.push((i.min(j), i.max(j)));
			}
		}
	}
	pairs
}

/// A static collision mesh in local space that moves without rotating, swinging
/// along `swing` about `origin` once every `period` seconds. Its BVH is reusable
/// across every position because animation changes only the transform.
#[derive(Clone, Debug)]
pub struct Platform {
	pub mesh: Mesh,
	pub origin: Vec3,
	pub swing: Vec3,
	pub period: f32,
}

impl Platform {
	pub fn new(mesh: Mesh, origin: Vec3, swing: Vec3, period: f32) -> Platform {
		mesh.prepare();
		Platform {
			mesh,
			origin,
			swing,
			period,
		}
	}

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
		mesh.prepare();
		World {
			mesh,
			platforms: Vec::new(),
			bodies: Vec::new(),
			gravity: vec3(0.0, -9.81, 0.0),
			time: 0.0,
		}
	}

	/// Prepares the static world and all attached platform meshes before simulation.
	/// Call again after adding platforms; animated transforms do not invalidate their BVHs.
	pub fn prepare(&self) {
		self.mesh.prepare();
		for platform in &self.platforms {
			platform.mesh.prepare();
		}
	}

	/// Moves every body against the meshes, then alternates resolving body pairs
	/// with projecting bodies the pairs pushed into a mesh back out, so a stack
	/// settles against the floor. Platforms move at constant velocity within a
	/// step, from where they are now to where they are next. Bodies are not
	/// swept against each other.
	///
	/// Pairs come from a sweep-and-prune broadphase over the bodies' bounds, so
	/// bodies far apart never test each other. A static body (mass zero) never
	/// moves; a body at rest sleeps after [`SLEEP_AFTER`] seconds and wakes when
	/// a moving platform comes under it or an awake body touches it.
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
		// A moving platform wakes whatever rests on it.
		for p in &self.platforms {
			let at = p.offset(self.time);
			if (p.offset(next) - at).length2() <= EPS * dt * dt {
				continue;
			}
			let (lo, hi) = p.mesh.bounds();
			for b in &mut self.bodies {
				if b.sleeping && crate::overlaps(body_bounds(b), lo + at, hi + at) {
					b.wake();
				}
			}
		}
		for b in &mut self.bodies {
			b.advance(&colliders, dt, self.gravity, filter);
		}
		let pairs = broadphase(&self.bodies);
		for _ in 0..SOLVER_ITERATIONS {
			for &(i, j) in &pairs {
				let (head, tail) = self.bodies.split_at_mut(j);
				collide_pair(&mut head[i], &mut tail[0]);
			}
			for b in &mut self.bodies {
				if !b.sleeping {
					b.resolve(&colliders, dt, filter);
				}
			}
		}
		// Bodies at rest go to sleep; the threshold is a step of gravity, the
		// velocity a resting body regains between one solve and the next.
		let still = self.gravity.length() * dt;
		for b in &mut self.bodies {
			if b.mass == 0.0 || b.sleeping {
				continue;
			}
			let extent = b.extent();
			// Recent support, not `grounded` this instant: resting at the
			// contact skin, a body's fall can stay inside it for a step.
			if b.since_grounded < 3.0 * dt && b.vel.length() < still && b.spin.length() * extent < still {
				b.sleep_timer += dt;
				if b.sleep_timer >= SLEEP_AFTER {
					b.sleeping = true;
					b.vel = Vec3::ZERO;
					b.spin = Vec3::ZERO;
				}
			} else {
				b.sleep_timer = 0.0;
			}
		}
		self.time = next;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{scenes, Mesh};
	use std::sync::Arc;
	use std::time::Instant;

	const DT: f32 = 1.0 / 60.0;

	/// A box hull body of half size `h`.
	fn hull_box(h: f32) -> Body {
		let mut m = Mesh::default();
		m.push_box(vec3(-h, -h, -h), vec3(h, h, h));
		Body::from_hull(crate::Hull::from_mesh(&m))
	}

	/// A box hull part of half size `h`, centered at `offset` in its body.
	fn part_box(h: Vec3, offset: Vec3) -> crate::Part {
		let mut m = Mesh::default();
		m.push_box(-h, h);
		crate::Part {
			offset,
			shape: crate::PartShape::Hull(Arc::new(crate::Hull::from_mesh(&m))),
		}
	}

	/// A compound body rests on the grid like any body: each part finds its own
	/// contacts, the parent never touches.
	#[test]
	fn compound_body_rests_and_sleeps() {
		let parts = crate::Parts::new(vec![
			part_box(vec3(0.4, 0.4, 0.4), vec3(-0.55, 0.0, 0.0)),
			part_box(vec3(0.4, 0.4, 0.4), vec3(0.55, 0.0, 0.0)),
		]);
		let mut w = World::new(scenes::grid(8));
		w.bodies.push(Body {
			pos: vec3(0.2, 3.0, 0.3),
			mass: 2.0,
			friction: 0.5,
			rotates: true,
			..Body::from_parts(parts)
		});
		for _ in 0..240 {
			w.step(DT, true);
		}
		let b = &w.bodies[0];
		assert!(b.grounded, "not grounded at {:?}", b.pos);
		assert!((b.pos.y - 0.4).abs() < 2e-2, "resting at {:?}", b.pos);
		assert!(b.vel.length() < 1e-2, "still moving: {:?}", b.vel);
		assert!(b.spin.length() < 1e-2, "still turning: {:?}", b.spin);
		assert!(b.sleeping, "never slept");
		assert!(b.penetration(&w.mesh) < 1e-3);
	}

	/// The concavity between parts is real against other bodies too: a sphere
	/// falls through a slot no single convex could leave.
	#[test]
	fn sphere_falls_through_the_slot_between_parts() {
		let parts = crate::Parts::new(vec![
			part_box(vec3(0.5, 0.5, 0.5), vec3(-0.8, 0.0, 0.0)),
			part_box(vec3(0.5, 0.5, 0.5), vec3(0.8, 0.0, 0.0)),
		]);
		let mut w = World::new(scenes::grid(8));
		// The compound hangs as static geometry at slot height.
		w.bodies.push(Body {
			pos: vec3(0.0, 0.5, 0.0),
			mass: 0.0,
			..Body::from_parts(parts)
		});
		w.bodies.push(Body {
			pos: vec3(0.0, 3.0, 0.0),
			radius: 0.25,
			..Body::default()
		});
		for _ in 0..240 {
			w.step(DT, true);
		}
		let b = &w.bodies[1];
		assert!(b.grounded, "not grounded at {:?}", b.pos);
		// On the floor under the compound, not on the compound: the parent
		// hull would have held it at 1.25.
		assert!(
			(b.pos.y - 0.25).abs() < 2e-2,
			"did not fall through the slot: {:?}",
			b.pos
		);
		assert_eq!(w.bodies[0].pos, vec3(0.0, 0.5, 0.0), "static body moved");
	}

	/// A part's inertia plus the parallel axis about the body's center.
	#[test]
	fn compound_inertia_spins_about_the_center() {
		// Two boxes at ±0.55, each 0.8 cubed, 2 kg together: about the y axis
		// each spins about the center through its arm, a dumbbell; about x
		// the arms lie along the axis and only the boxes' own moments stack.
		let parts = crate::Parts::new(vec![
			part_box(vec3(0.4, 0.4, 0.4), vec3(-0.55, 0.0, 0.0)),
			part_box(vec3(0.4, 0.4, 0.4), vec3(0.55, 0.0, 0.0)),
		]);
		let i = crate::Convex::inertia(&parts, 2.0);
		let own = 1.0 / 12.0 * (0.8 * 0.8 + 0.8 * 0.8);
		assert!((i.y - 2.0 * (own + 0.55 * 0.55)).abs() < 1e-3, "y {i:?}");
		assert!((i.x - 2.0 * own).abs() < 1e-3, "x {i:?}");
		assert!((i.z - i.y).abs() < 1e-6, "z {i:?}");
	}

	#[test]
	fn hull_box_rests_on_grid() {
		let mut w = World::new(scenes::grid(8));
		w.bodies.push(Body {
			pos: vec3(0.4, 3.0, 0.6),
			mass: 2.0,
			friction: 0.5,
			rotates: true,
			..hull_box(0.5)
		});
		for _ in 0..240 {
			w.step(DT, true);
		}
		let b = &w.bodies[0];
		assert!(b.grounded, "not grounded at {:?}", b.pos);
		assert!(b.vel.length() < 1e-2, "still moving: {:?}", b.vel);
		assert!(b.penetration(&w.mesh) < 1e-3);
	}

	#[test]
	fn hull_slides_into_wall_and_stops() {
		let mut w = World::new(scenes::wall());
		w.bodies.push(Body {
			pos: vec3(1.0, 0.5, 3.0),
			vel: vec3(4.0, 0.0, 0.0),
			mass: 2.0,
			..hull_box(0.5)
		});
		for _ in 0..180 {
			w.step(DT, true);
		}
		let b = &w.bodies[0];
		assert!(b.pos.x < 4.5 + 1e-2, "through the wall: {:?}", b.pos);
		assert!(b.pos.x > 4.4, "stopped short: {:?}", b.pos);
		assert!(b.vel.x.abs() < 1e-2, "still moving: {:?}", b.vel);
		assert!(b.penetration(&w.mesh) < 1e-3);
	}

	#[test]
	fn static_hull_blocks_a_falling_sphere() {
		let mut w = World::new(scenes::grid(8));
		w.bodies.push(Body {
			pos: vec3(2.0, 0.5, 2.0),
			mass: 0.0,
			..hull_box(0.5)
		});
		w.bodies.push(Body {
			pos: vec3(2.0, 3.0, 2.0),
			radius: 0.5,
			..Body::default()
		});
		let static_pos = w.bodies[0].pos;
		for _ in 0..240 {
			w.step(DT, true);
		}
		assert_eq!(w.bodies[0].pos, static_pos, "the static hull moved");
		let b = &w.bodies[1];
		assert!(b.grounded, "sphere never landed: {:?}", b.pos);
		assert!(
			(b.pos.y - 1.5).abs() < 1e-2,
			"rests at {:?}, not on the hull",
			b.pos
		);
		assert!(b.vel.length() < 1e-2, "still moving: {:?}", b.vel);
	}

	/// A sphere hitting a hull runs the GJK pair path; with no friction or
	/// gravity, the pair's linear momentum is conserved.
	#[test]
	fn sphere_on_hull_conserves_momentum() {
		let mut w = World::new(scenes::grid(8));
		w.gravity = Vec3::ZERO;
		w.bodies.push(Body {
			pos: vec3(0.0, 0.5, 0.0),
			vel: vec3(2.0, 0.0, 0.0),
			..Body::default()
		});
		w.bodies.push(Body {
			pos: vec3(3.0, 0.5, 0.0),
			mass: 2.0,
			rotates: true,
			..hull_box(0.5)
		});
		for _ in 0..180 {
			w.step(DT, true);
		}
		let px: f32 = w.bodies.iter().map(|b| b.mass * b.vel.x).sum();
		assert!((px - 2.0).abs() < 5e-3, "momentum {px}, was 2");
		assert!(w.bodies[1].vel.x > 1e-2, "hull never moved");
	}

	#[test]
	fn cylinder_body_rests_upright() {
		let mut w = World::new(scenes::grid(8));
		w.bodies.push(Body {
			pos: vec3(1.0, 3.0, 1.0),
			shape: Shape::Cylinder,
			radius: 0.3,
			half_height: 0.5,
			friction: 0.5,
			..Body::default()
		});
		for _ in 0..240 {
			w.step(DT, true);
		}
		let b = &w.bodies[0];
		assert!(b.grounded, "not grounded at {:?}", b.pos);
		assert!(
			(b.pos.y - 0.5).abs() < 1e-2,
			"rests at {:?}, not upright",
			b.pos
		);
		assert_eq!(b.orient, Quat::IDENTITY, "tipped: {:?}", b.orient);
		assert!(b.vel.length() < 1e-2, "still moving: {:?}", b.vel);
	}

	#[test]
	fn spheres_sleep_and_wake() {
		let mut w = World::new(scenes::grid(8));
		w.bodies.push(Body {
			pos: vec3(0.7, 0.5, 0.6),
			..Body::default()
		});
		for _ in 0..90 {
			w.step(DT, true);
		}
		assert!(w.bodies[0].sleeping, "never slept");
		// `wake` is the API for game code to stir a body.
		w.bodies[0].wake();
		assert!(!w.bodies[0].sleeping);
		// A body landing on a sleeper wakes it through the pair contact.
		for _ in 0..30 {
			w.step(DT, true);
		}
		assert!(w.bodies[0].sleeping, "did not settle back");
		w.bodies.push(Body {
			pos: vec3(0.7, 2.0, 0.6),
			..Body::default()
		});
		for _ in 0..60 {
			w.step(DT, true);
		}
		assert!(!w.bodies[0].sleeping, "contact did not wake it");
	}

	/// Sweep and prune finds exactly the pairs brute force does, including the
	/// static and sleeping skips.
	#[test]
	fn broadphase_matches_brute_force() {
		let mut seed = 3u32;
		let mut rnd = || {
			seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
			(seed >> 8) as f32 / (1u32 << 24) as f32 * 8.0 - 4.0
		};
		let bodies: Vec<Body> = (0..40)
			.map(|i| Body {
				pos: vec3(rnd(), rnd(), rnd()),
				radius: 0.5 + 0.5 * rnd().abs(),
				half_height: if i % 3 == 0 { 0.5 } else { 0.0 },
				mass: if i % 7 == 0 { 0.0 } else { 1.0 },
				sleeping: i % 11 == 0,
				..Body::default()
			})
			.collect();
		let boxes: Vec<(Vec3, Vec3)> = bodies.iter().map(body_bounds).collect();
		let active = |b: &Body| b.mass > 0.0 && !b.sleeping;
		let mut want = Vec::new();
		for i in 0..bodies.len() {
			for j in i + 1..bodies.len() {
				if crate::overlaps(boxes[i], boxes[j].0, boxes[j].1)
					&& (active(&bodies[i]) || active(&bodies[j]))
				{
					want.push((i, j));
				}
			}
		}
		want.sort();
		let mut got = broadphase(&bodies);
		got.sort();
		assert_eq!(got, want);
	}

	/// Steps a level full of mixed bodies and prints the cost per step: first
	/// while they fall and pile, then once sleep has emptied the active set.
	///
	/// Run in release, where the numbers mean anything:
	/// `cargo test -p physics --release -- --ignored --nocapture bench`
	#[test]
	#[ignore]
	fn bench_world_step() {
		let hull = {
			let mut m = Mesh::default();
			m.push_box(vec3(-0.4, -0.4, -0.4), vec3(0.4, 0.4, 0.4));
			Arc::new(crate::Hull::from_mesh(&m))
		};
		// The grid with a rim, so nothing rolls off the edge and falls forever:
		// a level holds its bodies, and the faller would never sleep.
		let mut level = scenes::grid(24);
		for (lo, hi) in [
			(vec3(-1.0, 0.0, -1.0), vec3(25.0, 1.0, 0.0)),
			(vec3(-1.0, 0.0, 24.0), vec3(25.0, 1.0, 25.0)),
			(vec3(-1.0, 0.0, 0.0), vec3(0.0, 1.0, 24.0)),
			(vec3(24.0, 0.0, 0.0), vec3(25.0, 1.0, 24.0)),
		] {
			level.push_box(lo, hi);
		}
		let mut w = World::new(level);
		// A lattice spacing wider than every body, so nothing starts overlapped
		// and jammed: a level's bodies spawn apart, and a settled level must
		// reach sleep, which a jam can never do.
		let mut spots = (0..20).flat_map(|i| (0..15).map(move |k| (i, k)));
		for _ in 0..300 {
			let (i, k) = spots.next().unwrap();
			let (x, z) = (0.6 + i as f32 * 1.2, 0.6 + k as f32 * 1.55);
			let y = 2.0 + (i % 5) as f32 * 0.9 + (k % 3) as f32 * 0.3;
			let body = match (i + k) % 4 {
				0 => Body {
					pos: vec3(x, y, z),
					shape: Shape::Hull(Arc::clone(&hull)),
					radius: 0.4,
					half_height: 0.4,
					mass: 2.0,
					friction: 0.5,
					rotates: true,
					..Body::default()
				},
				1 => Body {
					pos: vec3(x, y, z),
					shape: Shape::Cylinder,
					radius: 0.3,
					half_height: 0.5,
					mass: 1.5,
					friction: 0.5,
					rotates: true,
					..Body::default()
				},
				2 => Body {
					pos: vec3(x, y, z),
					shape: Shape::Capsule,
					radius: 0.3,
					half_height: 0.4,
					mass: 1.0,
					friction: 0.5,
					..Body::default()
				},
				_ => Body {
					pos: vec3(x, y, z),
					radius: 0.4,
					friction: 0.5,
					rotates: true,
					..Body::default()
				},
			};
			w.bodies.push(body);
		}
		let settle = Instant::now();
		for _ in 0..120 {
			w.step(DT, true);
		}
		let busy = settle.elapsed();
		// Phase costs over the same settled state `World::step` sees, by running
		// its phases once per body directly.
		let mut phase = (0u128, 0u128, 0u128, 0u128);
		for k in 0..480 {
			let colliders = [Placed {
				mesh: &w.mesh,
				offset: Vec3::ZERO,
				vel: Vec3::ZERO,
			}];
			let now = Instant::now();
			for b in &mut w.bodies {
				b.advance(&colliders, DT, w.gravity, true);
			}
			phase.0 += now.elapsed().as_micros();
			let now = Instant::now();
			let pairs = broadphase(&w.bodies);
			phase.1 += now.elapsed().as_micros();
			let now = Instant::now();
			for _ in 0..SOLVER_ITERATIONS {
				for &(i, j) in &pairs {
					let (head, tail) = w.bodies.split_at_mut(j);
					collide_pair(&mut head[i], &mut tail[0]);
				}
				for b in &mut w.bodies {
					if !b.sleeping {
						b.resolve(&colliders, DT, true);
					}
				}
			}
			phase.2 += now.elapsed().as_micros();
			let now = Instant::now();
			let still = w.gravity.length() * DT;
			for b in &mut w.bodies {
				if b.mass == 0.0 || b.sleeping {
					continue;
				}
				let extent = b.extent();
				if b.since_grounded < 3.0 * DT && b.vel.length() < still && b.spin.length() * extent < still
				{
					b.sleep_timer += DT;
					if b.sleep_timer >= SLEEP_AFTER {
						b.sleeping = true;
						b.vel = Vec3::ZERO;
						b.spin = Vec3::ZERO;
					}
				} else {
					b.sleep_timer = 0.0;
				}
			}
			phase.3 += now.elapsed().as_micros();
			w.time += if k == 0 { 0.0 } else { DT };
		}
		let idle = std::time::Duration::from_micros(phase.0 as u64)
			+ std::time::Duration::from_micros(phase.1 as u64)
			+ std::time::Duration::from_micros(phase.2 as u64)
			+ std::time::Duration::from_micros(phase.3 as u64);
		let awake = w
			.bodies
			.iter()
			.filter(|b| !b.sleeping && b.mass > 0.0)
			.count();
		let pairs = broadphase(&w.bodies).len();
		println!(
			"{} bodies, {} awake, {} candidate pairs: piling {:.0} µs/step, settled {:.0} µs/step (advance {:.0}, broadphase {:.0}, solve {:.0}, sleep {:.0})",
			w.bodies.len(),
			awake,
			pairs,
			busy.as_micros() / 120,
			idle.as_micros() / 480,
			phase.0 / 480,
			phase.1 / 480,
			phase.2 / 480,
			phase.3 / 480,
		);
	}
}
