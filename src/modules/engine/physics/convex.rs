//! Convex body shapes: the analytic cylinder and arbitrary mesh geometry as a hull.
//!
//! GJK only ever asks a shape for its support point along a direction, and a
//! triangle soup's support point is its convex hull's, so arbitrary mesh
//! geometry needs no hull algorithm: [`Hull`] keeps the soup and answers
//! supports from its vertices. Concavity is filled in, the standard trade for
//! a dynamic body; static concave geometry stays exact through [`crate::Mesh`].
//!
//! Every method answers in the body's local frame, `+y` the cylinder's axis,
//! a hull's vertices centered on the body (see [`Hull::from_tris`]). The mesh
//! queries in `lib.rs` rotate triangles into this frame and witnesses back out.

use crate::{Cylinder, Triangle, EPS};
use linalg::{vec3, Vec3};
use std::collections::HashSet;
use std::f32::consts::TAU;
use std::sync::Arc;

/// A convex shape in its body's local frame.
pub trait Convex {
	/// The point of the shape eroded by the distance `shrink` that lies farthest
	/// along `d`, which need not be unit length. The eroded shape must be a true
	/// convex shape — a support point slid `shrink` along `d` is not, and GJK
	/// would read the pseudo-shape as overlapping a plane it clears — so the
	/// cylinder erodes to the cylinder of radius and half height each less
	/// `shrink`, exact, and a hull scales about its center by the same fraction
	/// of its smallest half extent, never looser than its bounds. The vertex
	/// selection never changes, so GJK's witnesses stay meaningful.
	fn support(&self, d: Vec3, shrink: f32) -> Vec3;
	/// Local bounds around the origin, used for GJK's clipping box and queries.
	fn bounds(&self) -> (Vec3, Vec3);
	/// Diagonal moments of inertia of a solid of `mass` in local axes.
	fn inertia(&self, mass: f32) -> Vec3;

	/// Half extents of the clipping box: the symmetric box covering `bounds`.
	fn half(&self) -> Vec3 {
		let (lo, hi) = self.bounds();
		vec3(
			lo.x.abs().max(hi.x.abs()),
			lo.y.abs().max(hi.y.abs()),
			lo.z.abs().max(hi.z.abs()),
		)
	}
	/// Smallest half extent: shapes deflate at most this far before collapsing.
	fn min_half(&self) -> f32 {
		let h = self.half();
		h.x.min(h.y).min(h.z)
	}
	/// Largest deflation that can leave any volume.
	fn max_shrink(&self) -> f32 {
		self.min_half() * 0.999
	}
	/// How close to a triangle the shape rests on it: the contact skin. A percent
	/// of the shape's size, so resting is resolved before the surfaces touch and
	/// a rocking body keeps its contact, as a capsule's radius does for it.
	fn margin(&self) -> f32 {
		(self.min_half() * 1e-2).max(1e-5)
	}
}

impl Convex for Cylinder {
	fn support(&self, d: Vec3, shrink: f32) -> Vec3 {
		Cylinder::support(
			Cylinder {
				radius: self.radius - shrink,
				half_height: self.half_height - shrink,
			},
			Vec3::ZERO,
			d,
		)
	}

	fn bounds(&self) -> (Vec3, Vec3) {
		let r = vec3(self.radius, self.half_height, self.radius);
		(-r, r)
	}

	fn inertia(&self, mass: f32) -> Vec3 {
		// Solid cylinder: exact principal moments about the center of mass.
		let (r, l) = (self.radius, 2.0 * self.half_height);
		let perp = mass * (3.0 * r * r + l * l) / 12.0;
		let axis = mass * r * r / 2.0;
		vec3(perp, axis, perp)
	}
}

/// Arbitrary mesh geometry as a convex body: its triangles for reference and
/// rendering, its vertices for supports. The convex hull of the soup, never
/// computed: supports over the vertices are the hull's supports.
#[derive(Clone, Debug)]
pub struct Hull {
	/// Unique vertices, centered on the body's center of mass (the vertex
	/// centroid), local frame.
	pub verts: Vec<Vec3>,
	/// The source triangles over `verts`, shifted with them.
	pub tris: Vec<Triangle>,
	lo: Vec3,
	hi: Vec3,
}

impl Hull {
	/// Builds a hull from arbitrary triangles, duplicate vertices removed and
	/// the rest recentered on their centroid, so the body's `pos` is its center
	/// of mass and rotation is about it.
	pub fn from_tris(tris: Vec<Triangle>) -> Hull {
		assert!(!tris.is_empty(), "a hull needs triangles");
		let mut seen = HashSet::with_capacity(tris.len() * 3);
		let mut verts = Vec::with_capacity(tris.len() * 3);
		let index = |v: Vec3, seen: &mut HashSet<(u32, u32, u32)>, verts: &mut Vec<Vec3>| {
			let key = (v.x.to_bits(), v.y.to_bits(), v.z.to_bits());
			if seen.insert(key) {
				verts.push(v);
			}
		};
		for t in &tris {
			for &v in t {
				index(v, &mut seen, &mut verts);
			}
		}
		let centroid = verts.iter().fold(Vec3::ZERO, |p, &v| p + v) * (1.0 / verts.len() as f32);
		for v in &mut verts {
			*v -= centroid;
		}
		let tris = tris
			.into_iter()
			.map(|t| [t[0] - centroid, t[1] - centroid, t[2] - centroid])
			.collect();
		let lo = verts.iter().fold(verts[0], |m, &v| m.min(v));
		let hi = verts.iter().fold(verts[0], |m, &v| m.max(v));
		Hull {
			verts,
			tris,
			lo,
			hi,
		}
	}

	/// The hull of a mesh's triangles, for loading model geometry as a body.
	pub fn from_mesh(mesh: &crate::Mesh) -> Hull {
		Hull::from_tris(mesh.tris().to_vec())
	}
}

impl Convex for Hull {
	fn support(&self, d: Vec3, shrink: f32) -> Vec3 {
		let mut best = self.verts[0];
		let mut best_d = best.dot(d);
		for &v in &self.verts[1..] {
			let dv = v.dot(d);
			if dv > best_d {
				(best, best_d) = (v, dv);
			}
		}
		// Erosion by a distance has no cheap support over bare vertices, so
		// scale about the center by the fraction of the smallest half extent:
		// a true convex deflation, under-deflating a shape stockier than its
		// bounds, so depths measured through it err small.
		if shrink <= 0.0 {
			return best;
		}
		best * (1.0 - shrink / self.min_half().max(EPS))
	}

	fn bounds(&self) -> (Vec3, Vec3) {
		(self.lo, self.hi)
	}

	fn inertia(&self, mass: f32) -> Vec3 {
		// The bounding box's inertia: a solid of the same mass filling the
		// hull's local bounds. Exact for crates, close for anything stockier
		// than its box, and a body that does not rotate never asks.
		let s = self.hi - self.lo;
		let m = mass / 12.0;
		vec3(
			m * (s.y * s.y + s.z * s.z),
			m * (s.x * s.x + s.z * s.z),
			m * (s.x * s.x + s.y * s.y),
		)
	}
}

/// One convex piece of a compound body: a shape at a fixed offset from the
/// body's center, never rotating relative to its siblings.
#[derive(Clone, Debug)]
pub struct Part {
	/// The part's center in the body's local frame.
	pub offset: Vec3,
	/// The part's shape, in the part's own frame centered on `offset`.
	pub shape: PartShape,
}

/// A compound body's piece: an analytic cylinder or a hull of mesh geometry.
#[derive(Clone, Debug)]
pub enum PartShape {
	Cylinder(Cylinder),
	Hull(Arc<Hull>),
}

impl PartShape {
	/// The shape as `dyn Convex`, in the part's frame.
	pub fn get(&self) -> &dyn Convex {
		match self {
			PartShape::Cylinder(c) => c,
			PartShape::Hull(h) => &**h,
		}
	}

	/// The part's volume, weighting its share of the body's inertia.
	fn volume(&self) -> f32 {
		match self {
			// A cylinder's is exact; a hull is weighed as its bounds, over
			// for anything stockier than its box, and inertia is a feel, not
			// a guarantee.
			PartShape::Cylinder(c) => TAU * c.radius * c.radius * c.half_height,
			PartShape::Hull(h) => {
				let s = h.hi - h.lo;
				s.x * s.y * s.z
			}
		}
	}

	/// The shape's corner points in the part's frame, for the parent hull.
	fn corners(&self) -> Vec<Vec3> {
		match self {
			PartShape::Cylinder(c) => {
				// Two octagons: enough points that the parent hull's supports
				// round the cylinder no looser than the margin matters.
				(0..8)
					.flat_map(|k| {
						let (s, d) = (TAU * k as f32 / 8.0).sin_cos();
						[
							vec3(c.radius * d, -c.half_height, c.radius * s),
							vec3(c.radius * d, c.half_height, c.radius * s),
						]
					})
					.collect()
			}
			PartShape::Hull(h) => h.verts.clone(),
		}
	}
}

/// A compound body: convex parts filling a concave shape, plus the parent hull
/// over all their vertices. The parent answers every whole-body question —
/// broadphase bounds, the sweep, the sleep extents — and the parts answer the
/// contacts: GJK per part against the mesh and per part pair between bodies, so
/// a concave compound holds things in its concavity, where its parent hull
/// could not. The parts must together be the shape: whatever they leave
/// uncovered is not collided, so they come from a decomposition, not a slicing.
#[derive(Clone, Debug)]
pub struct Parts {
	/// The convex pieces, in the body's local frame.
	pub parts: Vec<Part>,
	parent: Hull,
}

impl Parts {
	/// Builds a compound body from its parts, computing the parent hull over
	/// their vertices and centering the whole on their centroid, so the body's
	/// `pos` is its center of mass and rotation is about it.
	pub fn new(parts: Vec<Part>) -> Parts {
		assert!(!parts.is_empty(), "a compound body needs parts");
		let verts: Vec<Vec3> = parts
			.iter()
			.flat_map(|p| p.shape.corners().into_iter().map(move |v| v + p.offset))
			.collect();
		let centroid = verts.iter().fold(Vec3::ZERO, |p, &v| p + v) * (1.0 / verts.len() as f32);
		// Degenerate triangles, one per vertex: `from_tris` dedupes and the
		// parent only supports and bounds, never renders or clips.
		let tris = verts
			.iter()
			.map(|&v| [v - centroid, v - centroid, v - centroid])
			.collect();
		let parent = Hull::from_tris(tris);
		let mut parts = parts;
		for p in &mut parts {
			p.offset -= centroid;
		}
		Parts { parts, parent }
	}
}

impl Convex for Parts {
	fn support(&self, d: Vec3, shrink: f32) -> Vec3 {
		self.parent.support(d, shrink)
	}

	fn bounds(&self) -> (Vec3, Vec3) {
		self.parent.bounds()
	}

	fn inertia(&self, mass: f32) -> Vec3 {
		// The parts' inertias by volume share plus the parallel axis about the
		// body's center: each part spins about the body, not itself alone.
		let total: f32 = self.parts.iter().map(|p| p.shape.volume()).sum();
		let mut diag = Vec3::ZERO;
		for p in &self.parts {
			let m = mass * p.shape.volume() / total.max(EPS);
			let local = p.shape.get().inertia(m);
			let (o, o2) = (p.offset, p.offset.length2());
			diag += vec3(
				local.x + m * (o2 - o.x * o.x),
				local.y + m * (o2 - o.y * o.y),
				local.z + m * (o2 - o.z * o.z),
			);
		}
		diag
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// `support` over a soup matches brute force over its vertices.
	#[test]
	fn support_is_the_farthest_vertex() {
		let hull = Hull::from_tris(vec![
			[
				vec3(0.0, 0.0, 0.0),
				vec3(2.0, 0.0, 0.0),
				vec3(0.0, 2.0, 0.0),
			],
			[
				vec3(0.0, 0.0, 1.0),
				vec3(2.0, 0.0, 1.0),
				vec3(0.0, 2.0, 1.0),
			],
			[
				vec3(0.5, 0.5, 5.0),
				vec3(0.5, 0.5, 5.0),
				vec3(0.5, 0.5, 5.0),
			],
		]);
		// `from_tris` recenters on the vertex centroid: the raw sum (4.5, 4.5, 8) over 7.
		let shift = vec3(4.5, 4.5, 8.0) * (1.0 / 7.0);
		let d = vec3(0.1, 0.2, 1.0);
		assert_eq!(hull.support(d, 0.0), vec3(0.5, 0.5, 5.0) - shift);
		let dd = vec3(-1.0, -0.1, 0.0);
		assert_eq!(hull.support(dd, 0.0), vec3(0.0, 0.0, 0.0) - shift);
		// Deflation is a true smaller shape; the vertex selection never changes.
		let s = hull.support(d, 0.5);
		assert_eq!(
			s,
			(vec3(0.5, 0.5, 5.0) - shift) * (1.0 - 0.5 / hull.min_half())
		);
	}

	/// A cylinder erodes exactly; a hull scales by the fraction of its smallest
	/// half extent.
	#[test]
	fn deflation_is_a_true_smaller_shape() {
		let cyl = Cylinder {
			radius: 1.0,
			half_height: 2.0,
		};
		assert_eq!(
			Convex::support(&cyl, vec3(0.0, -1.0, 0.0), 0.3),
			vec3(0.0, -1.7, 0.0)
		);
		assert_eq!(
			Convex::support(&cyl, vec3(1.0, 0.0, 0.0), 0.3),
			vec3(0.7, 1.7, 0.0)
		);
		let hull = Hull::from_tris(vec![
			[
				vec3(0.0, 0.0, 0.0),
				vec3(2.0, 0.0, 0.0),
				vec3(0.0, 2.0, 0.0),
			],
			[
				vec3(0.0, 0.0, 1.0),
				vec3(2.0, 0.0, 1.0),
				vec3(0.0, 2.0, 1.0),
			],
		]);
		let scale = 1.0 - 0.3 / hull.min_half();
		let d = vec3(0.0, 0.0, 1.0);
		assert_eq!(hull.support(d, 0.3), hull.support(d, 0.0) * scale);
	}

	/// Duplicate vertices collapse; the rest recenter on their centroid.
	#[test]
	fn from_tris_dedupes_and_recenters() {
		let hull = Hull::from_tris(vec![
			[
				vec3(0.0, 0.0, 0.0),
				vec3(1.0, 0.0, 0.0),
				vec3(0.0, 1.0, 0.0),
			],
			[
				vec3(0.0, 0.0, 0.0),
				vec3(1.0, 0.0, 0.0),
				vec3(1.0, 1.0, 0.0),
			],
		]);
		assert_eq!(hull.verts.len(), 4);
		let centroid = hull.verts.iter().fold(Vec3::ZERO, |p, &v| p + v) / 4.0;
		assert!(centroid.length() < 1e-6, "not centered: {centroid:?}");
	}
}
