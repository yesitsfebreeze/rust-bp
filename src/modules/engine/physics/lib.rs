//! Sphere and capsule collision against triangle meshes.
//!
//! Collision math is ported from Jakub Tomšů's Odin port of Christer Ericson's
//! Real-Time Collision Detection
//! (https://gist.github.com/jakubtomsu/2acd84731d3c2613c91e40c2e064ffe6), including
//! the corrected segment–segment clamp.
//!
//! Meshes are triangle soups: no adjacency, no welding. Internal-edge (ghost)
//! contacts are removed geometrically instead, so shared edges, unwelded seams and
//! T-junctions all behave the same. See [`Mesh::contacts`].
//!
//! Triangles are two-sided. Bodies sweep between steps ([`Mesh::sweep`]), so fast
//! bodies stop at thin geometry. Queries walk a bounding-volume hierarchy, and
//! contacts apply Coulomb friction ([`Body::friction`]).

use linalg::{vec3, Vec3};
use std::sync::OnceLock;

mod body;
pub use body::{Body, Platform, World};

pub type Triangle = [Vec3; 3];

/// Part of a triangle a closest point lies on. Edge `i` runs from vertex `i` to `(i + 1) % 3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
	Face,
	Edge(u8),
	Vertex(u8),
}

const EPS: f32 = 1e-6;

pub fn triangle_normal(tri: &Triangle) -> Vec3 {
	(tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize()
}

/// Closest point on segment `ab` to `point`, with `t` such that `point = a + t * (b - a)`.
pub fn closest_point_segment(point: Vec3, a: Vec3, b: Vec3) -> (f32, Vec3) {
	let ab = b - a;
	let t = ((point - a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
	(t, a + ab * t)
}

/// Closest points of S1(s) = p1 + s * (q1 - p1) and S2(t) = p2 + t * (q2 - p2).
///
/// Corrected from the gist revision the comments report: the unclamped numerator of
/// `t` is compared against `e` (not `[0, 1]`) and only divided by `e` once it is known
/// to lie inside S2. Clamping it undivided snapped closest points to segment ends.
pub fn closest_point_between_segments(
	p1: Vec3,
	q1: Vec3,
	p2: Vec3,
	q2: Vec3,
) -> ([f32; 2], [Vec3; 2]) {
	let d1 = q1 - p1; // Direction vector of segment S1
	let d2 = q2 - p2; // Direction vector of segment S2
	let r = p1 - p2;
	let a = d1.dot(d1); // Squared length of segment S1, always nonnegative
	let e = d2.dot(d2); // Squared length of segment S2, always nonnegative
	let f = d2.dot(r);

	let mut t = [0.0f32; 2];
	if a <= EPS && e <= EPS {
		// Both segments degenerate into points
		return (t, [p1, p2]);
	}
	if a <= EPS {
		// First segment degenerates into a point: s = 0 => t = (b*s + f) / e = f / e
		t[1] = (f / e).clamp(0.0, 1.0);
	} else {
		let c = d1.dot(r);
		if e <= EPS {
			// Second segment degenerates into a point: t = 0 => s = (b*t - c) / a = -c / a
			t[0] = (-c / a).clamp(0.0, 1.0);
		} else {
			// The general nondegenerate case starts here
			let b = d1.dot(d2);
			let denom = a * e - b * b; // Always nonnegative

			// If segments not parallel, compute closest point on L1 to L2 and
			// clamp to segment S1. Else pick arbitrary s (here 0)
			t[0] = if denom != 0.0 {
				((b * f - c * e) / denom).clamp(0.0, 1.0)
			} else {
				0.0
			};

			// Point on L2 closest to S1(s): t = (b*s + f) / e. Keep the numerator and
			// compare it against e, so [0, 1] in t is [0, e] in tnom.
			let tnom = b * t[0] + f;
			if tnom < 0.0 {
				t[1] = 0.0;
				t[0] = (-c / a).clamp(0.0, 1.0);
			} else if tnom > e {
				t[1] = 1.0;
				t[0] = ((b - c) / a).clamp(0.0, 1.0);
			} else {
				t[1] = tnom / e;
			}
		}
	}
	(t, [p1 + d1 * t[0], p2 + d2 * t[1]])
}

/// Closest point on `tri` to `point`, and the feature it lies on.
pub fn closest_point_triangle(point: Vec3, tri: &Triangle) -> (Vec3, Feature) {
	let [a, b, c] = *tri;
	let ab = b - a;
	let ac = c - a;
	let ap = point - a;
	let d1 = ab.dot(ap);
	let d2 = ac.dot(ap);
	if d1 <= 0.0 && d2 <= 0.0 {
		return (a, Feature::Vertex(0)); // barycentric coordinates (1,0,0)
	}

	// Check if P in vertex region outside B
	let bp = point - b;
	let d3 = ab.dot(bp);
	let d4 = ac.dot(bp);
	if d3 >= 0.0 && d4 <= d3 {
		return (b, Feature::Vertex(1)); // barycentric coordinates (0,1,0)
	}

	// Check if P in edge region of AB, if so return projection of P onto AB
	let vc = d1 * d4 - d3 * d2;
	if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
		let v = d1 / (d1 - d3);
		return (a + ab * v, Feature::Edge(0)); // barycentric coordinates (1-v,v,0)
	}

	// Check if P in vertex region outside C
	let cp = point - c;
	let d5 = ab.dot(cp);
	let d6 = ac.dot(cp);
	if d6 >= 0.0 && d5 <= d6 {
		return (c, Feature::Vertex(2)); // barycentric coordinates (0,0,1)
	}

	// Check if P in edge region of AC, if so return projection of P onto AC
	let vb = d5 * d2 - d1 * d6;
	if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
		let w = d2 / (d2 - d6);
		return (a + ac * w, Feature::Edge(2)); // barycentric coordinates (1-w,0,w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	let va = d3 * d6 - d5 * d4;
	if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
		let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return (b + (c - b) * w, Feature::Edge(1)); // barycentric coordinates (0,1-w,w)
	}

	// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
	let denom = 1.0 / (va + vb + vc);
	(a + ab * (vb * denom) + ac * (vc * denom), Feature::Face)
}

/// Parameter `t` where segment `pq` enters `tri` from its front side.
pub fn intersect_segment_triangle(p: Vec3, q: Vec3, tri: &Triangle) -> Option<f32> {
	let ab = tri[1] - tri[0];
	let ac = tri[2] - tri[0];
	let qp = p - q;
	let normal = ab.cross(ac);

	// If denom <= 0, segment is parallel to or points away from triangle
	let denom = qp.dot(normal);
	if denom <= 0.0 {
		return None;
	}
	// Intersection t with the plane; delay dividing by denom until it pierces
	let ap = p - tri[0];
	let t = ap.dot(normal);
	if t < 0.0 || t > denom {
		return None;
	}
	// Compute barycentric coordinate components and test if within bounds
	let e = qp.cross(ap);
	let v = ac.dot(e);
	if v < 0.0 || v > denom {
		return None;
	}
	let w = -ab.dot(e);
	if w < 0.0 || v + w > denom {
		return None;
	}
	Some(t / denom)
}

/// Closest approach between a segment (a capsule's axis, or a sphere's center when
/// both ends coincide) and one triangle.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
	pub seg_point: Vec3,
	pub tri_point: Vec3,
	pub feature: Feature,
	/// Distance between the points; negative depth below the plane when the segment pierces.
	pub dist: f32,
	/// Unit direction from `tri_point` toward `seg_point`.
	pub normal: Vec3,
}

/// Triangles are two-sided: the normal points to whichever side the segment is on.
pub fn segment_triangle(p0: Vec3, p1: Vec3, tri: &Triangle) -> Hit {
	let n = triangle_normal(tri);
	if intersect_segment_triangle(p0, p1, tri).is_some()
		|| intersect_segment_triangle(p1, p0, tri).is_some()
	{
		// Pierced: the side holding the segment's midpoint wins, and the end on the
		// other side is pushed back out along that side's normal.
		let (s0, s1) = ((p0 - tri[0]).dot(n), (p1 - tri[0]).dot(n));
		let n = if s0 + s1 >= 0.0 { n } else { -n };
		let (s0, s1) = ((p0 - tri[0]).dot(n), (p1 - tri[0]).dot(n));
		let (deep, sd) = if s0 < s1 { (p0, s0) } else { (p1, s1) };
		return Hit {
			seg_point: deep,
			tri_point: deep - n * sd,
			feature: Feature::Face,
			dist: sd,
			normal: n,
		};
	}

	let ends: &[Vec3] = if (p1 - p0).length2() <= EPS {
		&[p0]
	} else {
		&[p0, p1]
	};
	let mut best: Option<(Vec3, Vec3, Feature, f32)> = None;
	for &s in ends {
		let (q, feature) = closest_point_triangle(s, tri);
		let d = (s - q).length();
		if best.is_none_or(|b| d < b.3) {
			best = Some((s, q, feature, d));
		}
	}
	let (mut s, mut q, mut feature, mut d) = best.unwrap();
	if ends.len() == 2 {
		for i in 0..3 {
			let j = (i + 1) % 3;
			let (t, [ps, pt]) = closest_point_between_segments(p0, p1, tri[i], tri[j]);
			let de = (ps - pt).length();
			// Strictly closer only: on ties the endpoint (often a face point) wins.
			if de < d - EPS {
				(s, q, d) = (ps, pt, de);
				feature = if t[1] <= 0.0 {
					Feature::Vertex(i as u8)
				} else if t[1] >= 1.0 {
					Feature::Vertex(j as u8)
				} else {
					Feature::Edge(i as u8)
				};
			}
		}
	}

	let normal = if d > EPS { (s - q) * (1.0 / d) } else { n };
	Hit {
		seg_point: s,
		tri_point: q,
		feature,
		dist: d,
		normal,
	}
}

/// Fraction of `motion` after which the capsule `p0`–`p1` first comes within
/// `SWEEP_STOP * radius` of `tri`, if it does before `limit`.
///
/// The distance between two convex shapes under translation is convex in time, so
/// it never falls below its tangent line: stepping to where the tangent reaches
/// `SWEEP_TARGET * radius` can never overshoot, and from the left that converges like
/// Newton's method. A non-positive closing speed means the distance only grows from
/// there, which also ends grazing motion along a surface in one query.
fn sweep_triangle(
	p0: Vec3,
	p1: Vec3,
	radius: f32,
	motion: Vec3,
	tri: &Triangle,
	limit: f32,
) -> Option<f32> {
	let mut t = 0.0;
	for _ in 0..SWEEP_ITERATIONS {
		let hit = segment_triangle(p0 + motion * t, p1 + motion * t, tri);
		let closing = -motion.dot(hit.normal);
		if hit.dist < radius * SWEEP_STOP {
			// Already this close at the start: block only motion that goes deeper.
			return (t > 0.0 || closing > 0.0).then_some(t);
		}
		if closing <= 0.0 {
			return None;
		}
		t += (hit.dist - radius * SWEEP_TARGET) / closing;
		if t >= limit {
			return None;
		}
	}
	Some(t)
}

/// A sweep stops a body once it is this fraction of its radius from a triangle. The
/// contact solver resolves the remaining overlap; the center never reaches the
/// triangle, so it cannot cross it and the push-out side stays the one it came from.
const SWEEP_STOP: f32 = 0.5;
/// Where each sweep iteration aims, below `SWEEP_STOP` so iterations finish.
const SWEEP_TARGET: f32 = 0.25;
const SWEEP_ITERATIONS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
	pub tri: usize,
	pub hit: Hit,
	pub depth: f32,
}

/// Points closer than this to a triangle are on it.
const ON_SURFACE: f32 = 1e-4;

fn bounds(tri: &Triangle) -> (Vec3, Vec3) {
	(
		tri[0].min(tri[1]).min(tri[2]),
		tri[0].max(tri[1]).max(tri[2]),
	)
}

fn overlaps((lo, hi): (Vec3, Vec3), min: Vec3, max: Vec3) -> bool {
	lo.x <= max.x && hi.x >= min.x && lo.y <= max.y && hi.y >= min.y && lo.z <= max.z && hi.z >= min.z
}

/// Bounding-volume hierarchy over a mesh's triangles, nodes in depth-first order: an
/// inner node's left child follows it, `right` names the other. Leaves list
/// `order[first..first + count]`.
#[derive(Clone, Debug, Default)]
struct Bvh {
	nodes: Vec<Node>,
	order: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct Node {
	lo: Vec3,
	hi: Vec3,
	/// Leaf: first index into `order`. Inner: index of the right child.
	first: u32,
	/// Triangles in a leaf; zero for inner nodes.
	count: u32,
}

const LEAF_SIZE: usize = 4;

impl Bvh {
	fn build(tris: &[Triangle]) -> Bvh {
		let mut bvh = Bvh {
			nodes: Vec::new(),
			order: (0..tris.len() as u32).collect(),
		};
		if !tris.is_empty() {
			bvh.split(tris, 0, tris.len());
		}
		bvh
	}

	/// Median split along the longest axis of the triangle centroids.
	fn split(&mut self, tris: &[Triangle], start: usize, end: usize) {
		let items = &mut self.order[start..end];
		let (mut lo, mut hi) = bounds(&tris[items[0] as usize]);
		let centroid = |i: u32| {
			let t = &tris[i as usize];
			(t[0] + t[1] + t[2]) * (1.0 / 3.0)
		};
		let (mut clo, mut chi) = (centroid(items[0]), centroid(items[0]));
		for &i in items.iter() {
			let (a, b) = bounds(&tris[i as usize]);
			(lo, hi) = (lo.min(a), hi.max(b));
			let c = centroid(i);
			(clo, chi) = (clo.min(c), chi.max(c));
		}
		let at = self.nodes.len();
		self.nodes.push(Node {
			lo,
			hi,
			first: start as u32,
			count: (end - start) as u32,
		});
		if end - start <= LEAF_SIZE {
			return;
		}
		let e = chi - clo;
		let axis = |v: Vec3| {
			if e.x >= e.y && e.x >= e.z {
				v.x
			} else if e.y >= e.z {
				v.y
			} else {
				v.z
			}
		};
		let mid = (end - start) / 2;
		items.select_nth_unstable_by(mid, |&a, &b| {
			axis(centroid(a)).total_cmp(&axis(centroid(b)))
		});
		self.split(tris, start, start + mid);
		self.nodes[at].first = self.nodes.len() as u32;
		self.nodes[at].count = 0;
		self.split(tris, start + mid, end);
	}
}

/// Triangles whose bounds overlap a box, found by walking the [`Bvh`].
struct Near<'a> {
	tris: &'a [Triangle],
	bvh: &'a Bvh,
	min: Vec3,
	max: Vec3,
	stack: Vec<u32>,
	leaf: std::ops::Range<usize>,
}

impl<'a> Iterator for Near<'a> {
	type Item = (usize, &'a Triangle);

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			if let Some(k) = self.leaf.next() {
				let i = self.bvh.order[k] as usize;
				if overlaps(bounds(&self.tris[i]), self.min, self.max) {
					return Some((i, &self.tris[i]));
				}
				continue;
			}
			let at = self.stack.pop()? as usize;
			let node = self.bvh.nodes[at];
			if !overlaps((node.lo, node.hi), self.min, self.max) {
				continue;
			}
			if node.count > 0 {
				self.leaf = node.first as usize..(node.first + node.count) as usize;
			} else {
				self.stack.extend([node.first, at as u32 + 1]);
			}
		}
	}
}

/// Triangle soup. Queries go through a bounding-volume hierarchy, built on the first
/// query after the triangles last changed.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
	tris: Vec<Triangle>,
	bvh: OnceLock<Bvh>,
}

impl Mesh {
	pub fn tris(&self) -> &[Triangle] {
		&self.tris
	}

	pub fn push_triangle(&mut self, tri: Triangle) {
		self.tris.push(tri);
		self.bvh = OnceLock::new();
	}

	/// Adds quad `abcd` as two triangles, wound so the face points along `outward`.
	pub fn push_quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, outward: Vec3) {
		for tri in [[a, b, c], [a, c, d]] {
			let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
			self.push_triangle(if n.dot(outward) >= 0.0 {
				tri
			} else {
				[tri[0], tri[2], tri[1]]
			});
		}
	}

	/// Axis-aligned box with outward faces.
	pub fn push_box(&mut self, min: Vec3, max: Vec3) {
		let p = |x: bool, y: bool, z: bool| {
			vec3(
				if x { max.x } else { min.x },
				if y { max.y } else { min.y },
				if z { max.z } else { min.z },
			)
		};
		let (f, t) = (false, true);
		self.push_quad(
			p(f, t, f),
			p(t, t, f),
			p(t, t, t),
			p(f, t, t),
			vec3(0.0, 1.0, 0.0),
		);
		self.push_quad(
			p(f, f, f),
			p(t, f, f),
			p(t, f, t),
			p(f, f, t),
			vec3(0.0, -1.0, 0.0),
		);
		self.push_quad(
			p(t, f, f),
			p(t, t, f),
			p(t, t, t),
			p(t, f, t),
			vec3(1.0, 0.0, 0.0),
		);
		self.push_quad(
			p(f, f, f),
			p(f, t, f),
			p(f, t, t),
			p(f, f, t),
			vec3(-1.0, 0.0, 0.0),
		);
		self.push_quad(
			p(f, f, t),
			p(t, f, t),
			p(t, t, t),
			p(f, t, t),
			vec3(0.0, 0.0, 1.0),
		);
		self.push_quad(
			p(f, f, f),
			p(t, f, f),
			p(t, t, f),
			p(f, t, f),
			vec3(0.0, 0.0, -1.0),
		);
	}

	fn near(&self, min: Vec3, max: Vec3) -> Near<'_> {
		let bvh = self.bvh.get_or_init(|| Bvh::build(&self.tris));
		Near {
			tris: &self.tris,
			bvh,
			min,
			max,
			stack: if bvh.nodes.is_empty() {
				Vec::new()
			} else {
				vec![0]
			},
			leaf: 0..0,
		}
	}

	/// Fraction of `motion` the capsule `p0`–`p1` can travel before it comes within
	/// half its radius of a triangle; `None` when it clears the whole way.
	pub fn sweep(&self, p0: Vec3, p1: Vec3, radius: f32, motion: Vec3) -> Option<f32> {
		let r = vec3(radius, radius, radius);
		let (q0, q1) = (p0 + motion, p1 + motion);
		let mut first: Option<f32> = None;
		for (_, t) in self.near(
			p0.min(p1).min(q0).min(q1) - r,
			p0.max(p1).max(q0).max(q1) + r,
		) {
			if let Some(at) = sweep_triangle(p0, p1, radius, motion, t, first.unwrap_or(1.0)) {
				first = Some(at);
			}
		}
		first
	}

	/// Contacts of the capsule `p0`–`p1` with `radius`, deepest first.
	///
	/// With `filter`, an edge or vertex contact at point P is dropped when another
	/// triangle passing through P is strictly closer to the shape. Then P is not the
	/// closest point of that surface patch, so the contact is an internal-edge artifact,
	/// and the closer triangle supplies the real contact. Only point-on-triangle
	/// queries are used, so unwelded seams and T-junctions need no adjacency data.
	/// Convex edges survive: every triangle through P is equally far there.
	pub fn contacts(&self, p0: Vec3, p1: Vec3, radius: f32, filter: bool) -> Vec<Contact> {
		let r = vec3(radius, radius, radius);
		let mut out: Vec<Contact> = self
			.near(p0.min(p1) - r, p0.max(p1) + r)
			.filter_map(|(tri, t)| {
				let hit = segment_triangle(p0, p1, t);
				(hit.dist < radius).then_some(Contact {
					tri,
					hit,
					depth: radius - hit.dist,
				})
			})
			.collect();
		if filter {
			out.retain(|c| c.hit.feature == Feature::Face || !self.dominated(c, p0, p1));
		}
		out.sort_by(|a, b| b.depth.total_cmp(&a.depth));
		out
	}

	fn dominated(&self, c: &Contact, p0: Vec3, p1: Vec3) -> bool {
		let p = c.hit.tri_point;
		let e = vec3(ON_SURFACE, ON_SURFACE, ON_SURFACE);
		self.near(p - e, p + e).any(|(i, t)| {
			i != c.tri
				&& (closest_point_triangle(p, t).0 - p).length() <= ON_SURFACE
				&& segment_triangle(p0, p1, t).dist < c.hit.dist - EPS
		})
	}
}

/// Where a [`Mesh::cast_cylinder`] stopped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cast {
	/// Fraction of the motion travelled, in `0..=1`.
	pub fraction: f32,
	/// Unit direction from the blocking triangle toward the cylinder where it stopped.
	pub normal: Vec3,
	pub tri: usize,
}

/// Vertical cylinder: flat top and bottom, round sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cylinder {
	pub radius: f32,
	pub half_height: f32,
}

impl Cylinder {
	/// The point of the cylinder centred at `center` farthest along `d`.
	fn support(self, center: Vec3, d: Vec3) -> Vec3 {
		let flat = vec3(d.x, 0.0, d.z);
		let len = flat.length();
		let side = if len > EPS {
			flat * (self.radius / len)
		} else {
			Vec3::ZERO
		};
		let h = if d.y >= 0.0 {
			self.half_height
		} else {
			-self.half_height
		};
		center + side + vec3(0.0, h, 0.0)
	}
}

/// A vertex of the GJK simplex: a point of the Minkowski difference and the
/// cylinder point it came from.
#[derive(Clone, Copy, Debug)]
struct Support {
	w: Vec3,
	a: Vec3,
}

/// Shrinks the simplex `s` to the vertices spanning its point closest to the origin,
/// with their weights in `l`, and returns that point; `None` when the origin is inside.
fn reduce(s: &mut Vec<Support>, l: &mut Vec<f32>) -> Option<Vec3> {
	let all = s.clone();
	let weights: Vec<(usize, f32)> = match all.len() {
		1 => vec![(0, 1.0)],
		2 => {
			let (a, b) = (all[0].w, all[1].w);
			let ab = b - a;
			let t = (-a.dot(ab) / ab.length2().max(EPS * EPS)).clamp(0.0, 1.0);
			if t <= 0.0 {
				vec![(0, 1.0)]
			} else if t >= 1.0 {
				vec![(1, 1.0)]
			} else {
				vec![(0, 1.0 - t), (1, t)]
			}
		}
		3 => {
			let w = origin_triangle([all[0].w, all[1].w, all[2].w]);
			(0..3).filter(|&i| w[i] > 0.0).map(|i| (i, w[i])).collect()
		}
		_ => {
			// The closest of the faces that see the origin from outside. A tetrahedron
			// thinner than f32 resolves at its size has no reliable inside: then every
			// face is a candidate. Thin but resolvable ones, as a small shape against
			// a large triangle gives, keep their inside.
			let size = all.iter().fold(1.0f32, |m, p| m.max(p.w.length()));
			let flat = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)]
				.iter()
				.any(|&(i, j, k, o)| {
					let (a, b, c) = (all[i].w, all[j].w, all[k].w);
					let n = (b - a).cross(c - a);
					let len = n.length();
					len <= EPS * size * size || n.dot(all[o].w - a).abs() / len <= 1e-6 * size
				});
			let mut best: Option<(f32, Vec<(usize, f32)>)> = None;
			for (i, j, k, o) in [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)] {
				let (a, b, c) = (all[i].w, all[j].w, all[k].w);
				let n = (b - a).cross(c - a);
				if !flat && n.dot(-a) * n.dot(all[o].w - a) > 0.0 {
					continue;
				}
				let w = origin_triangle([a, b, c]);
				let d = (a * w[0] + b * w[1] + c * w[2]).length2();
				if best.as_ref().is_none_or(|b| d < b.0) {
					let idx = [i, j, k];
					best = Some((
						d,
						(0..3)
							.filter(|&m| w[m] > 0.0)
							.map(|m| (idx[m], w[m]))
							.collect(),
					));
				}
			}
			best?.1
		}
	};
	*s = weights.iter().map(|&(i, _)| all[i]).collect();
	*l = weights.iter().map(|&(_, x)| x).collect();
	Some(
		s.iter()
			.zip(l.iter())
			.fold(Vec3::ZERO, |p, (v, &x)| p + v.w * x),
	)
}

/// Barycentric weights of the point of triangle `t` closest to the origin
/// (Ericson, Real-Time Collision Detection 5.1.5).
fn origin_triangle(t: [Vec3; 3]) -> [f32; 3] {
	let [a, b, c] = t;
	let (ab, ac) = (b - a, c - a);
	let (d1, d2) = (ab.dot(-a), ac.dot(-a));
	if d1 <= 0.0 && d2 <= 0.0 {
		return [1.0, 0.0, 0.0];
	}
	let (d3, d4) = (ab.dot(-b), ac.dot(-b));
	if d3 >= 0.0 && d4 <= d3 {
		return [0.0, 1.0, 0.0];
	}
	let vc = d1 * d4 - d3 * d2;
	if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
		let v = d1 / (d1 - d3);
		return [1.0 - v, v, 0.0];
	}
	let (d5, d6) = (ab.dot(-c), ac.dot(-c));
	if d6 >= 0.0 && d5 <= d6 {
		return [0.0, 0.0, 1.0];
	}
	let vb = d5 * d2 - d1 * d6;
	if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
		let w = d2 / (d2 - d6);
		return [1.0 - w, 0.0, w];
	}
	let va = d3 * d6 - d5 * d4;
	if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
		let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return [0.0, 1.0 - w, w];
	}
	let denom = 1.0 / (va + vb + vc);
	let (v, w) = (vb * denom, vc * denom);
	[1.0 - v - w, v, w]
}

const GJK_ITERATIONS: usize = 64;

/// `poly` clipped to the box `±half` (Sutherland–Hodgman).
fn clip_to_box(poly: &[Vec3], half: Vec3) -> Vec<Vec3> {
	let axis = |p: Vec3, k: usize| match k {
		0 => p.x,
		1 => p.y,
		_ => p.z,
	};
	let mut out = poly.to_vec();
	for k in 0..3 {
		for sign in [1.0, -1.0] {
			let limit = axis(half, k);
			let input = std::mem::take(&mut out);
			for (i, &a) in input.iter().enumerate() {
				let b = input[(i + 1) % input.len()];
				let (da, db) = (sign * axis(a, k) - limit, sign * axis(b, k) - limit);
				if da <= 0.0 {
					out.push(a);
				}
				if (da <= 0.0) != (db <= 0.0) {
					out.push(a + (b - a) * (da / (da - db)));
				}
			}
			if out.is_empty() {
				return out;
			}
		}
	}
	out
}

/// Closest approach of the cylinder at `center` and `tri`, `None` when they overlap.
/// `seg_point` is the cylinder's closest point.
///
/// Exact closer than `reach`; farther apart, `dist` is `reach`, a lower bound. GJK
/// runs on the triangle clipped to a box `reach` around the cylinder, in the
/// cylinder's frame, so every point stays near the origin: against a triangle much
/// larger than the cylinder, f32 would otherwise lose the overlap entirely.
fn cylinder_triangle(center: Vec3, cyl: Cylinder, tri: &Triangle, reach: f32) -> Option<Hit> {
	let half = vec3(
		cyl.radius + reach,
		cyl.half_height + reach,
		cyl.radius + reach,
	);
	let poly = clip_to_box(&[tri[0] - center, tri[1] - center, tri[2] - center], half);
	if poly.is_empty() {
		let (q, feature) = closest_point_triangle(center, tri);
		return Some(Hit {
			seg_point: center,
			tri_point: q,
			feature,
			dist: reach,
			normal: (center - q).normalize(),
		});
	}
	let support = |d: Vec3| {
		let b = poly
			.iter()
			.copied()
			.max_by(|p, q| p.dot(-d).total_cmp(&q.dot(-d)))
			.unwrap();
		let a = cyl.support(Vec3::ZERO, d);
		Support { w: a - b, a }
	};
	let mut s = vec![support(poly[0])];
	let mut l = vec![1.0];
	let mut v = s[0].w;
	for _ in 0..GJK_ITERATIONS {
		let next = support(-v);
		// No support point comes meaningfully closer: `v` is the closest point.
		if v.length2() - v.dot(next.w) <= 1e-6 * v.length2() || s.iter().any(|p| p.w == next.w) {
			break;
		}
		s.push(next);
		v = reduce(&mut s, &mut l)?;
		// Closer than f32 resolves at the simplex's size: the shapes touch, and the
		// origin may lie on either side, so call it an overlap.
		let size = s.iter().fold(1.0f32, |m, p| m.max(p.w.length()));
		if v.length() <= 1e-5 * size {
			return None;
		}
	}
	let a = center + s.iter().zip(&l).fold(Vec3::ZERO, |p, (q, &x)| p + q.a * x);
	let tri_point = a - v;
	let n = triangle_normal(tri);
	let n = if n.dot(v) < 0.0 { -n } else { n };
	// Against the inside of a face, the exact answer: the cylinder's deepest point
	// toward the plane and its height above it. GJK's estimate drifts by a
	// fraction of a unit against a round rim, enough to read sliding along a slope
	// as moving into it.
	if v.normalize().dot(n) > 0.999 {
		let deepest = cyl.support(center, -n);
		let dist = n.dot(deepest - tri[0]);
		if dist <= 0.0 {
			return None;
		}
		return Some(Hit {
			seg_point: deepest,
			tri_point: deepest - n * dist,
			feature: Feature::Face,
			dist: dist.min(reach),
			normal: n,
		});
	}
	// A drifted direction cannot pick the side of the face: flipped by it, the
	// "deepest" point is the far end of the hull, whose projection crosses the
	// face as if the hull were sunk into it, and the normal points through the
	// surface. The face contact is whichever side holds a real separation.
	let face = |o: Vec3| -> Option<Hit> {
		let deepest = cyl.support(center, -o);
		let dist = o.dot(deepest - tri[0]);
		(dist > 0.0 && closest_point_triangle(deepest - o * dist, tri).1 == Feature::Face).then(|| {
			Hit {
				seg_point: deepest,
				tri_point: deepest - o * dist,
				feature: Feature::Face,
				dist: dist.min(reach),
				normal: o,
			}
		})
	};
	if let Some(hit) = face(n).or_else(|| face(-n)) {
		return Some(hit);
	}
	let feature = match closest_point_triangle(tri_point, tri).1 {
		Feature::Face => Feature::Edge(0),
		f => f,
	};
	Some(Hit {
		seg_point: a,
		tri_point,
		feature,
		dist: v.length().min(reach),
		normal: v.normalize(),
	})
}

/// How far `cyl` at `center` sinks into `tri` and the way out, or `None` when they
/// do not overlap. The cylinder shrinks evenly until it clears the triangle: the
/// shrink is the depth, exact against a face, and the shrunk cylinder's separating
/// direction is the way out.
fn cylinder_penetration(center: Vec3, cyl: Cylinder, tri: &Triangle) -> Option<(Vec3, f32)> {
	if cylinder_triangle(center, cyl, tri, 0.0).is_some() {
		return None;
	}
	let most = cyl.radius.min(cyl.half_height) * 0.999;
	let shrunk = |s: f32| Cylinder {
		radius: cyl.radius - s,
		half_height: cyl.half_height - s,
	};
	let reach = cyl.radius + cyl.half_height;
	let Some(_) = cylinder_triangle(center, shrunk(most), tri, reach) else {
		// The axis itself is inside: out along the plane, from its side.
		let n = triangle_normal(tri);
		let n = if n.dot(center - tri[0]) < 0.0 { -n } else { n };
		return Some((n, most + n.dot(tri[0] - center).abs()));
	};
	let (mut inside, mut clear) = (0.0, most);
	for _ in 0..24 {
		let mid = (inside + clear) * 0.5;
		match cylinder_triangle(center, shrunk(mid), tri, reach) {
			Some(_) => clear = mid,
			None => inside = mid,
		}
	}
	let hit = cylinder_triangle(center, shrunk(clear), tri, reach)?;
	Some((hit.normal, clear - hit.dist.min(clear)))
}

/// Conservative advancement of the cylinder along `motion` toward `tri`: the first
/// fraction at which it comes within `skin`, with the approach there, if before
/// `limit`. Each step aims at half the skin, which the distance, convex in time,
/// never passes.
///
/// Within `skin`, on arrival or from the start, only motion whose tangent line, a
/// lower bound on the distance, would close past a quarter of the skin is blocked:
/// one stopped at half the skin slides along the surface despite rounding in the
/// motion, and one reaching a seam whose normal has turned upright passes. One
/// overlapping the triangle is blocked outright, as a Quake trace starting in solid.
fn sweep_cylinder_triangle(
	center: Vec3,
	cyl: Cylinder,
	motion: Vec3,
	tri: &Triangle,
	skin: f32,
	limit: f32,
) -> Option<(f32, Hit)> {
	let reach = motion.length() * limit + skin;
	let at = |t: f32| cylinder_triangle(center + motion * t, cyl, tri, reach);
	let Some(mut hit) = at(0.0) else {
		// Already inside: block only motion that goes deeper, so a cylinder pushed
		// into a corner by a fast hit can still move out.
		let Some((normal, _)) = cylinder_penetration(center, cyl, tri) else {
			// Not inside after all: GJK called a grazing touch an overlap. Block
			// only motion into the face, along its outward side; one along it,
			// away from it, or out of it stays free, instead of wedging a hull
			// resting in the skin against the surface it rests on.
			let n = triangle_normal(tri);
			let n = if n.dot(center - tri[0]) < 0.0 { -n } else { n };
			if -motion.dot(n) <= 0.0 {
				return None;
			}
			let solid = Hit {
				seg_point: center,
				tri_point: center,
				feature: Feature::Face,
				dist: 0.0,
				normal: n,
			};
			return Some((0.0, solid));
		};
		let solid = Hit {
			seg_point: center,
			tri_point: center,
			feature: Feature::Face,
			dist: 0.0,
			normal,
		};
		return (-motion.dot(normal) > 0.0).then_some((0.0, solid));
	};
	let mut t = 0.0;
	for _ in 0..SWEEP_ITERATIONS {
		let closing = -motion.dot(hit.normal);
		if hit.dist < skin {
			let deeper = closing > 0.0 && hit.dist - closing * (limit - t) < skin * 0.25;
			return deeper.then_some((t, hit));
		}
		if closing <= 0.0 {
			return None;
		}
		let next = t + (hit.dist - skin * 0.5) / closing;
		if next >= limit {
			return None;
		}
		match at(next) {
			Some(h) => (t, hit) = (next, h),
			// GJK's normal on a curved side can understate the closing speed enough
			// to step through the skin: bisect back to a clear pose instead.
			None => {
				let mut over = next;
				for _ in 0..24 {
					let mid = (t + over) * 0.5;
					match at(mid) {
						Some(h) if h.dist >= skin * 0.25 => (t, hit) = (mid, h),
						Some(_) | None => over = mid,
					}
				}
				return Some((t, hit));
			}
		}
	}
	Some((t, hit))
}

impl Mesh {
	/// Sweeps `cyl` from `center` along `motion`: the first fraction at which it comes
	/// within `skin` of a triangle, or `None` when it clears the whole way. It never
	/// overlaps the mesh where it stops; moving along or away from a surface it rests
	/// on is free, and one that starts overlapping the mesh stops at once.
	///
	/// The flat bottom meets floors and step tops with an upright normal and the
	/// sides meet walls and step fronts with a level one. Internal-edge contacts are
	/// skipped as in [`Mesh::contacts`] with `filter`, so seams never catch it.
	pub fn cast_cylinder(
		&self,
		center: Vec3,
		cyl: Cylinder,
		motion: Vec3,
		skin: f32,
	) -> Option<Cast> {
		let grow = vec3(cyl.radius + skin, cyl.half_height + skin, cyl.radius + skin);
		let end = center + motion;
		let mut first: Option<Cast> = None;
		for (i, t) in self.near(center.min(end) - grow, center.max(end) + grow) {
			let limit = first.map_or(1.0, |c| c.fraction);
			let Some((at, mut hit)) = sweep_cylinder_triangle(center, cyl, motion, t, skin, limit) else {
				continue;
			};
			let mut tri = i;
			if hit.feature != Feature::Face {
				// An internal-edge hit stands for the closer surface through it: block
				// only if the rest of the motion would close in on that surface.
				if let Some((j, closer)) = self.dominated_cylinder(i, &hit, center + motion * at, cyl) {
					let closing = -motion.dot(closer.normal);
					if !(closing > 0.0 && closer.dist - closing * (limit - at) < skin * 0.25) {
						continue;
					}
					(tri, hit) = (j, closer);
				}
			}
			first = Some(Cast {
				fraction: at,
				normal: hit.normal,
				tri,
			});
		}
		first
	}

	/// The deepest overlap of `cyl` at `center` with the mesh: the direction out and
	/// the depth, or `None` when it overlaps nothing.
	pub fn cylinder_penetration(&self, center: Vec3, cyl: Cylinder) -> Option<(Vec3, f32)> {
		let grow = vec3(cyl.radius, cyl.half_height, cyl.radius);
		self
			.near(center - grow, center + grow)
			.filter_map(|(_, t)| cylinder_penetration(center, cyl, t))
			.max_by(|a, b| a.1.total_cmp(&b.1))
	}

	/// Whether `cyl` at `center` overlaps any triangle.
	pub fn cylinder_overlaps(&self, center: Vec3, cyl: Cylinder) -> bool {
		let grow = vec3(cyl.radius, cyl.half_height, cyl.radius);
		self
			.near(center - grow, center + grow)
			.any(|(_, t)| cylinder_triangle(center, cyl, t, 0.0).is_none())
	}

	/// The cylinder's version of `dominated`: a triangle through the hit point that is
	/// strictly closer, and its approach, making the edge or vertex hit an
	/// internal-edge artifact.
	fn dominated_cylinder(
		&self,
		tri: usize,
		hit: &Hit,
		center: Vec3,
		cyl: Cylinder,
	) -> Option<(usize, Hit)> {
		let p = hit.tri_point;
		let e = vec3(ON_SURFACE, ON_SURFACE, ON_SURFACE);
		self.near(p - e, p + e).find_map(|(i, t)| {
			if i == tri || (closest_point_triangle(p, t).0 - p).length() > ON_SURFACE {
				return None;
			}
			let closer = cylinder_triangle(center, cyl, t, hit.dist)?;
			(closer.dist < hit.dist - EPS).then_some((i, closer))
		})
	}
}

/// Test scenes.
pub mod scenes {
	use super::*;
	use linalg::Quat;

	/// `n`×`n` unit cells on y = 0, diagonals alternating so seams run both ways.
	pub fn grid(n: usize) -> Mesh {
		let mut m = Mesh::default();
		for i in 0..n {
			for k in 0..n {
				let (x, z) = (i as f32, k as f32);
				let c = [
					vec3(x, 0.0, z),
					vec3(x + 1.0, 0.0, z),
					vec3(x + 1.0, 0.0, z + 1.0),
					vec3(x, 0.0, z + 1.0),
				];
				let start = (i + k) % 2;
				m.push_quad(
					c[start],
					c[start + 1],
					c[start + 2],
					c[(start + 3) % 4],
					vec3(0.0, 1.0, 0.0),
				);
			}
		}
		m
	}

	/// Floor over x in [0, 8], z in [0, 4]: one 4×4 quad, then 1×1 cells whose
	/// vertices at x = 4, z = 1, 2, 3 sit in the middle of the big quad's edge.
	pub fn t_junction() -> Mesh {
		let up = vec3(0.0, 1.0, 0.0);
		let mut m = Mesh::default();
		m.push_quad(
			vec3(0.0, 0.0, 0.0),
			vec3(4.0, 0.0, 0.0),
			vec3(4.0, 0.0, 4.0),
			vec3(0.0, 0.0, 4.0),
			up,
		);
		for i in 4..8 {
			for k in 0..4 {
				let (x, z) = (i as f32, k as f32);
				m.push_quad(
					vec3(x, 0.0, z),
					vec3(x + 1.0, 0.0, z),
					vec3(x + 1.0, 0.0, z + 1.0),
					vec3(x, 0.0, z + 1.0),
					up,
				);
			}
		}
		m
	}

	/// 8×8 grid with a 1-unit box standing on it (unwelded) at x in [5, 6].
	pub fn wall() -> Mesh {
		let mut m = grid(8);
		m.push_box(vec3(5.0, 0.0, 1.0), vec3(6.0, 1.0, 7.0));
		m
	}

	/// Platform box, top at y = 1, ending at x = 4.
	pub fn ledge() -> Mesh {
		let mut m = Mesh::default();
		m.push_box(vec3(0.0, 0.0, 0.0), vec3(4.0, 1.0, 4.0));
		m
	}

	/// Inner and outer radius of the [`helix`] track, centered on the y axis.
	pub const HELIX_RADII: (f32, f32) = (2.0, 4.0);
	/// Angle the [`helix`] track turns through.
	pub const HELIX_TURN: f32 = 3.0 * std::f32::consts::PI;
	/// Height of the [`helix`] track where it starts, at angle zero.
	pub const HELIX_TOP: f32 = 5.0;

	/// A track spiralling down 1.5 turns, a capsule's height clear of itself, around the y axis, walled on both sides,
	/// above a floor at y = 0 that it runs out onto. The faceted walls give convex
	/// and concave internal edges at every segment, against a sloped floor.
	pub fn helix() -> Mesh {
		const SEGMENTS: usize = 48;
		const BOTTOM: f32 = 0.6;
		const WALL: f32 = 1.2;
		let (inner, outer) = HELIX_RADII;
		let up = vec3(0.0, 1.0, 0.0);
		let mut m = Mesh::default();
		let at = |a: f32, r: f32, y: f32| vec3(a.cos() * r, y, a.sin() * r);
		for i in 0..SEGMENTS {
			let (a0, a1) = (
				HELIX_TURN * i as f32 / SEGMENTS as f32,
				HELIX_TURN * (i + 1) as f32 / SEGMENTS as f32,
			);
			let y = |a: f32| HELIX_TOP + (BOTTOM - HELIX_TOP) * a / HELIX_TURN;
			let (y0, y1) = (y(a0), y(a1));
			let out = vec3(((a0 + a1) * 0.5).cos(), 0.0, ((a0 + a1) * 0.5).sin());
			m.push_quad(
				at(a0, inner, y0),
				at(a0, outer, y0),
				at(a1, outer, y1),
				at(a1, inner, y1),
				up,
			);
			m.push_quad(
				at(a0, outer, y0),
				at(a1, outer, y1),
				at(a1, outer, y1 + WALL),
				at(a0, outer, y0 + WALL),
				-out,
			);
			m.push_quad(
				at(a0, inner, y0),
				at(a1, inner, y1),
				at(a1, inner, y1 + WALL),
				at(a0, inner, y0 + WALL),
				out,
			);
		}
		m.push_quad(
			vec3(-6.0, 0.0, -6.0),
			vec3(6.0, 0.0, -6.0),
			vec3(6.0, 0.0, 6.0),
			vec3(-6.0, 0.0, 6.0),
			up,
		);
		m
	}

	/// Heights of the [`plates`].
	pub const PLATES: [f32; 3] = [4.0, 3.0, 2.0];

	/// Three single-quad plates stacked over a floor, each facing down: a body
	/// dropped on them meets their back sides, and each is thinner than one step
	/// of a fast fall.
	pub fn plates() -> Mesh {
		let mut m = grid(4);
		for y in PLATES {
			m.push_quad(
				vec3(0.0, y, 0.0),
				vec3(4.0, y, 0.0),
				vec3(4.0, y, 4.0),
				vec3(0.0, y, 4.0),
				vec3(0.0, -1.0, 0.0),
			);
		}
		m
	}

	pub struct Scene {
		pub name: &'static str,
		pub world: World,
	}

	fn one(mesh: Mesh, body: Body) -> World {
		World {
			bodies: vec![body],
			..World::new(mesh)
		}
	}

	/// A floor with a rimmed platform swinging along x above it. On the platform:
	/// rolling spheres, a capsule standing upright and one that tips over. On the
	/// floor, a sphere rolls into a heavier one.
	pub fn moving_platform() -> World {
		let mut deck = Mesh::default();
		deck.push_box(vec3(-1.5, -0.15, -1.5), vec3(1.5, 0.15, 1.5));
		for (lo, hi) in [
			(vec3(-1.5, 0.15, -1.5), vec3(1.5, 0.45, -1.4)),
			(vec3(-1.5, 0.15, 1.4), vec3(1.5, 0.45, 1.5)),
			(vec3(-1.5, 0.15, -1.4), vec3(-1.4, 0.45, 1.4)),
			(vec3(1.4, 0.15, -1.4), vec3(1.5, 0.45, 1.4)),
		] {
			deck.push_box(lo, hi);
		}
		let mut floor = Mesh::default();
		floor.push_box(vec3(-8.0, -1.0, -8.0), vec3(8.0, 0.0, 8.0));
		let ball = Body {
			radius: 0.35,
			friction: 0.6,
			rotates: true,
			..Body::default()
		};
		let upright = Body {
			half_height: 0.35,
			radius: 0.3,
			friction: 0.8,
			..Body::default()
		};
		let deck_top = 1.15;
		World {
			platforms: vec![Platform {
				mesh: deck,
				origin: vec3(0.0, 1.0, 0.0),
				swing: vec3(2.5, 0.0, 0.0),
				period: 5.0,
			}],
			bodies: vec![
				Body {
					pos: vec3(-0.6, deck_top + 0.35, -0.6),
					..ball
				},
				Body {
					pos: vec3(0.5, deck_top + 1.2, 0.4),
					..ball
				},
				Body {
					pos: vec3(-0.6, deck_top + 0.65, 0.7),
					..upright
				},
				Body {
					pos: vec3(0.7, deck_top + 0.75, -0.6),
					orient: Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), 0.25),
					rotates: true,
					..upright
				},
				Body {
					pos: vec3(-5.0, 0.35, 4.0),
					vel: vec3(5.0, 0.0, 0.0),
					..ball
				},
				Body {
					pos: vec3(0.0, 0.5, 4.05),
					radius: 0.5,
					mass: 3.0,
					..ball
				},
			],
			..World::new(floor)
		}
	}

	/// Each scene with the body launched into it.
	pub fn all() -> [Scene; 7] {
		let sphere = Body {
			pos: vec3(0.7, 0.5, 0.6),
			vel: vec3(4.0, 0.0, 1.3),
			..Body::default()
		};
		[
			Scene {
				name: "sphere on grid",
				world: one(grid(8), sphere),
			},
			Scene {
				name: "capsule over T-junctions",
				world: one(
					t_junction(),
					Body {
						pos: vec3(1.0, 0.9, 2.37),
						vel: vec3(1.5, 0.0, 0.0),
						half_height: 0.5,
						radius: 0.4,
						..sphere
					},
				),
			},
			Scene {
				name: "sphere into wall",
				world: one(
					wall(),
					Body {
						pos: vec3(1.0, 0.5, 3.3),
						vel: vec3(6.0, 0.0, 0.0),
						..sphere
					},
				),
			},
			Scene {
				name: "sphere off ledge",
				world: one(
					ledge(),
					Body {
						pos: vec3(4.1, 1.45, 2.0),
						vel: Vec3::ZERO,
						..sphere
					},
				),
			},
			Scene {
				name: "capsule down helix track",
				world: one(
					helix(),
					Body {
						pos: vec3(3.0, HELIX_TOP + 0.4 + 0.4 + 0.01, 0.0),
						vel: vec3(0.0, 0.0, 3.0),
						half_height: 0.4,
						radius: 0.4,
						friction: 0.05,
						..sphere
					},
				),
			},
			Scene {
				name: "fast drop onto thin plates",
				world: one(
					plates(),
					Body {
						pos: vec3(2.0, 8.0, 2.0),
						vel: vec3(0.0, -60.0, 0.0),
						..sphere
					},
				),
			},
			Scene {
				name: "bodies on a moving platform",
				world: moving_platform(),
			},
		]
	}

	/// Vertices lying strictly inside another triangle's edge.
	pub fn t_junctions(mesh: &Mesh) -> Vec<Vec3> {
		let mut out: Vec<Vec3> = Vec::new();
		for &p in mesh.tris.iter().flatten() {
			let inside = mesh.tris.iter().any(|t| {
				(0..3).any(|i| {
					let (u, q) = closest_point_segment(p, t[i], t[(i + 1) % 3]);
					u > 1e-4 && u < 1.0 - 1e-4 && (q - p).length() < ON_SURFACE
				})
			});
			if inside && !out.iter().any(|&o| (o - p).length() < ON_SURFACE) {
				out.push(p);
			}
		}
		out
	}
}

#[cfg(test)]
mod tests {
	use super::scenes::*;
	use super::*;
	use linalg::Quat;

	const DT: f32 = 1.0 / 60.0;
	const G: Vec3 = vec3(0.0, -9.81, 0.0);

	struct Run {
		min_vx: f32,
		max_vy: f32,
		max_pen: f32,
		end: Body,
	}

	fn slide(mesh: &Mesh, mut body: Body, steps: usize, filter: bool) -> Run {
		let mut run = Run {
			min_vx: f32::MAX,
			max_vy: 0.0,
			max_pen: 0.0,
			end: body,
		};
		for _ in 0..steps {
			body.step(mesh, DT, G, filter);
			run.min_vx = run.min_vx.min(body.vel.x);
			run.max_vy = run.max_vy.max(body.vel.y);
			run.max_pen = run.max_pen.max(body.penetration(mesh));
		}
		run.end = body;
		run
	}

	fn sphere(pos: Vec3, vel: Vec3) -> Body {
		Body {
			pos,
			vel,
			half_height: 0.0,
			radius: 0.5,
			..Body::default()
		}
	}

	fn capsule(pos: Vec3, vel: Vec3) -> Body {
		Body {
			pos,
			vel,
			half_height: 0.5,
			radius: 0.4,
			..Body::default()
		}
	}

	fn lcg(seed: &mut u32) -> f32 {
		*seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
		(*seed >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
	}

	#[test]
	fn segment_closest_points_match_brute_force() {
		let mut seed = 7;
		for _ in 0..2000 {
			let mut v = || vec3(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed));
			let (p1, q1, p2, q2) = (v(), v(), v(), v());
			let (_, [a, b]) = closest_point_between_segments(p1, q1, p2, q2);
			let got = (a - b).length();
			let mut best = f32::MAX;
			for i in 0..=200 {
				let s = p1 + (q1 - p1) * (i as f32 / 200.0);
				best = best.min((s - closest_point_segment(s, p2, q2).1).length());
			}
			assert!(got <= best + 1e-4, "{got} > {best}");
		}
	}

	#[test]
	fn sphere_crosses_grid_seams_without_bumps() {
		let body = sphere(vec3(0.7, 0.5, 0.6), vec3(4.0, 0.0, 1.3));
		let run = slide(&grid(8), body, 80, true);
		assert!(run.min_vx > 3.999, "lost speed: {}", run.min_vx);
		assert!(run.max_vy < 1e-3, "launched: {}", run.max_vy);
		assert!(run.max_pen < 1e-3);
		assert!(run.end.grounded);
	}

	#[test]
	fn unfiltered_grid_shows_ghost_contacts() {
		let body = sphere(vec3(0.7, 0.5, 0.6), vec3(4.0, 0.0, 1.3));
		let run = slide(&grid(8), body, 80, false);
		assert!(
			run.min_vx < 3.99,
			"expected ghost collisions, min vx {}",
			run.min_vx
		);
	}

	#[test]
	fn crosses_t_junctions() {
		for speed in [1.5, 4.0] {
			let steps = (5.0 / speed / DT) as usize;
			for z in [1.0, 2.0, 2.37] {
				let v = vec3(speed, 0.0, 0.0);
				for body in [sphere(vec3(1.0, 0.5, z), v), capsule(vec3(1.0, 0.9, z), v)] {
					let run = slide(&t_junction(), body, steps, true);
					assert!(
						run.min_vx > speed - 1e-3,
						"z {z}: lost speed {}",
						run.min_vx
					);
					assert!(run.max_vy < 1e-3, "z {z}: launched {}", run.max_vy);
					assert!(run.max_pen < 1e-3);
					assert!(run.end.pos.x > 5.5);
				}
			}
		}
	}

	#[test]
	fn unfiltered_t_junction_shows_ghost_contacts() {
		let body = capsule(vec3(1.0, 0.9, 2.37), vec3(1.5, 0.0, 0.0));
		let run = slide(&t_junction(), body, 200, false);
		assert!(
			run.min_vx < 1.49,
			"expected ghost collisions, min vx {}",
			run.min_vx
		);
	}

	#[test]
	fn capsule_crosses_grid_seams() {
		let body = capsule(vec3(0.7, 0.9, 0.6), vec3(4.0, 0.0, 1.3));
		let run = slide(&grid(8), body, 80, true);
		assert!(run.min_vx > 3.999, "lost speed: {}", run.min_vx);
		assert!(run.max_vy < 1e-3);
	}

	#[test]
	fn wall_still_blocks() {
		for body in [
			sphere(vec3(1.0, 0.5, 3.3), vec3(6.0, 0.0, 0.0)),
			capsule(vec3(1.0, 0.9, 3.3), vec3(6.0, 0.0, 0.0)),
		] {
			let run = slide(&wall(), body, 120, true);
			assert!(
				(run.end.pos.x - (5.0 - body.radius)).abs() < 1e-3,
				"x {}",
				run.end.pos.x
			);
			assert!(run.end.vel.x.abs() < 1e-3);
			assert!(run.max_vy < 1e-3, "climbed: {}", run.max_vy);
			assert!(run.max_pen < 1e-3);
		}
	}

	#[test]
	fn convex_ledge_edge_still_supports() {
		// Center just past the edge: only the convex edge holds it, pushing it outward.
		let body = sphere(vec3(4.1, 1.45, 2.0), Vec3::ZERO);
		let mut b = body;
		b.step(&ledge(), DT, G, true);
		assert!(b.penetration(&ledge()) < 1e-3);
		assert!(b.pos.y > 1.4, "fell through edge: {}", b.pos.y);
		assert!(b.vel.x > 0.0, "edge normal should push outward");
		let run = slide(&ledge(), body, 120, true);
		assert!(run.max_pen < 1e-3);
		assert!(run.end.pos.x > 4.5, "should roll off: {:?}", run.end.pos);
	}

	#[test]
	fn back_faces_block() {
		// Grid faces point up; come at them from below.
		let mut body = sphere(vec3(3.3, -2.0, 3.6), vec3(0.0, 5.0, 0.0));
		for _ in 0..60 {
			body.step(&grid(8), DT, Vec3::ZERO, true);
			assert!(body.penetration(&grid(8)) < 1e-3);
		}
		assert!((body.pos.y + 0.5).abs() < 1e-3, "y {}", body.pos.y);
	}

	#[test]
	fn fast_bodies_stop_at_thin_plates() {
		for (body, filter) in [
			(sphere(vec3(2.0, 8.0, 2.0), vec3(0.0, -60.0, 0.0)), true),
			(sphere(vec3(2.0, 8.0, 2.0), vec3(0.0, -60.0, 0.0)), false),
			(capsule(vec3(2.3, 9.0, 1.7), vec3(0.0, -200.0, 0.0)), true),
		] {
			let run = slide(&plates(), body, 60, filter);
			let rest = PLATES[0] + body.radius + body.half_height;
			assert!((run.end.pos.y - rest).abs() < 1e-3, "y {}", run.end.pos.y);
			assert!(run.max_pen < 1e-3);
		}
		// Sideways through a one-quad wall, faster than its radius per step.
		let mut wall = Mesh::default();
		wall.push_quad(
			vec3(5.0, 0.0, 0.0),
			vec3(5.0, 3.0, 0.0),
			vec3(5.0, 3.0, 4.0),
			vec3(5.0, 0.0, 4.0),
			vec3(1.0, 0.0, 0.0),
		);
		let run = slide(
			&wall,
			sphere(vec3(1.0, 1.5, 2.0), vec3(300.0, 0.0, 0.0)),
			30,
			true,
		);
		assert!((run.end.pos.x - 4.5).abs() < 1e-3, "x {}", run.end.pos.x);
	}

	#[test]
	fn bvh_matches_brute_force() {
		let mesh = helix();
		let mut seed = 11;
		for _ in 0..500 {
			let mut v = || vec3(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)) * 3.0;
			let (a, b) = (v(), v());
			let (min, max) = (a.min(b), a.max(b));
			let mut got: Vec<usize> = mesh.near(min, max).map(|(i, _)| i).collect();
			got.sort();
			let want: Vec<usize> = (0..mesh.tris.len())
				.filter(|&i| overlaps(bounds(&mesh.tris[i]), min, max))
				.collect();
			assert_eq!(got, want);
		}
	}

	#[test]
	fn friction_decelerates_at_mu_g() {
		let mut body = sphere(vec3(1.0, 0.5, 2.0), vec3(3.0, 0.0, 0.0));
		body.friction = 0.5;
		let run = slide(&grid(8), body, 30, true);
		let expected = 3.0 - 0.5 * 9.81 * 30.0 * DT;
		assert!(
			(run.end.vel.x - expected).abs() < 1e-3,
			"vx {}",
			run.end.vel.x
		);
		let run = slide(&grid(8), body, 120, true);
		assert!(run.end.vel.x.abs() < 1e-6, "should stop: {}", run.end.vel.x);
	}

	#[test]
	fn capsule_runs_the_helix_track() {
		let mut scene = all()
			.into_iter()
			.find(|s| s.name.contains("helix"))
			.unwrap();
		let mesh = std::mem::take(&mut scene.world.mesh);
		let mut body = scene.world.bodies[0];
		let (inner, outer) = HELIX_RADII;
		let mut max_pen = 0.0f32;
		let mut turned = 0.0f32;
		let mut last = body.pos.z.atan2(body.pos.x);
		let rest = body.radius + body.half_height;
		for _ in 0..1200 {
			body.step(&mesh, DT, G, true);
			if body.grounded && (body.pos.y - rest).abs() < 1e-3 {
				break;
			}
			max_pen = max_pen.max(body.penetration(&mesh));
			let a = body.pos.z.atan2(body.pos.x);
			let mut da = a - last;
			if da > std::f32::consts::PI {
				da -= std::f32::consts::TAU;
			} else if da < -std::f32::consts::PI {
				da += std::f32::consts::TAU;
			}
			(turned, last) = (turned + da, a);
			if turned < HELIX_TURN - 0.2 {
				let r = body.pos.x.hypot(body.pos.z);
				assert!(
					r > inner + body.radius - 1e-3 && r < outer - body.radius + 1e-3,
					"left the track at r {r}"
				);
			}
		}
		assert!(max_pen < 1e-3, "pen {max_pen}");
		assert!(turned > HELIX_TURN, "stalled after {turned} rad");
		assert!(
			(body.pos.y - rest).abs() < 1e-3,
			"not on the floor: {:?}",
			body.pos
		);
	}

	const CYL: Cylinder = Cylinder {
		radius: 0.4,
		half_height: 0.9,
	};
	const SKIN: f32 = 0.01;

	#[test]
	fn cylinder_distance_matches_closed_form() {
		let floor: Triangle = [
			vec3(-5.0, 0.0, -5.0),
			vec3(5.0, 0.0, -5.0),
			vec3(0.0, 0.0, 5.0),
		];
		let h = cylinder_triangle(vec3(0.3, 1.25, 0.1), CYL, &floor, 1000.0).unwrap();
		assert!(
			(h.dist - 0.35).abs() < 1e-4 && h.feature == Feature::Face,
			"{h:?}"
		);
		assert!((h.normal - vec3(0.0, 1.0, 0.0)).length() < 1e-3);
		// Beside a wall: the round side faces it.
		let wall: Triangle = [
			vec3(1.0, -5.0, -5.0),
			vec3(1.0, 5.0, -5.0),
			vec3(1.0, 0.0, 5.0),
		];
		let h = cylinder_triangle(Vec3::ZERO, CYL, &wall, 1000.0).unwrap();
		assert!((h.dist - 0.6).abs() < 1e-4, "{h:?}");
		assert!((h.normal - vec3(-1.0, 0.0, 0.0)).length() < 1e-3);
		// Diagonal from a vertical edge: radial.
		let edge: Triangle = [
			vec3(1.0, -5.0, 1.0),
			vec3(1.0, 5.0, 1.0),
			vec3(3.0, 0.0, 3.0),
		];
		let h = cylinder_triangle(Vec3::ZERO, CYL, &edge, 1000.0).unwrap();
		assert!((h.dist - (2f32.sqrt() - 0.4)).abs() < 1e-3, "{h:?}");
		assert!(cylinder_triangle(vec3(0.0, 0.5, 0.0), CYL, &floor, 1000.0).is_none());
	}

	#[test]
	fn cylinder_distance_to_a_tilted_face_matches_its_plane() {
		// A 60° face so large that the closest point is always inside it.
		let tri: Triangle = [
			vec3(0.0, 0.0, -4000.0),
			vec3(0.0, 0.0, 4000.0),
			vec3(1480.0, 2560.0, 0.0),
		];
		let n = triangle_normal(&tri);
		let n = if n.x > 0.0 { -n } else { n };
		let cyl = Cylinder {
			radius: 16.0,
			half_height: 36.0,
		};
		let mut seed = 3;
		for _ in 0..2000 {
			let x = 20.0 + (lcg(&mut seed) + 2.0) * 10.0;
			let c = vec3(
				x,
				x * 256.0 / 148.0 + 36.0 + 20.0 * lcg(&mut seed),
				20.0 * lcg(&mut seed),
			);
			// Lowest point toward the plane: rim along the level part of -n, bottom.
			let plane = n.dot(c - tri[0]) - cyl.radius * n.x.hypot(n.z) - cyl.half_height * n.y.abs();
			match cylinder_triangle(c, cyl, &tri, 1000.0) {
				Some(h) => assert!(
					(h.dist - plane).abs() < 1e-3,
					"gjk {} plane {plane} at {c:?}",
					h.dist
				),
				None => assert!(plane < 1e-3, "false overlap, plane {plane} at {c:?}"),
			}
		}
	}

	#[test]
	fn cylinder_penetration_points_out_and_measures_depth() {
		let mesh = scenes::wall();
		// Sunk 0.3 into the floor.
		let (n, depth) = mesh
			.cylinder_penetration(vec3(2.0, CYL.half_height - 0.3, 2.0), CYL)
			.unwrap();
		assert!(
			(n - vec3(0.0, 1.0, 0.0)).length() < 1e-3 && (depth - 0.3).abs() < 1e-3,
			"{n:?} {depth}"
		);
		// 0.2 into the wall at x = 5, clear of the floor.
		let c = vec3(5.0 - CYL.radius + 0.2, CYL.half_height + 0.5, 3.0);
		let (n, depth) = mesh.cylinder_penetration(c, CYL).unwrap();
		assert!(
			(n - vec3(-1.0, 0.0, 0.0)).length() < 1e-3 && (depth - 0.2).abs() < 1e-3,
			"{n:?} {depth}"
		);
		// From inside, moving out is free and moving deeper is blocked.
		assert!(mesh
			.cast_cylinder(c, CYL, vec3(-1.0, 0.0, 0.5), SKIN)
			.is_none());
		assert!(mesh
			.cast_cylinder(c, CYL, vec3(1.0, 0.0, 0.0), SKIN)
			.is_some_and(|h| h.fraction == 0.0));
		assert!(mesh
			.cylinder_penetration(vec3(3.0, 2.0, 3.0), CYL)
			.is_none());
	}

	#[test]
	fn cylinder_slides_over_grid_seams() {
		let mesh = scenes::grid(8);
		let c = vec3(0.5, CYL.half_height + SKIN * 0.75, 0.37);
		for dir in [
			vec3(7.0, 0.0, 0.0),
			vec3(0.0, 0.0, 7.0),
			vec3(6.5, 0.0, 6.5),
		] {
			assert!(
				mesh.cast_cylinder(c, CYL, dir, SKIN).is_none(),
				"seam blocked {dir:?}"
			);
		}
		let down = mesh
			.cast_cylinder(c, CYL, vec3(0.0, -1.0, 0.0), SKIN)
			.unwrap();
		assert!(down.fraction < SKIN && (down.normal - vec3(0.0, 1.0, 0.0)).length() < 1e-3);
	}

	#[test]
	fn cylinder_meets_a_ledge_side_level_and_its_top_upright() {
		// Platform box: top at y = 1, side at x = 4.
		let mesh = scenes::ledge();
		let side = mesh
			.cast_cylinder(vec3(5.0, 1.5, 2.0), CYL, vec3(-2.0, 0.0, 0.0), SKIN)
			.unwrap();
		assert!(
			(side.normal - vec3(1.0, 0.0, 0.0)).length() < 1e-3,
			"{side:?}"
		);
		// The flat bottom over the corner lands on the top, not on a rounded edge.
		let top = mesh
			.cast_cylinder(vec3(4.3, 3.0, 2.0), CYL, vec3(0.0, -2.0, 0.0), SKIN)
			.unwrap();
		assert!(
			(top.normal - vec3(0.0, 1.0, 0.0)).length() < 1e-3,
			"{top:?}"
		);
		let gap = 3.0 - 2.0 * top.fraction - CYL.half_height - 1.0;
		assert!((SKIN * 0.5 - 1e-4..SKIN).contains(&gap), "gap {gap}");
	}

	#[test]
	fn cylinder_never_ends_inside_the_mesh() {
		let mesh = scenes::helix();
		let cyl = Cylinder {
			radius: 0.3,
			half_height: 0.5,
		};
		let mut seed = 7;
		let mut moved = 0;
		for _ in 0..2000 {
			let c = vec3(
				lcg(&mut seed) * 3.0,
				lcg(&mut seed) * 2.0 + 3.0,
				lcg(&mut seed) * 3.0,
			);
			if mesh.cylinder_overlaps(c, cyl) {
				continue;
			}
			let d = vec3(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)) * 3.0;
			let t = mesh
				.cast_cylinder(c, cyl, d, SKIN)
				.map_or(1.0, |h| h.fraction);
			moved += (t > 0.0) as usize;
			let end = c + d * t;
			assert!(!mesh.cylinder_overlaps(end, cyl), "ended inside at {end:?}");
		}
		assert!(moved > 500, "only {moved} casts moved");
	}

	#[test]
	fn interpolation_spans_the_step() {
		let a = sphere(vec3(0.0, 1.0, 0.0), Vec3::ZERO);
		let b = sphere(vec3(2.0, 3.0, 0.0), Vec3::ZERO);
		assert_eq!(a.interpolate(&b, 0.0).pos, a.pos);
		assert_eq!(a.interpolate(&b, 1.0).pos, b.pos);
		assert_eq!(a.interpolate(&b, 0.25).pos, vec3(0.5, 1.5, 0.0));
	}

	/// Deepest overlap of any body with the static mesh or a platform.
	fn world_penetration(w: &World) -> f32 {
		let mut deepest = 0.0f32;
		for b in &w.bodies {
			deepest = deepest.max(b.penetration(&w.mesh));
			for p in &w.platforms {
				let local = Body {
					pos: b.pos - p.offset(w.time),
					..*b
				};
				deepest = deepest.max(local.penetration(&p.mesh));
			}
		}
		deepest
	}

	fn pair_gap(w: &World, i: usize, j: usize) -> f32 {
		let (a, b) = (&w.bodies[i], &w.bodies[j]);
		let (a0, a1) = a.segment();
		let (b0, b1) = b.segment();
		let (_, [pa, pb]) = closest_point_between_segments(a0, a1, b0, b1);
		(pa - pb).length() - a.radius - b.radius
	}

	#[test]
	fn bodies_collide_and_keep_momentum() {
		for heavy in [1.0, 3.0] {
			let mut w = World::new(grid(8));
			w.bodies = vec![
				sphere(vec3(1.0, 0.5, 4.0), vec3(4.0, 0.0, 0.0)),
				Body {
					mass: heavy,
					..sphere(vec3(3.0, 0.5, 4.0), Vec3::ZERO)
				},
			];
			for _ in 0..90 {
				w.step(DT, true);
				assert!(pair_gap(&w, 0, 1) > -1e-3, "overlap {}", pair_gap(&w, 0, 1));
				assert!(world_penetration(&w) < 1e-3);
			}
			let [a, b] = [w.bodies[0], w.bodies[1]];
			let momentum = a.vel.x + b.vel.x * heavy;
			assert!((momentum - 4.0).abs() < 1e-3, "momentum {momentum}");
			// Contacts remove the approach and nothing more: both end at the shared speed.
			let shared = 4.0 / (1.0 + heavy);
			assert!(
				(b.vel.x - shared).abs() < 1e-3,
				"heavy {heavy}: vx {}",
				b.vel.x
			);
		}
	}

	#[test]
	fn sphere_rests_on_sphere() {
		let mut w = World::new(grid(4));
		w.bodies = vec![
			sphere(vec3(2.0, 0.5, 2.0), Vec3::ZERO),
			sphere(vec3(2.0, 1.5, 2.0), Vec3::ZERO),
		];
		for _ in 0..300 {
			w.step(DT, true);
			assert!(pair_gap(&w, 0, 1) > -1e-3, "overlap {}", pair_gap(&w, 0, 1));
			assert!(world_penetration(&w) < 1e-3);
		}
		assert!(
			(w.bodies[1].pos.y - 1.5).abs() < 2e-3,
			"top at {}",
			w.bodies[1].pos.y
		);
		assert!(w.bodies[1].grounded, "standing on a body counts as ground");
	}

	#[test]
	fn rotating_sphere_rolls_at_five_sevenths() {
		let mut body = Body {
			friction: 0.3,
			rotates: true,
			..sphere(vec3(1.0, 0.5, 4.3), vec3(5.0, 0.0, 0.0))
		};
		for _ in 0..60 {
			body.step(&grid(8), DT, G, true);
			assert!(body.penetration(&grid(8)) < 1e-3);
		}
		let rolling = 5.0 * 5.0 / 7.0;
		// Friction is capped per step, so the slipping phase ends within a step.
		assert!((body.vel.x - rolling).abs() < 2e-2, "vx {}", body.vel.x);
		// Rolling without slipping: the contact point is at rest.
		assert!(
			(body.spin.z + body.vel.x / body.radius).abs() < 1e-2,
			"spin {:?}",
			body.spin
		);
	}

	#[test]
	fn rotating_capsule_tips_over() {
		let mut body = Body {
			friction: 0.8,
			rotates: true,
			orient: Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), 0.25),
			..capsule(vec3(4.0, 0.95, 4.0), Vec3::ZERO)
		};
		for _ in 0..240 {
			body.step(&grid(8), DT, G, true);
			assert!(body.penetration(&grid(8)) < 1e-3);
		}
		let axis = body.orient.rotate(vec3(0.0, 1.0, 0.0));
		assert!(axis.y.abs() < 0.02, "still standing: axis {axis:?}");
		assert!((body.pos.y - body.radius).abs() < 1e-2, "y {}", body.pos.y);
		assert!(
			body.spin.length() < 0.05,
			"still moving: spin {:?}",
			body.spin
		);
	}

	#[test]
	fn platform_carries_bodies_with_friction() {
		let mut deck = Mesh::default();
		deck.push_box(vec3(-1.5, -0.15, -1.5), vec3(1.5, 0.15, 1.5));
		let mut w = World::new(Mesh::default());
		w.platforms.push(Platform {
			mesh: deck,
			origin: vec3(0.0, 1.0, 0.0),
			swing: vec3(1.0, 0.0, 0.0),
			period: 5.0,
		});
		let rider = sphere(vec3(0.0, 1.65, -0.7), Vec3::ZERO);
		w.bodies = vec![
			Body {
				friction: 1.0,
				..rider
			},
			Body {
				pos: vec3(0.0, 1.65, 0.7),
				..rider
			},
		];
		// The swing starts at full speed, so the rider slips until friction (at most
		// mu g) has matched it; after that it keeps its place on the deck.
		let mut held = None;
		for step in 0..300 {
			w.step(DT, true);
			let gap = w.bodies[0].pos.x - w.platforms[0].offset(w.time).x;
			if step == 30 {
				held = Some(gap);
			}
			if let Some(held) = held {
				// Friction matches the deck's velocity at the end of each step, so the
				// rider moves at the previous step's deck velocity: it drifts by at most
				// the deck's speed range (2.5 m/s here) times one step.
				assert!((gap - held).abs() < 0.05, "slipped: gap {gap}, held {held}");
			}
			assert!(
				w.bodies[1].pos.x.abs() < 0.02,
				"frictionless body was dragged to {}",
				w.bodies[1].pos.x
			);
			assert!(world_penetration(&w) < 1e-3);
		}
		for b in &w.bodies {
			assert!((b.pos.y - 1.65).abs() < 1e-3, "left the deck: {:?}", b.pos);
		}
	}

	#[test]
	fn sweeps_follow_a_faceted_wall_at_speed() {
		// The inside of a 64-sided ring: crossing a facet takes one sweep.
		let mut ring = Mesh::default();
		let n = 64;
		for i in 0..n {
			let (a0, a1) = (
				std::f32::consts::TAU * i as f32 / n as f32,
				std::f32::consts::TAU * (i + 1) as f32 / n as f32,
			);
			let at = |a: f32, y: f32| vec3(a.cos() * 5.0, y, a.sin() * 5.0);
			let mid = (a0 + a1) * 0.5;
			ring.push_quad(
				at(a0, 0.0),
				at(a1, 0.0),
				at(a1, 2.0),
				at(a0, 2.0),
				vec3(-mid.cos(), 0.0, -mid.sin()),
			);
		}
		let speed = 200.0;
		let mut body = sphere(vec3(4.5, 1.0, 0.0), vec3(0.0, 0.0, speed));
		body.step(&ring, DT, Vec3::ZERO, true);
		let turned = body.pos.z.atan2(body.pos.x);
		let expected = speed * DT / 4.5;
		assert!(turned > 0.85 * expected, "turned {turned} of {expected}");
		assert!(body.penetration(&ring) < 1e-3);
	}

	#[test]
	fn deep_overlaps_exit_on_the_side_they_came_from() {
		let mut plate = Mesh::default();
		plate.push_quad(
			vec3(0.0, 1.0, 0.0),
			vec3(4.0, 1.0, 0.0),
			vec3(4.0, 1.0, 4.0),
			vec3(0.0, 1.0, 4.0),
			vec3(0.0, 1.0, 0.0),
		);
		for (y, vy, side) in [(0.95, -1.0, 1.0), (1.05, 1.0, -1.0)] {
			let mut body = sphere(vec3(2.0, y, 2.0), vec3(0.0, vy, 0.0));
			body.step(&plate, DT, Vec3::ZERO, true);
			let want = 1.0 + side * body.radius;
			assert!(
				(body.pos.y - want).abs() < 1e-3,
				"from {y} moving {vy}: y {}",
				body.pos.y
			);
		}
	}
}
