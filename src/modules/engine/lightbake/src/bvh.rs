//! Ray queries over a triangle soup: the same median-split, depth-first node
//! layout `physics` uses for its (private) hierarchy, but answering ray casts
//! for shadow and gather rays.

use crate::Triangle;
use linalg::{vec3, Vec3};

const LEAF_SIZE: usize = 4;

#[derive(Clone, Debug, Default)]
pub struct Bvh {
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

fn bounds(tri: &Triangle) -> (Vec3, Vec3) {
	(
		tri[0].min(tri[1]).min(tri[2]),
		tri[0].max(tri[1]).max(tri[2]),
	)
}

impl Bvh {
	pub fn build(tris: &[Triangle]) -> Bvh {
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

	/// The first triangle the ray from `o` along unit `d` hits before `max_t`,
	/// two-sided: its index and the distance along `d`.
	pub fn cast(&self, tris: &[Triangle], o: Vec3, d: Vec3, max_t: f32) -> Option<(usize, f32)> {
		if self.nodes.is_empty() {
			return None;
		}
		// Zero components become a huge inverse so the slab test still
		// separates: a parallel ray outside a slab sees two huge same-sign ends.
		let inv = d.map(|k| if k.abs() < 1e-12 { 1e30 } else { 1.0 / k });
		let mut stack = vec![0u32];
		let mut best = max_t;
		let mut hit = None;
		while let Some(at) = stack.pop() {
			let node = self.nodes[at as usize];
			if !enters(o, inv, node.lo, node.hi, 0.0, best) {
				continue;
			}
			if node.count > 0 {
				for k in node.first..node.first + node.count {
					let i = self.order[k as usize] as usize;
					if let Some(t) = hit_tri(o, d, &tris[i], 1e-4) {
						if t < best {
							(best, hit) = (t, Some(i));
						}
					}
				}
			} else {
				stack.extend([node.first, at + 1]);
			}
		}
		hit.map(|i| (i, best))
	}
}

/// Whether the ray from `o` (with precomputed `inv = 1/d`) crosses the box
/// between `tmin` and `tmax`.
fn enters(o: Vec3, inv: Vec3, lo: Vec3, hi: Vec3, tmin: f32, tmax: f32) -> bool {
	let axis = |o: f32, inv: f32, lo: f32, hi: f32, (t0, t1): (f32, f32)| -> (f32, f32) {
		let (a, b) = ((lo - o) * inv, (hi - o) * inv);
		let (near, far) = if a <= b { (a, b) } else { (b, a) };
		(t0.max(near), t1.min(far))
	};
	let (t0, t1) = (tmin, tmax);
	let (t0, t1) = axis(o.x, inv.x, lo.x, hi.x, (t0, t1));
	let (t0, t1) = axis(o.y, inv.y, lo.y, hi.y, (t0, t1));
	let (t0, t1) = axis(o.z, inv.z, lo.z, hi.z, (t0, t1));
	t0 <= t1
}

/// Two-sided Möller–Trumbore: the distance along unit `d` where the ray from
/// `o` pierces `tri`, if before `eps`. Degenerate triangles never pierce.
pub fn hit_tri(o: Vec3, d: Vec3, tri: &Triangle, eps: f32) -> Option<f32> {
	let e1 = tri[1] - tri[0];
	let e2 = tri[2] - tri[0];
	let p = d.cross(e2);
	let det = e1.dot(p);
	if det.abs() < 1e-9 {
		return None;
	}
	let inv = 1.0 / det;
	let t = o - tri[0];
	let u = t.dot(p) * inv;
	if !(0.0..=1.0).contains(&u) {
		return None;
	}
	let q = t.cross(e1);
	let v = d.dot(q) * inv;
	if v < 0.0 || u + v > 1.0 {
		return None;
	}
	let at = e2.dot(q) * inv;
	(at > eps).then_some(at)
}

/// The closest point on `tri` to `point` (Ericson, Real-Time Collision
/// Detection 5.1.5), used to evaluate chart texels that fall outside their
/// triangle at the nearest point that lies on it.
pub fn closest_point(point: Vec3, tri: &Triangle) -> Vec3 {
	let [a, b, c] = *tri;
	let (ab, ac) = (b - a, c - a);
	let ap = point - a;
	let (d1, d2) = (ab.dot(ap), ac.dot(ap));
	if d1 <= 0.0 && d2 <= 0.0 {
		return a;
	}
	let bp = point - b;
	let (d3, d4) = (ab.dot(bp), ac.dot(bp));
	if d3 >= 0.0 && d4 <= d3 {
		return b;
	}
	let vc = d1 * d4 - d3 * d2;
	if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
		return a + ab * (d1 / (d1 - d3));
	}
	let cp = point - c;
	let (d5, d6) = (ab.dot(cp), ac.dot(cp));
	if d6 >= 0.0 && d5 <= d6 {
		return c;
	}
	let vb = d5 * d2 - d1 * d6;
	if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
		return a + ac * (d2 / (d2 - d6));
	}
	let va = d3 * d6 - d5 * d4;
	if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
		return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
	}
	let denom = 1.0 / (va + vb + vc);
	a + ab * (vb * denom) + ac * (vc * denom)
}

/// Corners of the box spanning `tris`, for ray length bounds.
pub fn soup_bounds(tris: &[Triangle]) -> (Vec3, Vec3) {
	let mut lo = vec3(f32::MAX, f32::MAX, f32::MAX);
	let mut hi = vec3(f32::MIN, f32::MIN, f32::MIN);
	for t in tris {
		let (a, b) = bounds(t);
		(lo, hi) = (lo.min(a), hi.max(b));
	}
	if lo.x > hi.x {
		(Vec3::ZERO, Vec3::ZERO)
	} else {
		(lo, hi)
	}
}
