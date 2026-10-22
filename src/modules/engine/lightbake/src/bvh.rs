//! Ray queries over a triangle soup on the hierarchy `physics` builds,
//! answering ray casts for shadow and gather rays.

use crate::Triangle;
use linalg::{vec3, Vec3};
use physics::triangle_bounds;

/// [`physics::bvh::Bvh`], the hierarchy the collision queries walk, with the
/// bake's ray and GPU queries.
#[derive(Clone, Debug, Default)]
pub struct Bvh {
	tree: physics::bvh::Bvh,
}

impl Bvh {
	pub fn build(tris: &[Triangle]) -> Bvh {
		Bvh {
			tree: physics::bvh::Bvh::build(tris),
		}
	}

	/// The first triangle the ray from `o` along unit `d` hits before `max_t`,
	/// two-sided: its index and the distance along `d`.
	pub fn cast(&self, tris: &[Triangle], o: Vec3, d: Vec3, max_t: f32) -> Option<(usize, f32)> {
		if self.tree.nodes().is_empty() {
			return None;
		}
		// Zero components become a huge inverse so the slab test still
		// separates: a parallel ray outside a slab sees two huge same-sign ends.
		let inv = d.map(|k| if k.abs() < 1e-12 { 1e30 } else { 1.0 / k });
		let mut stack = vec![0u32];
		let mut best = max_t;
		let mut hit = None;
		while let Some(at) = stack.pop() {
			let node = self.tree.nodes()[at as usize];
			if !enters(o, inv, node.lo, node.hi, 0.0, best) {
				continue;
			}
			if node.count > 0 {
				for k in node.first..node.first + node.count {
					let i = self.tree.order()[k as usize] as usize;
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

	/// The nodes and triangle order as the GPU bake's storage buffer: each
	/// node `lo`, `first`, `hi`, `count` little endian, then the order array.
	#[cfg(feature = "gpu")]
	pub fn gpu(&self) -> (Vec<u8>, Vec<u32>) {
		let mut bytes = Vec::with_capacity(self.tree.nodes().len() * 32);
		for node in self.tree.nodes() {
			for v in [node.lo.x, node.lo.y, node.lo.z] {
				bytes.extend_from_slice(&v.to_bits().to_le_bytes());
			}
			bytes.extend_from_slice(&node.first.to_le_bytes());
			for v in [node.hi.x, node.hi.y, node.hi.z] {
				bytes.extend_from_slice(&v.to_bits().to_le_bytes());
			}
			bytes.extend_from_slice(&node.count.to_le_bytes());
		}
		(bytes, self.tree.order().to_vec())
	}

	/// Every triangle whose own bounds meet `lo`..`hi`, walking the same
	/// hierarchy `cast` does. The GPU bake's occluder sets gather their
	/// over-approximation from these.
	#[cfg(feature = "gpu")]
	pub fn in_box(&self, tris: &[Triangle], lo: Vec3, hi: Vec3) -> Vec<u32> {
		let mut out = Vec::new();
		if self.tree.nodes().is_empty() {
			return out;
		}
		let mut stack = vec![0u32];
		while let Some(at) = stack.pop() {
			let node = self.tree.nodes()[at as usize];
			if node.lo.x > hi.x
				|| node.hi.x < lo.x
				|| node.lo.y > hi.y
				|| node.hi.y < lo.y
				|| node.lo.z > hi.z
				|| node.hi.z < lo.z
			{
				continue;
			}
			if node.count > 0 {
				for k in node.first..node.first + node.count {
					let i = self.tree.order()[k as usize];
					let (a, b) = triangle_bounds(&tris[i as usize]);
					if a.x <= hi.x && b.x >= lo.x && a.y <= hi.y && b.y >= lo.y && a.z <= hi.z && b.z >= lo.z
					{
						out.push(i);
					}
				}
			} else {
				stack.extend([node.first, at + 1]);
			}
		}
		out
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

/// The closest point on `tri` to `point`, used to evaluate chart texels that
/// fall outside their triangle at the nearest point that lies on it.
pub fn closest_point(point: Vec3, tri: &Triangle) -> Vec3 {
	physics::closest_point_triangle(point, tri).0
}

/// Corners of the box spanning `tris`, for ray length bounds.
pub fn soup_bounds(tris: &[Triangle]) -> (Vec3, Vec3) {
	let mut lo = vec3(f32::MAX, f32::MAX, f32::MAX);
	let mut hi = vec3(f32::MIN, f32::MIN, f32::MIN);
	for t in tris {
		let (a, b) = triangle_bounds(t);
		(lo, hi) = (lo.min(a), hi.max(b));
	}
	if lo.x > hi.x {
		(Vec3::ZERO, Vec3::ZERO)
	} else {
		(lo, hi)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A soup of `n` small triangles scattered by a fixed LCG.
	fn soup(n: usize) -> Vec<Triangle> {
		let mut seed = 0x2545_F491u32;
		let mut next = move || {
			seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			(seed >> 8) as f32 / (1 << 24) as f32
		};
		(0..n)
			.map(|_| {
				let c = vec3(next(), next(), next()) * 100.0;
				let mut corner = || c + vec3(next(), next(), next()) * 4.0;
				[corner(), corner(), corner()]
			})
			.collect()
	}

	fn depth(bvh: &Bvh, at: usize) -> usize {
		let node = bvh.tree.nodes()[at];
		if node.count > 0 {
			return 0;
		}
		1 + depth(bvh, at + 1).max(depth(bvh, node.first as usize))
	}

	#[test]
	fn sah_casts_match_brute_force_and_stay_shallow() {
		let tris = soup(3000);
		let bvh = Bvh::build(&tris);
		assert!(depth(&bvh, 0) <= physics::bvh::MAX_DEPTH);
		let mut sorted = bvh.tree.order().to_vec();
		sorted.sort_unstable();
		assert!(sorted.iter().enumerate().all(|(k, &i)| k as u32 == i));
		for k in 0..200 {
			let f = k as f32;
			let o = vec3(-10.0, f * 0.5, 50.0 + (f * 0.37).sin() * 40.0);
			let d = vec3(1.0, (f * 0.13).cos() * 0.3, (f * 0.07).sin() * 0.3).normalize();
			let brute = tris
				.iter()
				.enumerate()
				.filter_map(|(i, t)| hit_tri(o, d, t, 1e-4).map(|t| (i, t)))
				.min_by(|a, b| a.1.total_cmp(&b.1));
			let cast = bvh.cast(&tris, o, d, f32::MAX);
			assert_eq!(cast.map(|c| c.1), brute.map(|b| b.1), "ray {k}");
		}
	}
}
