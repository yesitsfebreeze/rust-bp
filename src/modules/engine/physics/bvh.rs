//! Bounding-volume hierarchy over a triangle soup, split by binned SAH. The
//! mesh queries walk it on the CPU; the light bake walks the same layout for
//! its rays and uploads it to the GPU.

use crate::{triangle_bounds, Triangle};
use linalg::{vec3, Vec3};

const LEAF_SIZE: usize = 4;
/// Centroid bins per axis the split cost is weighed over.
const BINS: usize = 12;
/// The deepest a node sits. A traversal keeps at most one stack entry a level
/// plus one, so a fixed stack of `MAX_DEPTH + 2` never overflows.
pub const MAX_DEPTH: usize = 31;

/// Nodes in depth-first order: an inner node's left child follows it, `first`
/// names the right one. Leaves list `order[first..first + count]`.
#[derive(Clone, Debug, Default)]
pub struct Bvh {
	nodes: Vec<Node>,
	order: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
	pub lo: Vec3,
	pub hi: Vec3,
	/// Leaf: first index into `order`. Inner: index of the right child.
	pub first: u32,
	/// Triangles in a leaf; zero for inner nodes.
	pub count: u32,
}

fn axis_of(v: Vec3, axis: usize) -> f32 {
	match axis {
		0 => v.x,
		1 => v.y,
		_ => v.z,
	}
}

/// Half a box's surface area: what the chance a ray or a small box meets it
/// scales with.
fn half_area(lo: Vec3, hi: Vec3) -> f32 {
	let d = hi - lo;
	d.x * d.y + d.y * d.z + d.z * d.x
}

impl Bvh {
	pub fn build(tris: &[Triangle]) -> Bvh {
		let mut bvh = Bvh {
			nodes: Vec::new(),
			order: (0..tris.len() as u32).collect(),
		};
		if !tris.is_empty() {
			bvh.split(tris, 0, tris.len(), 0);
		}
		bvh
	}

	/// The nodes, root first; empty for an empty soup.
	pub fn nodes(&self) -> &[Node] {
		&self.nodes
	}

	/// Triangle indices, grouped by leaf.
	pub fn order(&self) -> &[u32] {
		&self.order
	}

	/// Splits where the surface area heuristic, weighed over centroid bins
	/// on every axis, says queries will test the fewest triangles; at the
	/// median of the longest axis when no bin boundary separates the centroids.
	fn split(&mut self, tris: &[Triangle], start: usize, end: usize, depth: usize) {
		let items = &mut self.order[start..end];
		let (mut lo, mut hi) = triangle_bounds(&tris[items[0] as usize]);
		let centroid = |i: u32| {
			let t = &tris[i as usize];
			(t[0] + t[1] + t[2]) * (1.0 / 3.0)
		};
		let (mut clo, mut chi) = (centroid(items[0]), centroid(items[0]));
		for &i in items.iter() {
			let (a, b) = triangle_bounds(&tris[i as usize]);
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
		if end - start <= LEAF_SIZE || depth >= MAX_DEPTH {
			return;
		}
		let e = chi - clo;
		let bin = |i: u32, axis: usize| {
			let f = (axis_of(centroid(i), axis) - axis_of(clo, axis)) / axis_of(e, axis);
			((f * BINS as f32) as usize).min(BINS - 1)
		};
		let empty = (
			vec3(f32::MAX, f32::MAX, f32::MAX),
			vec3(f32::MIN, f32::MIN, f32::MIN),
			0usize,
		);
		// (cost, axis, last bin on the left)
		let mut best = (f32::INFINITY, 0, 0);
		for axis in 0..3 {
			if axis_of(e, axis) <= 1e-6 {
				continue;
			}
			let mut boxes = [empty; BINS];
			for &i in items.iter() {
				let (a, b) = triangle_bounds(&tris[i as usize]);
				let slot = &mut boxes[bin(i, axis)];
				*slot = (slot.0.min(a), slot.1.max(b), slot.2 + 1);
			}
			// Right of each boundary, swept from the far end.
			let mut right = [(0.0f32, 0usize); BINS];
			let mut acc = empty;
			for k in (1..BINS).rev() {
				acc = (
					acc.0.min(boxes[k].0),
					acc.1.max(boxes[k].1),
					acc.2 + boxes[k].2,
				);
				right[k] = (half_area(acc.0, acc.1), acc.2);
			}
			let mut acc = empty;
			for k in 0..BINS - 1 {
				acc = (
					acc.0.min(boxes[k].0),
					acc.1.max(boxes[k].1),
					acc.2 + boxes[k].2,
				);
				let (area, count) = right[k + 1];
				if acc.2 == 0 || count == 0 {
					continue;
				}
				let cost = half_area(acc.0, acc.1) * acc.2 as f32 + area * count as f32;
				if cost < best.0 {
					best = (cost, axis, k);
				}
			}
		}
		let mid = if best.0.is_finite() {
			let mut mid = 0;
			for j in 0..items.len() {
				if bin(items[j], best.1) <= best.2 {
					items.swap(mid, j);
					mid += 1;
				}
			}
			mid
		} else {
			let axis = (0..3)
				.max_by(|&a, &b| axis_of(e, a).total_cmp(&axis_of(e, b)))
				.expect("three axes");
			let mid = items.len() / 2;
			items.select_nth_unstable_by(mid, |&a, &b| {
				axis_of(centroid(a), axis).total_cmp(&axis_of(centroid(b), axis))
			});
			mid
		};
		self.split(tris, start, start + mid, depth + 1);
		self.nodes[at].first = self.nodes.len() as u32;
		self.nodes[at].count = 0;
		self.split(tris, start + mid, end, depth + 1);
	}
}
