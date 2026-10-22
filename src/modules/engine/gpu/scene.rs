//! The map as the ray caster reads it: every face's fan triangles, each
//! carrying where its texture tiles. Light is not stored per triangle: the
//! renderer lights every hit itself, from the sun and its probe grid.

use crate::bsp::MapRender;
use lightbake::Triangle;
use linalg::Vec3;

/// One triangle as the shader's `Tri`: six `vec4`s, the corners' `w`
/// holding its unit normal.
pub const TRI_BYTES: usize = 6 * 16;

/// The packed map.
pub struct Scene {
	/// Corners, for the hierarchy.
	pub tris: Vec<Triangle>,
	/// Every triangle as `TRI_BYTES` for the storage buffer.
	pub bytes: Vec<u8>,
	/// Bounds of every corner.
	pub lo: Vec3,
	pub hi: Vec3,
}

/// `value = grad · p + offset` across a face's plane.
#[derive(Clone, Copy, Debug)]
struct Affine {
	grad: Vec3,
	offset: f32,
}

impl Affine {
	/// The affine map through three corners' values, constant along the normal.
	fn fit(p: [Vec3; 3], v: [f32; 3]) -> Affine {
		let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
		let n = e1.cross(e2);
		let grad = (e2.cross(n) * (v[1] - v[0]) + n.cross(e1) * (v[2] - v[0])) / n.length2();
		Affine {
			grad,
			offset: v[0] - grad.dot(p[0]),
		}
	}

	#[cfg(test)]
	fn at(&self, p: Vec3) -> f32 {
		self.grad.dot(p) + self.offset
	}
}

fn push(bytes: &mut Vec<u8>, v: [f32; 4]) {
	bytes.extend(v.iter().flat_map(|x| x.to_le_bytes()));
}

/// Packs `render`'s faces.
pub fn build(render: &MapRender) -> Scene {
	let fans: usize = render
		.faces
		.iter()
		.map(|f| f.verts.len().saturating_sub(2))
		.sum();
	let mut scene = Scene {
		tris: Vec::with_capacity(fans),
		bytes: Vec::with_capacity(fans * TRI_BYTES),
		lo: Vec3::splat(f32::MAX),
		hi: Vec3::splat(f32::MIN),
	};
	for face in &render.faces {
		let verts = &face.verts;
		if verts.len() < 3 {
			continue;
		}
		// The texture map fits once per face, through its widest fan triangle.
		let p0 = verts[0].pos;
		let k = (2..verts.len())
			.max_by(|&a, &b| {
				let area = |k: usize| (verts[k - 1].pos - p0).cross(verts[k].pos - p0).length2();
				area(a).total_cmp(&area(b))
			})
			.expect("three corners");
		let corners = [p0, verts[k - 1].pos, verts[k].pos];
		let fit =
			|value: &dyn Fn(usize) -> f32| Affine::fit(corners, [value(0), value(k - 1), value(k)]);
		let uv = [fit(&|i| verts[i].uv.0), fit(&|i| verts[i].uv.1)];
		for k in 1..verts.len() - 1 {
			let tri = [p0, verts[k].pos, verts[k + 1].pos];
			let bytes = &mut scene.bytes;
			let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
			let n = n / n.length().max(1e-12);
			for (p, w) in tri.into_iter().zip([n.x, n.y, n.z]) {
				push(bytes, [p.x, p.y, p.z, w]);
				scene.lo = scene.lo.min(p);
				scene.hi = scene.hi.max(p);
			}
			for a in uv {
				push(bytes, [a.grad.x, a.grad.y, a.grad.z, a.offset]);
			}
			let t = face.texture;
			bytes.extend([t.x, t.y, t.w, t.h].iter().flat_map(|x| x.to_le_bytes()));
			scene.tris.push(tri);
		}
	}
	scene
}

impl Scene {
	/// The triangles' bytes rearranged into a hierarchy's leaf `order`, so
	/// its leaves index them directly.
	pub fn bytes_in(&self, order: &[u32]) -> Vec<u8> {
		order
			.iter()
			.flat_map(|&i| &self.bytes[i as usize * TRI_BYTES..][..TRI_BYTES])
			.copied()
			.collect()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use linalg::vec3;

	#[test]
	fn affine_fit_reproduces_corner_values() {
		let p = [
			vec3(1.0, 2.0, 3.0),
			vec3(5.0, 2.0, 3.0),
			vec3(1.0, 7.0, 4.0),
		];
		let fit = Affine::fit(p, [0.25, 3.0, -2.0]);
		for (p, v) in p.iter().zip([0.25, 3.0, -2.0]) {
			assert!((fit.at(*p) - v).abs() < 1e-5);
		}
	}

	/// The bundled Dust2; skipped without it.
	#[test]
	fn dust2_packs_every_fan_triangle() {
		let path =
			std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/maps/de_dust2.bsp");
		let Ok(map) = crate::bsp::load(&path) else {
			return;
		};
		let scene = build(&map.render);
		let fans: usize = map
			.render
			.faces
			.iter()
			.map(|f| f.verts.len().saturating_sub(2))
			.sum();
		assert_eq!(scene.tris.len(), fans);
		assert_eq!(scene.bytes.len(), fans * TRI_BYTES);
		assert!(scene.lo.x < scene.hi.x && scene.lo.y < scene.hi.y);
	}
}
