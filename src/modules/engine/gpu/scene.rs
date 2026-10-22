//! The map as the ray caster reads it: every face's fan triangles, each
//! carrying where its texture tiles and its face's baked chart — the
//! lightmap grid the bake wrote, in world positions and atlas texels
//! alike, so every render path shades the same texel the map's own
//! lightdata sits on.

use crate::bsp::MapRender;
use crate::lightmap::FaceLm;
use lightbake::Triangle;
use linalg::Vec3;

/// One triangle as the shader's `Tri`: nine `vec4`s, the corners' `w`
/// holding its unit normal, the face's lightmap mapping riding with every
/// fan triangle.
pub const TRI_BYTES: usize = 9 * 16;
/// One corner for the surface pass (`raster.rs`): position, whole-tile uv
/// with the corner's lightmap texel in `zw`, the face's unit normal, its
/// atlas rectangle, its face's baked chart's rectangle — five `vec4`s.
pub const CORNER_BYTES: usize = 5 * 16;

/// The packed map.
pub struct Scene {
	/// Corners, for the hierarchy.
	pub tris: Vec<Triangle>,
	/// Every triangle as `TRI_BYTES` for the storage buffer.
	pub bytes: Vec<u8>,
	/// Every corner of every face as `CORNER_BYTES` for the surface pass's
	/// vertex buffer.
	pub corners: Vec<u8>,
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

/// Corners for faces drawn over the map for a frame — props, things that
/// move — onto `out`, as the surface pass reads the map's: they have no
/// baked chart, so `ambient`, the light around them, rides where a chart's
/// rectangle would, a height of -1 marking it.
pub fn loose_corners(faces: &[crate::bsp::Face], ambient: Vec3, out: &mut Vec<u8>) {
	for face in faces {
		let verts = &face.verts;
		if verts.len() < 3 {
			continue;
		}
		let n = face.normal();
		let t = face.texture;
		for v in 1..verts.len() - 1 {
			for corner in [0, v, v + 1] {
				let v = &verts[corner];
				push(out, [v.pos.x, v.pos.y, v.pos.z, face.finish.roughness]);
				push(out, [v.uv.0, v.uv.1, 0.0, 0.0]);
				push(out, [n.x, n.y, n.z, face.finish.metal]);
				push(out, [t.x as f32, t.y as f32, t.w as f32, t.h as f32]);
				push(out, [ambient.x, ambient.y, ambient.z, -1.0]);
			}
		}
	}
}

/// Packs `render`'s faces. `face_lms` holds each face's baked light
/// mapping, as `lightmap::face_lms` returned them: a corner's lightmap
/// texel is its position through its face's mapping, and the chart's
/// atlas rectangle rides along, one per corner. Faces without a mapping
/// pack zeroed lightmap data, and so does a `face_lms` that runs out
/// early.
pub fn build(render: &MapRender, face_lms: &[Option<FaceLm>]) -> Scene {
	let fans: usize = render
		.faces
		.iter()
		.map(|f| f.verts.len().saturating_sub(2))
		.sum();
	let mut scene = Scene {
		tris: Vec::with_capacity(fans),
		bytes: Vec::with_capacity(fans * TRI_BYTES),
		corners: Vec::with_capacity(fans * 3 * CORNER_BYTES),
		lo: Vec3::splat(f32::MAX),
		hi: Vec3::splat(f32::MIN),
	};
	for (i, face) in render.faces.iter().enumerate() {
		let verts = &face.verts;
		if verts.len() < 3 {
			continue;
		}
		let lm = face_lms.get(i).copied().flatten();
		// The face's own unit normal, the same winding its triangles carry:
		// the first fan triangle with any area, so a long thin face does
		// not normalize a zero cross.
		let n = face.normal();
		let t = face.texture;
		// Every fan triangle's corners, three per triangle: the hardware
		// meshes consecutive triples, not a face's whole vertex list.
		for v in 1..verts.len() - 1 {
			for corner in [0, v, v + 1] {
				let v = &verts[corner];
				// The corner's own BSP lm, interpolated across the fan
				// by the mesh: faces the BSP wound off one plane shade
				// right, where the planar mapping would not.
				let (lu, lv) = lm.map(|_| v.lm).unwrap_or((0.0, 0.0));
				let rect = lm
					.map(|l| {
						[
							l.rect.x as f32,
							l.rect.y as f32,
							l.rect.w as f32,
							l.rect.h as f32,
						]
					})
					.unwrap_or([0.0; 4]);
				// The face's finish rides in the spare lanes: roughness
				// with the position, metalness with the normal.
				push(
					&mut scene.corners,
					[v.pos.x, v.pos.y, v.pos.z, face.finish.roughness],
				);
				push(&mut scene.corners, [v.uv.0, v.uv.1, lu, lv]);
				push(&mut scene.corners, [n.x, n.y, n.z, face.finish.metal]);
				push(
					&mut scene.corners,
					[t.x as f32, t.y as f32, t.w as f32, t.h as f32],
				);
				push(&mut scene.corners, rect);
			}
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
			for (p, w) in tri.into_iter().zip([n.x, n.y, n.z]) {
				push(bytes, [p.x, p.y, p.z, w]);
				scene.lo = scene.lo.min(p);
				scene.hi = scene.hi.max(p);
			}
			for a in uv {
				push(bytes, [a.grad.x, a.grad.y, a.grad.z, a.offset]);
			}
			// The face's baked light mapping and chart, with every fan
			// triangle: fan triangles sharing a face share one chart.
			push(bytes, lm.map(|l| l.lmu).unwrap_or([0.0; 4]));
			push(bytes, lm.map(|l| l.lmv).unwrap_or([0.0; 4]));
			bytes.extend([t.x, t.y, t.w, t.h].iter().flat_map(|x| x.to_le_bytes()));
			bytes.extend(
				lm.map(|l| [l.rect.x, l.rect.y, l.rect.w, l.rect.h])
					.unwrap_or([0; 4])
					.iter()
					.flat_map(|x| x.to_le_bytes()),
			);
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
		let scene = build(&map.render, &[]);
		let fans: usize = map
			.render
			.faces
			.iter()
			.map(|f| f.verts.len().saturating_sub(2))
			.sum();
		assert_eq!(scene.tris.len(), fans);
		assert_eq!(scene.bytes.len(), fans * TRI_BYTES);
		assert_eq!(scene.corners.len(), fans * 3 * CORNER_BYTES);
		assert!(scene.lo.x < scene.hi.x && scene.lo.y < scene.hi.y);
	}

	/// Each face's corners carry their BSP lightmap coordinates, packed from
	/// the vertex itself and interpolated across the mesh — exactly how
	/// GoldSrc shaded it — and its chart's rectangle rides with them: the
	/// surface pass shades the same texel the map's own lightdata sits on.
	/// Only from the bake's cache: a fresh whole-map bake belongs to the
	/// ignored `bake` test, in release.
	#[test]
	fn corners_carry_their_faces_lightmap() {
		let path =
			std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/maps/de_dust2.bsp");
		let Ok(map) = crate::bsp::load(&path) else {
			return;
		};
		let cache = path.with_extension("lightbake");
		if !cache.exists() {
			return;
		}
		let (baked, cached) = crate::lightmap::bake_or_load(&map, &path).unwrap();
		assert!(cached, "a cache that exists must load");
		let lms = crate::lightmap::face_lms(&map.render, &baked);
		let scene = build(&map.render, &lms);
		let mut corner = 0;
		for (face, lm) in map.render.faces.iter().zip(&lms) {
			if face.verts.len() < 3 {
				continue;
			}
			// Where the BSP's lm grid carries the face, its chart's rectangle is
			// its lightmap block; slivers and faces wound off one plane carry a
			// synthetic chart of their own.
			let planar = crate::lightmap::grid(face).is_some();
			for v in 1..face.verts.len() - 1 {
				for k in [0, v, v + 1] {
					let vert = &face.verts[k];
					let bytes = &scene.corners[corner..corner + CORNER_BYTES];
					corner += CORNER_BYTES;
					let f = |i: usize| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
					let (lu, lv) = (f(6), f(7));
					assert_eq!((lu, lv), lm.map(|_| vert.lm).unwrap_or((0.0, 0.0)));
					if lm.is_some() && planar {
						let (w, h) = (f(18), f(19));
						assert_eq!((w as u32, h as u32), (face.lightmap.w, face.lightmap.h));
					}
				}
			}
		}
	}
}
