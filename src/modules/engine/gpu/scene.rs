//! The map as the raster draws it: every face's fan triangles as corners,
//! each carrying where its texture tiles and its face's baked chart — the
//! lightmap grid the bake wrote — so every pixel shades the same texel the
//! map's own lightdata sits on.

use crate::bsp::MapRender;
use crate::lightmap::FaceLm;
use lightbake::Triangle;
use linalg::Vec3;

/// One corner for the surface pass (`raster.rs`): position, whole-tile uv
/// with the corner's lightmap texel in `zw`, the face's unit normal, its
/// atlas rectangle, its face's baked chart's rectangle — five `vec4`s.
pub const CORNER_BYTES: usize = 5 * 16;

/// The packed map.
pub struct Scene {
	/// Every fan triangle's corners, for CPU ray casts.
	pub tris: Vec<Triangle>,
	/// Every corner of every face as `CORNER_BYTES` for the surface pass's
	/// vertex buffer.
	pub corners: Vec<u8>,
	/// Bounds of every corner.
	pub lo: Vec3,
	pub hi: Vec3,
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
		for k in 1..verts.len() - 1 {
			let tri = [verts[0].pos, verts[k].pos, verts[k + 1].pos];
			for p in tri {
				scene.lo = scene.lo.min(p);
				scene.hi = scene.hi.max(p);
			}
			scene.tris.push(tri);
		}
	}
	scene
}

#[cfg(test)]
mod tests {
	use super::*;

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
