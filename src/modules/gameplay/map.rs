//! Test arena in GoldSrc units, Y up, floor at y = 0: stairs, ramps, crates and a
//! duck tunnel, each sized to probe one movement rule. Every triangle has a
//! [`Surface`] the renderer draws it with.

use engine::Surface;
use engine::linalg::{vec3, Vec3};
use engine::physics::Mesh;

pub const SPAWN: Vec3 = vec3(0.0, 0.0, 0.0);
const HALF: f32 = 1024.0;
const TILE: f32 = 64.0;
/// The fenced physics pen's inner bounds ([`crate::showcase`]'s recycle zone).
pub const PEN: (Vec3, Vec3) = (vec3(640.0, 0.0, 512.0), vec3(960.0, 512.0, 832.0));
/// Where the pen's chute drops bodies from, above its slope.
pub const CHUTE_TOP: Vec3 = vec3(668.0, 308.0, 668.0);

/// The arena and the surface of each of its triangles, by index.
pub fn surfaced_arena() -> (Mesh, Vec<Surface>) {
	let mut m = Mesh::default();
	let mut surfaces = Vec::new();
	// Marks every triangle added since the last call as `surface`.
	let mut paint = |m: &Mesh, surface| surfaces.resize(m.tris().len(), surface);
	// Tiled floor: the seams exercise the cast's internal-edge filter.
	let tiles = (2.0 * HALF / TILE) as i32;
	for i in 0..tiles {
		for k in 0..tiles {
			let (x, z) = (-HALF + i as f32 * TILE, -HALF + k as f32 * TILE);
			m.push_quad(
				vec3(x, 0.0, z),
				vec3(x + TILE, 0.0, z),
				vec3(x + TILE, 0.0, z + TILE),
				vec3(x, 0.0, z + TILE),
				vec3(0.0, 1.0, 0.0),
			);
		}
	}
	paint(&m, Surface::Tile);
	// Walls.
	for (min, max) in [
		(
			vec3(-HALF - 32.0, 0.0, -HALF - 32.0),
			vec3(HALF + 32.0, 256.0, -HALF),
		),
		(
			vec3(-HALF - 32.0, 0.0, HALF),
			vec3(HALF + 32.0, 256.0, HALF + 32.0),
		),
		(vec3(-HALF - 32.0, 0.0, -HALF), vec3(-HALF, 256.0, HALF)),
		(vec3(HALF, 0.0, -HALF), vec3(HALF + 32.0, 256.0, HALF)),
	] {
		m.push_box(min, max);
	}
	paint(&m, Surface::Concrete);
	// Stairs of 16 unit steps, walkable, onto a 128 unit platform.
	for s in 0..8 {
		let x = 256.0 + s as f32 * 32.0;
		m.push_box(
			vec3(x, 0.0, -320.0),
			vec3(x + 32.0, 16.0 * (s + 1) as f32, -192.0),
		);
	}
	m.push_box(vec3(512.0, 0.0, -320.0), vec3(768.0, 128.0, -64.0));
	// Steps of 24 units: above sv_stepsize, so they need a jump.
	for s in 0..4 {
		let x = 256.0 + s as f32 * 48.0;
		m.push_box(
			vec3(x, 0.0, -512.0),
			vec3(x + 48.0, 24.0 * (s + 1) as f32, -384.0),
		);
	}
	paint(&m, Surface::Marble);
	// Crates: 32 is jumpable, 48 needs a duck jump, 64 is too high.
	for (i, h) in [32.0, 48.0, 64.0].into_iter().enumerate() {
		let x = -512.0 + i as f32 * 128.0;
		m.push_box(vec3(x, 0.0, 256.0), vec3(x + 64.0, h, 320.0));
	}
	paint(&m, Surface::Jade);
	// Duck tunnel: 48 units of clearance, under standing height 72.
	m.push_box(vec3(-512.0, 0.0, -512.0), vec3(-480.0, 96.0, -256.0));
	m.push_box(vec3(-416.0, 0.0, -512.0), vec3(-384.0, 96.0, -256.0));
	m.push_box(vec3(-480.0, 48.0, -512.0), vec3(-416.0, 96.0, -256.0));
	paint(&m, Surface::Steel);
	// A 30° ramp up to 128, walkable (normal y ≈ 0.87).
	ramp(&mut m, vec3(256.0, 0.0, 256.0), 222.0, 128.0, 128.0);
	paint(&m, Surface::Marble);
	// A surf wedge: 60° faces (normal y = 0.5) are too steep to stand on.
	wedge(&mut m, vec3(-256.0, 0.0, 512.0), 296.0, 256.0, 448.0);
	paint(&m, Surface::Jade);
	// The fenced physics pen: bodies drop down its chute onto a swinging deck and
	// a lift ([`crate::showcase`]). The walls are a step too tall to jump, so the
	// player watches from outside.
	for (min, max) in [
		(vec3(624.0, 0.0, 496.0), vec3(976.0, 128.0, 512.0)),
		(vec3(624.0, 0.0, 832.0), vec3(976.0, 128.0, 848.0)),
		(vec3(624.0, 0.0, 512.0), vec3(640.0, 128.0, 832.0)),
		(vec3(960.0, 0.0, 512.0), vec3(976.0, 128.0, 832.0)),
	] {
		m.push_box(min, max);
	}
	paint(&m, Surface::Concrete);
	// The chute: a slope from (656, 288) down to (768, 176), 64 wide, dropping
	// bodies east toward the deck, with rails and a back wall.
	m.push_quad(
		vec3(656.0, 288.0, 640.0),
		vec3(656.0, 288.0, 704.0),
		vec3(768.0, 176.0, 704.0),
		vec3(768.0, 176.0, 640.0),
		vec3(1.0, 1.0, 0.0),
	);
	m.push_quad(
		vec3(656.0, 288.0, 640.0),
		vec3(656.0, 304.0, 640.0),
		vec3(768.0, 192.0, 640.0),
		vec3(768.0, 176.0, 640.0),
		vec3(0.0, 0.0, -1.0),
	);
	m.push_quad(
		vec3(656.0, 288.0, 704.0),
		vec3(656.0, 304.0, 704.0),
		vec3(768.0, 192.0, 704.0),
		vec3(768.0, 176.0, 704.0),
		vec3(0.0, 0.0, 1.0),
	);
	m.push_box(vec3(652.0, 288.0, 640.0), vec3(660.0, 360.0, 704.0));
	m.push_box(vec3(672.0, 0.0, 664.0), vec3(688.0, 288.0, 680.0));
	m.push_box(vec3(736.0, 0.0, 664.0), vec3(752.0, 176.0, 680.0));
	paint(&m, Surface::Steel);
	(m, surfaces)
}

/// Prism rising along +x from `at` over `length` to `height`, `width` deep along z.
fn ramp(m: &mut Mesh, at: Vec3, length: f32, height: f32, width: f32) {
	let (a, b, c) = (
		at,
		at + vec3(length, height, 0.0),
		at + vec3(length, 0.0, 0.0),
	);
	let w = vec3(0.0, 0.0, width);
	m.push_quad(a, b, b + w, a + w, vec3(-height, length, 0.0));
	m.push_quad(b, c, c + w, b + w, vec3(1.0, 0.0, 0.0));
	m.push_triangle([a, c, b]);
	m.push_triangle([a + w, b + w, c + w]);
}

/// Symmetric triangular prism `width` across x, ridge `height` up, `depth` along z.
fn wedge(m: &mut Mesh, at: Vec3, width: f32, height: f32, depth: f32) {
	let (a, b, c) = (
		at,
		at + vec3(width * 0.5, height, 0.0),
		at + vec3(width, 0.0, 0.0),
	);
	let d = vec3(0.0, 0.0, depth);
	m.push_quad(a, b, b + d, a + d, vec3(-1.0, 0.0, 0.0));
	m.push_quad(b, c, c + d, b + d, vec3(1.0, 0.0, 0.0));
	m.push_triangle([a, c, b]);
	m.push_triangle([a + d, b + d, c + d]);
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn every_triangle_has_a_surface() {
		let (mesh, surfaces) = surfaced_arena();
		assert_eq!(mesh.tris().len(), surfaces.len());
		for s in Surface::ALL {
			assert!(surfaces.contains(&s), "{s:?} unused");
		}
	}
}
