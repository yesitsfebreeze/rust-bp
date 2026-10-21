//! Hitscan with 1.6-style punch-through: a bullet walks wall by wall, pays
//! damage per material it crosses, and stops when a wall is too thick or the
//! weapon has no walls left in it.

use crate::materials::SurfaceMaterial;
use crate::weapons::Weapon;
use engine::linalg::Vec3;
use engine::physics::{Mesh, Triangle};
use engine::Surface;

/// A hit closer than this to the last exit is the same wall's far side.
const EPSILON: f32 = 0.03125;

/// One wall a bullet crossed.
#[derive(Clone, Copy, Debug)]
pub struct Wall {
	pub surface: Surface,
	pub entry: Vec3,
	pub exit: Vec3,
}

/// Where a bullet went and the damage it still carries at its end.
#[derive(Clone, Debug)]
pub struct Shot {
	pub origin: Vec3,
	/// Where the bullet came to rest: the last wall that stopped it, or its
	/// range's end in empty space.
	pub end: Vec3,
	/// Every wall crossed, in order.
	pub walls: Vec<Wall>,
	/// Damage still carried at `end`, after range falloff and every wall's cut.
	pub damage: f32,
}

/// A two-sided ray-triangle hit: how far along the ray, and whether the ray
/// crossed the triangle's front (`normal` against the ray) or back side.
fn ray_tri(origin: Vec3, dir: Vec3, tri: &Triangle) -> Option<(f32, bool)> {
	let normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
	let denom = dir.dot(normal);
	if denom.abs() < 1e-8 {
		return None;
	}
	let t = (tri[0] - origin).dot(normal) / denom;
	if t < 0.0 {
		return None;
	}
	let point = origin + dir * t;
	// Barycentric inside test, on the triangle's plane.
	let (a, b, c) = (tri[0], tri[1], tri[2]);
	let ap = point - a;
	let (ab, ac) = (b - a, c - a);
	let d00 = ab.dot(ab);
	let d01 = ab.dot(ac);
	let d11 = ac.dot(ac);
	let d20 = ap.dot(ab);
	let d21 = ap.dot(ac);
	let denom2 = d00 * d11 - d01 * d01;
	if denom2.abs() < 1e-12 {
		return None;
	}
	let v = (d11 * d20 - d01 * d21) / denom2;
	let w = (d00 * d21 - d01 * d20) / denom2;
	if v < -1e-5 || w < -1e-5 || v + w > 1.0 + 1e-5 {
		return None;
	}
	Some((t, denom < 0.0))
}

/// The nearest triangle the ray crosses with the wanted side: (how far,
/// which triangle, front hit). `None` takes either side, `Some(false)` looks
/// for a wall's far side only.
fn nearest_hit(
	mesh: &Mesh,
	origin: Vec3,
	dir: Vec3,
	wanted: Option<bool>,
) -> Option<(f32, usize, bool)> {
	let mut best = None;
	for (i, tri) in mesh.tris().iter().enumerate() {
		let Some((t, front)) = ray_tri(origin, dir, tri) else {
			continue;
		};
		if t < EPSILON || wanted.is_some_and(|w| front != w) {
			continue;
		}
		if best.is_none_or(|(bt, _, _)| t < bt) {
			best = Some((t, i, front));
		}
	}
	best
}

/// Fires `weapon` from `origin` along `dir` through the world: entry and exit
/// of every punched wall, the resting point, and the damage left in the bullet.
/// With `penetration` off, the first wall stops it.
pub fn fire(
	world: &Mesh,
	surfaces: &[Surface],
	origin: Vec3,
	dir: Vec3,
	weapon: &Weapon,
	penetration: bool,
) -> Shot {
	let mut walls = Vec::new();
	let mut pen = weapon.pen;
	let mut at = origin;
	let mut travelled = 0.0;
	let end = loop {
		let Some((t, tri_index, front)) = nearest_hit(world, at, dir, None) else {
			// Empty air to the range's end.
			break at + dir * weapon.range;
		};
		if !front || !penetration || pen == 0 {
			// Entered from behind a wall, punch-through off, or the weapon has
			// none: the first wall (or the far side we are already inside) wins.
			break at + dir * t;
		}
		let entry = at + dir * t;
		let surface = surfaces[tri_index];
		// The wall's far side: the nearest back-facing crossing past the entry.
		// A wall face-to-face with another box still reads as its own thickness;
		// the next box is a wall of its own from there.
		let exit = nearest_hit(world, entry, dir, Some(false)).map(|(t2, _, _)| entry + dir * t2);
		let Some(exit) = exit else {
			break entry;
		};
		let thickness = (exit - entry).dot(dir);
		if thickness > surface.material().max_thickness {
			// Too thick for the material: the bullet stays in the wall.
			break entry;
		}
		pen -= 1;
		walls.push(Wall {
			surface,
			entry,
			exit,
		});
		travelled += t + thickness;
		at = exit;
	};
	travelled += (end - at).dot(dir);
	let damage = weapon.damage
		* weapon.range_mod.powf(travelled / 500.0)
		* walls
			.iter()
			.map(|w| w.surface.material().damage_mod)
			.product::<f32>();
	Shot {
		origin,
		end,
		walls,
		damage,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use engine::linalg::vec3;

	/// A box of `surface` from (0, 0, 0) to (`thickness`, 64, 64), and a second
	/// identical box at `second`, so rays along +x cross walls in order.
	fn two_walls(surface: Surface, thickness: f32, second: Option<f32>) -> (Mesh, Vec<Surface>) {
		let mut m = Mesh::default();
		m.push_box(vec3(0.0, 0.0, 0.0), vec3(thickness, 64.0, 64.0));
		if let Some(x) = second {
			m.push_box(vec3(x, 0.0, 0.0), vec3(x + thickness, 64.0, 64.0));
		}
		// One surface per triangle, as the arena's `paint` writes them.
		let surfaces = vec![surface; m.tris().len()];
		(m, surfaces)
	}

	/// The AWP is the only stock weapon with two walls in it; tests use it.
	fn awp() -> &'static Weapon {
		&table().weapons[7]
	}

	/// The default weapon table, parsed once for the tests.
	fn table() -> &'static crate::weapons::Table {
		static T: std::sync::OnceLock<crate::weapons::Table> = std::sync::OnceLock::new();
		T.get_or_init(|| crate::weapons::defaults().expect("weapons.toml parses"))
	}

	#[test]
	fn punches_through_a_thin_wall_with_less_damage() {
		let (world, surfaces) = two_walls(Surface::Jade, 16.0, None);
		let shot = fire(
			&world,
			&surfaces,
			vec3(-128.0, 32.0, 32.0),
			vec3(1.0, 0.0, 0.0),
			awp(),
			true,
		);
		assert_eq!(shot.walls.len(), 1);
		assert_eq!(shot.walls[0].surface, Surface::Jade);
		// Out the far side and on to the range's end.
		assert!(
			(shot.walls[0].exit.x - 16.0).abs() < 1e-3,
			"exit {:?}",
			shot.walls[0].exit
		);
		assert!(shot.end.x > 16.0, "still flying: {:?}", shot.end);
		let direct = awp().damage * awp().range_mod.powf((144.0 + awp().range) / 500.0);
		assert!(
			(shot.damage - direct * Surface::Jade.material().damage_mod).abs() < 1e-3,
			"the wall cut the damage: {} vs {direct} * 0.75",
			shot.damage
		);
	}

	#[test]
	fn a_wall_too_thick_stops_the_bullet() {
		let (world, surfaces) = two_walls(Surface::Concrete, 200.0, None);
		let shot = fire(
			&world,
			&surfaces,
			vec3(-128.0, 32.0, 32.0),
			vec3(1.0, 0.0, 0.0),
			awp(),
			true,
		);
		assert!(shot.walls.is_empty());
		assert!(
			(shot.end.x).abs() < 1e-3,
			"rests at the entry: {:?}",
			shot.end
		);
	}

	#[test]
	fn penetration_runs_out() {
		// The AWP carries two walls; a third stops it.
		let mut m = Mesh::default();
		m.push_box(vec3(0.0, 0.0, 0.0), vec3(8.0, 64.0, 64.0));
		m.push_box(vec3(128.0, 0.0, 0.0), vec3(136.0, 64.0, 64.0));
		m.push_box(vec3(256.0, 0.0, 0.0), vec3(264.0, 64.0, 64.0));
		let surfaces = vec![Surface::Jade; m.tris().len()];
		let shot = fire(
			&m,
			&surfaces,
			vec3(-128.0, 32.0, 32.0),
			vec3(1.0, 0.0, 0.0),
			awp(),
			true,
		);
		assert_eq!(shot.walls.len(), 2);
		assert!(
			(shot.end.x - 256.0).abs() < 1e-3,
			"rests at the third: {:?}",
			shot.end
		);
	}

	#[test]
	fn a_knife_cuts_no_walls() {
		let (world, surfaces) = two_walls(Surface::Jade, 8.0, None);
		let shot = fire(
			&world,
			&surfaces,
			vec3(-128.0, 32.0, 32.0),
			vec3(1.0, 0.0, 0.0),
			&crate::weapons::defaults().unwrap().weapons[0],
			true,
		);
		assert!(shot.walls.is_empty());
		assert!((shot.end.x).abs() < 1e-3);
	}

	#[test]
	fn range_bleeds_damage() {
		let (world, surfaces) = two_walls(Surface::Jade, 16.0, None);
		let near = fire(
			&world,
			&surfaces,
			vec3(-64.0, 32.0, 32.0),
			vec3(1.0, 0.0, 0.0),
			awp(),
			true,
		);
		let far = fire(
			&world,
			&surfaces,
			vec3(-2048.0, 32.0, 32.0),
			vec3(1.0, 0.0, 0.0),
			awp(),
			true,
		);
		assert!(far.damage < near.damage, "far {:?} near {:?}", far, near);
	}

	#[test]
	fn empty_air_flies_to_the_range() {
		let world = Mesh::default();
		let shot = fire(
			&world,
			&[],
			vec3(0.0, 0.0, 0.0),
			vec3(1.0, 0.0, 0.0),
			awp(),
			true,
		);
		assert_eq!(shot.walls.len(), 0);
		assert!((shot.end.x - awp().range).abs() < 1e-3);
	}
}
