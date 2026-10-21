//! The lightbake path for BSP maps: per-triangle albedo from the map's own
//! texture atlas and the sun from `light_environment`, fed to `lightbake`.
//! The textured path draws the map's own lightdata; this path is our baked
//! light instead, so destruction can rebake (`Baked::update`).

use crate::bsp::{Atlas, Map, Rect};
use linalg::{vec3, Vec3};

/// The map's albedo for every world triangle: mean linear color of each
/// triangle's face texture, taken from the packed atlas, matching the
/// shader's `to_linear`.
pub fn albedo(map: &Map) -> Vec<Vec3> {
	let mut out = Vec::with_capacity(map.world.tris().len());
	for face in &map.render.faces {
		// World triangles are pushed fan by fan, face order, solid faces
		// only; non-solid faces render but carry no physics.
		if !face.solid {
			continue;
		}
		let a = mean(&map.render.albedo, &face.texture);
		for _ in 0..face.verts.len().saturating_sub(2) {
			out.push(a);
		}
	}
	out
}

/// Mean linear color of an atlas rectangle.
fn mean(atlas: &Atlas, rect: &Rect) -> Vec3 {
	let to_linear = |v: u8| (v as f32 / 255.0).powf(2.2);
	let at = |x: u32, y: u32| 4 * (y * atlas.side + x) as usize;
	let mut sum = Vec3::ZERO;
	for y in rect.y..rect.y + rect.h {
		for x in rect.x..rect.x + rect.w {
			let i = at(x, y);
			sum += vec3(
				to_linear(atlas.rgba[i]),
				to_linear(atlas.rgba[i + 1]),
				to_linear(atlas.rgba[i + 2]),
			);
		}
	}
	let n = (rect.w * rect.h).max(1) as f32;
	sum / n
}

/// The value of `key` in one entity block, `None` when the block lacks it.
fn quoted(block: &str, key: &str) -> Option<String> {
	let key = format!("\"{key}\"");
	let at = block.find(&key)? + key.len();
	let rest = &block[at..];
	let open = rest.find('"')? + 1;
	let end = rest[open..].find('"')? + open;
	Some(rest[open..end].into())
}

/// Sun direction and color from the map file's `light_environment`, or the
/// neutral default `parse` uses without one. GoldSrc pitch counts down from
/// the horizon, so the sun sits `-pitch` above it; `_light` is
/// `r g b [brightness]` with 255-channel bytes and a brightness where 100 is
/// radiance 1.
pub fn sun(path: &std::path::Path) -> Result<(Vec3, Vec3), String> {
	let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
	let entities = entity_text(&bytes)?;
	Ok(
		entities
			.split('{')
			.find(|b| quoted(b, "classname").as_deref() == Some("light_environment"))
			.and_then(sun_from)
			.unwrap_or((Vec3::Y, vec3(1.0, 1.0, 1.0))),
	)
}

/// The entities lump of a BSP30 file.
fn entity_text(bytes: &[u8]) -> Result<&str, String> {
	if bytes.len() < 12 || i32_le(bytes, 0) != 30 {
		return Err("not a BSP30".to_string());
	}
	let at = 4; // the entities lump is index 0
	let (off, len) = (u32_le(bytes, at) as usize, u32_le(bytes, at + 4) as usize);
	let lump = bytes
		.get(off..off + len)
		.ok_or_else(|| "entities lump past the end".to_string())?;
	std::str::from_utf8(lump).map_err(|_| "entities are not UTF-8".to_string())
}

fn i32_le(b: &[u8], at: usize) -> i32 {
	i32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u32_le(b: &[u8], at: usize) -> u32 {
	u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The sun inside one `light_environment` block.
fn sun_from(block: &str) -> Option<(Vec3, Vec3)> {
	let angles: Vec<f32> = quoted(block, "angles")?
		.split_whitespace()
		.filter_map(|t| t.parse().ok())
		.collect();
	let pitch: f32 = quoted(block, "pitch")
		.and_then(|p| p.parse().ok())
		.unwrap_or(*angles.first()?);
	let yaw = *angles.get(1)?;
	let light: Vec<f32> = quoted(block, "_light")?
		.split_whitespace()
		.filter_map(|t| t.parse().ok())
		.collect();
	let channel = |i: usize| light.get(i).copied().unwrap_or(255.0) / 255.0;
	let bright = light.get(3).copied().unwrap_or(100.0) / 100.0;
	let (elev, yaw) = (-pitch.to_radians(), yaw.to_radians());
	// GoldSrc yaw 0 faces east (+x) and 90 north (+y); the engine is Y up.
	let dir = vec3(elev.cos() * yaw.cos(), elev.sin(), -elev.cos() * yaw.sin()).normalize();
	Some((dir, vec3(channel(0), channel(1), channel(2)) * bright))
}

/// Bakes the map with `lightbake`: our direct sun, sky and one bounce, ready
/// for partial rebakes when the map's geometry changes.
pub fn bake(map: &Map, path: &std::path::Path) -> Result<lightbake::Baked, String> {
	let albedo = albedo(map);
	if albedo.len() != map.world.tris().len() {
		return Err(format!(
			"{} world triangles but {} albedos",
			map.world.tris().len(),
			albedo.len()
		));
	}
	let (sun_dir, sun_color) = sun(path)?;
	lightbake::bake(
		map.world.tris(),
		&albedo,
		&lightbake::Config {
			sun_dir,
			sun_color,
			sky_color: vec3(
				crate::map_render::SKY.r,
				crate::map_render::SKY.g,
				crate::map_render::SKY.b,
			),
			ground_color: vec3(0.24, 0.23, 0.22),
			..lightbake::Config::default()
		},
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn dust2() -> Option<std::path::PathBuf> {
		let dir = std::env::var_os("CSTRIKE")
			.map(std::path::PathBuf::from)
			.unwrap_or_else(|| {
				std::path::PathBuf::from(r"C:\Program Files (x86)\Steam\steamapps\common\Half-Life\cstrike")
			});
		let path = dir.join("maps/de_dust2.bsp");
		path.exists().then_some(path)
	}

	#[test]
	fn albedo_covers_every_world_triangle() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let albedo = albedo(&map);
		assert_eq!(albedo.len(), map.world.tris().len());
		for a in &albedo {
			for c in [a.x, a.y, a.z] {
				assert!((0.0..=1.0).contains(&c), "albedo {a:?}");
			}
		}
		// de_dust2's light_environment: east, 60 degrees up, 255 255 128 at 70.
		let (dir, color) = sun(&path).unwrap();
		let want = vec3(0.5, 3.0f32.sqrt() / 2.0, 0.0);
		assert!((dir - want).length() < 1e-4, "{dir:?}");
		let want = vec3(0.7, 0.7, 0.7 * 128.0 / 255.0);
		assert!((color - want).length() < 1e-4, "{color:?}");
	}

	/// A real lightmap bake of the whole map. Not part of CI: `cargo test
	/// --release -p app -- --ignored bake` runs it with the BVH speed.
	#[test]
	#[ignore]
	fn bakes_de_dust2() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let start = std::time::Instant::now();
		let baked = bake(&map, &path).unwrap();
		println!(
			"de_dust2: {} charts, {}x{} atlas, {:?}",
			baked.charts(),
			baked.atlas().0,
			baked.atlas().0,
			start.elapsed()
		);
	}
}
