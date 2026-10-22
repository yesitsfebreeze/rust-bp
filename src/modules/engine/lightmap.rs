//! The lightbake path for BSP maps: per-triangle albedo from the map's own
//! texture atlas, the sun from `light_environment` and the map's `light`
//! and `light_spot` entities, and one chart per BSP face on the map's own
//! 16-unit luxel grid — the same grid qrad sampled, the same block sizes
//! the map's lightdata packs — fed to `lightbake`. The baked atlas is the
//! map's light for every render path; the embedded qrad lightdata stays
//! parsed for reference, and destruction can rebake (`Baked::update`).

use crate::bsp::{Atlas, Face, Map, MapRender, Rect};
use linalg::{vec3, Vec3};

/// The map's albedo for every rendered triangle: mean linear color of each
/// triangle's face texture, taken from the packed atlas, matching the
/// shader's `to_linear`.
pub fn albedo(map: &Map) -> Vec<Vec3> {
	let capacity = map
		.render
		.faces
		.iter()
		.map(|face| face.verts.len().saturating_sub(2))
		.sum();
	let mut out = Vec::with_capacity(capacity);
	for face in &map.render.faces {
		let a = mean(&map.render.albedo, &face.texture);
		for _ in 0..face.verts.len().saturating_sub(2) {
			out.push(a);
		}
	}
	out
}

fn triangles(map: &Map) -> Vec<lightbake::Triangle> {
	map
		.render
		.faces
		.iter()
		.flat_map(|face| {
			(1..face.verts.len().saturating_sub(1))
				.map(|k| [face.verts[0].pos, face.verts[k].pos, face.verts[k + 1].pos])
		})
		.collect()
}

/// Each fan triangle's chart: its face's. The triangles come in fan order,
/// so each face's triangles are one contiguous run.
fn tri_chart(render: &MapRender) -> Vec<u32> {
	let mut out = Vec::with_capacity(
		render
			.faces
			.iter()
			.map(|face| face.verts.len().saturating_sub(2))
			.sum(),
	);
	for (i, face) in render.faces.iter().enumerate() {
		out.extend(std::iter::repeat_n(
			i as u32,
			face.verts.len().saturating_sub(2),
		));
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

/// The face's own luxel grid, if its lightmap coordinates carry one: the
/// world point at luxel (0, 0) and the two world steps one luxel spans.
/// The lm coordinates are affine in world position, so two edges whose lm
/// deltas span the grid invert it exactly.
pub(crate) fn grid(face: &Face) -> Option<(Vec3, Vec3, Vec3)> {
	if face.lightmap.w == 0 || face.lightmap.h == 0 || face.verts.len() < 3 {
		return None;
	}
	let a = &face.verts[0];
	let (la0, la1) = a.lm;
	let delta = |v: &crate::bsp::Vert| (v.lm.0 - la0, v.lm.1 - la1);
	// The edge whose lm delta reaches farthest, then the longest one not
	// parallel to it: a snapped T-junction leaves a corner nearly on the
	// first vertex, and inverting through its hair-fine delta would read
	// the grid from floating point noise.
	let (p1, d1) = face.verts[1..]
		.iter()
		.map(|v| (v.pos - a.pos, delta(v)))
		.max_by(|(_, a), (_, b)| (a.0 * a.0 + a.1 * a.1).total_cmp(&(b.0 * b.0 + b.1 * b.1)))?;
	if d1.0 * d1.0 + d1.1 * d1.1 < 1e-8 {
		return None;
	}
	let (p2, d2) = face.verts[1..]
		.iter()
		.map(|v| (v.pos - a.pos, delta(v)))
		.filter(|&(_, d)| (d1.0 * d.1 - d1.1 * d.0).abs() > 1e-8)
		.max_by(|(_, a), (_, b)| (a.0 * a.0 + a.1 * a.1).total_cmp(&(b.0 * b.0 + b.1 * b.1)))?;
	let det = d1.0 * d2.1 - d1.1 * d2.0;
	// The world steps one luxel takes along each grid axis.
	let u = (p1 * d2.1 - p2 * d1.1) / det;
	let v = (p2 * d1.0 - p1 * d2.0) / det;
	// A sliver face's edges span one line in the world: the lm deltas
	// invert, but to steps that nearly or exactly cancel, a grid with no
	// area to read. The synthetic chart covers such a face instead.
	if u.cross(v).length2() < 1.0 {
		return None;
	}
	// The grid must lie in the face's plane: a face the BSP wound off one
	// plane — an arch's curve — maps its lm off it too, and no planar
	// chart follows such a face.
	let n = face.normal();
	if u.dot(n).abs() > 1e-3 * u.length() || v.dot(n).abs() > 1e-3 * v.length() {
		return None;
	}
	Some((a.pos - u * la0 - v * la1, u, v))
}

/// The bake's charts, one per face, on the map's own luxel grid: the face's
/// lightmap block size, grid axes inverted from its lm coordinates. Faces
/// without a lightmap block or with degenerate lm — the playground arena
/// builds its faces without either — get a synthetic dominant-axis chart
/// at the config's density instead.
pub fn face_charts(render: &MapRender, config: &lightbake::Config) -> Vec<lightbake::ChartSpec> {
	render
		.faces
		.iter()
		.map(|face| match grid(face) {
			Some((origin, u, v)) => lightbake::ChartSpec {
				origin,
				u,
				v,
				n: face.normal(),
				w: face.lightmap.w,
				h: face.lightmap.h,
			},
			None => lightbake::ChartSpec::fit_points(
				&face.verts.iter().map(|v| v.pos).collect::<Vec<_>>(),
				face.normal(),
				config,
			),
		})
		.collect()
}

/// One face's baked light mapping: where its chart sits in the baked atlas,
/// and the affine from world positions on the face to its luxel
/// coordinates, `lm_u(p) = p·lmu.xyz + lmu.w` and `lm_v` likewise. The
/// same mapping for every render path: the raster's corners, the voxel
/// paths' cubes.
#[derive(Clone, Copy, Debug)]
pub struct FaceLm {
	/// The face's chart in the baked atlas.
	pub rect: lightbake::Rect,
	pub lmu: [f32; 4],
	pub lmv: [f32; 4],
}

impl FaceLm {
	/// The luxel coordinates of world point `p`, unclamped.
	pub fn lm(&self, p: Vec3) -> (f32, f32) {
		let at = |g: &[f32; 4]| p.x * g[0] + p.y * g[1] + p.z * g[2] + g[3];
		(at(&self.lmu), at(&self.lmv))
	}
}

/// Each face's baked light mapping, `None` for faces the bake skipped.
/// Fresh and cache-loaded bakes carry their charts alike, so this reads
/// the same from either.
pub fn face_lms(render: &MapRender, baked: &lightbake::Baked) -> Vec<Option<FaceLm>> {
	render
		.faces
		.iter()
		.enumerate()
		.map(|(i, _)| {
			let view = baked.chart(i)?;
			// The dual basis of the chart's grid: the gradients mapping
			// world points to luxel coordinates.
			let (g_u, g_v) = view.grads();
			Some(FaceLm {
				rect: view.rect,
				lmu: [g_u.x, g_u.y, g_u.z, -g_u.dot(view.origin)],
				lmv: [g_v.x, g_v.y, g_v.z, -g_v.dot(view.origin)],
			})
		})
		.collect()
}

/// Sun direction and color from the map file's `light_environment`, or the
/// neutral default `parse` uses without one. The entity's angles say where
/// its light shines, as [`shine_dir`] reads them; the sun sits the other
/// way, back along the light. `_light` is
/// `r g b [brightness]` with 255-channel bytes and a brightness where 100 is
/// radiance 1.
pub fn sun(path: &std::path::Path) -> Result<(Vec3, Vec3), String> {
	let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
	let entities = crate::bsp::entity_text(&bytes)?;
	Ok(
		entities
			.split('{')
			.find(|b| crate::bsp::field(b, "classname").as_deref() == Some("light_environment"))
			.and_then(sun_from)
			.unwrap_or((Vec3::Y, vec3(1.0, 1.0, 1.0))),
	)
}

/// Direction a GoldSrc `elev yaw` pair points: `elev` is degrees above the
/// horizon, yaw 0 faces east (+x) and 90 north (+y); the engine is Y up.
/// The sun's pitch counts up while a spot's counts down, so callers sign it.
fn goldsrc_dir(elev: f32, yaw: f32) -> Vec3 {
	let (elev, yaw) = (elev.to_radians(), yaw.to_radians());
	vec3(elev.cos() * yaw.cos(), elev.sin(), -elev.cos() * yaw.sin()).normalize()
}

/// The direction a GoldSrc light entity shines, as qrad reads it: the yaw
/// from `angle` when set (-1 shines straight up, -2 straight down), else
/// from `angles`; the pitch from `pitch` when set, else from `angles`, else
/// `default_pitch`, counting up from the horizon — -90 shines straight
/// down.
fn shine_dir(block: &str, default_pitch: f32) -> Vec3 {
	let angles: Vec<f32> = crate::bsp::field(block, "angles")
		.map(|a| {
			a.split_whitespace()
				.filter_map(|t| t.parse().ok())
				.collect()
		})
		.unwrap_or_default();
	let angle: f32 = crate::bsp::field(block, "angle")
		.and_then(|a| a.parse().ok())
		.unwrap_or(0.0);
	if angle == -1.0 {
		return Vec3::Y;
	}
	if angle == -2.0 {
		return -Vec3::Y;
	}
	let yaw = if angle != 0.0 {
		angle
	} else {
		angles.get(1).copied().unwrap_or(0.0)
	};
	let pitch = crate::bsp::field(block, "pitch")
		.and_then(|p| p.parse().ok())
		.or_else(|| angles.first().copied())
		.unwrap_or(default_pitch);
	goldsrc_dir(pitch, yaw)
}

/// The sun inside one `light_environment` block.
fn sun_from(block: &str) -> Option<(Vec3, Vec3)> {
	let light: Vec<f32> = crate::bsp::field(block, "_light")?
		.split_whitespace()
		.filter_map(|t| t.parse().ok())
		.collect();
	let channel = |i: usize| light.get(i).copied().unwrap_or(255.0) / 255.0;
	let bright = light.get(3).copied().unwrap_or(100.0) / 100.0;
	Some((
		-shine_dir(block, -90.0),
		vec3(channel(0), channel(1), channel(2)) * bright,
	))
}

/// The map's baked light sources besides the sun: every `light` and
/// `light_spot` entity, its origin in engine coordinates, its radiance
/// scaled like the sun's, and a spot's cone from its angles. A light
/// without `_fade` keeps the default reach.
pub fn lights(path: &std::path::Path) -> Result<Vec<lightbake::Light>, String> {
	let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
	let entities = crate::bsp::entity_text(&bytes)?;
	let mut out = Vec::new();
	for block in entities.split('{').skip(1) {
		let spot = match crate::bsp::field(block, "classname").as_deref() {
			Some("light") => false,
			Some("light_spot") => true,
			_ => continue,
		};
		let Some(origin) = crate::bsp::field(block, "origin").and_then(|o| {
			let v: Vec<f32> = o
				.split_whitespace()
				.filter_map(|t| t.parse().ok())
				.collect();
			(v.len() == 3).then_some([v[0], v[1], v[2]])
		}) else {
			continue;
		};
		let light: Vec<f32> = crate::bsp::field(block, "_light")
			.map(|l| {
				l.split_whitespace()
					.filter_map(|t| t.parse().ok())
					.collect()
			})
			.unwrap_or_default();
		let channel = |i: usize| light.get(i).copied().unwrap_or(255.0) / 255.0;
		let bright = light.get(3).copied().unwrap_or(100.0) / 100.0;
		let fade = crate::bsp::field(block, "_fade")
			.and_then(|f| f.parse().ok())
			.unwrap_or(768.0);
		let cone = spot.then(|| {
			let dir = shine_dir(block, -90.0);
			let deg = crate::bsp::field(block, "_cone")
				.and_then(|c| c.parse().ok())
				.unwrap_or(10.0);
			(dir, deg)
		});
		out.push(lightbake::Light {
			pos: crate::bsp::to_our(origin),
			color: vec3(channel(0), channel(1), channel(2)) * bright * 4.5,
			fade,
			cone,
		});
	}
	Ok(out)
}

/// The bake's config for a map: its own sun and sky, at the BSP's own
/// density — 16 units per texel, the luxel grid qrad sampled, one chart per
/// face sized as the map's own lightmap blocks — with 512 rays and 3
/// bounces, sharp up close and smooth in the shade. `units_per_texel` and
/// `min_chart` only shape the synthetic charts faces without a lightmap
/// block get.
///
/// GoldSrc's `_light` brightness is not linear radiance: unscaled, the sun
/// lands weaker than the sky that fills every shadow, and the map inverts —
/// shade brighter than sunlit wall. The sun is scaled to the radiance the
/// shader's light model expects, and the sky comes from the visible
/// background color, linearized.
pub fn config(path: &std::path::Path) -> Result<lightbake::Config, String> {
	let (sun_dir, sun_color) = sun(path)?;
	Ok(lightbake::Config {
		atlas_size: 4096,
		units_per_texel: 16.0,
		min_chart: 1,
		rays: 512,
		bounces: 3,
		sun_dir,
		sun_color: sun_color * 4.5,
		lights: lights(path)?,
		sky_color: vec3(
			crate::gpu::SKY.r.powf(2.2),
			crate::gpu::SKY.g.powf(2.2),
			crate::gpu::SKY.b.powf(2.2),
		),
		ground_color: vec3(0.24, 0.23, 0.22),
		..lightbake::Config::default()
	})
}

/// Bakes the map with `lightbake` under `config`: one chart per face on the
/// map's own luxel grid, ready for partial rebakes when the map's geometry
/// changes. `bake` is this with the config a map file's entities describe.
pub fn bake_with(map: &Map, config: &lightbake::Config) -> Result<lightbake::Baked, String> {
	let tris = triangles(map);
	let albedo = albedo(map);
	let charts = face_charts(&map.render, config);
	let tri_chart = tri_chart(&map.render);
	if albedo.len() != tris.len() {
		return Err(format!(
			"{} rendered triangles but {} albedos",
			tris.len(),
			albedo.len()
		));
	}
	lightbake::bake(&tris, &albedo, &charts, &tri_chart, config)
}

/// Bakes the map with the config a map file's entities describe.
pub fn bake(map: &Map, path: &std::path::Path) -> Result<lightbake::Baked, String> {
	bake_with(map, &config(path)?)
}

/// Where `path`'s bake is cached: the map's own name, extension swapped.
fn cache_path(path: &std::path::Path) -> std::path::PathBuf {
	path.with_extension("lightbake")
}

/// Bakes the map, or loads the bake cached beside it when the map's bytes and
/// the config it was baked with both match. Returns the bake and whether it
/// came from the cache; a cache that cannot be written is ignored, one that
/// cannot be read falls back to a fresh bake. `bake` is the bake itself, so
/// the CPU and GPU paths share every other step. `extra` joins the map's
/// bytes in the cache's key: whatever was added to the map since it loaded
/// — a level's own file — so a changed level never loads a stale bake.
fn bake_or_load_with(
	map: &Map,
	path: &std::path::Path,
	extra: &[u8],
	bake: impl FnOnce(
		&[lightbake::Triangle],
		&[Vec3],
		&[lightbake::ChartSpec],
		&[u32],
		&lightbake::Config,
	) -> Result<lightbake::Baked, String>,
) -> Result<(lightbake::Baked, bool), String> {
	let mut bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
	bytes.extend_from_slice(extra);
	let config = config(path)?;
	let key = config.cache_key(&bytes);
	if let Ok(cached) = std::fs::read(cache_path(path)) {
		if let Ok(baked) = lightbake::Baked::from_bytes(&cached, key) {
			return Ok((baked, true));
		}
	}
	let tris = triangles(map);
	let albedo = albedo(map);
	let charts = face_charts(&map.render, &config);
	let tri_chart = tri_chart(&map.render);
	if albedo.len() != tris.len() {
		return Err(format!(
			"{} rendered triangles but {} albedos",
			tris.len(),
			albedo.len()
		));
	}
	let baked = bake(&tris, &albedo, &charts, &tri_chart, &config)?;
	// A read-only map directory just rebakes next time.
	let _ = std::fs::write(cache_path(path), baked.to_bytes(key));
	Ok((baked, false))
}

/// Bakes the map with `lightbake`: our direct sun, sky and bounces, ready
/// for partial rebakes when the map's geometry changes.
pub fn bake_or_load(map: &Map, path: &std::path::Path) -> Result<(lightbake::Baked, bool), String> {
	bake_or_load_with(map, path, &[], lightbake::bake)
}

/// The GPU bake: the same charts, atlas and cache as `bake_or_load`, with
/// the texel passes on wgpu compute. Direct terms match the CPU bake to
/// rounding; bounced light agrees within its Monte-Carlo noise. Only for
/// the baking tool — the game stays wgpu-free.
#[cfg(feature = "gpu-bake")]
pub fn bake_or_load_gpu(
	map: &Map,
	path: &std::path::Path,
) -> Result<(lightbake::Baked, bool), String> {
	bake_or_load_with(map, path, &[], lightbake::bake_gpu)
}

/// The map at `path`, with whatever a level added to it, baked or loaded
/// from the cache: `extra` is the level's own bytes, empty without one, so
/// the cache answers only for this map with this level. Bakes on the GPU —
/// seconds — and on the CPU only when no GPU bakes.
pub fn bake_or_load_level(
	map: &Map,
	path: &std::path::Path,
	extra: &[u8],
) -> Result<(lightbake::Baked, bool), String> {
	bake_or_load_with(map, path, extra, |tris, albedo, charts, tri_chart, config| {
		lightbake::bake_gpu(tris, albedo, charts, tri_chart, config)
			.or_else(|_| lightbake::bake(tris, albedo, charts, tri_chart, config))
	})
}

/// The bake's atlas as the renderer samples it: half floats, RGBA per
/// texel, little endian — RGB the bake's linear `S`, A its sun
/// visibility, so the surfaces can part the straight sun from the rest.
/// Half floats keep a dark room's light as smooth as a sunlit wall's:
/// bytes left only a handful of steps in the shade, and the eye's
/// adaptation stretched them into bands.
pub fn atlas(baked: &lightbake::Baked) -> (u32, Vec<u8>) {
	let (side, texels) = baked.atlas();
	let bytes = texels
		.iter()
		.flat_map(|texel| [texel[0], texel[1], texel[2], texel[3].clamp(0.0, 1.0)])
		.flat_map(|v| lightbake::f16_bits(v).to_le_bytes())
		.collect();
	(side, bytes)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn dust2() -> Option<std::path::PathBuf> {
		// The bundled copy first, then a full 1.6 install.
		let bundled =
			std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/maps/de_dust2.bsp");
		let installed = std::env::var_os("CSTRIKE")
			.map(std::path::PathBuf::from)
			.unwrap_or_else(|| {
				std::path::PathBuf::from(r"C:\Program Files (x86)\Steam\steamapps\common\Half-Life\cstrike")
			})
			.join("maps/de_dust2.bsp");
		[bundled, installed].into_iter().find(|p| p.exists())
	}

	#[test]
	fn albedo_covers_every_rendered_triangle() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let albedo = albedo(&map);
		assert_eq!(albedo.len(), triangles(&map).len());
		for a in &albedo {
			for c in [a.x, a.y, a.z] {
				assert!((0.0..=1.0).contains(&c), "albedo {a:?}");
			}
		}
		// de_dust2's light_environment: `angle` 43 and `pitch` -60, shining
		// 60 degrees down toward yaw 43 — so the sun stands 60 degrees up at
		// yaw 223 — 255 255 128 at 70. Its `angles` say 0 0 0: qrad takes
		// the yaw from `angle`.
		let (dir, color) = sun(&path).unwrap();
		let yaw = 43f32.to_radians();
		let want = vec3(-0.5 * yaw.cos(), 3.0f32.sqrt() / 2.0, 0.5 * yaw.sin());
		assert!((dir - want).length() < 1e-4, "{dir:?}");
		let want = vec3(0.7, 0.7, 0.7 * 128.0 / 255.0);
		assert!((color - want).length() < 1e-4, "{color:?}");
	}

	/// The sun stands where qrad lit Dust2 from: sun rays from the luxels
	/// of its sunward faces reach the sky about where the map's own
	/// lightmaps are bright. The sun's old reading, 223 degrees off, agreed
	/// on far fewer.
	#[test]
	fn sun_shadows_match_the_maps_own_lightmaps() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let (sun, _) = sun(&path).unwrap();
		let tris = triangles(&map);
		let bvh = lightbake::Bvh::build(&tris);
		let lm = &map.render.lightmap;
		let (mut agree, mut total) = (0, 0);
		for f in &map.render.faces {
			let n = f.normal();
			if f.lightmap.w == 0 || n.dot(sun) < 0.3 {
				continue;
			}
			let Some((origin, u, v)) = grid(f) else {
				continue;
			};
			for j in 0..f.lightmap.h {
				for i in 0..f.lightmap.w {
					let p = origin + u * i as f32 + v * j as f32;
					let at = (((f.lightmap.y + j) * lm.side + f.lightmap.x + i) * 4) as usize;
					let bright = lm.rgba[at] as f32 * 0.3
						+ lm.rgba[at + 1] as f32 * 0.59
						+ lm.rgba[at + 2] as f32 * 0.11;
					let lit = bvh.cast(&tris, p + n * 0.5, sun, 1e30).is_none();
					agree += ((bright > 65.0) == lit) as u32;
					total += 1;
				}
			}
		}
		let share = agree as f32 / total as f32;
		assert!(
			share > 0.8,
			"sun shadows agree with qrad on {share} of luxels"
		);
	}

	#[test]
	fn the_sun_outshines_the_sky_that_fills_shade() {
		let Some(path) = dust2() else {
			return;
		};
		let config = config(&path).unwrap();
		// Direct sun on a horizontal surface against the sky that a shaded
		// one is left with: sun must dominate, or shade reads as bright as
		// sunlight and shadows vanish.
		let sun = config.sun_color.y * config.sun_dir.y / std::f32::consts::PI;
		let sky = config.sky_color.y;
		assert!(sun > 1.5 * sky, "sun {sun} does not outshine sky {sky}");
	}

	#[test]
	fn dust2s_lights_parse_into_the_bake() {
		let Some(path) = dust2() else {
			return;
		};
		let lights = lights(&path).unwrap();
		// de_dust2 carries eleven `light` entities and one sun.
		assert_eq!(lights.len(), 11);
		for light in &lights {
			assert!(light.fade > 0.0, "fade {:?}", light.fade);
			for c in [light.color.x, light.color.y, light.color.z] {
				assert!(c.is_finite() && c >= 0.0, "radiance {c}");
			}
		}
	}

	#[test]
	fn qrad_light_follows_the_fan_winding() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let (sun_dir, _) = sun(&path).unwrap();
		// Mean brightness of each face's embedded qrad block, split by the
		// face fan's normal: sun-facing surfaces must carry more of the map's
		// own baked light than sun-away ones, or the winding is flipped and
		// the bake inverts — lit walls black, shaded walls lit.
		let lm = &map.render.lightmap;
		let at = |x: u32, y: u32| 4 * (y * lm.side + x) as usize;
		let (mut toward, mut away) = ((0.0f32, 0usize), (0.0f32, 0usize));
		for face in &map.render.faces {
			let [a, b, c] = [face.verts[0].pos, face.verts[1].pos, face.verts[2].pos];
			let n = (b - a).cross(c - a).normalize();
			let d = n.dot(sun_dir);
			let rect = face.lightmap;
			if rect.w == 0 || rect.h == 0 {
				continue;
			}
			let mut sum = 0.0;
			for y in rect.y..rect.y + rect.h {
				for x in rect.x..rect.x + rect.w {
					let i = at(x, y);
					sum +=
						(lm.rgba[i] as f32 + lm.rgba[i + 1] as f32 + lm.rgba[i + 2] as f32) / (3.0 * 255.0);
				}
			}
			let mean = sum / (rect.w * rect.h) as f32;
			if d > 0.4 {
				toward = (toward.0 + mean, toward.1 + 1);
			} else if d < -0.4 {
				away = (away.0 + mean, away.1 + 1);
			}
		}
		assert!(toward.1 > 0 && away.1 > 0, "no faces to compare");
		let (toward, away) = (toward.0 / toward.1 as f32, away.0 / away.1 as f32);
		assert!(
			toward > away * 1.1,
			"qrad light {toward} does not follow the winding against {away}"
		);
	}

	/// One chart per face, on the map's own luxel grid: the chart's block is
	/// the face's own lightmap block, its grid steps span 16 world units,
	/// and every corner's lm coordinates come back out of the chart's
	/// mapping — so the whole render pipeline, which lights by that mapping,
	/// reads the same luxel the map's own lightdata sits on.
	#[test]
	fn face_charts_fit_the_map_lightmap_blocks() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let config = config(&path).unwrap();
		let charts = face_charts(&map.render, &config);
		assert_eq!(charts.len(), map.render.faces.len());
		for (i, (face, chart)) in map.render.faces.iter().zip(&charts).enumerate() {
			// A sliver face's lm deltas invert to no grid (`grid` returns
			// None): its synthetic chart follows the config's density, not
			// qrad's block.
			if grid(face).is_none() {
				continue;
			}
			assert_eq!(
				(chart.w, chart.h),
				(face.lightmap.w, face.lightmap.h),
				"face {i}'s chart block is not its lightmap block"
			);
			// Every corner maps onto the chart's own grid: inverting the
			// mapping reproduces the BSP's lm coordinates. The grid steps'
			// lengths follow the face's texture scale, so they are not
			// asserted against 16 — only the mapping is exact. Inverting
			// near-parallel edges in f32 wobbles the corners by up to a
			// sixteenth of a luxel.
			let (g_u, g_v) = chart_grads(chart);
			for v in &face.verts {
				let rel = v.pos - chart.origin;
				let got = (g_u.dot(rel), g_v.dot(rel));
				assert!(
					(got.0 - v.lm.0).abs() < 1.0 / 64.0 && (got.1 - v.lm.1).abs() < 1.0 / 64.0,
					"face {i}'s corner maps to ({got:?}), the BSP says ({:?})",
					v.lm
				);
				assert!(
					got.0 >= -0.6 && got.0 <= chart.w as f32 - 1.0 + 0.6,
					"face {i}'s corner sits outside its block: {got:?}"
				);
			}
		}
	}

	/// The dual-basis gradients of a chart spec, as [`face_lms`] packs them.
	fn chart_grads(chart: &lightbake::ChartSpec) -> (Vec3, Vec3) {
		let d = chart.u.cross(chart.v).dot(chart.n);
		(chart.v.cross(chart.n) / d, chart.n.cross(chart.u) / d)
	}

	/// Every face finds its chart in the bake, on the block the map sized
	/// for it, and the affine [`face_lms`] packs maps the face's corners to
	/// their lm coordinates — so the render paths light a face by the very
	/// grid qrad sampled.
	#[test]
	fn face_lms_map_corners_onto_their_blocks() {
		let Some(path) = dust2() else {
			return;
		};
		// Only from the cache: a fresh whole-map bake belongs to the
		// ignored `bake` test, in release.
		if !cache_path(&path).exists() {
			return;
		}
		let map = crate::bsp::load(&path).unwrap();
		let (baked, cached) = bake_or_load(&map, &path).unwrap();
		assert!(cached, "a cache that exists must load");
		let lms = face_lms(&map.render, &baked);
		assert_eq!(lms.len(), map.render.faces.len());
		for (i, (face, lm)) in map.render.faces.iter().zip(&lms).enumerate() {
			let Some(lm) = lm else {
				continue;
			};
			// Slivers carry no lm grid, so their synthetic chart lights them
			// on its own density, not qrad's block.
			if grid(face).is_none() {
				continue;
			}
			let view = baked.chart(i).unwrap();
			assert_eq!((lm.rect.w, lm.rect.h), (face.lightmap.w, face.lightmap.h));
			for v in &face.verts {
				let got = lm.lm(v.pos);
				// The cache stores charts at f16, which wobbles the mapping
				// by a hundredth of a luxel; a sixteenth covers it.
				assert!(
					(got.0 - v.lm.0).abs() < 1.0 / 64.0 && (got.1 - v.lm.1).abs() < 1.0 / 64.0,
					"face {i}'s chart {:?} maps a corner to {got:?}, the BSP says {:?}",
					view.rect,
					v.lm
				);
			}
		}
	}

	/// Direct terms in isolation: both bakes with `bounces` zeroed. Charts
	/// agree to rounding except where a light ray grazes a blocker's edge
	/// — the shared edge is hit or missed on floating point alone, and on
	/// a skinny face whose chart clamps most texels to one ray, one
	/// flip moves the whole chart mean. The test bounds that population
	/// instead of demanding zero: a systematic divergence would show up
	/// in hundreds of charts, not a handful. Also needs the `gpu-bake`
	/// feature and a GPU.
	#[cfg(feature = "gpu-bake")]
	#[test]
	#[ignore]
	fn gpu_direct_bake_matches_cpu_on_dust2() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let mut config = config(&path).unwrap();
		config.bounces = 0;
		let tris = triangles(&map);
		let albedo = albedo(&map);
		let charts = face_charts(&map.render, &config);
		let tri_chart = tri_chart(&map.render);
		let cpu = lightbake::bake(&tris, &albedo, &charts, &tri_chart, &config).unwrap();
		let gpu = lightbake::bake_gpu(&tris, &albedo, &charts, &tri_chart, &config).unwrap();
		assert_eq!(gpu.charts(), cpu.charts());
		let (side, gpu_texels) = gpu.atlas();
		let (_, cpu_texels) = cpu.atlas();
		let mut worst = (0.0f32, 0usize);
		let mut over = 0usize;
		for face in 0..map.render.faces.len() {
			let (Some(a), Some(b)) = (cpu.chart(face), gpu.chart(face)) else {
				continue;
			};
			assert_eq!(a.rect, b.rect, "face {face} packs differently");
			let mut mean = (vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));
			for j in 0..a.rect.h {
				for i in 0..a.rect.w {
					let at = ((a.rect.y + j) * side + a.rect.x + i) as usize;
					mean.0 += vec3(cpu_texels[at][0], cpu_texels[at][1], cpu_texels[at][2]);
					mean.1 += vec3(gpu_texels[at][0], gpu_texels[at][1], gpu_texels[at][2]);
				}
			}
			let n = (a.rect.w * a.rect.h) as f32;
			let d = ((mean.0 - mean.1) / n).length() / ((mean.0 / n).length() + 1e-6);
			if d > worst.0 {
				worst = (d, face);
			}
			if d > 0.1 {
				over += 1;
			}
		}
		println!(
			"direct-only: worst mean gap {:.1}% on face {}, {} charts over 10%",
			worst.0 * 100.0,
			worst.1,
			over
		);
		assert!(
			over < cpu.charts() / 100,
			"{over} charts' direct terms differ by more than 10%"
		);
	}

	/// The GPU bake against the whole-map CPU bake cached beside the bundled
	/// map: identical charts, direct terms within rounding, bounced light
	/// within its Monte-Carlo noise. Needs a GPU and the `gpu-bake`
	/// feature: `cargo test --release -p engine --features gpu-bake --
	/// --ignored gpu --nocapture`.
	#[cfg(feature = "gpu-bake")]
	#[test]
	#[ignore]
	fn gpu_bake_matches_dust2s_cached_cpu_bake() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let config = config(&path).unwrap();
		let bytes = std::fs::read(&path).unwrap();
		let cpu = lightbake::Baked::from_bytes(
			&std::fs::read(cache_path(&path)).unwrap(),
			config.cache_key(&bytes),
		)
		.unwrap();
		let start = std::time::Instant::now();
		let gpu = lightbake::bake_gpu(
			&triangles(&map),
			&albedo(&map),
			&face_charts(&map.render, &config),
			&tri_chart(&map.render),
			&config,
		)
		.unwrap();
		println!("de_dust2 GPU bake: {:?}", start.elapsed());
		assert_eq!(gpu.charts(), cpu.charts());
		let (side, gpu_texels) = gpu.atlas();
		let (_, cpu_texels) = cpu.atlas();
		let mut flipped = 0;
		let mut texels = 0;
		// Diagnostics first: how far apart the two bakes land per chart,
		// and where, before any single chart fails an assertion.
		let mut worst = (0.0f32, 0usize);
		let mut over = 0usize;
		for face in 0..map.render.faces.len() {
			let (Some(a), Some(b)) = (cpu.chart(face), gpu.chart(face)) else {
				continue;
			};
			assert_eq!(a.rect, b.rect, "face {face} packs differently");
			let mut mean = (vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));
			let mut chart_flips = 0;
			for j in 0..a.rect.h {
				for i in 0..a.rect.w {
					let at = ((a.rect.y + j) * side + a.rect.x + i) as usize;
					mean.0 += vec3(cpu_texels[at][0], cpu_texels[at][1], cpu_texels[at][2]);
					mean.1 += vec3(gpu_texels[at][0], gpu_texels[at][1], gpu_texels[at][2]);
					texels += 1;
					if cpu_texels[at][3] != gpu_texels[at][3] {
						flipped += 1;
						chart_flips += 1;
					}
				}
			}
			let n = (a.rect.w * a.rect.h) as f32;
			let (cpu_mean, gpu_mean) = (mean.0 / n, mean.1 / n);
			let d = (cpu_mean - gpu_mean).length() / (cpu_mean.length() + 1e-6);
			if d > worst.0 {
				worst = (d, face);
			}
			if d > 0.1 {
				over += 1;
				println!(
					"chart {face}: {d:.2} gap, {} of {} texels flip sun visibility",
					chart_flips,
					a.rect.w * a.rect.h
				);
			}
		}
		println!(
			"worst mean gap {:.1}% on face {}, {} charts over 10%, {} of {} vis flips",
			worst.0 * 100.0,
			worst.1,
			over,
			flipped,
			texels
		);
		// The knife-edge charts from `gpu_direct_bake_matches_cpu_on_dust2`
		// carry over; here their one flipped direct ray also perturbs what
		// the bounce passes gather, so the same handful stays out of line.
		assert!(
			over < cpu.charts() / 100,
			"{over} charts differ by more than 10%"
		);
		// Sun visibility is deterministic; the flips left are rays that
		// graze a shared edge and fall on either side by rounding.
		assert!(
			(flipped as f32) < texels as f32 * 0.01,
			"{flipped} of {texels} texels disagree on sun visibility"
		);
	}

	/// A real lightmap bake of the whole map, and the cache it leaves behind.
	/// Not part of CI: `cargo test --release -p engine -- --ignored bake`
	/// runs it with the BVH speed; the first run bakes and writes the cache,
	/// later runs load it.
	#[test]
	#[ignore]
	fn bakes_de_dust2() {
		let Some(path) = dust2() else {
			return;
		};
		let map = crate::bsp::load(&path).unwrap();
		let start = std::time::Instant::now();
		let (baked, cached) = bake_or_load(&map, &path).unwrap();
		println!(
			"de_dust2: {} charts, {}x{} atlas, cached: {cached}, {:?}",
			baked.charts(),
			baked.atlas().0,
			baked.atlas().0,
			start.elapsed()
		);
		if !cached {
			let start = std::time::Instant::now();
			let (_, cached) = bake_or_load(&map, &path).unwrap();
			assert!(cached, "a fresh bake must load back from its cache");
			println!("cache load: {:?}", start.elapsed());
		}
	}
}
