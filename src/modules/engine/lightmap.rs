//! The lightbake path for BSP maps: per-triangle albedo from the map's own
//! texture atlas, the sun from `light_environment` and the map's `light`
//! and `light_spot` entities, fed to `lightbake`. The textured path draws
//! the map's own lightdata; this path is our baked light instead, so
//! destruction can rebake (`Baked::update`).

use crate::bsp::{Atlas, Map, MapRender, Rect};
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

/// Sun direction and color from the map file's `light_environment`, or the
/// neutral default `parse` uses without one. GoldSrc pitch counts down from
/// the horizon, so the sun sits `-pitch` above it; `_light` is
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

/// The sun inside one `light_environment` block.
fn sun_from(block: &str) -> Option<(Vec3, Vec3)> {
	let angles: Vec<f32> = crate::bsp::field(block, "angles")?
		.split_whitespace()
		.filter_map(|t| t.parse().ok())
		.collect();
	let pitch: f32 = crate::bsp::field(block, "pitch")
		.and_then(|p| p.parse().ok())
		.unwrap_or(*angles.first()?);
	let yaw = *angles.get(1)?;
	let light: Vec<f32> = crate::bsp::field(block, "_light")?
		.split_whitespace()
		.filter_map(|t| t.parse().ok())
		.collect();
	let channel = |i: usize| light.get(i).copied().unwrap_or(255.0) / 255.0;
	let bright = light.get(3).copied().unwrap_or(100.0) / 100.0;
	Some((
		goldsrc_dir(-pitch, yaw),
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
			let angles: Vec<f32> = crate::bsp::field(block, "angles")
				.map(|a| {
					a.split_whitespace()
						.filter_map(|t| t.parse().ok())
						.collect()
				})
				.unwrap_or_default();
			let pitch = crate::bsp::field(block, "pitch")
				.and_then(|p| p.parse().ok())
				.or_else(|| angles.first().copied())
				.unwrap_or(-90.0);
			let yaw = angles.get(1).copied().unwrap_or(0.0);
			// Spot pitch counts down from the horizon: -90 shines straight down.
			let dir = goldsrc_dir(pitch, yaw);
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

/// The bake's config for a map: its own sun and sky, at a density the
/// BSP's lightdata never had — 4 units per texel (the 4096 atlas, the
/// largest the vertex packing addresses, fits no finer), 32 rays and 2
/// bounces, sharp up close and smooth in the shade.
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
		units_per_texel: 4.0,
		rays: 32,
		bounces: 2,
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

/// Bakes the map with `lightbake`: our direct sun, sky and bounces, ready
/// for partial rebakes when the map's geometry changes.
pub fn bake(map: &Map, path: &std::path::Path) -> Result<lightbake::Baked, String> {
	let tris = triangles(map);
	let albedo = albedo(map);
	if albedo.len() != tris.len() {
		return Err(format!(
			"{} rendered triangles but {} albedos",
			tris.len(),
			albedo.len()
		));
	}
	lightbake::bake(&tris, &albedo, &config(path)?)
}

/// Where `path`'s bake is cached: the map's own name, extension swapped.
fn cache_path(path: &std::path::Path) -> std::path::PathBuf {
	path.with_extension("lightbake")
}

/// Bakes the map, or loads the bake cached beside it when the map's bytes and
/// the config it was baked with both match. Returns the bake and whether it
/// came from the cache; a cache that cannot be written is ignored, one that
/// cannot be read falls back to a fresh bake. `bake` is the bake itself, so
/// the CPU and GPU paths share every other step.
fn bake_or_load_with(
	map: &Map,
	path: &std::path::Path,
	bake: impl FnOnce(
		&[lightbake::Triangle],
		&[Vec3],
		&lightbake::Config,
	) -> Result<lightbake::Baked, String>,
) -> Result<(lightbake::Baked, bool), String> {
	let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
	let config = config(path)?;
	let key = config.cache_key(&bytes);
	if let Ok(cached) = std::fs::read(cache_path(path)) {
		if let Ok(baked) = lightbake::Baked::from_bytes(&cached, key) {
			return Ok((baked, true));
		}
	}
	let tris = triangles(map);
	let albedo = albedo(map);
	if albedo.len() != tris.len() {
		return Err(format!(
			"{} rendered triangles but {} albedos",
			tris.len(),
			albedo.len()
		));
	}
	let baked = bake(&tris, &albedo, &config)?;
	// A read-only map directory just rebakes next time.
	let _ = std::fs::write(cache_path(path), baked.to_bytes(key));
	Ok((baked, false))
}

/// Bakes the map with `lightbake`: our direct sun, sky and bounces, ready
/// for partial rebakes when the map's geometry changes.
pub fn bake_or_load(map: &Map, path: &std::path::Path) -> Result<(lightbake::Baked, bool), String> {
	bake_or_load_with(map, path, lightbake::bake)
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
	bake_or_load_with(map, path, lightbake::bake_gpu)
}

/// The radiance one baked atlas byte spans: the atlas stores linear `S`
/// divided by this, and the shader multiplies it back, so sunlit plus
/// lamplit texels past 2.0 keep their range instead of clamping white.
pub const BAKED_SPAN: f32 = 4.0;

/// The bake's atlas as the [`Atlas`] the game's renderer and the splat viewer
/// sample. RGB is linear `S / BAKED_SPAN` in bytes; alpha, the bake's sun
/// visibility, is dropped — it is for partial rebakes, not for shading.
pub fn atlas(baked: &lightbake::Baked) -> Atlas {
	let (side, texels) = baked.atlas();
	Atlas {
		side,
		rgba: texels
			.iter()
			.flat_map(|texel| {
				[
					(texel[0] / BAKED_SPAN * 255.0).round() as u8,
					(texel[1] / BAKED_SPAN * 255.0).round() as u8,
					(texel[2] / BAKED_SPAN * 255.0).round() as u8,
					255,
				]
			})
			.collect(),
	}
}

/// One fan triangle of a face with its baked chart: everything a sampler needs
/// to light a point on the face, whose triangles each hold their own chart.
#[derive(Clone, Copy, Debug)]
pub struct FaceChart {
	/// The triangle's corners, wound as the face's fan built them.
	pub tri: [Vec3; 3],
	pub view: lightbake::ChartView,
}

impl FaceChart {
	/// Whether `p` lies inside `tri`, boundary included: fan triangles tile
	/// their face, so a point on the face falls in at least one.
	pub fn contains(&self, p: Vec3) -> bool {
		let [a, b, c] = self.tri;
		let (ab, ac, ap) = (b - a, c - a, p - a);
		let (d00, d01, d11) = (ab.dot(ab), ab.dot(ac), ac.dot(ac));
		let (d20, d21) = (ap.dot(ab), ap.dot(ac));
		let denom = d00 * d11 - d01 * d01;
		if denom.abs() < 1e-9 {
			return false;
		}
		let v = (d11 * d20 - d01 * d21) / denom;
		let w = (d00 * d21 - d01 * d20) / denom;
		let u = 1.0 - v - w;
		u >= 0.0 && v >= 0.0 && w >= 0.0
	}

	/// `p`'s chart texel, clamped into the chart as the game's vertices are.
	pub fn texel(&self, p: Vec3) -> (f32, f32) {
		let at = |axis: Vec3, min: f32, span: u32| {
			((p.dot(axis) - min) / self.view.units - 0.5).clamp(0.0, span as f32 - 1.0)
		};
		(
			at(self.view.u, self.view.u_min, self.view.rect.w),
			at(self.view.v, self.view.v_min, self.view.rect.h),
		)
	}
}

/// Each face's fan triangles with their baked charts, in fan order: the same
/// triangle numbering the bake and the renderer's meshes use, so a sampler
/// finds a face's chart for a point by the triangle containing it. Faces
/// whose triangles hold no chart come back empty.
pub fn face_charts(render: &MapRender, baked: &lightbake::Baked) -> Vec<Vec<FaceChart>> {
	let mut tri = 0;
	render
		.faces
		.iter()
		.map(|face| {
			let charts = (1..face.verts.len().saturating_sub(1))
				.filter_map(|k| {
					let corners = [face.verts[0].pos, face.verts[k].pos, face.verts[k + 1].pos];
					baked
						.chart(tri)
						.map(|view| FaceChart { tri: corners, view })
				})
				.collect();
			tri += face.verts.len().saturating_sub(2);
			charts
		})
		.collect()
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
		// de_dust2's light_environment: east, 60 degrees up, 255 255 128 at 70.
		let (dir, color) = sun(&path).unwrap();
		let want = vec3(0.5, 3.0f32.sqrt() / 2.0, 0.0);
		assert!((dir - want).length() < 1e-4, "{dir:?}");
		let want = vec3(0.7, 0.7, 0.7 * 128.0 / 255.0);
		assert!((color - want).length() < 1e-4, "{color:?}");
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

	/// Direct terms in isolation: both bakes with `bounces` zeroed. Charts
	/// agree to rounding except where a light ray grazes a blocker's edge
	/// — the shared edge is hit or missed on floating point alone, and on
	/// a skinny triangle whose chart clamps most texels to one ray, one
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
		let cpu = lightbake::bake(&tris, &albedo, &config).unwrap();
		let gpu = lightbake::bake_gpu(&tris, &albedo, &config).unwrap();
		assert_eq!(gpu.charts(), cpu.charts());
		let (side, gpu_texels) = gpu.atlas();
		let (_, cpu_texels) = cpu.atlas();
		let mut worst = (0.0f32, 0usize);
		let mut over = 0usize;
		for tri in 0..tris.len() {
			let (Some(a), Some(b)) = (cpu.chart(tri), gpu.chart(tri)) else {
				continue;
			};
			assert_eq!(a.rect, b.rect, "triangle {tri} packs differently");
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
				worst = (d, tri);
			}
			if d > 0.1 {
				over += 1;
			}
		}
		println!(
			"direct-only: worst mean gap {:.1}% on triangle {}, {} charts over 10%",
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
		let gpu = lightbake::bake_gpu(&triangles(&map), &albedo(&map), &config).unwrap();
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
		for tri in 0..map
			.render
			.faces
			.iter()
			.map(|f| f.verts.len().saturating_sub(2))
			.sum::<usize>()
		{
			let (Some(a), Some(b)) = (cpu.chart(tri), gpu.chart(tri)) else {
				continue;
			};
			assert_eq!(a.rect, b.rect, "triangle {tri} packs differently");
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
				worst = (d, tri);
			}
			if d > 0.1 {
				over += 1;
				println!(
					"chart {tri}: {d:.2} gap, {} of {} texels flip sun visibility",
					chart_flips,
					a.rect.w * a.rect.h
				);
			}
		}
		println!(
			"worst mean gap {:.1}% on triangle {}, {} charts over 10%, {} of {} vis flips",
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
