//! Baked lightmaps for static level geometry, reusable across levels: a
//! triangle soup plus per-triangle diffuse albedo and one caller-supplied
//! [`ChartSpec`] per surface goes in, one atlas comes out.
//!
//! A chart is a rectangle of atlas texels holding the diffuse light
//! arriving at one surface, on a grid the caller places: the spec names the
//! world point at texel (0, 0)'s center, the two world step vectors one
//! texel along the grid, and the surface's normal — so a renderer that
//! already maps its geometry onto a lightmap grid (a BSP face's luxels, say)
//! hands that grid to the bake and reads the texels back by the same
//! mapping. A texel stores `S` such that the lit diffuse term is
//! `kd * albedo * S`:
//!
//! - direct sun, `sun_color · max(n·l, 0) · visibility / π`, visibility raycast
//!   with the shadowing triangle remembered per texel;
//! - the sky/ground hemisphere, `mix(ground, sky, d.y · 0.5 + 0.5)` along
//!   each cosine-distributed gather ray `d` that escapes the level, so only
//!   the sky a texel sees lands: corners, overhangs and interiors darken as
//!   they close in (with zero `rays`, the open hemisphere around the normal
//!   lands unoccluded instead);
//! - the config's [`Light`]s, shadow-rayed per texel and faded by distance;
//! - `bounces` of indirect light, gathered with the same rays, whose hits
//!   sample the charts themselves: each bounce carries on only the light the
//!   one before it added, so no order of light is counted twice. The
//!   gathered light is then smoothed across neighbouring texels on the same
//!   plane, chart borders and all, so each surface's Monte-Carlo noise does
//!   not show as blotches that change face by face; direct light stays
//!   as sharp as its texels.
//!
//! Geometry edits keep triangle indices stable: destroyed geometry becomes a
//! zero-area triangle at the same index, new geometry is appended, and every
//! created, moved or destroyed index is passed to [`Baked::update`] with the
//! charts recomputed for the edited level. It rebakes only the changed
//! charts and the texels whose sunlight the edit shadowed or unshadowed, and
//! returns the atlas rectangles that changed, for a partial texture upload
//! plus a mipmap regeneration. Indirect light from surfaces beyond the edit
//! is not re-gathered elsewhere; call [`bake`] again for an exact result.

use linalg::{vec3, Vec3};
use pack::Packer;
use physics::{triangle_bounds, triangle_normal};
use std::f32::consts::PI;

mod bvh;
pub mod pack;

/// The ray-cast hierarchy the bake traces against, also packed for GPU ray
/// casters by `Bvh::gpu`.
pub use bvh::Bvh;

#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::bake_gpu;

/// A level triangle, the same shape as `physics::Triangle`, so a mesh's
/// `tris()` passes straight in.
pub use physics::Triangle;

/// No triangle shadows a texel's sun ray.
const NO_SHADOW: u32 = u32::MAX;
/// Sun and gather rays start this far off the surface, in world units, so
/// grazing rays do not hit the triangle they were cast from.
const OFFSET: f32 = 0.1;
/// How far the gathered light is smoothed, in texels of the config's
/// `units_per_texel`: the radius of the Gaussian that averages each texel
/// with its neighbours on the same plane.
const SMOOTH_TEXELS: f32 = 2.5;

/// A light source baked into the atlas besides the sun. `cone` turns it
/// into a spotlight: the emission direction (unit) and its half-angle in
/// degrees.
#[derive(Clone, Debug)]
pub struct Light {
	pub pos: Vec3,
	/// Linear radiance at the source; faded by distance from there.
	pub color: Vec3,
	/// The distance the light fades to nothing across.
	pub fade: f32,
	pub cone: Option<(Vec3, f32)>,
}

/// What to bake and at what density.
#[derive(Clone, Debug)]
pub struct Config {
	/// Atlas side in texels; a power of two, so uploaded mips regenerate.
	pub atlas_size: u32,
	/// World units one lightmap texel covers, for
	/// [`ChartSpec::fit_triangle`] and callers deriving their own grids.
	pub units_per_texel: f32,
	/// Smallest chart side, so tiny triangles still receive light.
	pub min_chart: u32,
	/// Toward the sun, normalized.
	pub sun_dir: Vec3,
	/// Linear sun radiance.
	pub sun_color: Vec3,
	/// Linear sky and ground hemisphere colors.
	pub sky_color: Vec3,
	pub ground_color: Vec3,
	/// Indirect bounces gathered per texel.
	pub bounces: usize,
	/// Gather rays per texel per bounce.
	pub rays: u32,
	/// Light sources besides the sun: point lights and spotlights.
	pub lights: Vec<Light>,
	/// Seed of the gather ray pattern.
	pub seed: u32,
}

impl Default for Config {
	fn default() -> Config {
		Config {
			atlas_size: 1024,
			units_per_texel: 16.0,
			min_chart: 4,
			sun_dir: vec3(0.4, 1.0, 0.25).normalize(),
			sun_color: vec3(3.2, 3.0, 2.7),
			sky_color: vec3(0.30, 0.38, 0.50),
			ground_color: vec3(0.16, 0.14, 0.12),
			bounces: 1,
			rays: 8,
			lights: Vec::new(),
			seed: 1,
		}
	}
}

/// A rectangle of atlas texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
	pub x: u32,
	pub y: u32,
	pub w: u32,
	pub h: u32,
}

impl Rect {
	/// This rectangle grown by one texel on every side, staying in the atlas.
	fn grown(self, side: u32) -> Rect {
		let x = self.x.saturating_sub(1);
		let y = self.y.saturating_sub(1);
		Rect {
			x,
			y,
			w: (self.w + 2).min(side - x),
			h: (self.h + 2).min(side - y),
		}
	}
}

/// One surface's chart, placed by the caller: a rectangle of atlas texels
/// and the world plane it samples. `origin` is the world point at texel
/// `(0, 0)`'s center; `u` and `v` step one texel along the grid, in world
/// units; `n` is the surface's outward unit normal, which the grid spans.
///
/// A world point `p` on the surface sits at chart texel
/// `lm(p) = (g_u · (p - origin), g_v · (p - origin))`, where `g_u` and `g_v`
/// are the dual basis [`Chart::grads`] returns; the chart's texel `(i, j)`
/// conversely centers at `origin + u · i + v · j`.
#[derive(Clone, Copy, Debug)]
pub struct ChartSpec {
	pub origin: Vec3,
	pub u: Vec3,
	pub v: Vec3,
	pub n: Vec3,
	pub w: u32,
	pub h: u32,
}

impl ChartSpec {
	/// The chart the surface's own triangle makes on its dominant world axis
	/// — the same axis choice the game renderer uses for its tiling
	/// textures — sized by the triangle's extent at the config's density,
	/// never below `min_chart` texels.
	pub fn fit_triangle(t: &Triangle, config: &Config) -> ChartSpec {
		Self::fit_points(t, triangle_normal(t), config)
	}

	/// The chart a set of surface points makes on the dominant world axis of
	/// their `n`, sized by their extent at the config's density, never
	/// below `min_chart` texels.
	pub fn fit_points(points: &[Vec3], n: Vec3, config: &Config) -> ChartSpec {
		let (u, v, axis) = dominant(n);
		let units = config.units_per_texel;
		let (mut s_lo, mut s_hi, mut t_lo, mut t_hi) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
		for p in points {
			let (s, t) = (p.dot(u), p.dot(v));
			s_lo = s_lo.min(s);
			s_hi = s_hi.max(s);
			t_lo = t_lo.min(t);
			t_hi = t_hi.max(t);
		}
		let extent = |lo: f32, hi: f32| {
			(((hi - lo) / units).ceil() as u32)
				.max(config.min_chart)
				.min(config.atlas_size - 1)
		};
		ChartSpec {
			// Texel (0, 0)'s center: half a texel in from the extent corner.
			origin: u * (s_lo + units * 0.5)
				+ v * (t_lo + units * 0.5)
				+ axis * points.first().copied().unwrap_or(Vec3::ZERO).dot(axis),
			u: u * units,
			v: v * units,
			n,
			w: extent(s_lo, s_hi),
			h: extent(t_lo, t_hi),
		}
	}
}

/// One chart's texels: where they sit in the atlas and the caller's plane
/// mapping onto them, as [`ChartSpec`] places it.
#[derive(Clone)]
struct Chart {
	rect: Rect,
	spec: ChartSpec,
	/// The triangle shadowing each texel's sun ray, or [`NO_SHADOW`].
	shadow: Vec<u32>,
	/// Every triangle that blocks any of the chart's texels' light rays,
	/// sun aside: destroying or moving one of these reshadows the chart.
	occluders: Vec<u32>,
}

impl Chart {
	/// The world point at the center of texel `(i, j)`.
	fn world(&self, i: u32, j: u32) -> Vec3 {
		self.spec.origin + self.spec.u * (i as f32) + self.spec.v * (j as f32)
	}

	/// The dual basis mapping world points on the chart's plane to texel
	/// coordinates: `lm_u(p) = g_u · (p - origin)`, `lm_v(p)` likewise.
	fn grads(&self) -> (Vec3, Vec3) {
		let d = self.spec.u.cross(self.spec.v).dot(self.spec.n);
		(
			self.spec.v.cross(self.spec.n).map(|x| x / d),
			self.spec.n.cross(self.spec.u).map(|x| x / d),
		)
	}

	/// The world box the chart spans: the min and max of its four corners.
	fn bounds(&self) -> (Vec3, Vec3) {
		let (w, h) = (self.rect.w as f32, self.rect.h as f32);
		let mut lo = self.world(0, 0);
		let mut hi = lo;
		for &(i, j) in &[(w, 0.0), (0.0, h), (w, h)] {
			let c = self.spec.origin + self.spec.u * i + self.spec.v * j;
			lo = lo.min(c);
			hi = hi.max(c);
		}
		(lo, hi)
	}
}

/// The atlas mapping of one surface, for a renderer building per-vertex
/// lightmap coordinates or sampling the chart by world point.
#[derive(Clone, Copy, Debug)]
pub struct ChartView {
	/// The world point at texel `(0, 0)`'s center.
	pub origin: Vec3,
	/// The world step one texel along the grid, in each direction.
	pub u: Vec3,
	pub v: Vec3,
	/// The surface's outward unit normal.
	pub n: Vec3,
	/// The chart's rectangle, atlas texels.
	pub rect: Rect,
}

impl ChartView {
	/// The dual basis mapping world points on the surface to chart texel
	/// coordinates.
	pub fn grads(&self) -> (Vec3, Vec3) {
		let d = self.u.cross(self.v).dot(self.n);
		(
			self.v.cross(self.n).map(|x| x / d),
			self.n.cross(self.u).map(|x| x / d),
		)
	}
}

/// The level geometry one bake or rebake pass runs against.
struct Scene<'a> {
	tris: &'a [Triangle],
	albedo: &'a [Vec3],
	bvh: &'a bvh::Bvh,
	/// Every ray's maximum travel.
	span: f32,
	/// Each chart's triangles, `(first, count)` into `tris`.
	chart_tris: &'a [(u32, u32)],
	/// World units one texel covers, the config's `units_per_texel`.
	units: f32,
}

/// A baked level: one atlas of radiance texels and one chart per surface.
pub struct Baked {
	config: Config,
	side: u32,
	/// `side` × `side` RGBA, row major, row 0 at v = 0. RGB is `S`, A is sun
	/// visibility. Each chart's edges are duplicated into its gutter.
	texels: Vec<[f32; 4]>,
	charts: Vec<Option<Chart>>,
	/// Each triangle's chart, so gather hits and edits find the surface they
	/// landed on.
	tri_chart: Vec<u32>,
	packer: Packer,
}

/// Bakes `tris` into one atlas under `config`. `albedo` is each triangle's
/// diffuse albedo, linear, metallic contribution already removed — indirect
/// rays treat every surface as a diffuse reflector. `charts` places one
/// chart per surface at whatever density the caller maps geometry with;
/// `tri_chart` names each triangle's chart.
pub fn bake(
	tris: &[Triangle],
	albedo: &[Vec3],
	charts: &[ChartSpec],
	tri_chart: &[u32],
	config: &Config,
) -> Result<Baked, String> {
	if tris.len() != albedo.len() {
		return Err(format!(
			"{} triangles but {} albedos",
			tris.len(),
			albedo.len()
		));
	}
	if tris.len() != tri_chart.len() {
		return Err(format!(
			"{} triangles but {} chart indices",
			tris.len(),
			tri_chart.len()
		));
	}
	if config.atlas_size.count_ones() != 1 {
		return Err(format!(
			"atlas_size {} is not a power of two",
			config.atlas_size
		));
	}
	let (mut baked, order) = packed(tris, charts, tri_chart, config)?;
	let bvh = bvh::Bvh::build(tris);
	let span = ray_span(tris);
	let chart_tris = tri_chart_ranges(tri_chart, charts.len())?;
	let scene = Scene {
		tris,
		albedo,
		bvh: &bvh,
		span,
		chart_tris: &chart_tris,
		units: config.units_per_texel,
	};
	let mut rng = config.seed;
	// Pass 0 writes direct sun and lights. Each gather pass then adds one
	// order of light, gathered from only what the pass before it added —
	// the first from the direct light, and it alone sees the sky — all of
	// a pass landing in a copy first so no chart is gathered from its own
	// half-written bounce. Zero bounces still runs the first pass, for the
	// sky, its hits gathering nothing.
	for &i in &order {
		baked.bake_chart(&scene, i, &mut rng, false);
	}
	if config.rays > 0 {
		let mut prev: Vec<Option<Vec<[f32; 4]>>> = vec![None; charts.len()];
		for &i in &order {
			prev[i] = Some(baked.chart_values(i));
		}
		let direct = prev.clone();
		for pass in 0..config.bounces.max(1) {
			let hits = pass < config.bounces;
			let mut added: Vec<Option<Vec<[f32; 4]>>> = vec![None; charts.len()];
			for &i in &order {
				added[i] =
					Some(
						baked.gather_chart(&scene, i, &mut rng, pass == 0, |hit, c, q| {
							match &prev[hit] {
								Some(v) if hits => sample_grid(c, q, |x, y| v[(y * c.rect.w + x) as usize]),
								_ => Vec3::ZERO,
							}
						}),
					);
			}
			for &i in &order {
				baked.add_values(i, added[i].as_ref().unwrap());
			}
			prev = added;
		}
		baked.smooth(&direct);
	}
	Ok(baked)
}

/// The empty [`Baked`] with every live chart packed, tallest first so the
/// shelf packer's rows stay shallow, plus the packed chart order. The CPU
/// bake and the GPU bake share this setup, so both place charts identically.
fn packed(
	tris: &[Triangle],
	charts: &[ChartSpec],
	tri_chart: &[u32],
	config: &Config,
) -> Result<(Baked, Vec<usize>), String> {
	let side = config.atlas_size;
	let ranges = tri_chart_ranges(tri_chart, charts.len())?;
	let mut baked = Baked {
		config: config.clone(),
		side,
		texels: vec![[0.0; 4]; (side * side) as usize],
		charts: vec![None; charts.len()],
		tri_chart: tri_chart.to_vec(),
		packer: Packer::new(side),
	};
	// A chart is live when it has triangles and any of them has area; the
	// rest never sample.
	let alive = |i: usize| -> bool {
		let (first, count) = ranges[i];
		count > 0
			&& tris[first as usize..(first + count) as usize]
				.iter()
				.any(|t| area(t) > 1e-12)
			&& charts[i].w > 0
			&& charts[i].h > 0
	};
	let mut order: Vec<usize> = (0..charts.len()).filter(|&i| alive(i)).collect();
	order.sort_by(|&a, &b| {
		charts[b]
			.h
			.cmp(&charts[a].h)
			.then(charts[b].w.cmp(&charts[a].w))
	});
	for &i in &order {
		baked.add_chart(i, &charts[i])?;
	}
	Ok((baked, order))
}

/// Each chart's triangles as `(first, count)` into the soup, so a chart
/// covers one contiguous run of triangles — the fan order every caller
/// builds its soup in. `Err` when an index is out of range or a chart's
/// triangles are not contiguous.
fn tri_chart_ranges(tri_chart: &[u32], charts: usize) -> Result<Vec<(u32, u32)>, String> {
	let mut out = vec![(0u32, 0u32); charts];
	for (i, &c) in tri_chart.iter().enumerate() {
		let Some(slot) = out.get_mut(c as usize) else {
			return Err(format!("triangle {i} claims chart {c} of {charts}"));
		};
		if slot.1 == 0 {
			*slot = (i as u32, 1);
		} else if slot.0 + slot.1 == i as u32 {
			slot.1 += 1;
		} else {
			return Err(format!(
				"chart {c}'s triangles are not contiguous: {i} follows {}",
				slot.0 + slot.1 - 1
			));
		}
	}
	Ok(out)
}

impl Baked {
	/// The atlas: its side and the RGBA texels, ready for a float texture.
	pub fn atlas(&self) -> (u32, &[[f32; 4]]) {
		(self.side, &self.texels)
	}

	/// How many surfaces hold a chart.
	pub fn charts(&self) -> usize {
		self.charts.iter().filter(|c| c.is_some()).count()
	}

	/// Surface `chart`'s atlas mapping, `None` for surfaces without one —
	/// past the bake's last surface too, as geometry added since has none.
	pub fn chart(&self, chart: usize) -> Option<ChartView> {
		let c = self.charts.get(chart)?.as_ref()?;
		Some(ChartView {
			origin: c.spec.origin,
			u: c.spec.u,
			v: c.spec.v,
			n: c.spec.n,
			rect: c.rect,
		})
	}

	/// Normalized atlas coordinates of world point `p` on surface `chart`,
	/// clamped into its chart, at texel centers.
	pub fn uv(&self, chart: usize, p: Vec3) -> Option<[f32; 2]> {
		let c = self.charts[chart].as_ref()?;
		let (g_u, g_v) = c.grads();
		let rel = p - c.spec.origin;
		let (x, y) = (
			g_u.dot(rel).clamp(0.0, c.rect.w as f32 - 1.0),
			g_v.dot(rel).clamp(0.0, c.rect.h as f32 - 1.0),
		);
		Some([
			(c.rect.x as f32 + x + 0.5) / self.side as f32,
			(c.rect.y as f32 + y + 0.5) / self.side as f32,
		])
	}

	/// The chart under world point `p` and its baked light there, `None`
	/// away from every surface. A CPU query for gameplay lighting; the shader
	/// samples the atlas directly.
	pub fn at(&self, p: Vec3) -> Option<(usize, Vec3)> {
		for (i, chart) in self.charts.iter().enumerate() {
			let Some(c) = chart else { continue };
			let tol = 0.75 * c.spec.u.length().min(c.spec.v.length());
			if (p - c.spec.origin).dot(c.spec.n).abs() > tol {
				continue;
			}
			let (g_u, g_v) = c.grads();
			let rel = p - c.spec.origin;
			let (x, y) = (g_u.dot(rel), g_v.dot(rel));
			if x < 0.0 || x > c.rect.w as f32 - 1.0 || y < 0.0 || y > c.rect.h as f32 - 1.0 {
				continue;
			}
			return Some((i, self.chart_sample(c, p)));
		}
		None
	}

	/// Rebakes after geometry edits: `changed` lists every triangle index
	/// that was created, moved, destroyed (replaced by a zero-area triangle)
	/// or appended to the soup, with `charts` and `tri_chart` recomputed for
	/// the edited level. Returns the atlas rectangles that changed, for
	/// partial texture uploads. See the crate docs for the index contract.
	pub fn update(
		&mut self,
		tris: &[Triangle],
		albedo: &[Vec3],
		charts: &[ChartSpec],
		tri_chart: &[u32],
		changed: &[usize],
	) -> Result<Vec<Rect>, String> {
		if tris.len() != albedo.len() {
			return Err(format!(
				"{} triangles but {} albedos",
				tris.len(),
				albedo.len()
			));
		}
		if tris.len() != tri_chart.len() {
			return Err(format!(
				"{} triangles but {} chart indices",
				tris.len(),
				tri_chart.len()
			));
		}
		let chart_tris = tri_chart_ranges(tri_chart, charts.len())?;
		self.charts.resize(charts.len(), None);
		self.tri_chart = tri_chart.to_vec();
		let mut changed: Vec<usize> = changed
			.iter()
			.copied()
			.filter(|&i| i < tris.len())
			.collect();
		changed.sort_unstable();
		changed.dedup();
		let bvh = bvh::Bvh::build(tris);
		let span = ray_span(tris);
		let scene = Scene {
			tris,
			albedo,
			bvh: &bvh,
			span,
			chart_tris: &chart_tris,
			units: self.config.units_per_texel,
		};
		let mut rng = self.config.seed.wrapping_mul(2_025_617).wrapping_add(1);

		// The surfaces the edit touched: alive ones rebake, dead ones free
		// their chart. A surface dies when every triangle of it did.
		let mut touched = Vec::new();
		for &i in &changed {
			let c = tri_chart[i] as usize;
			if !touched.contains(&c) {
				touched.push(c);
			}
		}
		let mut rects = Vec::new();
		let mut rebuild = Vec::new();
		for &c in &touched {
			let (first, count) = chart_tris[c];
			let alive = tris[first as usize..(first + count) as usize]
				.iter()
				.any(|t| area(t) > 1e-12);
			if alive {
				if self.charts[c].is_none() {
					self.add_chart(c, &charts[c])?;
				}
				rebuild.push(c);
			} else if let Some(chart) = self.charts[c].take() {
				self.clear_rect(&chart.rect);
				rects.push(chart.rect.grown(self.side));
			}
		}
		// Sun visibility recorded per texel names the triangle that shadowed
		// it; destroying or moving that one unshadows or reshadows the texel.
		// The per-chart occluder sets do the same for light rays.
		for (i, chart) in self.charts.iter().enumerate() {
			let Some(c) = chart else { continue };
			if !rebuild.contains(&i)
				&& (c
					.shadow
					.iter()
					.any(|&s| s != NO_SHADOW && changed.binary_search(&(s as usize)).is_ok())
					|| c
						.occluders
						.iter()
						.any(|&s| changed.binary_search(&(s as usize)).is_ok()))
			{
				rebuild.push(i);
			}
		}
		// A moved or added triangle also shadows texels whose recorded shadow
		// triangle predates it, so charts inside the swept shadow volume are
		// re-raycast too: once along the sun's rays, once per light, out to
		// the light's fade distance.
		for &i in &changed {
			if area(&tris[i]) <= 1e-12 {
				continue;
			}
			let (lo, hi) = triangle_bounds(&tris[i]);
			let sun = self.config.sun_dir;
			let mut sweeps = vec![(lo.min(lo - sun * span), hi.max(hi - sun * span))];
			for light in &self.config.lights {
				let (mut s_lo, mut s_hi) = (lo, hi);
				for p in &tris[i] {
					let to = *p - light.pos;
					let len = to.length();
					if len > 1e-6 {
						let far = *p + to / len * light.fade;
						s_lo = s_lo.min(far);
						s_hi = s_hi.max(far);
					}
				}
				sweeps.push((s_lo, s_hi));
			}
			for (j, chart) in self.charts.iter().enumerate() {
				let Some(c) = chart else { continue };
				let (a, b) = c.bounds();
				if sweeps.iter().any(|&(sw_lo, sw_hi)| {
					a.x <= sw_hi.x
						&& b.x >= sw_lo.x
						&& a.y <= sw_hi.y
						&& b.y >= sw_lo.y
						&& a.z <= sw_hi.z
						&& b.z >= sw_lo.z
				}) && !rebuild.contains(&j)
				{
					rebuild.push(j);
				}
			}
		}
		rebuild.sort_unstable();
		rebuild.dedup();
		for &i in &rebuild {
			self.bake_chart(&scene, i, &mut rng, true);
			let rect = self.charts[i].as_ref().unwrap().rect;
			rects.push(rect.grown(self.side));
		}
		Ok(rects)
	}

	/// Direct sun and point lights at one surface point: `S`, sun
	/// visibility, the triangle that shadowed the sun ray, and every triangle
	/// that blocked a light ray. The sky comes from the gather rays; only a
	/// config without any adds the open hemisphere around `n` here, so walls
	/// facing away from the sun keep their fill either way.
	fn direct(
		&self,
		s: &Scene,
		p: Vec3,
		n: Vec3,
		lights: &[&Light],
	) -> ([f32; 3], f32, u32, Vec<u32>) {
		let c = &self.config;
		let mut out = if c.rays == 0 {
			c.ground_color.lerp(c.sky_color, n.y * 0.5 + 0.5)
		} else {
			Vec3::ZERO
		};
		let mut vis = 1.0;
		let mut shadow = NO_SHADOW;
		let n_l = n.dot(c.sun_dir).max(0.0);
		if n_l > 0.0 {
			match s.bvh.cast(s.tris, p + n * OFFSET, c.sun_dir, s.span) {
				Some((i, _)) => {
					vis = 0.0;
					shadow = i as u32;
				}
				None => out += c.sun_color * (n_l / PI),
			}
		}
		let mut occluders = Vec::new();
		for light in lights {
			let (lo, occluder) = light_at(s, light, p, n);
			out += lo;
			if let Some(i) = occluder {
				occluders.push(i);
			}
		}
		([out.x, out.y, out.z], vis, shadow, occluders)
	}

	/// The light gathered at every texel of chart `chart` by cosine-
	/// distributed rays: a ray that escapes the level brings the sky when
	/// `sky`, one that hits a surface brings its albedo times
	/// `light(chart, view, point)`, the light the gather carries on from
	/// there.
	fn gather_chart(
		&self,
		s: &Scene,
		chart: usize,
		rng: &mut u32,
		sky: bool,
		light: impl Fn(usize, &Chart, Vec3) -> Vec3,
	) -> Vec<[f32; 4]> {
		let c = &self.config;
		let chart_ref = self.charts[chart].as_ref().unwrap();
		let n = chart_ref.spec.n;
		let mut out = vec![[0.0f32; 4]; (chart_ref.rect.w * chart_ref.rect.h) as usize];
		if c.rays == 0 {
			return out;
		}
		for j in 0..chart_ref.rect.h {
			for i in 0..chart_ref.rect.w {
				let p = face_point(s, s.chart_tris[chart], chart_ref.world(i, j));
				let mut sum = Vec3::ZERO;
				for _ in 0..c.rays {
					let d = cosine(n, rng);
					let o = p + n * OFFSET;
					sum += match s.bvh.cast(s.tris, o, d, s.span) {
						Some((hit, at)) => {
							let h = self.tri_chart[hit] as usize;
							self.charts[h]
								.as_ref()
								.map(|view| mul(s.albedo[hit], light(h, view, o + d * at)))
								.unwrap_or_default()
						}
						None if sky => c.ground_color.lerp(c.sky_color, d.y * 0.5 + 0.5),
						None => Vec3::ZERO,
					};
				}
				let mean = sum / c.rays as f32;
				out[(j * chart_ref.rect.w + i) as usize] = [mean.x, mean.y, mean.z, 0.0];
			}
		}
		out
	}

	/// Bakes chart `chart` in place: direct sun and point lights, plus the
	/// shadow table, and with `gather` the sky and one bounce of the atlas
	/// as it stands — every order of light it holds at once, the closest a
	/// lone chart's rebake comes to the full bake's passes.
	fn bake_chart(&mut self, s: &Scene, chart: usize, rng: &mut u32, gather: bool) {
		let (mut values, shadow, occluders) = {
			let c = &self.config;
			let chart_ref = self.charts[chart].as_ref().unwrap();
			let n = chart_ref.spec.n;
			let count = (chart_ref.rect.w * chart_ref.rect.h) as usize;
			// Only lights that can reach the chart's bounds are raycast per
			// texel; the rest are out of range for every texel of it.
			let (lo, hi) = chart_ref.bounds();
			let near: Vec<&Light> = c
				.lights
				.iter()
				.filter(|light| {
					let q = lo.max(light.pos.min(hi));
					(q - light.pos).length2() <= light.fade * light.fade
				})
				.collect();
			let mut values = vec![[0.0f32; 4]; count];
			let mut shadow = vec![NO_SHADOW; count];
			let mut occluders = Vec::new();
			for j in 0..chart_ref.rect.h {
				for i in 0..chart_ref.rect.w {
					let k = (j * chart_ref.rect.w + i) as usize;
					let p = face_point(s, s.chart_tris[chart], chart_ref.world(i, j));
					let (light, vis, sh, mut blocked) = self.direct(s, p, n, &near);
					occluders.append(&mut blocked);
					shadow[k] = sh;
					values[k] = [light[0], light[1], light[2], vis];
				}
			}
			occluders.sort_unstable();
			occluders.dedup();
			(values, shadow, occluders)
		};
		if gather {
			let bounce = self.config.bounces > 0;
			let indirect = self.gather_chart(s, chart, rng, true, |_, c, q| {
				if bounce {
					self.chart_sample(c, q)
				} else {
					Vec3::ZERO
				}
			});
			for (v, g) in values.iter_mut().zip(&indirect) {
				v[0] += g[0];
				v[1] += g[1];
				v[2] += g[2];
			}
		}
		self.commit(chart, &values, &shadow, &occluders);
	}

	/// Chart `chart`'s texels as the atlas holds them, row major, gutter
	/// aside.
	fn chart_values(&self, chart: usize) -> Vec<[f32; 4]> {
		let rect = self.charts[chart].as_ref().unwrap().rect;
		(0..rect.h)
			.flat_map(|j| (0..rect.w).map(move |i| (i, j)))
			.map(|(i, j)| self.texels[((rect.y + j) * self.side + rect.x + i) as usize])
			.collect()
	}

	/// Writes a chart's values and shadow table to the atlas, duplicating its
	/// edges into the gutter so bilinear filtering stays inside the chart.
	fn commit(&mut self, chart: usize, values: &[[f32; 4]], shadow: &[u32], occluders: &[u32]) {
		self.charts[chart].as_mut().unwrap().shadow = shadow.to_vec();
		self.charts[chart].as_mut().unwrap().occluders = occluders.to_vec();
		self.write_values(chart, values);
	}

	/// Writes a chart's values to the atlas, duplicating its edges into the
	/// gutter.
	fn write_values(&mut self, chart: usize, values: &[[f32; 4]]) {
		let rect = self.charts[chart].as_ref().unwrap().rect;
		for j in 0..rect.h {
			for i in 0..rect.w {
				let v = values[(j * rect.w + i) as usize];
				let (x, y) = (rect.x + i, rect.y + j);
				let at = |x: u32, y: u32| (y * self.side + x) as usize;
				self.texels[at(x, y)] = v;
				if i + 1 == rect.w {
					self.texels[at(x + 1, y)] = v;
				}
				if j + 1 == rect.h {
					self.texels[at(x, y + 1)] = v;
				}
				if i + 1 == rect.w && j + 1 == rect.h {
					self.texels[at(x + 1, y + 1)] = v;
				}
			}
		}
	}

	/// Smooths the gathered light — what each chart's texels hold past
	/// `direct`, its direct values — with a Gaussian over the texels near
	/// each one in the world on the same plane, whichever chart holds them,
	/// so neighbouring surfaces share one smooth field instead of each
	/// carrying its own noise. Direct light and sun visibility stay as
	/// they are.
	fn smooth(&mut self, direct: &[Option<Vec<[f32; 4]>>]) {
		let units = self.config.units_per_texel;
		let radius = SMOOTH_TEXELS * units;
		// Every texel's world point, normal and gathered light, bucketed
		// by a grid one radius wide so the neighbours are in the 27 cells
		// around a texel's own.
		struct Texel {
			p: Vec3,
			n: Vec3,
			gathered: Vec3,
		}
		let mut texels = Vec::new();
		let mut cells: std::collections::HashMap<(i32, i32, i32), Vec<u32>> = Default::default();
		let cell = |p: Vec3| {
			(
				(p.x / radius).floor() as i32,
				(p.y / radius).floor() as i32,
				(p.z / radius).floor() as i32,
			)
		};
		for (k, chart) in self.charts.iter().enumerate() {
			let (Some(c), Some(d)) = (chart, &direct[k]) else {
				continue;
			};
			let rect = c.rect;
			for j in 0..rect.h {
				for i in 0..rect.w {
					let total = self.texels[((rect.y + j) * self.side + rect.x + i) as usize];
					let d = d[(j * rect.w + i) as usize];
					let p = c.world(i, j);
					cells.entry(cell(p)).or_default().push(texels.len() as u32);
					texels.push(Texel {
						p,
						n: c.spec.n,
						gathered: vec3(total[0] - d[0], total[1] - d[1], total[2] - d[2]),
					});
				}
			}
		}
		let sigma2 = (radius * 0.5) * (radius * 0.5);
		let smoothed: Vec<Vec3> = texels
			.iter()
			.map(|t| {
				let (cx, cy, cz) = cell(t.p);
				let (mut sum, mut weight) = (Vec3::ZERO, 0.0);
				for dz in -1..=1 {
					for dy in -1..=1 {
						for dx in -1..=1 {
							let Some(near) = cells.get(&(cx + dx, cy + dy, cz + dz)) else {
								continue;
							};
							for &o in near {
								let o = &texels[o as usize];
								let rel = o.p - t.p;
								let d2 = rel.length2();
								// Only the same plane: facing alike, and off
								// it by less than a quarter texel.
								if d2 > radius * radius || t.n.dot(o.n) < 0.9 || rel.dot(t.n).abs() > units * 0.25 {
									continue;
								}
								let w = (-d2 / (2.0 * sigma2)).exp();
								sum += o.gathered * w;
								weight += w;
							}
						}
					}
				}
				// A texel always counts itself, so the weight never vanishes.
				sum / weight
			})
			.collect();
		let mut at = 0;
		for (k, dk) in direct.iter().enumerate() {
			let (Some(c), Some(d)) = (&self.charts[k], dk) else {
				continue;
			};
			let values: Vec<[f32; 4]> = d
				.iter()
				.zip(&smoothed[at..at + d.len()])
				.map(|(d, g)| [d[0] + g.x, d[1] + g.y, d[2] + g.z, d[3]])
				.collect();
			debug_assert_eq!(d.len(), (c.rect.w * c.rect.h) as usize);
			at += d.len();
			self.write_values(k, &values);
		}
	}

	/// Adds gathered values to a chart's texels, gutters included.
	fn add_values(&mut self, chart: usize, values: &[[f32; 4]]) {
		let rect = self.charts[chart].as_ref().unwrap().rect;
		for j in 0..rect.h {
			for i in 0..rect.w {
				let v = values[(j * rect.w + i) as usize];
				let (x, y) = (rect.x + i, rect.y + j);
				let at = |x: u32, y: u32| (y * self.side + x) as usize;
				for t in [0, 1, 2] {
					self.texels[at(x, y)][t] += v[t];
					if i + 1 == rect.w {
						self.texels[at(x + 1, y)][t] += v[t];
					}
					if j + 1 == rect.h {
						self.texels[at(x, y + 1)][t] += v[t];
					}
					if i + 1 == rect.w && j + 1 == rect.h {
						self.texels[at(x + 1, y + 1)][t] += v[t];
					}
				}
			}
		}
	}

	/// Zeroes a freed chart's texels, gutter included.
	fn clear_rect(&mut self, rect: &Rect) {
		for y in rect.y..rect.y + rect.h + 1 {
			for x in rect.x..rect.x + rect.w + 1 {
				self.texels[(y * self.side + x) as usize] = [0.0; 4];
			}
		}
	}

	/// Packs and stores chart `chart`, as the caller's spec places it.
	fn add_chart(&mut self, chart: usize, spec: &ChartSpec) -> Result<(), String> {
		let Some((x, y)) = self.packer.place(spec.w, spec.h) else {
			return Err(format!(
				"atlas {}×{} full: chart {chart} needs a {}×{} rectangle",
				self.side, self.side, spec.w, spec.h
			));
		};
		self.charts[chart] = Some(Chart {
			rect: Rect {
				x,
				y,
				w: spec.w,
				h: spec.h,
			},
			spec: *spec,
			shadow: vec![NO_SHADOW; (spec.w * spec.h) as usize],
			occluders: Vec::new(),
		});
		Ok(())
	}

	/// Bilinear sample of the atlas at world point `p` as seen by chart `c`.
	fn chart_sample(&self, c: &Chart, p: Vec3) -> Vec3 {
		sample_grid(c, p, |i, j| {
			self.texels[((c.rect.y + j) * self.side + c.rect.x + i) as usize]
		})
	}

	/// The bake as bytes for a cache file: the atlas, every chart with its
	/// shadow table, the triangle-to-chart map, and the packer cursor, tagged
	/// with `key`, the caller's identity for the level and config baked.
	/// Atlas texels store rounded f16 — half the bytes, error below one part
	/// in two thousand, past what a tone mapped lightmap shows.
	pub fn to_bytes(&self, key: [u8; 16]) -> Vec<u8> {
		let mut out = Vec::with_capacity(48 + self.texels.len() * 8);
		out.extend_from_slice(CACHE_MAGIC);
		out.extend_from_slice(&key);
		self.config.write(&mut out);
		out.extend_from_slice(&self.side.to_le_bytes());
		for v in self.packer.save() {
			out.extend_from_slice(&v.to_le_bytes());
		}
		out.extend_from_slice(&(self.texels.len() as u64).to_le_bytes());
		for texel in &self.texels {
			for c in texel {
				out.extend_from_slice(&f16_bits(*c).to_le_bytes());
			}
		}
		out.extend_from_slice(&(self.charts.len() as u64).to_le_bytes());
		for chart in &self.charts {
			let Some(c) = chart else {
				out.push(0);
				continue;
			};
			out.push(1);
			let Rect { x, y, w, h } = c.rect;
			for v in [x, y, w, h] {
				out.extend_from_slice(&v.to_le_bytes());
			}
			for v in [c.spec.origin, c.spec.u, c.spec.v, c.spec.n] {
				for k in [v.x, v.y, v.z] {
					out.extend_from_slice(&k.to_bits().to_le_bytes());
				}
			}
			out.extend_from_slice(&(c.shadow.len() as u64).to_le_bytes());
			for s in &c.shadow {
				out.extend_from_slice(&s.to_le_bytes());
			}
			out.extend_from_slice(&(c.occluders.len() as u64).to_le_bytes());
			for s in &c.occluders {
				out.extend_from_slice(&s.to_le_bytes());
			}
		}
		out.extend_from_slice(&(self.tri_chart.len() as u64).to_le_bytes());
		for c in &self.tri_chart {
			out.extend_from_slice(&c.to_le_bytes());
		}
		out
	}

	/// The bake [`to_bytes`](Self::to_bytes) wrote, `Err` unless the magic
	/// matches, the key is the one it was tagged with, and every field reads
	/// whole from the bytes.
	pub fn from_bytes(bytes: &[u8], key: [u8; 16]) -> Result<Baked, String> {
		let mut read = Read { bytes, at: 0 };
		if read.take(CACHE_MAGIC.len(), "magic")? != CACHE_MAGIC {
			return Err("not a lightbake cache".into());
		}
		if read.take(16, "key")? != key {
			return Err("baked for another level or config".into());
		}
		let config = Config::read(&mut read)?;
		let side = read.u32("atlas side")?;
		if side != config.atlas_size || side.count_ones() != 1 || side > CACHE_MAX_SIDE {
			return Err(format!("atlas side {side} disagrees with the config"));
		}
		let packer = Packer::restore([
			read.u32("packer size")?,
			read.u32("packer x")?,
			read.u32("packer y")?,
			read.u32("packer row")?,
		]);
		let count = read.u64("texel count")?;
		if count != side as u64 * side as u64 {
			return Err(format!("{count} texels in a {side}x{side} atlas"));
		}
		let mut texels = Vec::with_capacity(count as usize);
		for _ in 0..count {
			let mut texel = [0.0f32; 4];
			for c in &mut texel {
				*c = f32_from(read.u16("texel")?);
			}
			texels.push(texel);
		}
		let charts = read.u64("chart count")? as usize;
		let mut out = Vec::with_capacity(charts);
		for _ in 0..charts {
			if read.u8("chart marker")? == 0 {
				out.push(None);
				continue;
			}
			let rect = Rect {
				x: read.u32("chart x")?,
				y: read.u32("chart y")?,
				w: read.u32("chart w")?,
				h: read.u32("chart h")?,
			};
			let vec = |read: &mut Read| -> Result<Vec3, String> {
				Ok(vec3(read.f32("x")?, read.f32("y")?, read.f32("z")?))
			};
			out.push(Some(Chart {
				rect,
				spec: ChartSpec {
					origin: vec(&mut read)?,
					u: vec(&mut read)?,
					v: vec(&mut read)?,
					n: vec(&mut read)?,
					w: rect.w,
					h: rect.h,
				},
				shadow: {
					let n = read.u64("shadow count")?;
					if n > read.remaining() / 4 {
						return Err("truncated cache: shadow table".into());
					}
					(0..n)
						.map(|_| read.u32("shadow"))
						.collect::<Result<Vec<_>, _>>()?
				},
				occluders: {
					let n = read.u64("occluder count")?;
					if n > read.remaining() / 4 {
						return Err("truncated cache: occluder table".into());
					}
					(0..n)
						.map(|_| read.u32("occluder"))
						.collect::<Result<Vec<_>, _>>()?
				},
			}));
		}
		let tri_count = read.u64("chart index count")?;
		if tri_count > read.remaining() / 4 {
			return Err("truncated cache: chart index table".into());
		}
		let tri_chart = (0..tri_count)
			.map(|_| read.u32("chart index"))
			.collect::<Result<Vec<_>, _>>()?;
		for (i, &c) in tri_chart.iter().enumerate() {
			if c as usize >= out.len() {
				return Err(format!("triangle {i} claims chart {c} of {}", out.len()));
			}
		}
		if read.at != bytes.len() {
			return Err("trailing bytes in lightbake cache".into());
		}
		Ok(Baked {
			config,
			side,
			texels,
			charts: out,
			tri_chart,
			packer,
		})
	}
}

/// The cache file's signature; the digit is the format version.
const CACHE_MAGIC: &[u8] = b"lightbake-cache-7\n";
/// Largest atlas side a cache may claim, so a corrupt length cannot ask for a
/// wild allocation before the bytes run out.
const CACHE_MAX_SIDE: u32 = 16384;

impl Config {
	/// Every field, little endian, floats as raw bits: the same bytes both go
	/// into a file and hash into a [`cache_key`](Self::cache_key).
	fn write(&self, out: &mut Vec<u8>) {
		out.extend_from_slice(&self.atlas_size.to_le_bytes());
		out.extend_from_slice(&self.units_per_texel.to_bits().to_le_bytes());
		out.extend_from_slice(&self.min_chart.to_le_bytes());
		for v in [
			self.sun_dir,
			self.sun_color,
			self.sky_color,
			self.ground_color,
		] {
			for c in [v.x, v.y, v.z] {
				out.extend_from_slice(&c.to_bits().to_le_bytes());
			}
		}
		out.extend_from_slice(&(self.bounces as u64).to_le_bytes());
		out.extend_from_slice(&self.rays.to_le_bytes());
		out.extend_from_slice(&(self.lights.len() as u64).to_le_bytes());
		for light in &self.lights {
			for v in [light.pos, light.color] {
				for c in [v.x, v.y, v.z] {
					out.extend_from_slice(&c.to_bits().to_le_bytes());
				}
			}
			out.extend_from_slice(&light.fade.to_bits().to_le_bytes());
			match light.cone {
				None => out.push(0),
				Some((dir, deg)) => {
					out.push(1);
					for c in [dir.x, dir.y, dir.z] {
						out.extend_from_slice(&c.to_bits().to_le_bytes());
					}
					out.extend_from_slice(&deg.to_bits().to_le_bytes());
				}
			}
		}
		out.extend_from_slice(&self.seed.to_le_bytes());
	}

	fn read(read: &mut Read) -> Result<Config, String> {
		let mut config = Config {
			atlas_size: read.u32("atlas size")?,
			units_per_texel: read.f32("units per texel")?,
			min_chart: read.u32("min chart")?,
			sun_dir: vec3(read.f32("sun x")?, read.f32("sun y")?, read.f32("sun z")?),
			sun_color: vec3(
				read.f32("sun color r")?,
				read.f32("sun color g")?,
				read.f32("sun color b")?,
			),
			sky_color: vec3(read.f32("sky r")?, read.f32("sky g")?, read.f32("sky b")?),
			ground_color: vec3(
				read.f32("ground r")?,
				read.f32("ground g")?,
				read.f32("ground b")?,
			),
			bounces: read.u64("bounces")? as usize,
			rays: read.u32("rays")?,
			lights: Vec::new(),
			// The seed trails the light table in `write`; read it back after.
			seed: 0,
		};
		let count = read.u64("light count")?;
		if count > read.remaining() / 28 {
			return Err("truncated cache: light table".into());
		}
		for _ in 0..count {
			let pos = vec3(
				read.f32("light x")?,
				read.f32("light y")?,
				read.f32("light z")?,
			);
			let color = vec3(
				read.f32("light color r")?,
				read.f32("light color g")?,
				read.f32("light color b")?,
			);
			let fade = read.f32("light fade")?;
			let cone = if read.u8("light cone marker")? == 0 {
				None
			} else {
				Some((
					vec3(
						read.f32("cone dir x")?,
						read.f32("cone dir y")?,
						read.f32("cone dir z")?,
					),
					read.f32("cone degrees")?,
				))
			};
			config.lights.push(Light {
				pos,
				color,
				fade,
				cone,
			});
		}
		config.seed = read.u32("seed")?;
		Ok(config)
	}

	/// The cache key of this config over `level`: a 128-bit identity covering
	/// every config field and the level's bytes, so a changed map or config
	/// rebakes instead of loading stale light. Two FNV-1a lanes with different
	/// seeds carry the two halves.
	pub fn cache_key(&self, level: &[u8]) -> [u8; 16] {
		let mut config = Vec::new();
		self.write(&mut config);
		let lane = |seed: u64| {
			level
				.iter()
				.chain(&config)
				.fold(seed, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
		};
		let mut key = [0u8; 16];
		key[..8].copy_from_slice(&lane(0xcbf29ce484222325).to_le_bytes());
		key[8..].copy_from_slice(&lane(0x6c62272e07bb0142).to_le_bytes());
		key
	}
}

/// A bounds-checked cursor over cache bytes; `what` names the field in errors.
struct Read<'a> {
	bytes: &'a [u8],
	at: usize,
}

impl Read<'_> {
	fn take(&mut self, n: usize, what: &str) -> Result<&[u8], String> {
		let end = self
			.at
			.checked_add(n)
			.filter(|&end| end <= self.bytes.len())
			.ok_or_else(|| format!("truncated cache: {what}"))?;
		let out = &self.bytes[self.at..end];
		self.at = end;
		Ok(out)
	}

	fn remaining(&self) -> u64 {
		(self.bytes.len() - self.at) as u64
	}

	fn u8(&mut self, what: &str) -> Result<u8, String> {
		Ok(self.take(1, what)?[0])
	}

	fn u16(&mut self, what: &str) -> Result<u16, String> {
		Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
	}

	fn u32(&mut self, what: &str) -> Result<u32, String> {
		Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
	}

	fn u64(&mut self, what: &str) -> Result<u64, String> {
		Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
	}

	fn f32(&mut self, what: &str) -> Result<f32, String> {
		Ok(f32::from_bits(self.u32(what)?))
	}
}

/// Rounded f16 bits of `v`: half the bytes for a cache, every value within
/// one part in two thousand of the f32 it came from. Light smaller than an
/// f16's smallest subnormal is black anyway.
pub fn f16_bits(v: f32) -> u16 {
	let bits = v.to_bits();
	let sign = ((bits >> 16) & 0x8000) as u16;
	if v.is_nan() {
		return sign | 0x7e00;
	}
	let exp = ((bits >> 23) & 0xff) as i32 - 127;
	if exp > 15 {
		return sign | 0x7bff; // saturate at f16's largest finite
	}
	if exp < -14 {
		return sign; // rounds to zero
	}
	let mant = bits & 0x7f_ffff;
	// Round to nearest even: half an ulp, one part in two thousand.
	let rounded = (mant + 0x0fff + ((mant >> 13) & 1)) >> 13;
	if exp == 15 && rounded > 0x3ff {
		return sign | 0x7bff; // rounds past f16's largest finite
	}
	sign | (((exp + 15) as u16) << 10) | rounded as u16
}

/// The f32 a [`f16_bits`] value stands for.
fn f32_from(h: u16) -> f32 {
	let sign = ((h >> 15) as u32) << 31;
	let exp = ((h >> 10) & 0x1f) as i32;
	let mant = (h & 0x3ff) as u32;
	if exp == 0 {
		let v = (mant as f32) * (-24.0f32).exp2();
		return if sign == 0 { v } else { -v };
	}
	if exp == 31 {
		return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
	}
	f32::from_bits(sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13))
}

/// Bilinear sample at world point `p` of chart `c`'s grid, whose texel
/// `(i, j)` `texel` reads: the atlas, or a pass's values for the chart
/// alone.
fn sample_grid(c: &Chart, p: Vec3, texel: impl Fn(u32, u32) -> [f32; 4]) -> Vec3 {
	let (g_u, g_v) = c.grads();
	let rel = p - c.spec.origin;
	let (x, y) = (
		g_u.dot(rel).clamp(0.0, c.rect.w as f32 - 1.0),
		g_v.dot(rel).clamp(0.0, c.rect.h as f32 - 1.0),
	);
	let (x0, y0) = (x.floor(), y.floor());
	let (tx, ty) = (x - x0, y - y0);
	let (x1, y1) = (
		(x0 + 1.0).min(c.rect.w as f32 - 1.0),
		(y0 + 1.0).min(c.rect.h as f32 - 1.0),
	);
	let row = |j: f32| {
		let (p, q) = (texel(x0 as u32, j as u32), texel(x1 as u32, j as u32));
		vec3(
			p[0] + (q[0] - p[0]) * tx,
			p[1] + (q[1] - p[1]) * tx,
			p[2] + (q[2] - p[2]) * tx,
		)
	};
	row(y0).lerp(row(y1), ty)
}

/// Component-wise product, what a diffuse reflector does to arriving light.
fn mul(a: Vec3, b: Vec3) -> Vec3 {
	vec3(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// The nearest point to `p` on the chart's own triangles, so texels the
/// chart's rectangle places outside the surface clamp onto it — and then
/// [`EDGE_INSET`] of a texel on into the triangle: a clamped point lies on the face's
/// edge, in the plane of the face meeting it there, and rays cast from it
/// along that face slip through it and light the corner from outside.
fn face_point(s: &Scene, tris: (u32, u32), p: Vec3) -> Vec3 {
	let mut best = p;
	let mut best_d = f32::INFINITY;
	let mut centroid = p;
	for t in &s.tris[tris.0 as usize..(tris.0 + tris.1) as usize] {
		let q = bvh::closest_point(p, t);
		let d = (q - p).length2();
		if d < best_d {
			best_d = d;
			best = q;
			centroid = (t[0] + t[1] + t[2]) * (1.0 / 3.0);
		}
	}
	inset(p, best, centroid, EDGE_INSET * s.units)
}

/// How far a texel clamped onto its face's edge is moved on into the face,
/// in texels: a sixteenth, one unit on a GoldSrc luxel.
const EDGE_INSET: f32 = 1.0 / 16.0;

/// `q`, the face point nearest `p`, moved `by` world units toward its
/// triangle's `centroid` when `p` lay off the face — never past halfway.
fn inset(p: Vec3, q: Vec3, centroid: Vec3, by: f32) -> Vec3 {
	if (q - p).length2() < 1e-6 {
		return q;
	}
	let to = centroid - q;
	let d = to.length();
	if d < 1e-6 {
		return q;
	}
	q + to * (by.min(d * 0.5) / d)
}

/// One light's diffuse radiance arriving at `p`, and the triangle that
/// blocks it, if any: shadow-rayed against the geometry, faded to nothing
/// across its `fade` distance, and cut off outside a spotlight's cone.
fn light_at(s: &Scene, light: &Light, p: Vec3, n: Vec3) -> (Vec3, Option<u32>) {
	let to = light.pos - p;
	let d = to.length();
	if d >= light.fade || d < 1e-4 {
		return (Vec3::ZERO, None);
	}
	let l = to / d;
	let n_l = n.dot(l).max(0.0);
	if n_l == 0.0 {
		return (Vec3::ZERO, None);
	}
	if let Some((dir, deg)) = light.cone {
		if (-l).dot(dir) < (deg * PI / 180.0).cos() {
			return (Vec3::ZERO, None);
		}
	}
	// Shadowed when anything stands between the texel and the light.
	if let Some((i, _)) = s.bvh.cast(s.tris, p + n * OFFSET, l, (d - OFFSET).max(0.0)) {
		return (Vec3::ZERO, Some(i as u32));
	}
	let fade = 1.0 - d / light.fade;
	(light.color * (n_l * fade * fade / PI), None)
}

/// Twice the soup's diagonal: every ray's maximum travel.
fn ray_span(tris: &[Triangle]) -> f32 {
	let (lo, hi) = bvh::soup_bounds(tris);
	(hi - lo).length() * 2.0
}

fn area(t: &Triangle) -> f32 {
	(t[1] - t[0]).cross(t[2] - t[0]).length() * 0.5
}

/// The world axes a triangle's chart projects on, and its dominant axis: the
/// same choice the game renderer makes for its tiling textures.
fn dominant(n: Vec3) -> (Vec3, Vec3, Vec3) {
	let a = n.map(f32::abs);
	if a.y >= a.x && a.y >= a.z {
		(Vec3::X, Vec3::Z, Vec3::Y)
	} else if a.x >= a.z {
		(Vec3::Z, Vec3::Y, Vec3::X)
	} else {
		(Vec3::X, Vec3::Y, Vec3::Z)
	}
}

/// A cosine-distributed direction around `n` from `rng`.
fn cosine(n: Vec3, rng: &mut u32) -> Vec3 {
	let (r1, r2) = (rand(rng), rand(rng));
	let (phi, cos) = (2.0 * PI * r1, (1.0 - r2).sqrt());
	let sin = r2.sqrt();
	let a = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
	let t = (a - n * a.dot(n)).normalize();
	let b = n.cross(t);
	t * (phi.cos() * sin) + n * cos + b * (phi.sin() * sin)
}

fn rand(rng: &mut u32) -> f32 {
	*rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
	(*rng >> 8) as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
	use super::*;

	fn quad(tris: &mut Vec<Triangle>, a: Vec3, b: Vec3, c: Vec3, d: Vec3, outward: Vec3) {
		for t in [[a, b, c], [a, c, d]] {
			let n = (t[1] - t[0]).cross(t[2] - t[0]);
			tris.push(if n.dot(outward) >= 0.0 {
				t
			} else {
				[t[0], t[2], t[1]]
			});
		}
	}

	/// An 8×8 floor of 1-unit tiles, `2 * n * n` triangles.
	fn floor_tiles(tris: &mut Vec<Triangle>, n: usize) {
		for i in 0..n {
			for k in 0..n {
				let (x, z) = (i as f32, k as f32);
				quad(
					tris,
					vec3(x, 0.0, z),
					vec3(x + 1.0, 0.0, z),
					vec3(x + 1.0, 0.0, z + 1.0),
					vec3(x, 0.0, z + 1.0),
					vec3(0.0, 1.0, 0.0),
				);
			}
		}
	}

	/// Floor tiles and a 2-high slab standing on them at x = 4, z in
	/// [2, 5]. The sun slants toward +x, so the slab's shadow falls on the
	/// floor at x in [2, 4], z in [2, 5] — wider than the [3, 4] band the
	/// tests read, so a neighbouring tile's chart, whose edge texels reach
	/// into the band, reads shadow there too. Zero sky, zero bounces:
	/// every lit texel is pure direct sun. Returns the slab's triangle
	/// indices.
	fn shadow_scene() -> (Vec<Triangle>, Vec<usize>) {
		let mut tris = Vec::new();
		floor_tiles(&mut tris, 8);
		let slab = tris.len();
		quad(
			&mut tris,
			vec3(4.0, 0.0, 2.0),
			vec3(4.0, 2.0, 2.0),
			vec3(4.0, 2.0, 5.0),
			vec3(4.0, 0.0, 5.0),
			vec3(1.0, 0.0, 0.0),
		);
		let count = tris.len();
		(tris, (slab..count).collect())
	}

	fn config() -> Config {
		Config {
			atlas_size: 128,
			units_per_texel: 1.0,
			min_chart: 2,
			sun_dir: vec3(1.0, 1.0, 0.0).normalize(),
			sun_color: vec3(1.0, 1.0, 1.0),
			sky_color: Vec3::ZERO,
			ground_color: Vec3::ZERO,
			bounces: 0,
			rays: 4,
			lights: Vec::new(),
			seed: 1,
		}
	}

	fn white(tris: &[Triangle]) -> Vec<Vec3> {
		vec![vec3(0.8, 0.8, 0.8); tris.len()]
	}

	/// One chart per triangle, the old default the convenience fit makes.
	fn specs(tris: &[Triangle], config: &Config) -> Vec<ChartSpec> {
		tris
			.iter()
			.map(|t| ChartSpec::fit_triangle(t, config))
			.collect()
	}

	/// Every triangle its own chart, in soup order.
	fn identity(n: usize) -> Vec<u32> {
		(0..n as u32).collect()
	}

	/// The shadow_scene baked, one chart per triangle.
	fn baked(tris: &[Triangle], config: &Config) -> Baked {
		bake(
			tris,
			&white(tris),
			&specs(tris, config),
			&identity(tris.len()),
			config,
		)
		.unwrap()
	}

	#[test]
	fn charts_pack_without_overlap() {
		let (tris, _) = shadow_scene();
		let baked = baked(&tris, &config());
		assert_eq!(baked.charts(), tris.len());
		let rects: Vec<Rect> = (0..tris.len())
			.filter_map(|i| baked.chart(i).map(|c| c.rect))
			.collect();
		for (i, &r) in rects.iter().enumerate() {
			assert!(
				r.x + r.w < 128 && r.y + r.h < 128,
				"chart {i} outside: {r:?}"
			);
			for &a in rects.iter().take(i) {
				let overlap = r.x < a.x + a.w && a.x < r.x + r.w && r.y < a.y + a.h && a.y < r.y + r.h;
				assert!(!overlap, "charts {i} and an earlier one overlap");
			}
		}
	}

	#[test]
	fn sun_direct_hits_unshadowed_floor() {
		let (tris, _) = shadow_scene();
		let baked = baked(&tris, &config());
		let (_, s) = baked.at(vec3(1.5, 0.0, 1.5)).unwrap();
		let want = config().sun_dir.y / PI;
		assert!(
			(s - vec3(want, want, want)).length() < 1e-3,
			"S {s:?}, want {want}"
		);
	}

	#[test]
	fn slab_shadows_the_floor_behind_it() {
		let (tris, _) = shadow_scene();
		let baked = baked(&tris, &config());
		let (_, lit) = baked.at(vec3(6.5, 0.0, 3.5)).unwrap();
		let (_, dark) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		let want = config().sun_dir.y / PI;
		assert!((lit.x - want).abs() < 1e-3, "lit {lit:?}");
		assert!(dark.x < 1e-3, "shadowed floor too bright: {dark:?}");
	}

	#[test]
	fn destroying_the_slab_unshadows_rebaked_texels() {
		let (tris, slab) = shadow_scene();
		let config = config();
		let mut baked = baked(&tris, &config);
		let (_, dark) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		assert!(
			dark.x < 1e-3,
			"expected shadow before destruction: {dark:?}"
		);
		// Destroyed geometry: zero-area triangles at the same indices.
		let mut edited = tris.clone();
		for &i in &slab {
			edited[i] = [vec3(0.0, -100.0, 0.0); 3];
		}
		let rects = baked
			.update(
				&edited,
				&white(&edited),
				&specs(&edited, &config),
				&identity(edited.len()),
				&slab,
			)
			.unwrap();
		assert!(!rects.is_empty(), "destruction should dirty the atlas");
		let (_, lit) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		let want = config.sun_dir.y / PI;
		assert!(
			(lit.x - want).abs() < 1e-3,
			"shadow should be gone: {lit:?}, want {want}"
		);
	}

	#[test]
	fn adding_an_occluder_shadows_rebaked_texels() {
		let mut tris = Vec::new();
		floor_tiles(&mut tris, 8);
		let config = config();
		let mut baked = baked(&tris, &config);
		let (_, lit) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		let want = config.sun_dir.y / PI;
		assert!((lit.x - want).abs() < 1e-3, "floor starts lit: {lit:?}");
		let slab = tris.len();
		quad(
			&mut tris,
			vec3(4.0, 0.0, 2.0),
			vec3(4.0, 2.0, 2.0),
			vec3(4.0, 2.0, 5.0),
			vec3(4.0, 0.0, 5.0),
			vec3(1.0, 0.0, 0.0),
		);
		let albedo = white(&tris);
		let changed = (slab..tris.len()).collect::<Vec<_>>();
		let rects = baked
			.update(
				&tris,
				&albedo,
				&specs(&tris, &config),
				&identity(tris.len()),
				&changed,
			)
			.unwrap();
		assert!(!rects.is_empty(), "the new chart should dirty the atlas");
		let (_, dark) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		assert!(dark.x < 1e-3, "new slab should shadow: {dark:?}");
	}

	/// The shadow_scene's floor with the sun and sky off and one point light
	/// over the slab's left side: only baked light is the light's.
	fn light_config() -> Config {
		let mut config = config();
		config.sun_dir = Vec3::Y;
		config.sun_color = Vec3::ZERO;
		config.lights = vec![Light {
			pos: vec3(3.5, 2.0, 3.5),
			color: vec3(1.0, 1.0, 1.0),
			fade: 10.0,
			cone: None,
		}];
		config
	}

	#[test]
	fn a_point_light_casts_a_shadow_behind_the_slab() {
		let (tris, _) = shadow_scene();
		let baked = baked(&tris, &light_config());
		let (_, lit) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		// The light one fade away at 45 degrees: the floor under it reads
		// near the analytic value; `at` may claim a neighboring chart, so
		// only the order of magnitude is asserted here.
		let want = (1.0f32 - 2.0 / 10.0).powi(2) / PI;
		assert!(
			(lit.x - want).abs() < 0.1 * want,
			"lit {lit:?}, want {want}"
		);
		let (_, dark) = baked.at(vec3(6.5, 0.0, 3.5)).unwrap();
		assert!(dark.x < 1e-3, "slab should shadow the light: {dark:?}");
	}

	#[test]
	fn destroying_the_slab_unshadows_the_point_light() {
		let (tris, slab) = shadow_scene();
		let config = light_config();
		let mut baked = baked(&tris, &config);
		let (_, dark) = baked.at(vec3(6.5, 0.0, 3.5)).unwrap();
		assert!(
			dark.x < 1e-3,
			"expected shadow before destruction: {dark:?}"
		);
		let mut edited = tris.clone();
		for &i in &slab {
			edited[i] = [vec3(0.0, -100.0, 0.0); 3];
		}
		baked
			.update(
				&edited,
				&white(&edited),
				&specs(&edited, &config),
				&identity(edited.len()),
				&slab,
			)
			.unwrap();
		let (_, lit) = baked.at(vec3(6.5, 0.0, 3.5)).unwrap();
		// Straight-line distance from the light, its fade and its cosine;
		// `at` may claim a neighboring chart, so only the order of
		// magnitude is asserted here.
		let d = (9.0f32 + 4.0).sqrt();
		let want = (1.0 - d / 10.0).powi(2) * (2.0 / d) / PI;
		assert!(
			(lit.x - want).abs() < 0.3 * want,
			"light shadow should be gone: {lit:?}, want {want}"
		);
	}

	#[test]
	fn a_spotlight_only_lights_its_cone() {
		let mut tris = Vec::new();
		floor_tiles(&mut tris, 8);
		let mut config = light_config();
		config.lights = vec![Light {
			pos: vec3(3.5, 4.0, 3.5),
			color: vec3(1.0, 1.0, 1.0),
			fade: 12.0,
			cone: Some((vec3(0.0, -1.0, 0.0), 30.0)),
		}];
		let baked = baked(&tris, &config);
		let (_, lit) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		assert!(lit.x > 0.05, "cone center should be lit: {lit:?}");
		// 45 degrees off the axis: outside a 30-degree cone.
		let (_, dark) = baked.at(vec3(3.5, 0.0, 7.5)).unwrap();
		assert!(dark.x < 1e-3, "outside the cone should be dark: {dark:?}");
	}

	#[test]
	fn open_sky_lands_evenly_across_charts() {
		// A floor of many small charts under nothing but sky, gathered with
		// a handful of rays each: every texel's own estimate is noisy, but
		// the smoothed field lands even across the chart borders.
		let mut tris = Vec::new();
		floor_tiles(&mut tris, 8);
		let mut config = config();
		config.sun_color = Vec3::ZERO;
		config.sky_color = vec3(0.3, 0.3, 0.3);
		config.rays = 4;
		let baked = baked(&tris, &config);
		let mut seen = Vec::new();
		for x in 0..12 {
			for z in 0..12 {
				let p = vec3(2.0 + x as f32 * 0.33, 0.0, 2.0 + z as f32 * 0.33);
				seen.push(baked.at(p).unwrap().1.x);
			}
		}
		let mean = seen.iter().sum::<f32>() / seen.len() as f32;
		// The cosine-weighted mean of `0.3 · (d.y · 0.5 + 0.5)` up the
		// normal: 0.3 · (2/3 · 0.5 + 0.5).
		assert!((mean - 0.25).abs() < 0.02, "mean sky {mean}");
		let spread = seen.iter().map(|s| (s - mean).abs()).fold(0.0, f32::max);
		assert!(spread < 0.02, "sky varies by {spread} around {mean}");
	}

	#[test]
	fn a_bounce_brightens_surfaces_the_sun_misses() {
		// A floor and one wall facing away from the slanted sun: no direct
		// light reaches the wall's front, only light bounced from the floor.
		let mut tris = Vec::new();
		floor_tiles(&mut tris, 8);
		quad(
			&mut tris,
			vec3(7.0, 0.0, 2.0),
			vec3(7.0, 4.0, 2.0),
			vec3(7.0, 4.0, 5.0),
			vec3(7.0, 0.0, 5.0),
			vec3(-1.0, 0.0, 0.0),
		);
		let mut config = config();
		config.bounces = 0;
		let flat = baked(&tris, &config);
		config.bounces = 1;
		config.rays = 64;
		let bounced = baked(&tris, &config);
		let (_, s) = flat.at(vec3(6.99, 2.0, 3.5)).unwrap();
		let (_, t) = bounced.at(vec3(6.99, 2.0, 3.5)).unwrap();
		assert!(s.x < 1e-3, "the sun misses the wall: {s:?}");
		assert!(t.x > s.x + 1e-3, "bounce should add light: {s:?} -> {t:?}");
	}

	#[test]
	fn uv_maps_chart_points_into_the_chart() {
		let (tris, _) = shadow_scene();
		let config = config();
		let baked = baked(&tris, &config);
		for (i, tri) in tris.iter().enumerate() {
			let Some(view) = baked.chart(i) else { continue };
			// The world point at the chart's middle, clamped onto the
			// triangle, maps to atlas coordinates inside the chart.
			let world = view.origin
				+ view.u * ((view.rect.w - 1) as f32 * 0.5)
				+ view.v * ((view.rect.h - 1) as f32 * 0.5);
			let q = bvh::closest_point(world, tri);
			let uv = baked.uv(i, q).unwrap();
			assert!(
				uv[0] * 128.0 >= view.rect.x as f32 && uv[0] * 128.0 <= (view.rect.x + view.rect.w) as f32,
				"u {} outside chart {:?}",
				uv[0],
				view.rect
			);
			assert!(
				uv[1] * 128.0 >= view.rect.y as f32 && uv[1] * 128.0 <= (view.rect.y + view.rect.h) as f32,
				"v {} outside chart {:?}",
				uv[1],
				view.rect
			);
		}
	}

	#[test]
	fn same_seed_bakes_identically() {
		let (tris, _) = shadow_scene();
		let config = config();
		let a = baked(&tris, &config);
		let b = baked(&tris, &config);
		assert_eq!(a.atlas().1, b.atlas().1);
	}

	#[test]
	fn f16_keeps_light_within_one_part_in_two_thousand() {
		for v in [1e-3, 0.25, 0.7, 1.0, 2.0, 3.2, 100.0, 65504.0] {
			let back = f32_from(f16_bits(v));
			assert!(
				(back - v).abs() <= v / 2048.0,
				"{v} became {back}, past one ulp in two thousand"
			);
		}
		assert_eq!(f32_from(f16_bits(0.0)), 0.0);
		assert_eq!(f32_from(f16_bits(1e-9)), 0.0, "f16 has no room for this");
		assert_eq!(f32_from(f16_bits(1e5)), 65504.0, "saturates");
	}

	#[test]
	fn cache_round_trips_a_bake() {
		let (tris, _) = shadow_scene();
		let mut config = config();
		config.lights = vec![
			Light {
				pos: vec3(3.5, 2.0, 3.5),
				color: vec3(1.0, 1.0, 1.0),
				fade: 10.0,
				cone: None,
			},
			Light {
				pos: vec3(2.0, 4.0, 2.0),
				color: vec3(0.5, 0.5, 0.25),
				fade: 8.0,
				cone: Some((vec3(0.0, -1.0, 0.0), 30.0)),
			},
		];
		let baked = baked(&tris, &config);
		let key = config.cache_key(b"the level");
		let bytes = baked.to_bytes(key);
		let back = Baked::from_bytes(&bytes, key).unwrap();
		assert_eq!(back.charts(), baked.charts());
		assert_eq!(back.atlas().0, baked.atlas().0);
		assert_eq!(back.tri_chart, baked.tri_chart);
		for (a, b) in back.atlas().1.iter().zip(baked.atlas().1) {
			for (a, b) in a.iter().zip(b) {
				assert!(
					(a - b).abs() <= b.abs() / 2048.0 + 1e-6,
					"texel {b} became {a}"
				);
			}
		}
		for i in 0..tris.len() {
			assert_eq!(
				back.chart(i).map(|c| c.rect),
				baked.chart(i).map(|c| c.rect)
			);
		}
		// A wrong key, a wrong config: refused, so stale light never loads.
		assert!(Baked::from_bytes(&bytes, config.cache_key(b"another level")).is_err());
		let mut other = config;
		other.units_per_texel += 1.0;
		assert!(Baked::from_bytes(&bytes, other.cache_key(b"the level")).is_err());
		let mut truncated = bytes.clone();
		truncated.truncate(truncated.len() - 1);
		assert!(Baked::from_bytes(&truncated, key).is_err());
		let mut trailing = bytes.clone();
		trailing.push(0);
		assert!(Baked::from_bytes(&trailing, key).is_err());
	}

	#[test]
	fn a_loaded_bake_rebakes_like_a_fresh_one() {
		let (tris, slab) = shadow_scene();
		let config = config();
		let first = baked(&tris, &config);
		let key = config.cache_key(b"the level");
		let mut loaded = Baked::from_bytes(&first.to_bytes(key), key).unwrap();
		// Destroyed geometry: zero-area triangles at the same indices.
		let mut edited = tris.clone();
		for &i in &slab {
			edited[i] = [vec3(0.0, -100.0, 0.0); 3];
		}
		let mut fresh = baked(&tris, &config);
		let loaded_rects = loaded
			.update(
				&edited,
				&white(&edited),
				&specs(&edited, &config),
				&identity(edited.len()),
				&slab,
			)
			.unwrap();
		let fresh_rects = fresh
			.update(
				&edited,
				&white(&edited),
				&specs(&edited, &config),
				&identity(edited.len()),
				&slab,
			)
			.unwrap();
		assert!(!loaded_rects.is_empty() && !fresh_rects.is_empty());
		let (_, lit) = loaded.at(vec3(3.5, 0.0, 3.5)).unwrap();
		let want = config.sun_dir.y / PI;
		assert!(
			(lit.x - want).abs() < 1e-2,
			"loaded bake lost its shadow table: {lit:?}, want {want}"
		);
	}
}
