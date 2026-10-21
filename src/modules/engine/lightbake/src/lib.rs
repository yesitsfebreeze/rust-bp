//! Baked lightmaps for static level geometry, reusable across levels: any
//! triangle soup plus per-triangle diffuse albedo goes in, one atlas and a
//! chart per triangle come out.
//!
//! A chart is a small rectangle of atlas texels holding the diffuse light
//! arriving at one triangle, projected on the triangle's dominant axis — the
//! same axis choice the game renderer uses for its tiling textures, so a
//! renderer maps world positions to chart texels without new vertex data. A
//! texel stores `S` such that the lit diffuse term is `kd * albedo * S`:
//!
//! - direct sun, `sun_color · max(n·l, 0) · visibility / π`, visibility raycast
//!   with the shadowing triangle remembered per texel;
//! - the sky/ground hemisphere, `mix(ground, sky, n.y · 0.5 + 0.5)`;
//! - `bounces` of indirect light, gathered with cosine-distributed rays whose
//!   hits sample the charts themselves.
//!
//! Geometry edits keep triangle indices stable: destroyed geometry becomes a
//! zero-area triangle at the same index, new geometry is appended, and every
//! created, moved or destroyed index is passed to [`Baked::update`]. It
//! rebakes only the changed charts and the texels whose sunlight the edit
//! shadowed or unshadowed, and returns the atlas rectangles that changed, for
//! a partial texture upload plus a mipmap regeneration. Indirect light from
//! surfaces beyond the edit is not re-gathered elsewhere; call [`bake`] again
//! for an exact result.

use linalg::{vec3, Vec3};
use pack::Packer;
use std::f32::consts::PI;

mod bvh;
mod pack;

/// A level triangle, the same shape as `physics::Triangle`, so a mesh's
/// `tris()` passes straight in.
pub type Triangle = [Vec3; 3];

/// No triangle shadows a texel's sun ray.
const NO_SHADOW: u32 = u32::MAX;
/// Sun and gather rays start this far off the surface, in world units, so
/// grazing rays do not hit the triangle they were cast from.
const OFFSET: f32 = 0.1;

/// What to bake and at what density.
#[derive(Clone, Debug)]
pub struct Config {
	/// Atlas side in texels; a power of two, so uploaded mips regenerate.
	pub atlas_size: u32,
	/// World units one lightmap texel covers.
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

/// One triangle's texels: where they sit in the atlas and how the triangle's
/// plane maps onto them. `u` and `v` are world axes, so a world point's chart
/// texel is `((p·u - u_min) / units - 0.5, (p·v - v_min) / units - 0.5)`.
#[derive(Clone)]
struct Chart {
	rect: Rect,
	u: Vec3,
	v: Vec3,
	/// The dominant world axis; every point of the triangle has `p·axis = plane`.
	axis: Vec3,
	plane: f32,
	u_min: f32,
	v_min: f32,
	/// The triangle shadowing each texel's sun ray, or [`NO_SHADOW`].
	shadow: Vec<u32>,
}

impl Chart {
	/// The world point at the center of texel `(i, j)`.
	fn world(&self, i: u32, j: u32, units: f32) -> Vec3 {
		let at = |k: u32, min: f32| min + (k as f32 + 0.5) * units;
		self.u * at(i, self.u_min) + self.v * at(j, self.v_min) + self.axis * self.plane
	}

	/// The world box the chart spans; `u`, `v` and `axis` are world axes, so it
	/// is axis aligned.
	fn bounds(&self, units: f32) -> (Vec3, Vec3) {
		let (u0, u1) = (self.u_min, self.u_min + self.rect.w as f32 * units);
		let (v0, v1) = (self.v_min, self.v_min + self.rect.h as f32 * units);
		let mut lo = Vec3::ZERO;
		let mut hi = Vec3::ZERO;
		for &(a, b) in &[(u0, u1), (v0, v1)] {
			lo = lo.min(self.u * a + self.v * b).min(self.u * b + self.v * a);
			hi = hi.max(self.u * a + self.v * b).max(self.u * b + self.v * a);
		}
		(lo + self.axis * self.plane, hi + self.axis * self.plane)
	}
}

/// The atlas mapping of one triangle, for a renderer building per-vertex
/// lightmap coordinates. A world point `p` on the triangle sits at chart
/// texel `((p·u - u_min) / units, (p·v - v_min) / units)`.
#[derive(Clone, Copy, Debug)]
pub struct ChartView {
	pub u: Vec3,
	pub v: Vec3,
	/// The triangle's dominant world axis, on which every point of it has
	/// `p·axis = plane`.
	pub axis: Vec3,
	pub plane: f32,
	pub u_min: f32,
	pub v_min: f32,
	/// World units per texel.
	pub units: f32,
	/// The chart's rectangle, atlas texels.
	pub rect: Rect,
}

/// The level geometry one bake or rebake pass runs against.
struct Scene<'a> {
	tris: &'a [Triangle],
	albedo: &'a [Vec3],
	bvh: &'a bvh::Bvh,
	/// Every ray's maximum travel.
	span: f32,
}

/// A baked level: one atlas of radiance texels and a chart per triangle.
pub struct Baked {
	config: Config,
	side: u32,
	/// `side` × `side` RGBA, row major, row 0 at v = 0. RGB is `S`, A is sun
	/// visibility. Each chart's edges are duplicated into its gutter.
	texels: Vec<[f32; 4]>,
	charts: Vec<Option<Chart>>,
	packer: Packer,
}

/// Bakes `tris` into one atlas under `config`. `albedo` is each triangle's
/// diffuse albedo, linear, metallic contribution already removed — indirect
/// rays treat every surface as a diffuse reflector.
pub fn bake(tris: &[Triangle], albedo: &[Vec3], config: &Config) -> Result<Baked, String> {
	if tris.len() != albedo.len() {
		return Err(format!(
			"{} triangles but {} albedos",
			tris.len(),
			albedo.len()
		));
	}
	if config.atlas_size.count_ones() != 1 {
		return Err(format!(
			"atlas_size {} is not a power of two",
			config.atlas_size
		));
	}
	let side = config.atlas_size;
	let mut baked = Baked {
		config: config.clone(),
		side,
		texels: vec![[0.0; 4]; (side * side) as usize],
		charts: vec![None; tris.len()],
		packer: Packer::new(side),
	};
	// Tallest charts first keep the shelf packer's rows shallow.
	let mut order: Vec<usize> = (0..tris.len())
		.filter(|&i| area(&tris[i]) > 1e-12)
		.collect();
	let size_of = |i: usize| chart_size(config, &tris[i]);
	order.sort_by(|&a, &b| {
		size_of(b)
			.1
			.cmp(&size_of(a).1)
			.then(size_of(b).0.cmp(&size_of(a).0))
	});
	for &i in &order {
		baked.add_chart(tris, i)?;
	}
	let bvh = bvh::Bvh::build(tris);
	let span = ray_span(tris);
	let scene = Scene {
		tris,
		albedo,
		bvh: &bvh,
		span,
	};
	let mut rng = config.seed;
	// Pass 0 writes direct sun and sky. Each later pass gathers one bounce
	// from what the earlier passes wrote, all of a pass landing in a copy
	// first so no chart is gathered from its own half-written bounce.
	for &i in &order {
		baked.bake_chart(&scene, i, &mut rng, false);
	}
	for _ in 0..config.bounces {
		let mut pass = Vec::with_capacity(order.len());
		for &i in &order {
			pass.push(baked.gather_chart(&scene, i, &mut rng));
		}
		for (&i, values) in order.iter().zip(&pass) {
			baked.add_values(i, values);
		}
	}
	Ok(baked)
}

impl Baked {
	/// The atlas: its side and the RGBA texels, ready for a float texture.
	pub fn atlas(&self) -> (u32, &[[f32; 4]]) {
		(self.side, &self.texels)
	}

	/// How many triangles hold a chart.
	pub fn charts(&self) -> usize {
		self.charts.iter().filter(|c| c.is_some()).count()
	}

	/// Triangle `tri`'s atlas mapping, `None` for degenerate triangles.
	pub fn chart(&self, tri: usize) -> Option<ChartView> {
		let c = self.charts[tri].as_ref()?;
		Some(ChartView {
			u: c.u,
			v: c.v,
			axis: c.axis,
			plane: c.plane,
			u_min: c.u_min,
			v_min: c.v_min,
			units: self.config.units_per_texel,
			rect: c.rect,
		})
	}

	/// Normalized atlas coordinates of world point `p` on triangle `tri`,
	/// clamped into its chart, at texel centers.
	pub fn uv(&self, tri: usize, p: Vec3) -> Option<[f32; 2]> {
		let c = self.charts[tri].as_ref()?;
		let units = self.config.units_per_texel;
		let at = |axis: Vec3, min: f32, span: u32| {
			((p.dot(axis) - min) / units - 0.5).clamp(0.0, span as f32 - 1.0)
		};
		let i = at(c.u, c.u_min, c.rect.w);
		let j = at(c.v, c.v_min, c.rect.h);
		Some([
			(c.rect.x as f32 + i + 0.5) / self.side as f32,
			(c.rect.y as f32 + j + 0.5) / self.side as f32,
		])
	}

	/// The triangle under world point `p` and its baked light there, `None`
	/// away from every surface. A CPU query for gameplay lighting; the shader
	/// samples the atlas directly.
	pub fn at(&self, p: Vec3) -> Option<(usize, Vec3)> {
		let units = self.config.units_per_texel;
		for (i, c) in self.charts.iter().enumerate() {
			let Some(c) = c else { continue };
			if (p.dot(c.axis) - c.plane).abs() > units * 0.75 {
				continue;
			}
			let (uu, vv) = (p.dot(c.u), p.dot(c.v));
			let (u1, v1) = (
				c.u_min + c.rect.w as f32 * units,
				c.v_min + c.rect.h as f32 * units,
			);
			if uu < c.u_min || uu > u1 || vv < c.v_min || vv > v1 {
				continue;
			}
			return Some((i, self.chart_sample(c, p)));
		}
		None
	}

	/// Rebakes after geometry edits: `changed` lists every triangle index that
	/// was created, moved, destroyed (replaced by a zero-area triangle) or
	/// appended to the soup. Returns the atlas rectangles that changed, for
	/// partial texture uploads. See the crate docs for the index contract.
	pub fn update(
		&mut self,
		tris: &[Triangle],
		albedo: &[Vec3],
		changed: &[usize],
	) -> Result<Vec<Rect>, String> {
		if tris.len() != albedo.len() {
			return Err(format!(
				"{} triangles but {} albedos",
				tris.len(),
				albedo.len()
			));
		}
		self.charts.resize(tris.len(), None);
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
		};
		let mut rng = self.config.seed.wrapping_mul(2_025_617).wrapping_add(1);

		// Charts of destroyed triangles go black; created ones get packed.
		let mut rects = Vec::new();
		let mut rebuild = Vec::new();
		for &i in &changed {
			if area(&tris[i]) > 1e-12 {
				if self.charts[i].is_none() {
					self.add_chart(tris, i)?;
				}
				rebuild.push(i);
			} else if let Some(c) = self.charts[i].take() {
				self.clear_rect(&c.rect);
				rects.push(c.rect.grown(self.side));
			}
		}
		// Sun visibility recorded per texel names the triangle that shadowed
		// it; destroying or moving that one unshadows or reshadows the texel.
		for (i, c) in self.charts.iter().enumerate() {
			let Some(c) = c else { continue };
			if !rebuild.contains(&i)
				&& c
					.shadow
					.iter()
					.any(|&s| s != NO_SHADOW && changed.binary_search(&(s as usize)).is_ok())
			{
				rebuild.push(i);
			}
		}
		// A moved or added triangle also shadows texels whose recorded shadow
		// triangle predates it, so charts inside the swept shadow volume are
		// re-raycast too.
		for &i in &changed {
			if area(&tris[i]) <= 1e-12 {
				continue;
			}
			let (lo, hi) = tri_bounds(&tris[i]);
			let sun = self.config.sun_dir;
			let (lo, hi) = (lo.min(lo - sun * span), hi.max(hi - sun * span));
			for (j, c) in self.charts.iter().enumerate() {
				let Some(c) = c else { continue };
				let (a, b) = c.bounds(self.config.units_per_texel);
				if a.x <= hi.x
					&& b.x >= lo.x
					&& a.y <= hi.y
					&& b.y >= lo.y
					&& a.z <= hi.z
					&& b.z >= lo.z
					&& !rebuild.contains(&j)
				{
					rebuild.push(j);
				}
			}
		}
		rebuild.sort_unstable();
		rebuild.dedup();
		for &i in &rebuild {
			self.bake_chart(&scene, i, &mut rng, self.config.bounces > 0);
			let rect = self.charts[i].as_ref().unwrap().rect;
			rects.push(rect.grown(self.side));
		}
		Ok(rects)
	}

	/// Direct sun and sky at one surface point: `S`, sun visibility, and the
	/// triangle that shadowed the sun ray.
	fn direct(&self, s: &Scene, p: Vec3, n: Vec3) -> ([f32; 3], f32, u32) {
		let c = &self.config;
		let n_l = n.dot(c.sun_dir).max(0.0);
		if n_l == 0.0 {
			return ([0.0; 3], 1.0, NO_SHADOW);
		}
		let (vis, shadow) = match s.bvh.cast(s.tris, p + n * OFFSET, c.sun_dir, s.span) {
			Some((i, _)) => (0.0, i as u32),
			None => (1.0, NO_SHADOW),
		};
		let sun = c.sun_color * (n_l * vis / PI);
		let sky = c.ground_color.lerp(c.sky_color, n.y * 0.5 + 0.5);
		([sun.x + sky.x, sun.y + sky.y, sun.z + sky.z], vis, shadow)
	}

	/// One bounce of indirect light gathered at every texel of chart `tri`,
	/// reading the atlas as it stands.
	fn gather_chart(&self, s: &Scene, tri: usize, rng: &mut u32) -> Vec<[f32; 4]> {
		let c = &self.config;
		let chart = self.charts[tri].as_ref().unwrap();
		let t = &s.tris[tri];
		let n = normal(t);
		let mut out = vec![[0.0f32; 4]; (chart.rect.w * chart.rect.h) as usize];
		if c.rays == 0 {
			return out;
		}
		for j in 0..chart.rect.h {
			for i in 0..chart.rect.w {
				let p = bvh::closest_point(chart.world(i, j, c.units_per_texel), t);
				let mut sum = Vec3::ZERO;
				for _ in 0..c.rays {
					let d = cosine(n, rng);
					let o = p + n * OFFSET;
					let lo = match s.bvh.cast(s.tris, o, d, s.span) {
						Some((hit, at)) => {
							let q = o + d * at;
							self.charts[hit]
								.as_ref()
								.map(|h| mul(s.albedo[hit], self.chart_sample(h, q)))
								.unwrap_or_default()
						}
						None => c.ground_color.lerp(c.sky_color, d.y * 0.5 + 0.5),
					};
					sum += lo;
				}
				let mean = sum / c.rays as f32;
				out[(j * chart.rect.w + i) as usize] = [mean.x, mean.y, mean.z, 0.0];
			}
		}
		out
	}

	/// Bakes chart `tri` in place: direct sun, sky, and optionally one
	/// gathered bounce, plus the shadow table.
	fn bake_chart(&mut self, s: &Scene, tri: usize, rng: &mut u32, gather: bool) {
		let units = self.config.units_per_texel;
		let (values, indirect, shadow) = {
			let c = &self.config;
			let chart = self.charts[tri].as_ref().unwrap();
			let t = &s.tris[tri];
			let n = normal(t);
			let count = (chart.rect.w * chart.rect.h) as usize;
			let mut values = vec![[0.0f32; 4]; count];
			let mut indirect = vec![[0.0f32; 3]; count];
			let mut shadow = vec![NO_SHADOW; count];
			for j in 0..chart.rect.h {
				for i in 0..chart.rect.w {
					let k = (j * chart.rect.w + i) as usize;
					let p = bvh::closest_point(chart.world(i, j, units), t);
					let (light, vis, sh) = self.direct(s, p, n);
					shadow[k] = sh;
					if gather && c.rays > 0 {
						let mut sum = Vec3::ZERO;
						for _ in 0..c.rays {
							let d = cosine(n, rng);
							let o = p + n * OFFSET;
							let lo = match s.bvh.cast(s.tris, o, d, s.span) {
								Some((hit, at)) => {
									let q = o + d * at;
									self.charts[hit]
										.as_ref()
										.map(|h| mul(s.albedo[hit], self.chart_sample(h, q)))
										.unwrap_or_default()
								}
								None => c.ground_color.lerp(c.sky_color, d.y * 0.5 + 0.5),
							};
							sum += lo;
						}
						let mean = sum / c.rays as f32;
						indirect[k] = [mean.x, mean.y, mean.z];
					}
					values[k] = [light[0], light[1], light[2], vis];
				}
			}
			(values, indirect, shadow)
		};
		let mut values = values;
		if gather {
			for (v, g) in values.iter_mut().zip(&indirect) {
				v[0] += g[0];
				v[1] += g[1];
				v[2] += g[2];
			}
		}
		self.commit(tri, &values, &shadow);
	}

	/// Writes a chart's values and shadow table to the atlas, duplicating its
	/// edges into the gutter so bilinear filtering stays inside the chart.
	fn commit(&mut self, tri: usize, values: &[[f32; 4]], shadow: &[u32]) {
		let rect = self.charts[tri].as_ref().unwrap().rect;
		self.charts[tri].as_mut().unwrap().shadow = shadow.to_vec();
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

	/// Adds gathered values to a chart's texels, gutters included.
	fn add_values(&mut self, tri: usize, values: &[[f32; 4]]) {
		let rect = self.charts[tri].as_ref().unwrap().rect;
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

	/// Packs and stores triangle `i`'s chart.
	fn add_chart(&mut self, tris: &[Triangle], i: usize) -> Result<(), String> {
		let (w, h) = chart_size(&self.config, &tris[i]);
		let Some((x, y)) = self.packer.place(w, h) else {
			return Err(format!(
				"atlas {}×{} full: triangle {i} needs a {}×{} chart",
				self.side, self.side, w, h
			));
		};
		let t = &tris[i];
		let (u, v, axis) = dominant(normal(t));
		let us = [t[0].dot(u), t[1].dot(u), t[2].dot(u)];
		let vs = [t[0].dot(v), t[1].dot(v), t[2].dot(v)];
		self.charts[i] = Some(Chart {
			rect: Rect { x, y, w, h },
			u,
			v,
			axis,
			plane: t[0].dot(axis),
			u_min: us[0].min(us[1]).min(us[2]),
			v_min: vs[0].min(vs[1]).min(vs[2]),
			shadow: vec![NO_SHADOW; (w * h) as usize],
		});
		Ok(())
	}

	/// Bilinear sample of the atlas at world point `p` as seen by chart `c`.
	fn chart_sample(&self, c: &Chart, p: Vec3) -> Vec3 {
		let units = self.config.units_per_texel;
		let at = |axis: Vec3, min: f32| (p.dot(axis) - min) / units - 0.5;
		let (x, y) = (
			at(c.u, c.u_min).clamp(0.0, c.rect.w as f32 - 1.0),
			at(c.v, c.v_min).clamp(0.0, c.rect.h as f32 - 1.0),
		);
		let (x0, y0) = (x.floor(), y.floor());
		let (tx, ty) = (x - x0, y - y0);
		let (x1, y1) = (
			(x0 + 1.0).min(c.rect.w as f32 - 1.0),
			(y0 + 1.0).min(c.rect.h as f32 - 1.0),
		);
		let texel = |i: f32, j: f32| {
			let (a, b) = (c.rect.x as f32 + i, c.rect.y as f32 + j);
			self.texels[(b as u32 * self.side + a as u32) as usize]
		};
		let row = |j: f32| {
			let (p, q) = (texel(x0, j), texel(x1, j));
			vec3(
				p[0] + (q[0] - p[0]) * tx,
				p[1] + (q[1] - p[1]) * tx,
				p[2] + (q[2] - p[2]) * tx,
			)
		};
		row(y0).lerp(row(y1), ty)
	}
}

/// Component-wise product, what a diffuse reflector does to arriving light.
fn mul(a: Vec3, b: Vec3) -> Vec3 {
	vec3(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// Twice the soup's diagonal: every ray's maximum travel.
fn ray_span(tris: &[Triangle]) -> f32 {
	let (lo, hi) = bvh::soup_bounds(tris);
	(hi - lo).length() * 2.0
}

fn tri_bounds(t: &Triangle) -> (Vec3, Vec3) {
	(t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]))
}

fn area(t: &Triangle) -> f32 {
	(t[1] - t[0]).cross(t[2] - t[0]).length() * 0.5
}

fn normal(t: &Triangle) -> Vec3 {
	(t[1] - t[0]).cross(t[2] - t[0]).normalize()
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

/// A triangle's chart size: its extent on the projection axes, at the baked
/// density, never below `min_chart`.
fn chart_size(config: &Config, t: &Triangle) -> (u32, u32) {
	let (u, v, _) = dominant(normal(t));
	let us = [t[0].dot(u), t[1].dot(u), t[2].dot(u)];
	let vs = [t[0].dot(v), t[1].dot(v), t[2].dot(v)];
	let extent = |c: [f32; 3]| {
		(((c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])) / config.units_per_texel).ceil() as u32)
			.max(config.min_chart)
			.min(config.atlas_size - 1)
	};
	(extent(us), extent(vs))
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

	/// Floor tiles and a 2-high slab standing on them at x in [4, 5], z in
	/// [3, 4]. The sun slants toward +x, so the slab's shadow falls on the
	/// floor at x in [2, 4], z in [3, 4]. Zero sky, zero bounces: every lit
	/// texel is pure direct sun. Returns the slab's triangle indices.
	fn shadow_scene() -> (Vec<Triangle>, Vec<usize>) {
		let mut tris = Vec::new();
		floor_tiles(&mut tris, 8);
		let slab = tris.len();
		quad(
			&mut tris,
			vec3(4.0, 0.0, 3.0),
			vec3(4.0, 2.0, 3.0),
			vec3(4.0, 2.0, 4.0),
			vec3(4.0, 0.0, 4.0),
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
			seed: 1,
		}
	}

	fn white(tris: &[Triangle]) -> Vec<Vec3> {
		vec![vec3(0.8, 0.8, 0.8); tris.len()]
	}

	#[test]
	fn charts_pack_without_overlap() {
		let (tris, _) = shadow_scene();
		let baked = bake(&tris, &white(&tris), &config()).unwrap();
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
		let baked = bake(&tris, &white(&tris), &config()).unwrap();
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
		let baked = bake(&tris, &white(&tris), &config()).unwrap();
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
		let mut baked = bake(&tris, &white(&tris), &config).unwrap();
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
		let rects = baked.update(&edited, &white(&edited), &slab).unwrap();
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
		let mut baked = bake(&tris, &white(&tris), &config()).unwrap();
		let (_, lit) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		let want = config().sun_dir.y / PI;
		assert!((lit.x - want).abs() < 1e-3, "floor starts lit: {lit:?}");
		let slab = tris.len();
		quad(
			&mut tris,
			vec3(4.0, 0.0, 3.0),
			vec3(4.0, 2.0, 3.0),
			vec3(4.0, 2.0, 4.0),
			vec3(4.0, 0.0, 4.0),
			vec3(1.0, 0.0, 0.0),
		);
		let albedo = white(&tris);
		let changed = (slab..tris.len()).collect::<Vec<_>>();
		let rects = baked.update(&tris, &albedo, &changed).unwrap();
		assert!(!rects.is_empty(), "the new chart should dirty the atlas");
		let (_, dark) = baked.at(vec3(3.5, 0.0, 3.5)).unwrap();
		assert!(dark.x < 1e-3, "new slab should shadow: {dark:?}");
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
		let flat = bake(&tris, &white(&tris), &config).unwrap();
		config.bounces = 1;
		config.rays = 64;
		let bounced = bake(&tris, &white(&tris), &config).unwrap();
		let (_, s) = flat.at(vec3(6.99, 2.0, 3.5)).unwrap();
		let (_, t) = bounced.at(vec3(6.99, 2.0, 3.5)).unwrap();
		assert!(s.x < 1e-3, "the sun misses the wall: {s:?}");
		assert!(t.x > s.x + 1e-3, "bounce should add light: {s:?} -> {t:?}");
	}

	#[test]
	fn uv_maps_chart_points_into_the_chart() {
		let (tris, _) = shadow_scene();
		let baked = bake(&tris, &white(&tris), &config()).unwrap();
		for (i, tri) in tris.iter().enumerate() {
			let Some(view) = baked.chart(i) else { continue };
			// The world point at the chart's middle, clamped onto the
			// triangle, maps to atlas coordinates inside the chart.
			let mid = |span: u32, min: f32| min + span as f32 * 0.5 * view.units;
			let world = view.u * mid(view.rect.w, view.u_min)
				+ view.v * mid(view.rect.h, view.v_min)
				+ view.axis * view.plane;
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
		let a = bake(&tris, &white(&tris), &config()).unwrap();
		let b = bake(&tris, &white(&tris), &config()).unwrap();
		assert_eq!(a.atlas().1, b.atlas().1);
	}
}
