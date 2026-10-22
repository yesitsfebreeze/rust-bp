//! The world: the map's corners, texture atlas and baked light on the GPU,
//! drawn by the raster (`raster.rs`) into the frame's own color and depth,
//! then carried to the window shaded darker where its depth shows matter
//! standing near, fogged and smoothed (`ao.rs`).

use super::util::{self, floats, words};
use super::{ao, raster, scene, Color, View, NEAR};
use crate::bsp::Map;
use linalg::Vec3;

/// How the baked light's ambient part lands: the light the sun does not
/// bring straight — sky, bounces and lamps.
#[derive(Clone, Copy, PartialEq)]
pub struct Indirect {
	/// Scale on the bake's ambient light: 1 as baked, 0 sunlight alone.
	pub gi: f32,
	/// How strongly the frame's ambient occlusion darkens that light: 0
	/// off, 1 full.
	pub ao: f32,
	/// How far around a pixel the occlusion looks, world units.
	pub radius: f32,
	/// Whether the sun's shadow is read from the shadow map, sharp; else
	/// the lightmap's, as coarse as its texels.
	pub sharp: bool,
}

/// The air between the eye and what it sees: fog thick near the ground and
/// thinning with height, lit by the sky and, looking toward the sun, by the
/// sun's glow through it. Laid over the frame by its depth (`ao.wgsl`).
#[derive(Clone, Copy, PartialEq)]
pub struct Fog {
	/// Extinction per world unit at height 0: 0 clear air.
	pub density: f32,
	/// The height over which the fog thins to 1/e, world units.
	pub height: f32,
	/// Tint on the light the fog glows with: white as the sky and sun
	/// light it.
	pub color: Vec3,
}

/// A light that moves or flickers, lit fresh every frame by the surfaces
/// pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynLight {
	pub pos: Vec3,
	/// Linear radiance a quarter of its reach away.
	pub color: Vec3,
	/// How far it reaches, world units: nothing past it.
	pub reach: f32,
}

/// Dynamic lights the surfaces pass takes per frame, past which the rest
/// are dropped; and moving shapes casting shadows.
pub const MAX_LIGHTS: usize = 16;
pub const MAX_OCCLUDERS: usize = 32;

/// The playground's spheres at most: center and radius, `w` 0 for none.
const MAX_BODIES: u64 = 64;

/// The map's light: where the sun is and how bright, the sky, the ground.
pub struct Sources {
	pub sun_dir: Vec3,
	pub sun_color: Vec3,
	pub sky: Vec3,
	pub ground: Vec3,
}

impl Sources {
	/// The light the map's entities and the bake's settings describe.
	pub fn of(config: &lightbake::Config) -> Sources {
		Sources {
			sun_dir: config.sun_dir.normalize(),
			sun_color: config.sun_color,
			sky: config.sky_color,
			ground: config.ground_color,
		}
	}
}

/// What loading a world found.
#[derive(Clone, Copy, Debug)]
pub struct WorldStats {
	pub triangles: usize,
}

pub struct World {
	/// `View` in raster.wgsl and ao.wgsl.
	view: wgpu::Buffer,
	/// `Lighting` in raster.wgsl and ao.wgsl.
	lighting: wgpu::Buffer,
	bodies: wgpu::Buffer,
	/// `Dynamic` in raster.wgsl.
	dynamic: wgpu::Buffer,
	/// This frame's loose faces — props — as surface corners, grown as
	/// needed, and how many corners it holds.
	loose: Option<wgpu::Buffer>,
	loose_corners: u32,
	raster: raster::Raster,
	/// The frame's ambient occlusion pass over the world's own color.
	ambient: ao::Ambient,
	triangles: usize,
}

impl World {
	pub fn new(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		format: wgpu::TextureFormat,
		depth: wgpu::TextureFormat,
		map: &Map,
		sources: Sources,
		baked: &lightbake::Baked,
	) -> World {
		// The bake's light mapping per face, so every corner shades the same
		// luxel the map's own lightdata sits on.
		let face_lms = crate::lightmap::face_lms(&map.render, baked);
		let scene = scene::build(&map.render, &face_lms);
		let atlas = &map.render.albedo;
		let (_, albedo_srgb) =
			util::image(device, queue, "albedo", atlas.side, atlas.side, &atlas.rgba);
		// The bake's light atlas, linear radiance in half floats.
		let (side, light) = crate::lightmap::atlas(baked);
		let baked_light = util::half_image(device, queue, "baked light", side, side, &light);
		let lighting = util::buffer(
			device,
			"lighting",
			4 * 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		let view = util::buffer(
			device,
			"view",
			10 * 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		// The playground's spheres, written whole every frame.
		let bodies = util::buffer(
			device,
			"bodies",
			MAX_BODIES * 16,
			wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
		);
		// `Dynamic` in raster.wgsl: counts and switches, then the lights,
		// then the moving shapes, written whole every frame.
		let dynamic = util::buffer(
			device,
			"dynamic",
			(16 + MAX_LIGHTS * 32 + MAX_OCCLUDERS * 32) as u64,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		let s = &sources;
		queue.write_buffer(
			&lighting,
			0,
			&floats(&[
				s.sun_dir.x,
				s.sun_dir.y,
				s.sun_dir.z,
				0.0,
				s.sun_color.x,
				s.sun_color.y,
				s.sun_color.z,
				0.0,
				s.sky.x,
				s.sky.y,
				s.sky.z,
				0.0,
				s.ground.x,
				s.ground.y,
				s.ground.z,
				0.0,
			]),
		);
		let raster = raster::Raster::new(
			device,
			queue,
			format,
			depth,
			raster::Shared {
				view: &view,
				albedo_srgb: &albedo_srgb,
				baked: &baked_light,
				lighting: &lighting,
				dynamic: &dynamic,
				bodies: &bodies,
				body_slots: MAX_BODIES as u32,
			},
			&scene.corners,
			(scene.lo, scene.hi),
			s.sun_dir,
		);
		let ambient = ao::Ambient::new(device, format);
		World {
			view,
			lighting,
			bodies,
			dynamic,
			loose: None,
			loose_corners: 0,
			raster,
			ambient,
			triangles: scene.tris.len(),
		}
	}

	pub fn stats(&self) -> WorldStats {
		WorldStats {
			triangles: self.triangles,
		}
	}

	/// This frame's dynamic lights and moving shapes — capsules, `a` to `b`
	/// swept by `radius`, a sphere when the ends meet — for the surfaces
	/// pass, past [`MAX_LIGHTS`] and [`MAX_OCCLUDERS`] dropped; with `pbr`
	/// the surfaces reflect by their finish, with `shadows` the shapes
	/// cast shadows.
	pub fn set_dynamic(
		&self,
		queue: &wgpu::Queue,
		lights: &[DynLight],
		occluders: &[(Vec3, Vec3, f32)],
		pbr: bool,
		shadows: bool,
	) {
		let lights = &lights[..lights.len().min(MAX_LIGHTS)];
		let occluders = &occluders[..occluders.len().min(MAX_OCCLUDERS)];
		let mut bytes = words(&[
			lights.len() as u32,
			occluders.len() as u32,
			pbr as u32,
			shadows as u32,
		]);
		let mut slots = vec![[0.0f32; 8]; MAX_LIGHTS];
		for (slot, l) in slots.iter_mut().zip(lights) {
			*slot = [
				l.pos.x, l.pos.y, l.pos.z, l.reach, l.color.x, l.color.y, l.color.z, 0.0,
			];
		}
		bytes.extend(floats(&slots.concat()));
		let mut slots = vec![[0.0f32; 8]; MAX_OCCLUDERS];
		for (slot, &(a, b, r)) in slots.iter_mut().zip(occluders) {
			*slot = [a.x, a.y, a.z, r, b.x, b.y, b.z, 0.0];
		}
		bytes.extend(floats(&slots.concat()));
		queue.write_buffer(&self.dynamic, 0, &bytes);
	}

	/// This frame's loose faces, `bytes` of corners as
	/// `scene::loose_corners` packs them, for the surfaces pass to draw
	/// over the map.
	pub fn set_loose(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) {
		self.loose_corners = (bytes.len() / scene::CORNER_BYTES) as u32;
		if bytes.is_empty() {
			return;
		}
		if self.loose.as_ref().is_none_or(|b| b.size() < bytes.len() as u64) {
			self.loose = Some(util::buffer(
				device,
				"loose corners",
				(bytes.len() as u64).next_power_of_two(),
				wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
			));
		}
		queue.write_buffer(self.loose.as_ref().unwrap(), 0, bytes);
	}

	/// The playground's spheres: center and radius each, in `bodies`.
	pub fn set_bodies(&self, queue: &wgpu::Queue, bodies: &[(Vec3, f32)]) {
		let mut slots = vec![[0.0f32; 4]; MAX_BODIES as usize];
		for (slot, &(pos, radius)) in slots.iter_mut().zip(bodies) {
			*slot = [pos.x, pos.y, pos.z, radius];
		}
		queue.write_buffer(&self.bodies, 0, &floats(&slots.concat()));
	}

	/// Draws the frame from `view` into the pass's color and depth: the
	/// map's triangles, this frame's loose faces and the spheres, shown as
	/// debug view `mode` (0 lit). With `smooth`, [`World::occlude`] smooths
	/// the frame's edges as it carries it over.
	#[allow(clippy::too_many_arguments)]
	pub fn draw(
		&mut self,
		queue: &wgpu::Queue,
		pass: &mut wgpu::RenderPass<'_>,
		view: &View,
		size: (f32, f32),
		sky: Color,
		exposure: f32,
		encode: bool,
		mode: u32,
		indirect: Indirect,
		fog: Fog,
		smooth: bool,
	) {
		let (w, h) = size;
		let tan = (view.fovy * 0.5).tan();
		let (e, r, u, f) = (view.eye, view.right, view.up, view.forward);
		let linear = |c: f32| c.powf(2.2);
		#[rustfmt::skip]
		let uniform = [
			e.x, e.y, e.z, 0.0,
			r.x, r.y, r.z, tan * w / h,
			u.x, u.y, u.z, tan,
			f.x, f.y, f.z, 0.0,
			linear(sky.r), linear(sky.g), linear(sky.b), 0.0,
			exposure, encode as u32 as f32, NEAR, mode as f32,
			w, h, 0.0, 0.0,
			indirect.gi, indirect.ao, indirect.radius, indirect.sharp as u32 as f32,
			fog.density, fog.height, smooth as u32 as f32, 0.0,
			fog.color.x, fog.color.y, fog.color.z, 0.0,
		];
		queue.write_buffer(&self.view, 0, &floats(&uniform));
		let loose = self.loose.as_ref().map(|b| (b, self.loose_corners));
		self.raster.draw(pass, loose);
	}

	/// Carries the frame the world pass drew into `scene` over into
	/// `target`, each drawn pixel shaded darker where `depth` shows the
	/// map's own matter standing near it — unless `occlude` is off — and
	/// fogged by the air its depth looks through. The textures come as
	/// views because resize replaces them. `size` is the frame's, in
	/// pixels.
	#[allow(clippy::too_many_arguments)]
	pub fn occlude(
		&mut self,
		device: &wgpu::Device,
		encoder: &mut wgpu::CommandEncoder,
		target: &wgpu::TextureView,
		scene: &wgpu::TextureView,
		depth: &wgpu::TextureView,
		size: (u32, u32),
		occlude: bool,
	) {
		self.ambient.draw(
			device,
			encoder,
			target,
			scene,
			depth,
			&self.view,
			&self.lighting,
			size,
			occlude,
		);
	}
}
