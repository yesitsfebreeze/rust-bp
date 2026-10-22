//! The traced world: the map's triangles and hierarchy on the GPU, its
//! texture atlas, the voxel field's stored colors, and the full-screen pass
//! that casts every pixel's ray (`bvh.wgsl` + `material.wgsl` +
//! `voxel.wgsl` + `trace.wgsl`), or in its place the shell field's
//! marched cubes (`shell.rs`), or the triangles rasterized as themselves
//! (`raster.rs`) — then the frame's own color is carried to the window
//! shaded darker where its depth shows matter standing near (`ao.rs`).

use super::util::{self, entry, floats, words};
use super::{ao, raster, scene, shell, Color, View, NEAR};
use crate::bsp::Map;
use linalg::Vec3;
pub use shell::Lod;

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
	/// Whether the sun's shadow is traced per pixel; else the bake's,
	/// as coarse as its texels.
	pub traced: bool,
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

/// The smallest voxel, world units: the finest matter, right at the eye. A
/// sixteenth of a unit keeps cubes `K` pixels wide even at arm's length.
const BASE_VOXEL: f32 = 1.0 / 16.0;
/// Slots storing the voxels' colors: a power of two, more than a screen of
/// voxels at any setting; and each slot's bytes, as `Slot` in voxel.wgsl.
const FIELD_SLOTS: u32 = 1 << 21;
const SLOT_BYTES: u64 = 32;
/// The playground's spheres at most: center and radius, `w` 0 for none.
const MAX_BODIES: u64 = 64;
/// Dents the spheres left in the map at most: center and radius, `w` 0 past
/// the last.
pub const MAX_DENTS: usize = 256;

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
	view: wgpu::Buffer,
	/// `Lighting` in material.wgsl, as every pass reads it.
	lighting: wgpu::Buffer,
	bodies: wgpu::Buffer,
	dents: wgpu::Buffer,
	/// `Dynamic` in raster.wgsl.
	dynamic: wgpu::Buffer,
	/// This frame's loose faces — props — as surface corners, grown as
	/// needed, and how many corners it holds.
	loose: Option<wgpu::Buffer>,
	loose_corners: u32,
	group: wgpu::BindGroup,
	pipeline: wgpu::RenderPipeline,
	/// The shell field, voxelizing the map ahead of the view and drawing
	/// the cubes' exposed faces instead, when asked.
	shell: shell::Shell,
	/// The map's triangles rasterized as themselves, when asked.
	raster: raster::Raster,
	/// The frame's ambient occlusion pass over the world's own color.
	ambient: ao::Ambient,
	triangles: usize,
	/// Frames drawn so far: the voxel field's clock.
	frame: u32,
	/// How many times the dents changed: the voxels' stored colors key on
	/// it, so a new dent recolors the matter it reshaped.
	dent_changes: u32,
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
		// The bake's light mapping per face, so every render path — the
		// raster's corners, the voxel paths' cubes — shades the same luxel
		// the map's own lightdata sits on.
		let face_lms = crate::lightmap::face_lms(&map.render, baked);
		let scene = scene::build(&map.render, &face_lms);
		let bvh = lightbake::Bvh::build(&scene.tris);
		let (nodes, order) = bvh.gpu();
		// The triangles in the hierarchy's leaf order, so its leaves index
		// them directly and no order buffer is read per test.
		let tris = util::storage(device, "tris", &scene.bytes_in(&order));
		let nodes = util::storage(device, "nodes", &nodes);
		let atlas = &map.render.albedo;
		// sRGB bytes, read raw and through an sRGB view.
		let (albedo, albedo_srgb) =
			util::image(device, queue, "albedo", atlas.side, atlas.side, &atlas.rgba);
		// The bake's light atlas, linear radiance in half floats.
		let (side, light) = crate::lightmap::atlas(baked);
		let baked_light = util::half_image(device, queue, "baked light", side, side, &light);
		// Where the raycast voxel field stores its colors.
		let slots = util::buffer(
			device,
			"voxel colors",
			FIELD_SLOTS as u64 * SLOT_BYTES,
			wgpu::BufferUsages::STORAGE,
		);
		let lighting = util::buffer(
			device,
			"lighting",
			4 * 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		let view = util::buffer(
			device,
			"trace view",
			11 * 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		// The playground's spheres, written whole every frame.
		let bodies = util::buffer(
			device,
			"bodies",
			MAX_BODIES * 16,
			wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
		);
		// `Dynamic` in raster.wgsl: counts and switches, then the lights,
		// then the moving shapes, written whole every frame.
		let dynamic = util::buffer(
			device,
			"dynamic",
			(16 + MAX_LIGHTS * 32 + MAX_OCCLUDERS * 32) as u64,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		// The dents, written whole when they change.
		let dents = util::buffer(
			device,
			"dents",
			MAX_DENTS as u64 * 16,
			wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
		);
		// `Lighting` in material.wgsl.
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

		let fragment = wgpu::ShaderStages::FRAGMENT;
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("trace"),
			entries: &[
				util::buffer_entry(0, fragment, util::UNIFORM),
				util::buffer_entry(1, fragment, util::READ),
				util::buffer_entry(2, fragment, util::READ),
				util::texture_entry(4, fragment),
				util::texture_entry(5, fragment),
				util::texture_entry(6, fragment),
				util::buffer_entry(11, fragment, util::UNIFORM),
				util::buffer_entry(12, fragment, util::READ),
				util::buffer_entry(13, fragment, util::WRITE),
				util::buffer_entry(14, fragment, util::READ),
			],
		});
		let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("trace"),
			layout: &layout,
			entries: &[
				util::bind(0, &view),
				util::bind(1, &tris),
				util::bind(2, &nodes),
				entry(4, wgpu::BindingResource::TextureView(&albedo)),
				entry(5, wgpu::BindingResource::TextureView(&albedo_srgb)),
				entry(6, wgpu::BindingResource::TextureView(&baked_light)),
				util::bind(11, &lighting),
				util::bind(12, &bodies),
				util::bind(13, &slots),
				util::bind(14, &dents),
			],
		});
		let shared = [
			include_str!("bvh.wgsl"),
			include_str!("material.wgsl"),
			include_str!("voxel.wgsl"),
		];
		let shader = util::shader(
			device,
			"trace",
			&[&shared[..], &[include_str!("trace.wgsl")]].concat(),
		);

		let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("trace"),
			layout: Some(&util::layout(device, "trace", &[&layout])),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: Some("vs"),
				compilation_options: Default::default(),
				buffers: &[],
			},
			primitive: Default::default(),
			depth_stencil: Some(wgpu::DepthStencilState {
				format: depth,
				depth_write_enabled: true,
				depth_compare: wgpu::CompareFunction::Always,
				stencil: Default::default(),
				bias: Default::default(),
			}),
			multisample: Default::default(),
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: Some("fs"),
				compilation_options: Default::default(),
				targets: &[Some(format.into())],
			}),
			multiview: None,
			cache: None,
		});
		let shell = shell::Shell::new(
			device,
			format,
			depth,
			shell::Shared {
				view: &view,
				tris: &tris,
				nodes: &nodes,
				albedo: &albedo,
				albedo_srgb: &albedo_srgb,
				baked: &baked_light,
				lighting: &lighting,
				slots: &slots,
			},
			scene.tris.len(),
			MAX_DENTS,
		);
		let raster = raster::Raster::new(
			device,
			format,
			depth,
			raster::Shared {
				view: &view,
				tris: &tris,
				nodes: &nodes,
				albedo_srgb: &albedo_srgb,
				baked: &baked_light,
				lighting: &lighting,
				dynamic: &dynamic,
				bodies: &bodies,
				body_slots: MAX_BODIES as u32,
			},
			&scene.corners,
		);
		let ambient = ao::Ambient::new(device, format);
		World {
			view,
			lighting,
			bodies,
			dents,
			dynamic,
			loose: None,
			loose_corners: 0,
			group,
			pipeline,
			shell,
			raster,
			ambient,
			triangles: scene.tris.len(),
			frame: 0,
			dent_changes: 0,
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
		queue.write_buffer(&self.bodies, 0, &spheres(bodies, MAX_BODIES as usize));
		self.shell.set_bodies(queue, bodies);
	}

	/// The dents carved into the map: center and radius each, in `dents`;
	/// past [`MAX_DENTS`] the rest are dropped.
	pub fn set_dents(&mut self, queue: &wgpu::Queue, dents: &[(Vec3, f32)]) {
		queue.write_buffer(&self.dents, 0, &spheres(dents, MAX_DENTS));
		self.shell.set_dents(queue, dents);
		self.dent_changes = self.dent_changes.wrapping_add(1);
	}

	/// Brings the shell field along to `view` for this frame's draw on a
	/// screen `size` pixels: cube sizes change over distance as `lod` says,
	/// its `near` 0 for where the smallest cubes grow past `voxel` pixels.
	/// Before the frame's [`World::draw`] with `shell`.
	pub fn prepare(
		&mut self,
		queue: &wgpu::Queue,
		encoder: &mut wgpu::CommandEncoder,
		view: &View,
		size: (f32, f32),
		voxel: f32,
		lod: Lod,
	) {
		let pixel = 2.0 * (view.fovy * 0.5).tan() / size.1;
		let lod = Lod {
			near: if lod.near > 0.0 {
				lod.near
			} else {
				BASE_VOXEL / (pixel * voxel)
			},
			curve: lod.curve,
		};
		self.shell.prepare(
			queue,
			encoder,
			view.eye,
			pixel,
			lod,
			BASE_VOXEL,
			self.dent_changes,
			size,
		);
	}

	/// Draws the frame from `view` into the pass's color and depth: with
	/// `surface`, the map's triangles rasterized as themselves; with
	/// `shell`, the shell field's cubes [`World::prepare`] brought along;
	/// else a ray per pixel, traced per hit. With `smooth`,
	/// [`World::occlude`] smooths the frame's edges as it carries it over.
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
		voxel: f32,
		mode: u32,
		surface: bool,
		shell: bool,
		rays: u32,
		wire: bool,
		indirect: Indirect,
		fog: Fog,
		smooth: bool,
	) {
		self.frame = self.frame.wrapping_add(1);
		let (w, h) = size;
		let tan = (view.fovy * 0.5).tan();
		let pixel = 2.0 * tan / h;
		let (e, r, u, f) = (view.eye, view.right, view.up, view.forward);
		let linear = |c: f32| c.powf(2.2);
		#[rustfmt::skip]
		let uniform = [
			e.x, e.y, e.z, pixel,
			r.x, r.y, r.z, tan * w / h,
			u.x, u.y, u.z, tan,
			f.x, f.y, f.z, voxel,
			linear(sky.r), linear(sky.g), linear(sky.b), BASE_VOXEL,
			exposure, encode as u32 as f32, NEAR, mode as f32,
			w, h, rays as f32, wire as u32 as f32,
		];
		// The frame number goes as words: as f32 it would stop growing past
		// 2^24 and freeze the field's clock.
		let mut bytes = floats(&uniform);
		bytes.extend(words(&[self.frame, self.dent_changes, 0, 0]));
		bytes.extend(floats(&[
			indirect.gi,
			indirect.ao,
			indirect.radius,
			indirect.traced as u32 as f32,
			fog.density,
			fog.height,
			smooth as u32 as f32,
			0.0,
			fog.color.x,
			fog.color.y,
			fog.color.z,
			0.0,
		]));
		queue.write_buffer(&self.view, 0, &bytes);
		if surface {
			self.raster.draw(pass);
			if let Some(loose) = &self.loose {
				self.raster.draw_loose(pass, loose, self.loose_corners);
			}
			if wire {
				self.shell.wire(pass);
			}
			return;
		}
		if shell {
			self.shell.draw(pass, wire);
			return;
		}
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &self.group, &[]);
		pass.draw(0..3, 0..1);
	}

	/// Carries the frame the world pass drew into `scene` over into
	/// `target`, each drawn pixel shaded darker where `depth` shows the
	/// map's own matter standing near it, and fogged by the air its depth
	/// looks through. The textures come as views
	/// because resize replaces them.
	/// `size` is the frame's, in pixels.
	pub fn occlude(
		&mut self,
		device: &wgpu::Device,
		encoder: &mut wgpu::CommandEncoder,
		target: &wgpu::TextureView,
		scene: &wgpu::TextureView,
		depth: &wgpu::TextureView,
		size: (u32, u32),
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
		);
	}
}

/// Spheres as `count` vec4s, center and radius, the slots past them zero.
fn spheres(spheres: &[(Vec3, f32)], count: usize) -> Vec<u8> {
	let mut slots = vec![[0.0f32; 4]; count];
	for (slot, &(pos, radius)) in slots.iter_mut().zip(spheres) {
		*slot = [pos.x, pos.y, pos.z, radius];
	}
	floats(&slots.concat())
}
