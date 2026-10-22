//! The traced world: the map's triangles and hierarchy on the GPU, its
//! texture atlas, the voxel field's stored colors, and the full-screen pass
//! that casts every pixel's ray (`bvh.wgsl` + `material.wgsl` +
//! `voxel.wgsl` + `trace.wgsl`), or in its place the march through the
//! shell field (`shell.rs`).

use super::util::{self, entry, floats, words};
use super::{scene, shell, Color, View, NEAR};
use crate::bsp::Map;
use linalg::Vec3;

/// The smallest voxel, world units: the finest matter, right at the eye. A
/// sixteenth of a unit keeps cubes `K` pixels wide even at arm's length.
const BASE_VOXEL: f32 = 1.0 / 16.0;
/// Slots storing the voxels' colors: a power of two, more than a screen of
/// voxels at any setting; and each slot's bytes, as `Slot` in voxel.wgsl.
const FIELD_SLOTS: u32 = 1 << 20;
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
	bodies: wgpu::Buffer,
	dents: wgpu::Buffer,
	group: wgpu::BindGroup,
	pipeline: wgpu::RenderPipeline,
	/// The map voxelized ahead of the view, marched instead of traced when
	/// asked.
	shell: shell::Shell,
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
	) -> World {
		let scene = scene::build(&map.render);
		let (nodes, order) = lightbake::Bvh::build(&scene.tris).gpu();
		// The triangles in the hierarchy's leaf order, so its leaves index
		// them directly and no order buffer is read per test.
		let tris = util::storage(device, "tris", &scene.bytes_in(&order));
		let nodes = util::storage(device, "nodes", &nodes);
		let atlas = &map.render.albedo;
		// sRGB bytes, read raw and through an sRGB view.
		let (albedo, albedo_srgb) =
			util::image(device, queue, "albedo", atlas.side, atlas.side, &atlas.rgba);
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
			8 * 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		// The playground's spheres, written whole every frame.
		let bodies = util::buffer(
			device,
			"bodies",
			MAX_BODIES * 16,
			wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
				lighting: &lighting,
				bodies: &bodies,
				slots: &slots,
				dents: &dents,
			},
			scene.tris.len(),
			MAX_DENTS,
		);
		World {
			view,
			bodies,
			dents,
			group,
			pipeline,
			shell,
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

	/// The playground's spheres: center and radius each, in `bodies`.
	pub fn set_bodies(&self, queue: &wgpu::Queue, bodies: &[(Vec3, f32)]) {
		queue.write_buffer(&self.bodies, 0, &spheres(bodies, MAX_BODIES as usize));
	}

	/// The dents carved into the map: center and radius each, in `dents`;
	/// past [`MAX_DENTS`] the rest are dropped.
	pub fn set_dents(&mut self, queue: &wgpu::Queue, dents: &[(Vec3, f32)]) {
		queue.write_buffer(&self.dents, 0, &spheres(dents, MAX_DENTS));
		self.dent_changes = self.dent_changes.wrapping_add(1);
	}

	/// Brings the shell field along to `view` for this frame's march,
	/// cubes `voxel` pixels wide on a screen `size` pixels. Before the
	/// frame's [`World::draw`] with `shell`.
	pub fn prepare(
		&mut self,
		queue: &wgpu::Queue,
		encoder: &mut wgpu::CommandEncoder,
		view: &View,
		size: (f32, f32),
		voxel: f32,
	) {
		let pixel = 2.0 * (view.fovy * 0.5).tan() / size.1;
		self.shell.prepare(
			queue,
			encoder,
			view.eye,
			pixel,
			voxel,
			BASE_VOXEL,
			self.dent_changes,
		);
	}

	/// Traces the frame from `view` into the pass's color and depth; with
	/// `shell`, marches the shell field [`World::prepare`] brought along.
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
		shell: bool,
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
			w, h, 0.0, 0.0,
		];
		// The frame number goes as words: as f32 it would stop growing past
		// 2^24 and freeze the field's clock.
		let mut bytes = floats(&uniform);
		bytes.extend(words(&[self.frame, self.dent_changes, 0, 0]));
		queue.write_buffer(&self.view, 0, &bytes);
		if shell {
			self.shell.draw(pass);
			return;
		}
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &self.group, &[]);
		pass.draw(0..3, 0..1);
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
