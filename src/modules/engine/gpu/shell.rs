//! The shell field (`shell.wgsl`): the map voxelized ahead of the view into
//! one sparse shell per cube size, brought along as the eye moves, and the
//! full-screen pass that marches it in place of the trace.
//!
//! A shell holds the cells a cube size is drawn at — about R to 2R from the
//! eye, R = side / (pixel angle · K) — padded by [`PAD`] cells either way.
//! Once the eye has moved half that, the shell takes in the cells that
//! entered it; every [`REBUILD`] moves, or when the dents, the pixel angle
//! or `K` change, it is cleared and built whole.

use super::util::{self, entry, words};
use linalg::Vec3;

/// Shells, one per cube size from the smallest voxel up, as `SHELLS` in
/// shell.wgsl: the twelfth holds every cube past the eleventh's.
const SHELLS: usize = 12;
/// Entries one shell's table holds and each one's bytes, as `CAPACITY` and
/// `Entry` in shell.wgsl.
const CAPACITY: u64 = 1 << 19;
const ENTRY_BYTES: u64 = 16;
/// Bricks a shell's keys count along each axis from its origin, as
/// `KEY_SPAN` in shell.wgsl; cells past them are not kept.
const KEY_SPAN: i32 = 1024;
/// Words of one shell's pyramid of occupied blocks, as `PYRAMID_WORDS` in
/// shell.wgsl.
const PYRAMID_WORDS: u64 = 532608;
/// Cells a shell reaches past where its cubes are drawn, either way: how
/// far the eye may move before the shell is brought along.
const PAD: f32 = 16.0;
/// Moves a shell takes in new cells for before it is built whole again,
/// dropping the cells it left behind.
const REBUILD: u32 = 8;
/// Bytes between the per-shell dispatch parameters: the uniform offset
/// alignment every adapter accepts.
const WORK_STRIDE: u64 = 256;
/// Triangles one dispatch dimension holds, and the workgroups sharing one
/// triangle's or dent's columns, as `ROW` and `SLICES` in shell.wgsl.
const ROW: u32 = 65535;
const SLICES: u32 = 16;

/// The buffers the shell pass reads with the trace, at the trace's
/// bindings.
pub struct Shared<'a> {
	pub view: &'a wgpu::Buffer,
	pub tris: &'a wgpu::Buffer,
	pub nodes: &'a wgpu::Buffer,
	pub albedo: &'a wgpu::TextureView,
	pub albedo_srgb: &'a wgpu::TextureView,
	pub lighting: &'a wgpu::Buffer,
	pub bodies: &'a wgpu::Buffer,
	pub slots: &'a wgpu::Buffer,
	pub dents: &'a wgpu::Buffer,
}

/// What a shell was last filled for.
#[derive(Clone, Copy)]
struct Filled {
	/// The eye it last took cells in for.
	eye: Vec3,
	/// The brick its keys count from.
	origin: [i32; 3],
	/// Moves since it was built whole.
	moves: u32,
	pixel: f32,
	voxel: f32,
	dents: u32,
}

pub struct Shell {
	table: wgpu::Buffer,
	pyramid: wgpu::Buffer,
	shells: wgpu::Buffer,
	work: wgpu::Buffer,
	group: wgpu::BindGroup,
	pipeline: wgpu::RenderPipeline,
	fill: wgpu::ComputePipeline,
	bowl: wgpu::ComputePipeline,
	triangles: u32,
	dents: u32,
	filled: [Option<Filled>; SHELLS],
}

impl Shell {
	pub fn new(
		device: &wgpu::Device,
		format: wgpu::TextureFormat,
		depth: wgpu::TextureFormat,
		shared: Shared<'_>,
		triangles: usize,
		dents: usize,
	) -> Shell {
		let table = util::buffer(
			device,
			"shell table",
			SHELLS as u64 * CAPACITY * ENTRY_BYTES,
			wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
		);
		let pyramid = util::buffer(
			device,
			"shell pyramid",
			SHELLS as u64 * PYRAMID_WORDS * 4,
			wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
		);
		let shells = util::buffer(
			device,
			"shells",
			SHELLS as u64 * 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		let work = util::buffer(
			device,
			"shell work",
			SHELLS as u64 * WORK_STRIDE,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);

		let stages = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("shell"),
			entries: &[
				util::buffer_entry(0, stages, util::UNIFORM),
				util::buffer_entry(1, stages, util::READ),
				util::buffer_entry(2, stages, util::READ),
				util::texture_entry(4, stages),
				util::texture_entry(5, stages),
				util::buffer_entry(11, stages, util::UNIFORM),
				util::buffer_entry(12, stages, util::READ),
				util::buffer_entry(13, stages, util::WRITE),
				util::buffer_entry(14, stages, util::READ),
				util::buffer_entry(15, stages, util::WRITE),
				util::buffer_entry(16, stages, util::UNIFORM),
				wgpu::BindGroupLayoutEntry {
					binding: 17,
					visibility: stages,
					ty: wgpu::BindingType::Buffer {
						ty: util::UNIFORM,
						has_dynamic_offset: true,
						min_binding_size: wgpu::BufferSize::new(64),
					},
					count: None,
				},
				util::buffer_entry(18, stages, util::WRITE),
			],
		});
		let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("shell"),
			layout: &layout,
			entries: &[
				util::bind(0, shared.view),
				util::bind(1, shared.tris),
				util::bind(2, shared.nodes),
				entry(4, wgpu::BindingResource::TextureView(shared.albedo)),
				entry(5, wgpu::BindingResource::TextureView(shared.albedo_srgb)),
				util::bind(11, shared.lighting),
				util::bind(12, shared.bodies),
				util::bind(13, shared.slots),
				util::bind(14, shared.dents),
				util::bind(15, &table),
				util::bind(16, &shells),
				util::bind(18, &pyramid),
				entry(
					17,
					wgpu::BindingResource::Buffer(wgpu::BufferBinding {
						buffer: &work,
						offset: 0,
						size: wgpu::BufferSize::new(64),
					}),
				),
			],
		});
		let shader = util::shader(
			device,
			"shell",
			&[
				include_str!("bvh.wgsl"),
				include_str!("material.wgsl"),
				include_str!("voxel.wgsl"),
				include_str!("shell.wgsl"),
			],
		);
		let pipeline_layout = util::layout(device, "shell", &[&layout]);
		let compute = |entry: &str| {
			device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
				label: Some(entry),
				layout: Some(&pipeline_layout),
				module: &shader,
				entry_point: Some(entry),
				compilation_options: Default::default(),
				cache: None,
			})
		};
		let (fill, bowl) = (compute("shell_fill"), compute("shell_bowl"));
		let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("shell"),
			layout: Some(&pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: Some("shell_vs"),
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
				entry_point: Some("shell_fs"),
				compilation_options: Default::default(),
				targets: &[Some(format.into())],
			}),
			multiview: None,
			cache: None,
		});
		Shell {
			table,
			pyramid,
			shells,
			work,
			group,
			pipeline,
			fill,
			bowl,
			triangles: triangles as u32,
			dents: dents as u32,
			filled: [None; SHELLS],
		}
	}

	/// Brings every shell along to `eye`: cubes `base · 2^l` wide, `voxel`
	/// pixels each where one pixel spans `pixel` radians; `dents` counts
	/// the dents' changes. Before the frame's draw.
	#[allow(clippy::too_many_arguments)]
	pub fn prepare(
		&mut self,
		queue: &wgpu::Queue,
		encoder: &mut wgpu::CommandEncoder,
		eye: Vec3,
		pixel: f32,
		voxel: f32,
		base: f32,
		dents: u32,
	) {
		let mut due = Vec::new();
		for (l, filled) in self.filled.iter_mut().enumerate() {
			let s = base * (1u32 << l) as f32;
			let r = s / (pixel * voxel);
			let pad = PAD * s;
			let near = if l == 0 { 0.0 } else { (r - pad).max(0.0) };
			let far = if l == SHELLS - 1 {
				1e30
			} else {
				2.0 * r + 0.87 * s + pad
			};
			let brick = 4.0 * s;
			// Whether the shell's keys still reach every brick it holds.
			let keyed = |origin: [i32; 3]| {
				let reach = far.min(65536.0) + s;
				[eye.x, eye.y, eye.z].iter().zip(origin).all(|(&e, o)| {
					((e - reach) / brick).floor() as i32 >= o
						&& (((e + reach) / brick).floor() as i32) < o + KEY_SPAN
				})
			};
			let whole = match filled {
				None => true,
				Some(f) => {
					f.pixel != pixel
						|| f.voxel != voxel
						|| f.dents != dents
						|| f.moves >= REBUILD
						|| !keyed(f.origin)
				}
			};
			let moved = filled.is_none_or(|f| (eye - f.eye).length() > pad * 0.5);
			if !whole && !moved {
				continue;
			}
			let (last, origin, moves) = match filled {
				Some(f) if !whole => (Some(f.eye), f.origin, f.moves + 1),
				// Keys count from a multiple of 64 bricks, so the pyramid's
				// blocks of every grain count from a whole one.
				_ => (
					None,
					[eye.x, eye.y, eye.z].map(|e| (e / (64.0 * brick)).floor() as i32 * 64 - KEY_SPAN / 2),
					0,
				),
			};
			*filled = Some(Filled {
				eye,
				origin,
				moves,
				pixel,
				voxel,
				dents,
			});
			// `Work` in shell.wgsl.
			let back = last.unwrap_or(eye);
			let mut bytes = util::floats(&[
				eye.x,
				eye.y,
				eye.z,
				s,
				back.x,
				back.y,
				back.z,
				last.is_some() as u32 as f32,
				near,
				far,
				0.0,
				0.0,
			]);
			bytes.extend(words(&[l as u32, 0, 0, 0]));
			queue.write_buffer(&self.work, l as u64 * WORK_STRIDE, &bytes);
			if whole {
				let size = CAPACITY * ENTRY_BYTES;
				encoder.clear_buffer(&self.table, l as u64 * size, Some(size));
				let size = PYRAMID_WORDS * 4;
				encoder.clear_buffer(&self.pyramid, l as u64 * size, Some(size));
			}
			due.push(l);
		}
		if due.is_empty() {
			return;
		}
		let origins: Vec<u32> = self
			.filled
			.iter()
			.flat_map(|f| match f {
				Some(f) => [
					f.origin[0] as u32,
					f.origin[1] as u32,
					f.origin[2] as u32,
					1,
				],
				None => [0; 4],
			})
			.collect();
		queue.write_buffer(&self.shells, 0, &words(&origins));
		let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
			label: Some("shell"),
			timestamp_writes: None,
		});
		let tris = self.triangles;
		for l in due {
			pass.set_bind_group(0, &self.group, &[(l as u64 * WORK_STRIDE) as u32]);
			pass.set_pipeline(&self.fill);
			pass.dispatch_workgroups(tris.min(ROW), SLICES, tris.div_ceil(ROW));
			pass.set_pipeline(&self.bowl);
			pass.dispatch_workgroups(self.dents, SLICES, 1);
		}
	}

	/// Marches the frame through the shells into the pass's color and
	/// depth, from the view the trace's uniform holds.
	pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &self.group, &[0]);
		pass.draw(0..3, 0..1);
	}
}
