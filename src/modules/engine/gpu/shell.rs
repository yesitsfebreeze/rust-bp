//! The shell field (`shell.wgsl`): the map voxelized ahead of the view into
//! one sparse shell per cube size, brought along as the eye moves, and the
//! full-screen pass that marches it in place of the trace.
//!
//! A shell holds the cells a cube size may be drawn at, about its R to
//! the next size's from the eye, with half a band more either way where
//! neighbouring sizes overlap to blend, padded by [`PAD`] cells. The Rs
//! follow a curve over distance ([`Lod`]): the smallest cubes out to
//! `near`, then `curve` sizes up per doubling of distance.
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
/// Words of one shell's pyramid of occupied blocks, as `PYRAMID_WORDS` in
/// shell.wgsl.
const PYRAMID_WORDS: u64 = 532608;
/// Pixels whose march start is kept, as `REACH_PIXELS` in shell.wgsl: a
/// 3840 by 2160 screen.
const REACH_PIXELS: u64 = 3840 * 2160;
/// Bricks a shell's keys count along each axis from its origin, as
/// `KEY_SPAN` in shell.wgsl; cells past them are not kept.
const KEY_SPAN: i32 = 1024;
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
/// The playground's spheres at most, as in world.rs.
const BODIES: u32 = 64;

/// The buffers the shell pass reads with the trace, at the trace's
/// bindings.
pub struct Shared<'a> {
	pub view: &'a wgpu::Buffer,
	pub tris: &'a wgpu::Buffer,
	pub nodes: &'a wgpu::Buffer,
	pub albedo: &'a wgpu::TextureView,
	pub albedo_srgb: &'a wgpu::TextureView,
	/// The bake's light atlas: linear radiance in half floats, one chart
	/// per face, read unfiltered at its texels.
	pub baked: &'a wgpu::TextureView,
	pub lighting: &'a wgpu::Buffer,
	pub slots: &'a wgpu::Buffer,
}

/// Where the cube sizes change over distance: the smallest cubes are
/// drawn out to `near` world units, then `curve` sizes up per doubling of
/// distance, so size `l` begins at `near · 2^(l / curve)`. A curve of 1
/// keeps cubes the same width on screen; below 1 they grow slower and
/// hold detail farther.
#[derive(Clone, Copy, PartialEq)]
pub struct Lod {
	pub near: f32,
	pub curve: f32,
}

impl Lod {
	/// Where size `l` begins.
	fn r(self, l: usize) -> f32 {
		self.near * (l as f32 / self.curve).exp2()
	}

	/// Half the overlap of neighbouring sizes, as a share of the R they
	/// switch at: half the band below it.
	fn overlap(self) -> f32 {
		0.5 * (1.0 - (-1.0 / self.curve).exp2())
	}
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
	lod: Lod,
	dents: u32,
}

pub struct Shell {
	table: wgpu::Buffer,
	pyramid: wgpu::Buffer,
	/// Per pixel, where its march starts, and the rows' least between the
	/// passes that take it.
	/// Kept so the bind group's buffers outlive it.
	_reach: [wgpu::Buffer; 2],
	/// The spheres and the dents, as the fill and the march read them:
	/// uniforms, a fixed count each.
	bodies: wgpu::Buffer,
	dents: wgpu::Buffer,
	shells: wgpu::Buffer,
	work: wgpu::Buffer,
	group: wgpu::BindGroup,
	pipeline: wgpu::RenderPipeline,
	/// The map's triangles as pink wireframe, drawn over the march.
	wire_pipeline: wgpu::RenderPipeline,
	fill: wgpu::ComputePipeline,
	bowl: wgpu::ComputePipeline,
	/// Each pixel's nearest triangle, then the least of it around: where
	/// its march starts.
	starts: [wgpu::ComputePipeline; 3],
	triangles: u32,
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
		// The dents the fill carves with, and the spheres the march walks,
		// as uniform arrays.
		let dents = util::buffer(
			device,
			"shell dents",
			dents as u64 * 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		let bodies = util::buffer(
			device,
			"shell bodies",
			BODIES as u64 * 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		let reach = ["shell reach", "shell reach rows"]
			.map(|label| util::buffer(device, label, REACH_PIXELS * 4, wgpu::BufferUsages::STORAGE));
		let shells = util::buffer(
			device,
			"shells",
			SHELLS as u64 * 16 + 16,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		let work = util::buffer(
			device,
			"shell work",
			SHELLS as u64 * WORK_STRIDE,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);

		let all = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
		// The vertex stages read the view and the triangles: the wireframe
		// projects the mesh's corners itself.
		let seen = all | wgpu::ShaderStages::VERTEX;
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("shell"),
			entries: &[
				util::buffer_entry(0, seen, util::UNIFORM),
				util::buffer_entry(1, seen, util::READ),
				util::buffer_entry(2, all, util::READ),
				util::texture_entry(4, all),
				util::texture_entry(5, all),
				util::texture_entry(6, all),
				util::buffer_entry(11, all, util::UNIFORM),
				util::buffer_entry(12, all, util::UNIFORM),
				util::buffer_entry(13, all, util::WRITE),
				util::buffer_entry(14, all, util::UNIFORM),
				util::buffer_entry(15, all, util::WRITE),
				util::buffer_entry(16, seen, util::UNIFORM),
				wgpu::BindGroupLayoutEntry {
					binding: 17,
					visibility: all,
					ty: wgpu::BindingType::Buffer {
						ty: util::UNIFORM,
						has_dynamic_offset: true,
						min_binding_size: wgpu::BufferSize::new(64),
					},
					count: None,
				},
				util::buffer_entry(18, all, util::WRITE),
				util::buffer_entry(
					19,
					wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
					util::WRITE,
				),
				util::buffer_entry(20, wgpu::ShaderStages::COMPUTE, util::WRITE),
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
				entry(6, wgpu::BindingResource::TextureView(shared.baked)),
				util::bind(11, shared.lighting),
				util::bind(12, &bodies),
				util::bind(13, shared.slots),
				util::bind(14, &dents),
				util::bind(15, &table),
				util::bind(16, &shells),
				entry(
					17,
					wgpu::BindingResource::Buffer(wgpu::BufferBinding {
						buffer: &work,
						offset: 0,
						size: wgpu::BufferSize::new(64),
					}),
				),
				util::bind(18, &pyramid),
				util::bind(19, &reach[0]),
				util::bind(20, &reach[1]),
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
		let compute = |entry_point: &str| {
			device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
				label: Some(entry_point),
				layout: Some(&pipeline_layout),
				module: &shader,
				entry_point: Some(entry_point),
				compilation_options: Default::default(),
				cache: None,
			})
		};
		let (fill, bowl) = (compute("shell_fill"), compute("shell_bowl"));
		let starts = [
			compute("shell_reach"),
			compute("shell_reach_rows"),
			compute("shell_reach_columns"),
		];
		// Reversed depth, as the overlays draw: the march writes its own
		// near / view depth per pixel.
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
		// The map's triangles as pink lines over the march: every triangle
		// three edges, straight out of the trace's triangle buffer, drawn
		// whatever depth the march left.
		let wire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("wire_vs"),
			layout: Some(&pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: Some("wire_vs"),
				compilation_options: Default::default(),
				buffers: &[],
			},
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::LineList,
				..Default::default()
			},
			depth_stencil: Some(wgpu::DepthStencilState {
				format: depth,
				depth_write_enabled: false,
				depth_compare: wgpu::CompareFunction::Always,
				stencil: Default::default(),
				bias: Default::default(),
			}),
			multisample: Default::default(),
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: Some("wire_fs"),
				compilation_options: Default::default(),
				targets: &[Some(format.into())],
			}),
			multiview: None,
			cache: None,
		});
		Shell {
			table,
			pyramid,
			_reach: reach,
			bodies,
			dents,
			shells,
			work,
			group,
			pipeline,
			wire_pipeline,
			fill,
			bowl,
			starts,
			triangles: triangles as u32,
			filled: [None; SHELLS],
		}
	}

	/// The playground's spheres, as the march reads them; called with the
	/// spheres the trace's pass reads.
	pub fn set_bodies(&self, queue: &wgpu::Queue, bodies: &[(Vec3, f32)]) {
		let count = self.bodies.size() as usize / 16;
		let mut slots = vec![[0.0f32; 4]; count];
		for (slot, &(pos, radius)) in slots.iter_mut().zip(bodies) {
			*slot = [pos.x, pos.y, pos.z, radius];
		}
		queue.write_buffer(&self.bodies, 0, &util::floats(&slots.concat()));
	}

	/// The dents carved into the map, as the fill reads them; called with
	/// the dents the trace's pass reads.
	pub fn set_dents(&self, queue: &wgpu::Queue, dents: &[(Vec3, f32)]) {
		let count = self.dents.size() as usize / 16;
		let mut slots = vec![[0.0f32; 4]; count];
		for (slot, &(pos, radius)) in slots.iter_mut().zip(dents) {
			*slot = [pos.x, pos.y, pos.z, radius];
		}
		queue.write_buffer(&self.dents, 0, &util::floats(&slots.concat()));
	}

	/// Brings every shell along to `eye`: cubes `base · 2^l` wide, drawn
	/// over distance as `lod` says, one pixel spanning `pixel` radians;
	/// `dents` counts the dents' changes. Before the frame's draw.
	#[allow(clippy::too_many_arguments)]
	pub fn prepare(
		&mut self,
		queue: &wgpu::Queue,
		encoder: &mut wgpu::CommandEncoder,
		eye: Vec3,
		pixel: f32,
		lod: Lod,
		base: f32,
		dents: u32,
		size: (f32, f32),
	) {
		let mut due = Vec::new();
		let a = lod.overlap();
		for (l, filled) in self.filled.iter_mut().enumerate() {
			let s = base * (1u32 << l) as f32;
			let pad = PAD * s;
			// Drawn from its cut, at the nearest the overlap's start before
			// its R, to its parent's at the farthest, give or take a cell's
			// corner.
			let near = if l == 0 {
				0.0
			} else {
				(lod.r(l) * (1.0 - a) - 2.87 * s - pad).max(0.0)
			};
			let far = if l == SHELLS - 1 {
				1e30
			} else {
				lod.r(l + 1) * (1.0 + a) + 4.87 * s + pad
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
				Some(f) => f.lod != lod || f.dents != dents || f.moves >= REBUILD || !keyed(f.origin),
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
				lod,
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
		// `Shells` in shell.wgsl: the origins, then the curve, the overlap,
		// and the most pixels a cube spans where it is drawn nearest.
		let widest = (1..SHELLS)
			.map(|l| base * (1u32 << l) as f32 / (lod.r(l) * (1.0 - a) * pixel))
			.fold(1.0f32, f32::max);
		if !due.is_empty() {
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
			let mut bytes = words(&origins);
			bytes.extend(util::floats(&[lod.near, lod.curve, a, widest]));
			queue.write_buffer(&self.shells, 0, &bytes);
		}
		let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
			label: Some("shell"),
			timestamp_writes: None,
		});
		if !due.is_empty() {
			let tris = self.triangles;
			let bowls = self.dents.size() as u32 / 16;
			for l in due {
				pass.set_bind_group(0, &self.group, &[(l as u64 * WORK_STRIDE) as u32]);
				pass.set_pipeline(&self.fill);
				pass.dispatch_workgroups(tris.min(ROW), SLICES, tris.div_ceil(ROW));
				pass.set_pipeline(&self.bowl);
				pass.dispatch_workgroups(bowls, SLICES, 1);
			}
		}
		// Every pixel's march starts just before the nearest triangle
		// around it, the least over a cube's width of pixels.
		pass.set_bind_group(0, &self.group, &[0]);
		let groups = ((size.0 as u32).div_ceil(8), (size.1 as u32).div_ceil(8));
		for start in &self.starts {
			pass.set_pipeline(start);
			pass.dispatch_workgroups(groups.0, groups.1, 1);
		}
	}

	/// Marches the frame through the shells into the pass's color and
	/// depth, from the view the trace's uniform holds, and draws the map's
	/// triangles as a pink wireframe over it when `wire`.
	pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, wire: bool) {
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &self.group, &[0]);
		pass.draw(0..3, 0..1);
		if wire {
			self.wire(pass);
		}
	}

	/// The map's triangles as a pink wireframe, over whatever the pass
	/// already holds.
	pub fn wire(&self, pass: &mut wgpu::RenderPass<'_>) {
		pass.set_pipeline(&self.wire_pipeline);
		pass.set_bind_group(0, &self.group, &[0]);
		pass.draw(0..6 * self.triangles, 0..1);
	}
}
