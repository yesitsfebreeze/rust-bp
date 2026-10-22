//! The map's surfaces rasterized as themselves (`raster.wgsl`): every face's
//! corners in a vertex buffer built at load, drawn at the screen's native
//! resolution into the hardware's own depth test — a triangle off every
//! screen edge never leaves the setup engine — and shaded per pixel from
//! two hardware-filtered taps — the albedo atlas and the baked light,
//! sampled at the corner's lightmap coordinates — and one ray toward the
//! sun through the map's hierarchy (`bvh.wgsl`), so the sun's shadow is as
//! sharp as the screen instead of as coarse as the bake's luxels. The
//! frame's dynamic lights and moving things' shapes come in one uniform,
//! and the playground's spheres draw here too: one unit sphere mesh,
//! instanced from their buffer.

use super::scene::CORNER_BYTES;
use super::util::{self, entry};

/// The buffers the raster draws with, at the trace's bindings minus the ray
/// marchers'.
pub struct Shared<'a> {
	pub view: &'a wgpu::Buffer,
	/// The map's triangles and their hierarchy, for the sun's shadow ray.
	pub tris: &'a wgpu::Buffer,
	pub nodes: &'a wgpu::Buffer,
	pub albedo_srgb: &'a wgpu::TextureView,
	/// The bake's atlas: linear radiance in half floats, per chart.
	pub baked: &'a wgpu::TextureView,
	pub lighting: &'a wgpu::Buffer,
	/// `Dynamic` in raster.wgsl: the frame's lights, moving shapes and
	/// switches.
	pub dynamic: &'a wgpu::Buffer,
	/// The playground's spheres, center and radius each, `bodies` slots.
	pub bodies: &'a wgpu::Buffer,
	pub body_slots: u32,
}

pub struct Raster {
	group: wgpu::BindGroup,
	vertices: wgpu::Buffer,
	pipeline: wgpu::RenderPipeline,
	corners: u32,
	/// The spheres: their pipeline, the unit sphere's corners and how many,
	/// and the instance buffer with its slot count.
	body_pipeline: wgpu::RenderPipeline,
	sphere: wgpu::Buffer,
	sphere_corners: u32,
	bodies: wgpu::Buffer,
	body_slots: u32,
}

/// The unit sphere's triangles as corner positions: `STACKS` bands from
/// pole to pole, `SLICES` around.
fn unit_sphere() -> Vec<[f32; 3]> {
	const STACKS: u32 = 12;
	const SLICES: u32 = 18;
	let at = |i: u32, j: u32| {
		let theta = std::f32::consts::PI * i as f32 / STACKS as f32;
		let phi = std::f32::consts::TAU * j as f32 / SLICES as f32;
		[
			theta.sin() * phi.cos(),
			theta.cos(),
			theta.sin() * phi.sin(),
		]
	};
	let mut out = Vec::new();
	for i in 0..STACKS {
		for j in 0..SLICES {
			let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
			out.extend([a, b, c, a, c, d]);
		}
	}
	out
}

impl Raster {
	/// `corners` is the scene's packed vertex stream, as `scene::build` wrote
	/// it: position, whole-tile uv, the face's unit normal, its atlas
	/// rectangle, per corner.
	pub fn new(
		device: &wgpu::Device,
		format: wgpu::TextureFormat,
		depth: wgpu::TextureFormat,
		shared: Shared<'_>,
		corners: &[u8],
	) -> Raster {
		// The vertex stage projects from the view; the fragment shades the
		// drawn pixel from two filtered taps: the albedo atlas and the
		// baked light.
		let both = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
		let fragment = wgpu::ShaderStages::FRAGMENT;
		// Linear between texels, nearest between mip levels: neither atlas
		// has any, and the taps are held inside their tiles and charts.
		let atlas = device.create_sampler(&wgpu::SamplerDescriptor {
			label: Some("raster"),
			address_mode_u: wgpu::AddressMode::ClampToEdge,
			address_mode_v: wgpu::AddressMode::ClampToEdge,
			address_mode_w: wgpu::AddressMode::ClampToEdge,
			mag_filter: wgpu::FilterMode::Linear,
			min_filter: wgpu::FilterMode::Linear,
			mipmap_filter: wgpu::FilterMode::Nearest,
			..Default::default()
		});
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("raster"),
			entries: &[
				util::buffer_entry(0, both, util::UNIFORM),
				util::buffer_entry(1, fragment, util::READ),
				util::buffer_entry(2, fragment, util::READ),
				util::sampler_entry(3, fragment),
				util::texture_entry(5, fragment),
				util::texture_entry(6, fragment),
				util::buffer_entry(7, fragment, util::UNIFORM),
				util::buffer_entry(11, fragment, util::UNIFORM),
			],
		});
		let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("raster"),
			layout: &layout,
			entries: &[
				util::bind(0, shared.view),
				util::bind(1, shared.tris),
				util::bind(2, shared.nodes),
				entry(3, wgpu::BindingResource::Sampler(&atlas)),
				entry(5, wgpu::BindingResource::TextureView(shared.albedo_srgb)),
				entry(6, wgpu::BindingResource::TextureView(shared.baked)),
				util::bind(7, shared.dynamic),
				util::bind(11, shared.lighting),
			],
		});
		let shader = util::shader(
			device,
			"raster",
			&[include_str!("bvh.wgsl"), include_str!("raster.wgsl")],
		);
		// Position, whole-tile uv with the corner's lightmap texel in `zw`,
		// the face's unit normal, its atlas rectangle, its lightmap chart's
		// rectangle, as scene.rs packed them: five vec4s.
		let at = |shader_location, offset, format| wgpu::VertexAttribute {
			offset,
			format,
			shader_location,
		};
		let buffers = &[wgpu::VertexBufferLayout {
			array_stride: CORNER_BYTES as u64,
			step_mode: wgpu::VertexStepMode::Vertex,
			attributes: &[
				at(0, 0, wgpu::VertexFormat::Float32x4),
				at(1, 16, wgpu::VertexFormat::Float32x4),
				at(2, 32, wgpu::VertexFormat::Float32x4),
				at(3, 48, wgpu::VertexFormat::Float32x4),
				at(4, 64, wgpu::VertexFormat::Float32x4),
			],
		}];
		let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("raster"),
			layout: Some(&util::layout(device, "raster", &[&layout])),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: Some("vs"),
				compilation_options: Default::default(),
				buffers,
			},
			primitive: Default::default(),
			depth_stencil: Some(wgpu::DepthStencilState {
				format: depth,
				depth_write_enabled: true,
				// Reversed depth, as the overlays draw.
				depth_compare: wgpu::CompareFunction::GreaterEqual,
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
		// The spheres: the unit sphere's corners per vertex, each body's
		// center and radius per instance, lit as the surfaces are.
		let sphere_layout = [
			wgpu::VertexBufferLayout {
				array_stride: 12,
				step_mode: wgpu::VertexStepMode::Vertex,
				attributes: &[at(0, 0, wgpu::VertexFormat::Float32x3)],
			},
			wgpu::VertexBufferLayout {
				array_stride: 16,
				step_mode: wgpu::VertexStepMode::Instance,
				attributes: &[at(1, 0, wgpu::VertexFormat::Float32x4)],
			},
		];
		let body_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("raster bodies"),
			layout: Some(&util::layout(device, "raster bodies", &[&layout])),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: Some("vs_body"),
				compilation_options: Default::default(),
				buffers: &sphere_layout,
			},
			primitive: Default::default(),
			depth_stencil: Some(wgpu::DepthStencilState {
				format: depth,
				depth_write_enabled: true,
				depth_compare: wgpu::CompareFunction::GreaterEqual,
				stencil: Default::default(),
				bias: Default::default(),
			}),
			multisample: Default::default(),
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: Some("fs_body"),
				compilation_options: Default::default(),
				targets: &[Some(format.into())],
			}),
			multiview: None,
			cache: None,
		});
		let sphere = unit_sphere();
		Raster {
			group,
			vertices: util::vertices(device, "raster corners", corners),
			pipeline,
			corners: (corners.len() / CORNER_BYTES) as u32,
			body_pipeline,
			sphere: util::vertices(device, "unit sphere", &util::floats(&sphere.concat())),
			sphere_corners: sphere.len() as u32,
			bodies: shared.bodies.clone(),
			body_slots: shared.body_slots,
		}
	}

	/// Draws `count` loose corners from `buffer` — props, packed by
	/// `scene::loose_corners` — as the map's triangles draw.
	pub fn draw_loose(&self, pass: &mut wgpu::RenderPass<'_>, buffer: &wgpu::Buffer, count: u32) {
		if count == 0 {
			return;
		}
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &self.group, &[]);
		pass.set_vertex_buffer(0, buffer.slice(..));
		pass.draw(0..count, 0..1);
	}

	/// Draws the map's triangles into the pass's color and depth, from the
	/// view the uniform already holds.
	pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &self.group, &[]);
		pass.set_vertex_buffer(0, self.vertices.slice(..));
		pass.draw(0..self.corners, 0..1);
		pass.set_pipeline(&self.body_pipeline);
		pass.set_vertex_buffer(0, self.sphere.slice(..));
		pass.set_vertex_buffer(1, self.bodies.slice(..));
		pass.draw(0..self.sphere_corners, 0..self.body_slots);
	}
}

#[cfg(test)]
mod tests {
	use super::super::{scene, util};

	/// The surface pass's sun ray — `bvh.wgsl`'s `occluded`, on the
	/// renderer's own triangle and hierarchy buffers — agrees with the CPU
	/// hierarchy the bake casts its sun rays through, from a point on every
	/// sunward fan triangle of the bundled Dust2. Skipped without the map
	/// or a GPU.
	#[test]
	fn sun_rays_match_the_bakes() {
		let path =
			std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/maps/de_dust2.bsp");
		let Ok(map) = crate::bsp::load(&path) else {
			return;
		};
		let (sun, _) = crate::lightmap::sun(&path).unwrap();
		let sun = sun.normalize();
		let scene = scene::build(&map.render, &[]);
		let bvh = lightbake::Bvh::build(&scene.tris);
		let (nodes, order) = bvh.gpu();
		let mut rays = Vec::new();
		let mut want = Vec::new();
		for t in &scene.tris {
			let n = physics::triangle_normal(t);
			if n.dot(sun) <= 0.0 || (t[1] - t[0]).cross(t[2] - t[0]).length2() < 1.0 {
				continue;
			}
			let o = (t[0] + t[1] + t[2]) * (1.0 / 3.0) + n * 0.125;
			want.push(bvh.cast(&scene.tris, o, sun, 1e30).is_some() as u32);
			for v in [o.x, o.y, o.z, 0.0, sun.x, sun.y, sun.z, 0.0] {
				rays.extend_from_slice(&f32::to_le_bytes(v));
			}
		}
		let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
		let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
			return;
		};
		let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
		let tris = util::storage(&device, "tris", &scene.bytes_in(&order));
		let nodes = util::storage(&device, "nodes", &nodes);
		let rays_buf = util::storage(&device, "rays", &rays);
		let size = want.len() as u64 * 4;
		let out = util::buffer(
			&device,
			"out",
			size,
			wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
		);
		let read = util::buffer(
			&device,
			"read",
			size,
			wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
		);
		let shader = util::shader(
			&device,
			"sun rays",
			&[
				include_str!("bvh.wgsl"),
				"struct Ray { o: vec4f, d: vec4f }
				@group(0) @binding(3) var<storage, read> rays: array<Ray>;
				@group(0) @binding(4) var<storage, read_write> out: array<u32>;
				@compute @workgroup_size(64)
				fn main(@builtin(global_invocation_id) g: vec3u) {
					if (g.x >= arrayLength(&rays)) { return; }
					out[g.x] = select(0u, 1u, occluded(rays[g.x].o.xyz, rays[g.x].d.xyz, MISS));
				}",
			],
		);
		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: None,
			layout: None,
			module: &shader,
			entry_point: Some("main"),
			compilation_options: Default::default(),
			cache: None,
		});
		let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &pipeline.get_bind_group_layout(0),
			entries: &[
				util::bind(1, &tris),
				util::bind(2, &nodes),
				util::bind(3, &rays_buf),
				util::bind(4, &out),
			],
		});
		let mut encoder = device.create_command_encoder(&Default::default());
		{
			let mut pass = encoder.begin_compute_pass(&Default::default());
			pass.set_pipeline(&pipeline);
			pass.set_bind_group(0, &group, &[]);
			pass.dispatch_workgroups((want.len() as u32).div_ceil(64), 1, 1);
		}
		encoder.copy_buffer_to_buffer(&out, 0, &read, 0, size);
		queue.submit([encoder.finish()]);
		read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
		device.poll(wgpu::PollType::Wait).unwrap();
		let got: Vec<u32> = read
			.slice(..)
			.get_mapped_range()
			.chunks(4)
			.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect();
		let wrong = got.iter().zip(&want).filter(|(g, w)| g != w).count();
		let shadowed = want.iter().filter(|&&w| w == 1).count();
		assert!(
			wrong * 100 <= want.len(),
			"{wrong} of {} sun rays disagree; the CPU shadows {shadowed}, the GPU {}",
			want.len(),
			got.iter().filter(|&&g| g == 1).count()
		);
	}
}
