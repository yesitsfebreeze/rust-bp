//! The world's one renderer (`raster.wgsl`): every face's corners in a
//! vertex buffer built at load, drawn at the screen's native resolution
//! into the hardware's own depth test — a triangle off every screen edge
//! never leaves the setup engine. Each frame draws the map's depth first,
//! then lights it testing depth equal, so every pixel is shaded once. The
//! lit fragment takes three hardware-filtered taps — the albedo atlas, the
//! baked light at the corner's lightmap coordinates, and the sun's shadow
//! map — and casts no ray: the map and the sun stand still, so the sun's
//! shadow is baked once at load, the map drawn from the sun into a depth
//! map over its whole extent. The frame's dynamic lights and moving
//! things' shapes come in one uniform, and the playground's spheres draw
//! here too: one unit sphere mesh, instanced from their buffer.

use super::scene::CORNER_BYTES;
use super::util::{self, entry};
use linalg::{vec3, Vec3};

/// The buffers the raster draws with.
pub struct Shared<'a> {
	pub view: &'a wgpu::Buffer,
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

/// The sun's shadow map: its side in texels, at most, and its depth
/// format — sixteen bits hold the map's depth to a fraction of a unit,
/// half the memory of thirty-two.
const SHADOW_SIDE: u32 = 4096;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;

pub struct Raster {
	group: wgpu::BindGroup,
	vertices: wgpu::Buffer,
	/// The depth pass and the lit draw over it.
	depth_pipeline: wgpu::RenderPipeline,
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

/// `Sun` in raster.wgsl: the orthographic view from the sun toward
/// `sun_dir`'s opposite, fit around the box `lo`..`hi` — three rows taking
/// a world point to clip space, x and y -1 to 1 across the box, z 0 on the
/// sun's side to 1 past it — and the world size of one of `side` texels.
fn sun_view(sun_dir: Vec3, lo: Vec3, hi: Vec3, side: u32) -> [f32; 16] {
	let forward = sun_dir.normalize() * -1.0;
	let up = if forward.y.abs() > 0.99 {
		vec3(1.0, 0.0, 0.0)
	} else {
		vec3(0.0, 1.0, 0.0)
	};
	let right = forward.cross(up).normalize();
	let up = right.cross(forward);
	let mut min = Vec3::splat(f32::MAX);
	let mut max = Vec3::splat(f32::MIN);
	for i in 0..8 {
		let p = vec3(
			if i & 1 == 0 { lo.x } else { hi.x },
			if i & 2 == 0 { lo.y } else { hi.y },
			if i & 4 == 0 { lo.z } else { hi.z },
		);
		let at = vec3(p.dot(right), p.dot(up), p.dot(forward));
		min = min.min(at);
		max = max.max(at);
	}
	// A unit of margin on every side, so the box's own faces stand inside.
	let (min, max) = (min - Vec3::splat(1.0), max + Vec3::splat(1.0));
	let row = |axis: Vec3, lo: f32, hi: f32, to: (f32, f32)| {
		let scale = (to.1 - to.0) / (hi - lo);
		let a = axis * scale;
		[a.x, a.y, a.z, to.0 - lo * scale]
	};
	let texel = ((max.x - min.x).max(max.y - min.y)) / side as f32;
	[
		row(right, min.x, max.x, (-1.0, 1.0)),
		row(up, min.y, max.y, (-1.0, 1.0)),
		row(forward, min.z, max.z, (0.0, 1.0)),
		[texel, 0.0, 0.0, 0.0],
	]
	.concat()
	.try_into()
	.unwrap()
}

impl Raster {
	/// `corners` is the scene's packed vertex stream, as `scene::build` wrote
	/// it: position, whole-tile uv, the face's unit normal, its atlas
	/// rectangle, per corner, bounded by `lo`..`hi`. The sun's shadow map is
	/// baked here, toward `sun_dir`.
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		format: wgpu::TextureFormat,
		depth: wgpu::TextureFormat,
		shared: Shared<'_>,
		corners: &[u8],
		bounds: (Vec3, Vec3),
		sun_dir: Vec3,
	) -> Raster {
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
		// The shadow map's comparison, filtered over its four nearest
		// texels.
		let shadow_cmp = device.create_sampler(&wgpu::SamplerDescriptor {
			label: Some("shadow"),
			address_mode_u: wgpu::AddressMode::ClampToEdge,
			address_mode_v: wgpu::AddressMode::ClampToEdge,
			mag_filter: wgpu::FilterMode::Linear,
			min_filter: wgpu::FilterMode::Linear,
			compare: Some(wgpu::CompareFunction::LessEqual),
			..Default::default()
		});
		let side = SHADOW_SIDE.min(device.limits().max_texture_dimension_2d);
		let sun = util::buffer(
			device,
			"sun view",
			64,
			wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		);
		queue.write_buffer(&sun, 0, &util::floats(&sun_view(sun_dir, bounds.0, bounds.1, side)));
		let shadow_map = device
			.create_texture(&wgpu::TextureDescriptor {
				label: Some("shadow map"),
				size: wgpu::Extent3d {
					width: side,
					height: side,
					depth_or_array_layers: 1,
				},
				mip_level_count: 1,
				sample_count: 1,
				dimension: wgpu::TextureDimension::D2,
				format: SHADOW_FORMAT,
				usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
				view_formats: &[],
			})
			.create_view(&Default::default());
		let shader = util::shader(device, "raster", &[include_str!("raster.wgsl")]);
		let vertices = util::vertices(device, "raster corners", corners);
		let corner_count = (corners.len() / CORNER_BYTES) as u32;
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
		let pipeline = |label: &str,
		                layout: &wgpu::PipelineLayout,
		                vs: &str,
		                fs: &str,
		                vertex: &[wgpu::VertexBufferLayout],
		                depth: wgpu::DepthStencilState,
		                color: Option<wgpu::ColorTargetState>| {
			let targets = [color.clone()];
			device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
				label: Some(label),
				layout: Some(layout),
				vertex: wgpu::VertexState {
					module: &shader,
					entry_point: Some(vs),
					compilation_options: Default::default(),
					buffers: vertex,
				},
				primitive: Default::default(),
				depth_stencil: Some(depth),
				multisample: Default::default(),
				fragment: Some(wgpu::FragmentState {
					module: &shader,
					entry_point: Some(fs),
					compilation_options: Default::default(),
					targets: if color.is_some() { &targets } else { &[] },
				}),
				multiview: None,
				cache: None,
			})
		};
		let depth_state = |format, write, compare, bias| wgpu::DepthStencilState {
			format,
			depth_write_enabled: write,
			depth_compare: compare,
			stencil: Default::default(),
			bias,
		};

		// The shadow map's bake: the map from the sun, see-through texels
		// dropped, nearest depth kept, sloped faces pushed back a little so
		// they never shadow themselves.
		let bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("shadow bake"),
			entries: &[
				util::sampler_entry(3, fragment),
				util::texture_entry(5, fragment),
				util::buffer_entry(8, wgpu::ShaderStages::VERTEX, util::UNIFORM),
			],
		});
		let bake_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("shadow bake"),
			layout: &bake_layout,
			entries: &[
				entry(3, wgpu::BindingResource::Sampler(&atlas)),
				entry(5, wgpu::BindingResource::TextureView(shared.albedo_srgb)),
				util::bind(8, &sun),
			],
		});
		let bake = pipeline(
			"shadow bake",
			&util::layout(device, "shadow bake", &[&bake_layout]),
			"vs_shadow",
			"fs_depth",
			buffers,
			depth_state(
				SHADOW_FORMAT,
				true,
				wgpu::CompareFunction::Less,
				wgpu::DepthBiasState {
					constant: 2,
					slope_scale: 1.5,
					clamp: 0.0,
				},
			),
			None,
		);
		let mut encoder = device.create_command_encoder(&Default::default());
		{
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("shadow bake"),
				color_attachments: &[],
				depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
					view: &shadow_map,
					depth_ops: Some(wgpu::Operations {
						load: wgpu::LoadOp::Clear(1.0),
						store: wgpu::StoreOp::Store,
					}),
					stencil_ops: None,
				}),
				timestamp_writes: None,
				occlusion_query_set: None,
			});
			pass.set_pipeline(&bake);
			pass.set_bind_group(0, &bake_group, &[]);
			pass.set_vertex_buffer(0, vertices.slice(..));
			pass.draw(0..corner_count, 0..1);
		}
		queue.submit([encoder.finish()]);

		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("raster"),
			entries: &[
				util::buffer_entry(0, both, util::UNIFORM),
				util::sampler_entry(3, fragment),
				util::texture_entry(5, fragment),
				util::texture_entry(6, fragment),
				util::buffer_entry(7, fragment, util::UNIFORM),
				util::buffer_entry(8, fragment, util::UNIFORM),
				wgpu::BindGroupLayoutEntry {
					binding: 9,
					visibility: fragment,
					ty: wgpu::BindingType::Texture {
						sample_type: wgpu::TextureSampleType::Depth,
						view_dimension: wgpu::TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				wgpu::BindGroupLayoutEntry {
					binding: 10,
					visibility: fragment,
					ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
					count: None,
				},
				util::buffer_entry(11, fragment, util::UNIFORM),
			],
		});
		let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("raster"),
			layout: &layout,
			entries: &[
				util::bind(0, shared.view),
				entry(3, wgpu::BindingResource::Sampler(&atlas)),
				entry(5, wgpu::BindingResource::TextureView(shared.albedo_srgb)),
				entry(6, wgpu::BindingResource::TextureView(shared.baked)),
				util::bind(7, shared.dynamic),
				util::bind(8, &sun),
				entry(9, wgpu::BindingResource::TextureView(&shadow_map)),
				entry(10, wgpu::BindingResource::Sampler(&shadow_cmp)),
				util::bind(11, shared.lighting),
			],
		});
		let raster_layout = util::layout(device, "raster", &[&layout]);
		// Reversed depth, as the overlays draw: nearer is greater.
		let depth_pipeline = pipeline(
			"raster depth",
			&raster_layout,
			"vs_depth",
			"fs_depth",
			buffers,
			depth_state(depth, true, wgpu::CompareFunction::GreaterEqual, Default::default()),
			// The pass's color, untouched: depth alone is written.
			Some(wgpu::ColorTargetState {
				format,
				blend: None,
				write_mask: wgpu::ColorWrites::empty(),
			}),
		);
		let lit = pipeline(
			"raster",
			&raster_layout,
			"vs",
			"fs",
			buffers,
			depth_state(depth, false, wgpu::CompareFunction::Equal, Default::default()),
			Some(format.into()),
		);
		// The spheres: the unit sphere's corners per vertex, each body's
		// center and radius per instance, lit as the surfaces are, tested
		// and written in one go — few pixels, no depth pass.
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
		let body_pipeline = pipeline(
			"raster bodies",
			&raster_layout,
			"vs_body",
			"fs_body",
			&sphere_layout,
			depth_state(depth, true, wgpu::CompareFunction::GreaterEqual, Default::default()),
			Some(format.into()),
		);
		let sphere = unit_sphere();
		Raster {
			group,
			vertices,
			depth_pipeline,
			pipeline: lit,
			corners: corner_count,
			body_pipeline,
			sphere: util::vertices(device, "unit sphere", &util::floats(&sphere.concat())),
			sphere_corners: sphere.len() as u32,
			bodies: shared.bodies.clone(),
			body_slots: shared.body_slots,
		}
	}

	/// Draws the map's triangles and `count` loose corners from `loose` —
	/// props, packed by `scene::loose_corners` — into the pass's color and
	/// depth, from the view the uniform already holds: their depth first,
	/// then lit over it, then the playground's spheres.
	pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, loose: Option<(&wgpu::Buffer, u32)>) {
		let loose = loose.filter(|&(_, count)| count > 0);
		pass.set_bind_group(0, &self.group, &[]);
		for pipeline in [&self.depth_pipeline, &self.pipeline] {
			pass.set_pipeline(pipeline);
			pass.set_vertex_buffer(0, self.vertices.slice(..));
			pass.draw(0..self.corners, 0..1);
			if let Some((buffer, count)) = loose {
				pass.set_vertex_buffer(0, buffer.slice(..));
				pass.draw(0..count, 0..1);
			}
		}
		pass.set_pipeline(&self.body_pipeline);
		pass.set_vertex_buffer(0, self.sphere.slice(..));
		pass.set_vertex_buffer(1, self.bodies.slice(..));
		pass.draw(0..self.sphere_corners, 0..self.body_slots);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The sun's view takes the box it was fit around inside its clip
	/// space, the sun's side nearest.
	#[test]
	fn sun_view_holds_the_map() {
		let (lo, hi) = (vec3(-3000.0, -200.0, -2500.0), vec3(2000.0, 600.0, 3500.0));
		let sun = vec3(0.4, 1.0, 0.25).normalize();
		let m = sun_view(sun, lo, hi, 4096);
		let clip = |p: Vec3| {
			[0, 4, 8].map(|r| m[r] * p.x + m[r + 1] * p.y + m[r + 2] * p.z + m[r + 3])
		};
		for i in 0..8 {
			let p = vec3(
				if i & 1 == 0 { lo.x } else { hi.x },
				if i & 2 == 0 { lo.y } else { hi.y },
				if i & 4 == 0 { lo.z } else { hi.z },
			);
			let [x, y, z] = clip(p);
			assert!(x.abs() < 1.0 && y.abs() < 1.0 && (0.0..1.0).contains(&z), "{p:?}");
		}
		let mid = (lo + hi) * 0.5;
		assert!(clip(mid + sun * 100.0)[2] < clip(mid)[2], "nearer the sun is shallower");
	}
}
