//! What draws over the traced world, recorded through the frame and drawn
//! at its end: world-space lines and boxes tested against the world's depth;
//! the viewmodel's textured triangles and stand-in boxes over everything,
//! hidden only behind themselves; then the flat HUD. Every batch goes up in
//! one write to one vertex buffer, kept and grown across frames.

use super::text::{self, Glyphs};
use super::util::floats;
use super::{Color, View, NEAR};
use linalg::{vec3, Vec3};
use std::ops::Range;

/// One overlay vertex: world units for world draws, pixels for the HUD.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
	pub pos: [f32; 3],
	pub uv: [f32; 2],
	pub color: [f32; 4],
}

impl Vertex {
	pub fn new(pos: Vec3, uv: [f32; 2], color: Color) -> Vertex {
		Vertex {
			pos: [pos.x, pos.y, pos.z],
			uv,
			color: [color.r, color.g, color.b, color.a],
		}
	}
}

/// An image registered with [`Overlay::texture`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureId(usize);

fn bytes(vertices: &[Vertex]) -> &[u8] {
	// SAFETY: `Vertex` is `repr(C)` with no padding (nine `f32`s), so every
	// byte of the slice is initialized.
	unsafe { std::slice::from_raw_parts(vertices.as_ptr().cast(), std::mem::size_of_val(vertices)) }
}

/// A screen quad's corners, cut to `clip` — x0, y0, x1, y1 — when there
/// is one, as triangles: the polygon left after each of the clip's four
/// edges cuts it, fanned from its first corner, its texture coordinates
/// and colors carried along each cut. A quad wholly outside leaves nothing.
fn quad(
	screen: &mut Vec<Vertex>,
	corners: [[f32; 2]; 4],
	uv: [[f32; 2]; 4],
	colors: [Color; 4],
	clip: Option<[f32; 4]>,
) {
	let vertex = |i: usize| {
		let c = colors[i];
		[corners[i][0], corners[i][1], uv[i][0], uv[i][1], c.r, c.g, c.b, c.a]
	};
	let mut poly: Vec<[f32; 8]> = (0..4).map(vertex).collect();
	if let Some([x0, y0, x1, y1]) = clip {
		// Each edge as (axis, bound, keep below it).
		for (axis, bound, below) in [(0, x0, false), (0, x1, true), (1, y0, false), (1, y1, true)] {
			let inside = |v: &[f32; 8]| if below { v[axis] <= bound } else { v[axis] >= bound };
			let mut out = Vec::with_capacity(poly.len() + 2);
			for k in 0..poly.len() {
				let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
				if inside(&a) {
					out.push(a);
				}
				if inside(&a) != inside(&b) {
					let t = (bound - a[axis]) / (b[axis] - a[axis]);
					let mut m = [0.0; 8];
					for j in 0..8 {
						m[j] = a[j] + (b[j] - a[j]) * t;
					}
					out.push(m);
				}
			}
			poly = out;
			if poly.len() < 3 {
				return;
			}
		}
	}
	for k in 1..poly.len() - 1 {
		for v in [poly[0], poly[k], poly[k + 1]] {
			screen.push(Vertex {
				pos: [v[0], v[1], 0.0],
				uv: [v[2], v[3]],
				color: [v[4], v[5], v[6], v[7]],
			});
		}
	}
}

fn span(start: usize, batch: &[Vertex]) -> Range<u32> {
	start as u32..(start + batch.len()) as u32
}

/// The 36 corners of a box's twelve triangles.
fn box_corners(center: Vec3, size: Vec3) -> impl Iterator<Item = Vec3> {
	let h = size * 0.5;
	let corner = move |i: usize| {
		center
			+ vec3(
				if i & 1 == 1 { h.x } else { -h.x },
				if i & 2 == 2 { h.y } else { -h.y },
				if i & 4 == 4 { h.z } else { -h.z },
			)
	};
	// Each face as two triangles over its corner indices.
	const FACES: [[usize; 4]; 6] = [
		[0, 2, 6, 4],
		[1, 5, 7, 3],
		[0, 4, 5, 1],
		[2, 3, 7, 6],
		[0, 1, 3, 2],
		[4, 6, 7, 5],
	];
	FACES
		.into_iter()
		.flat_map(|[a, b, c, d]| [a, b, c, a, c, d])
		.map(corner)
}

pub struct Overlay {
	camera: wgpu::Buffer,
	camera_group: wgpu::BindGroup,
	image_layout: wgpu::BindGroupLayout,
	repeat: wgpu::Sampler,
	textures: Vec<wgpu::BindGroup>,
	glyph_texture: wgpu::Texture,
	glyph_group: wgpu::BindGroup,
	glyphs: Glyphs,
	lines: Vec<Vertex>,
	boxes: Vec<Vertex>,
	front_boxes: Vec<Vertex>,
	/// Every mesh's vertices, and each mesh's image and range of them.
	mesh_vertices: Vec<Vertex>,
	meshes: Vec<(TextureId, Range<u32>)>,
	screen: Vec<Vertex>,
	/// Where screen drawing is cut to, x0, y0, x1, y1; none draws anywhere.
	clip: Option<[f32; 4]>,
	/// The frame's batches end to end, and the buffer they go up in.
	all: Vec<Vertex>,
	vertices: Option<wgpu::Buffer>,
	line_pipeline: wgpu::RenderPipeline,
	box_pipeline: wgpu::RenderPipeline,
	mesh_pipeline: wgpu::RenderPipeline,
	screen_pipeline: wgpu::RenderPipeline,
}

impl Overlay {
	pub fn new(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		format: wgpu::TextureFormat,
		depth: wgpu::TextureFormat,
	) -> Result<Overlay, String> {
		let camera = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("overlay camera"),
			size: 5 * 16,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("overlay camera"),
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::VERTEX,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
		});
		let camera_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("overlay camera"),
			layout: &camera_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: camera.as_entire_binding(),
			}],
		});
		let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("overlay image"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture {
						sample_type: wgpu::TextureSampleType::Float { filterable: true },
						view_dimension: wgpu::TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
					count: None,
				},
			],
		});
		let linear = |address: wgpu::AddressMode| {
			device.create_sampler(&wgpu::SamplerDescriptor {
				address_mode_u: address,
				address_mode_v: address,
				mag_filter: wgpu::FilterMode::Linear,
				min_filter: wgpu::FilterMode::Linear,
				..Default::default()
			})
		};
		let repeat = linear(wgpu::AddressMode::Repeat);
		let clamp = linear(wgpu::AddressMode::ClampToEdge);

		// The glyph atlas, its solid corner filled.
		let glyph_texture = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("glyphs"),
			size: wgpu::Extent3d {
				width: text::SIDE,
				height: text::SIDE,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::R8Unorm,
			usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
			view_formats: &[],
		});
		let solid = [255u8; (text::SOLID * text::SOLID) as usize];
		write_glyphs(
			queue,
			&glyph_texture,
			0,
			0,
			text::SOLID,
			text::SOLID,
			&solid,
		);
		let glyph_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("glyphs"),
			layout: &image_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::TextureView(
						&glyph_texture.create_view(&Default::default()),
					),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::Sampler(&clamp),
				},
			],
		});

		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("overlay"),
			source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
		});
		let plain = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("overlay plain"),
			bind_group_layouts: &[&camera_layout],
			push_constant_ranges: &[],
		});
		let imaged = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("overlay imaged"),
			bind_group_layouts: &[&camera_layout, &image_layout],
			push_constant_ranges: &[],
		});
		let vertex = [wgpu::VertexBufferLayout {
			array_stride: std::mem::size_of::<Vertex>() as u64,
			step_mode: wgpu::VertexStepMode::Vertex,
			attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4],
		}];
		let blended = Some(wgpu::ColorTargetState {
			format,
			blend: Some(wgpu::BlendState::ALPHA_BLENDING),
			write_mask: wgpu::ColorWrites::ALL,
		});
		let tested = Some(wgpu::DepthStencilState {
			format: depth,
			depth_write_enabled: true,
			// Reversed depth: nearer is greater.
			depth_compare: wgpu::CompareFunction::GreaterEqual,
			stencil: Default::default(),
			bias: Default::default(),
		});
		let pipeline = |label: &str,
		                layout: &wgpu::PipelineLayout,
		                vs: &str,
		                fs: &str,
		                topology: wgpu::PrimitiveTopology,
		                depth: Option<wgpu::DepthStencilState>| {
			device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
				label: Some(label),
				layout: Some(layout),
				vertex: wgpu::VertexState {
					module: &shader,
					entry_point: Some(vs),
					compilation_options: Default::default(),
					buffers: &vertex,
				},
				primitive: wgpu::PrimitiveState {
					topology,
					..Default::default()
				},
				depth_stencil: depth,
				multisample: Default::default(),
				fragment: Some(wgpu::FragmentState {
					module: &shader,
					entry_point: Some(fs),
					compilation_options: Default::default(),
					targets: std::slice::from_ref(&blended),
				}),
				multiview: None,
				cache: None,
			})
		};
		use wgpu::PrimitiveTopology::{LineList, TriangleList};
		Ok(Overlay {
			line_pipeline: pipeline(
				"lines",
				&plain,
				"vs_world",
				"fs_color",
				LineList,
				tested.clone(),
			),
			box_pipeline: pipeline(
				"boxes",
				&plain,
				"vs_world",
				"fs_color",
				TriangleList,
				tested.clone(),
			),
			mesh_pipeline: pipeline(
				"meshes",
				&imaged,
				"vs_world",
				"fs_textured",
				TriangleList,
				tested,
			),
			screen_pipeline: pipeline(
				"screen",
				&imaged,
				"vs_screen",
				"fs_screen",
				TriangleList,
				None,
			),
			camera,
			camera_group,
			image_layout,
			repeat,
			textures: Vec::new(),
			glyph_texture,
			glyph_group,
			glyphs: Glyphs::new()?,
			lines: Vec::new(),
			boxes: Vec::new(),
			front_boxes: Vec::new(),
			mesh_vertices: Vec::new(),
			meshes: Vec::new(),
			screen: Vec::new(),
			clip: None,
			all: Vec::new(),
			vertices: None,
		})
	}

	pub fn texture(
		&mut self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		width: u32,
		height: u32,
		rgba: &[u8],
	) -> TextureId {
		let view = super::util::image(device, queue, "overlay image", width, height, rgba).0;
		self
			.textures
			.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("overlay image"),
				layout: &self.image_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(&view),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::Sampler(&self.repeat),
					},
				],
			}));
		TextureId(self.textures.len() - 1)
	}

	pub fn clear_textures(&mut self) {
		self.textures.clear();
	}

	pub fn line_3d(&mut self, from: Vec3, to: Vec3, color: Color) {
		self.lines.push(Vertex::new(from, [0.0; 2], color));
		self.lines.push(Vertex::new(to, [0.0; 2], color));
	}

	pub fn cube(&mut self, center: Vec3, size: Vec3, color: Color) {
		self
			.boxes
			.extend(box_corners(center, size).map(|p| Vertex::new(p, [0.0; 2], color)));
	}

	pub fn front_cube(&mut self, center: Vec3, size: Vec3, color: Color) {
		self
			.front_boxes
			.extend(box_corners(center, size).map(|p| Vertex::new(p, [0.0; 2], color)));
	}

	pub fn mesh(&mut self, texture: TextureId, vertices: impl IntoIterator<Item = Vertex>) {
		let start = self.mesh_vertices.len() as u32;
		self.mesh_vertices.extend(vertices);
		self
			.meshes
			.push((texture, start..self.mesh_vertices.len() as u32));
	}

	fn solid_uv() -> [[f32; 2]; 4] {
		let at = 0.5 * text::SOLID as f32 / text::SIDE as f32;
		[[at, at]; 4]
	}

	pub fn rect(&mut self, x: f32, y: f32, width: f32, height: f32, color: Color) {
		let corners = [
			[x, y],
			[x + width, y],
			[x + width, y + height],
			[x, y + height],
		];
		quad(&mut self.screen, corners, Self::solid_uv(), [color; 4], self.clip);
	}

	/// A rectangle shaded between its corners' colors: top left, top right,
	/// bottom right, bottom left.
	pub fn gradient(&mut self, x: f32, y: f32, width: f32, height: f32, colors: [Color; 4]) {
		let corners = [
			[x, y],
			[x + width, y],
			[x + width, y + height],
			[x, y + height],
		];
		quad(&mut self.screen, corners, Self::solid_uv(), colors, self.clip);
	}

	/// Cuts the screen drawing that follows to `rect`, x, y, width, height,
	/// or lets it draw anywhere again with none.
	pub fn set_clip(&mut self, rect: Option<[f32; 4]>) {
		self.clip = rect.map(|[x, y, w, h]| [x, y, x + w, y + h]);
	}

	pub fn line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, color: Color) {
		let (dx, dy) = (x2 - x1, y2 - y1);
		let length = (dx * dx + dy * dy).sqrt().max(1e-6);
		let (nx, ny) = (
			-dy / length * thickness * 0.5,
			dx / length * thickness * 0.5,
		);
		let corners = [
			[x1 + nx, y1 + ny],
			[x2 + nx, y2 + ny],
			[x2 - nx, y2 - ny],
			[x1 - nx, y1 - ny],
		];
		quad(&mut self.screen, corners, Self::solid_uv(), [color; 4], self.clip);
	}

	pub fn text(&mut self, queue: &wgpu::Queue, line: &str, x: f32, y: f32, size: f32, color: Color) {
		let set = glyph_set(&mut self.glyphs, &self.glyph_texture, queue, size);
		let side = text::SIDE as f32;
		let mut pen = x;
		for c in line.chars() {
			let g = text::glyph(set, c);
			let [gx, gy, w, h] = g.rect.map(|v| v as f32);
			if w > 0.0 {
				let left = (pen + g.xmin).round();
				let top = (y - h - g.ymin).round();
				let corners = [
					[left, top],
					[left + w, top],
					[left + w, top + h],
					[left, top + h],
				];
				let uv = [
					[gx / side, gy / side],
					[(gx + w) / side, gy / side],
					[(gx + w) / side, (gy + h) / side],
					[gx / side, (gy + h) / side],
				];
				quad(&mut self.screen, corners, uv, [color; 4], self.clip);
			}
			pen += g.advance;
		}
	}

	pub fn text_width(&mut self, queue: &wgpu::Queue, line: &str, size: f32) -> f32 {
		let set = glyph_set(&mut self.glyphs, &self.glyph_texture, queue, size);
		line.chars().map(|c| text::glyph(set, c).advance).sum()
	}

	pub fn clear(&mut self) {
		self.lines.clear();
		self.boxes.clear();
		self.front_boxes.clear();
		self.mesh_vertices.clear();
		self.meshes.clear();
		self.screen.clear();
		self.clip = None;
	}

	/// Draws the frame's batches: with a view, the world overlays and then
	/// the viewmodel over a cleared depth; the HUD over everything.
	#[allow(clippy::too_many_arguments)]
	pub fn draw(
		&mut self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		encoder: &mut wgpu::CommandEncoder,
		target: &wgpu::TextureView,
		depth: &wgpu::TextureView,
		view: Option<&View>,
		size: (f32, f32),
	) {
		let mut camera = [0.0f32; 20];
		if let Some(view) = view {
			let tan = (view.fovy * 0.5).tan();
			let (e, r, u, f) = (view.eye, view.right, view.up, view.forward);
			#[rustfmt::skip]
			let world = [
				e.x, e.y, e.z, NEAR,
				r.x, r.y, r.z, tan * size.0 / size.1,
				u.x, u.y, u.z, tan,
				f.x, f.y, f.z, 0.0,
			];
			camera[..16].copy_from_slice(&world);
		}
		camera[16] = size.0;
		camera[17] = size.1;
		queue.write_buffer(&self.camera, 0, &floats(&camera));

		// Every batch end to end, in one write.
		self.all.clear();
		let lines = span(self.all.len(), &self.lines);
		self.all.extend_from_slice(&self.lines);
		let boxes = span(self.all.len(), &self.boxes);
		self.all.extend_from_slice(&self.boxes);
		let front = span(self.all.len(), &self.front_boxes);
		self.all.extend_from_slice(&self.front_boxes);
		let meshes_at = self.all.len() as u32;
		self.all.extend_from_slice(&self.mesh_vertices);
		let screen = span(self.all.len(), &self.screen);
		self.all.extend_from_slice(&self.screen);
		if self.all.is_empty() {
			return;
		}
		let needed = std::mem::size_of_val(self.all.as_slice()) as u64;
		if self.vertices.as_ref().is_none_or(|b| b.size() < needed) {
			self.vertices = Some(device.create_buffer(&wgpu::BufferDescriptor {
				label: Some("overlay vertices"),
				size: needed.next_power_of_two(),
				usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			}));
		}
		let buffer = self.vertices.as_ref().expect("sized above");
		queue.write_buffer(buffer, 0, bytes(&self.all));

		if view.is_some() {
			if !lines.is_empty() || !boxes.is_empty() {
				let mut pass = overlay_pass(encoder, target, Some(depth), false);
				pass.set_bind_group(0, &self.camera_group, &[]);
				pass.set_vertex_buffer(0, buffer.slice(..));
				pass.set_pipeline(&self.line_pipeline);
				pass.draw(lines, 0..1);
				pass.set_pipeline(&self.box_pipeline);
				pass.draw(boxes, 0..1);
			}
			if !front.is_empty() || !self.meshes.is_empty() {
				let mut pass = overlay_pass(encoder, target, Some(depth), true);
				pass.set_bind_group(0, &self.camera_group, &[]);
				pass.set_vertex_buffer(0, buffer.slice(..));
				pass.set_pipeline(&self.box_pipeline);
				pass.draw(front, 0..1);
				pass.set_pipeline(&self.mesh_pipeline);
				for (texture, range) in &self.meshes {
					pass.set_bind_group(1, &self.textures[texture.0], &[]);
					pass.draw(range.start + meshes_at..range.end + meshes_at, 0..1);
				}
			}
		}
		if !screen.is_empty() {
			let mut pass = overlay_pass(encoder, target, None, false);
			pass.set_pipeline(&self.screen_pipeline);
			pass.set_bind_group(0, &self.camera_group, &[]);
			pass.set_bind_group(1, &self.glyph_group, &[]);
			pass.set_vertex_buffer(0, buffer.slice(..));
			pass.draw(screen, 0..1);
		}
	}
}

/// A pass over the frame so far, keeping or clearing its depth.
fn overlay_pass<'a>(
	encoder: &'a mut wgpu::CommandEncoder,
	target: &wgpu::TextureView,
	depth: Option<&wgpu::TextureView>,
	clear_depth: bool,
) -> wgpu::RenderPass<'a> {
	encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
		label: Some("overlay"),
		color_attachments: &[Some(wgpu::RenderPassColorAttachment {
			view: target,
			depth_slice: None,
			resolve_target: None,
			ops: wgpu::Operations {
				load: wgpu::LoadOp::Load,
				store: wgpu::StoreOp::Store,
			},
		})],
		depth_stencil_attachment: depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
			view,
			depth_ops: Some(wgpu::Operations {
				load: if clear_depth {
					wgpu::LoadOp::Clear(0.0)
				} else {
					wgpu::LoadOp::Load
				},
				store: wgpu::StoreOp::Store,
			}),
			stencil_ops: None,
		}),
		timestamp_writes: None,
		occlusion_query_set: None,
	})
}

/// The glyphs at `size` pixels, uploading them into `texture` the first time.
fn glyph_set<'a>(
	glyphs: &'a mut Glyphs,
	texture: &wgpu::Texture,
	queue: &wgpu::Queue,
	size: f32,
) -> &'a [text::Glyph] {
	let px = size.round().clamp(6.0, 96.0) as u16;
	glyphs.set(px, |gx, gy, w, h, coverage| {
		write_glyphs(queue, texture, gx, gy, w, h, coverage)
	})
}

fn write_glyphs(
	queue: &wgpu::Queue,
	texture: &wgpu::Texture,
	x: u32,
	y: u32,
	w: u32,
	h: u32,
	coverage: &[u8],
) {
	queue.write_texture(
		wgpu::TexelCopyTextureInfo {
			texture,
			mip_level: 0,
			origin: wgpu::Origin3d { x, y, z: 0 },
			aspect: wgpu::TextureAspect::All,
		},
		coverage,
		wgpu::TexelCopyBufferLayout {
			offset: 0,
			bytes_per_row: Some(w),
			rows_per_image: Some(h),
		},
		wgpu::Extent3d {
			width: w,
			height: h,
			depth_or_array_layers: 1,
		},
	);
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_box_is_twelve_triangles_on_its_bounds() {
		let corners: Vec<Vec3> = box_corners(vec3(1.0, 2.0, 3.0), vec3(2.0, 4.0, 6.0)).collect();
		assert_eq!(corners.len(), 36);
		for c in corners {
			assert!((c.x - 1.0).abs() == 1.0 && (c.y - 2.0).abs() == 2.0 && (c.z - 3.0).abs() == 3.0);
		}
	}

	#[test]
	fn vertices_pack_without_padding() {
		assert_eq!(std::mem::size_of::<Vertex>(), 36);
	}
	#[test]
	fn clipped_quads_stay_inside_and_keep_their_texture_mapping() {
		let corners = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
		let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
		let white = [Color::new(1.0, 1.0, 1.0, 1.0); 4];
		// Unclipped: the two triangles.
		let mut out = Vec::new();
		quad(&mut out, corners, uv, white, None);
		assert_eq!(out.len(), 6);
		// Cut to its right half: every vertex inside, the texture coordinate
		// following the position.
		let mut out = Vec::new();
		quad(&mut out, corners, uv, white, Some([5.0, -1.0, 20.0, 20.0]));
		assert!(!out.is_empty());
		for v in &out {
			assert!(v.pos[0] >= 5.0 - 1e-4 && v.pos[0] <= 10.0 + 1e-4);
			assert!((v.uv[0] - v.pos[0] / 10.0).abs() < 1e-5);
		}
		// Wholly outside: nothing.
		let mut out = Vec::new();
		quad(&mut out, corners, uv, white, Some([20.0, 20.0, 30.0, 30.0]));
		assert!(out.is_empty());
	}
}
