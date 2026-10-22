//! The frame's ambient occlusion (`ao.wgsl`): two full-screen passes over
//! the world's own color and depth. The first writes each drawn pixel's
//! openness — how little matter its depth shows standing near it — into a
//! texture of its own, each pixel of a 4×4 tile turning its taps its own
//! way; the second blurs that over the tile among pixels on the same
//! surface, shades the color with it and fogs it by the air its depth looks
//! through on the way to the window. No ray
//! is cast. The frame's textures are replaced on resize, so the groups are
//! built per draw, and the openness texture follows the frame's size.

use super::util::{self, entry};

/// The openness texture's format: one channel, 0 walled in to 1 open.
const OPEN: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

pub struct Ambient {
	occlusion_layout: wgpu::BindGroupLayout,
	composite_layout: wgpu::BindGroupLayout,
	occlusion: wgpu::RenderPipeline,
	composite: wgpu::RenderPipeline,
	/// The openness the first pass writes, and its size.
	open: Option<(wgpu::TextureView, (u32, u32))>,
}

impl Ambient {
	pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Ambient {
		let fragment = wgpu::ShaderStages::FRAGMENT;
		let occlusion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("ao occlusion"),
			entries: &[
				util::buffer_entry(0, fragment, util::UNIFORM),
				util::raw_texture_entry(1, fragment),
			],
		});
		let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("ao composite"),
			entries: &[
				util::buffer_entry(0, fragment, util::UNIFORM),
				util::raw_texture_entry(1, fragment),
				util::raw_texture_entry(2, fragment),
				util::raw_texture_entry(3, fragment),
				util::buffer_entry(4, fragment, util::UNIFORM),
			],
		});
		let shader = util::shader(device, "ao", &[include_str!("ao.wgsl")]);
		let pipeline = |label: &str,
		                layout: &wgpu::BindGroupLayout,
		                entry_point: &str,
		                format: wgpu::TextureFormat| {
			device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
				label: Some(label),
				layout: Some(&util::layout(device, label, &[layout])),
				vertex: wgpu::VertexState {
					module: &shader,
					entry_point: Some("vs"),
					compilation_options: Default::default(),
					buffers: &[],
				},
				primitive: Default::default(),
				depth_stencil: None,
				multisample: Default::default(),
				fragment: Some(wgpu::FragmentState {
					module: &shader,
					entry_point: Some(entry_point),
					compilation_options: Default::default(),
					targets: &[Some(format.into())],
				}),
				multiview: None,
				cache: None,
			})
		};
		let occlusion = pipeline("ao occlusion", &occlusion_layout, "occlusion", OPEN);
		let composite = pipeline("ao composite", &composite_layout, "composite", format);
		Ambient {
			occlusion_layout,
			composite_layout,
			occlusion,
			composite,
			open: None,
		}
	}

	/// The openness texture for a frame of `size`, made anew when the frame
	/// changed size.
	fn open(&mut self, device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
		if self.open.as_ref().is_none_or(|(_, at)| *at != size) {
			let view = device
				.create_texture(&wgpu::TextureDescriptor {
					label: Some("ao openness"),
					size: wgpu::Extent3d {
						width: size.0,
						height: size.1,
						depth_or_array_layers: 1,
					},
					mip_level_count: 1,
					sample_count: 1,
					dimension: wgpu::TextureDimension::D2,
					format: OPEN,
					usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
					view_formats: &[],
				})
				.create_view(&Default::default());
			self.open = Some((view, size));
		}
		self.open.as_ref().unwrap().0.clone()
	}

	/// Carries the world pass's `scene` into `target`, each drawn pixel
	/// shaded by what its `depth` shows standing near it, read as `view`,
	/// the world pass's uniform, and fogged in the sun and sky `lighting`
	/// holds. `size` is the frame's, in pixels. Without `occlude` the
	/// occlusion pass is skipped: the composite reads no openness when the
	/// view's occlusion strength is 0.
	#[allow(clippy::too_many_arguments)]
	pub fn draw(
		&mut self,
		device: &wgpu::Device,
		encoder: &mut wgpu::CommandEncoder,
		target: &wgpu::TextureView,
		scene: &wgpu::TextureView,
		depth: &wgpu::TextureView,
		view: &wgpu::Buffer,
		lighting: &wgpu::Buffer,
		size: (u32, u32),
		occlude: bool,
	) {
		let open = self.open(device, size);
		let pass = |encoder: &mut wgpu::CommandEncoder,
		            label: &str,
		            to: &wgpu::TextureView,
		            pipeline: &wgpu::RenderPipeline,
		            group: &wgpu::BindGroup| {
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some(label),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: to,
					depth_slice: None,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
			});
			pass.set_pipeline(pipeline);
			pass.set_bind_group(0, group, &[]);
			pass.draw(0..3, 0..1);
		};
		if occlude {
			let occlusion = device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("ao occlusion"),
				layout: &self.occlusion_layout,
				entries: &[
					util::bind(0, view),
					entry(1, wgpu::BindingResource::TextureView(depth)),
				],
			});
			pass(encoder, "ao occlusion", &open, &self.occlusion, &occlusion);
		}
		let composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("ao composite"),
			layout: &self.composite_layout,
			entries: &[
				util::bind(0, view),
				entry(1, wgpu::BindingResource::TextureView(depth)),
				entry(2, wgpu::BindingResource::TextureView(scene)),
				entry(3, wgpu::BindingResource::TextureView(&open)),
				util::bind(4, lighting),
			],
		});
		pass(encoder, "ao composite", target, &self.composite, &composite);
	}
}
