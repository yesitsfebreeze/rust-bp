//! Small helpers every GPU module of the renderer shares.

use wgpu::util::DeviceExt;

/// `f32`s as the little-endian bytes a uniform or storage buffer holds.
pub fn floats(values: &[f32]) -> Vec<u8> {
	values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `u32`s as little-endian bytes.
pub fn words(values: &[u32]) -> Vec<u8> {
	values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A bind group entry.
pub fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
	wgpu::BindGroupEntry { binding, resource }
}

/// A bind group entry for a whole buffer.
pub fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
	entry(binding, buffer.as_entire_binding())
}

/// A layout entry for a buffer seen by `stages`.
pub fn buffer_entry(
	binding: u32,
	stages: wgpu::ShaderStages,
	ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: stages,
		ty: wgpu::BindingType::Buffer {
			ty,
			has_dynamic_offset: false,
			min_binding_size: None,
		},
		count: None,
	}
}

/// A layout entry for a filterable 2D texture seen by `stages`.
pub fn texture_entry(binding: u32, stages: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: stages,
		ty: wgpu::BindingType::Texture {
			sample_type: wgpu::TextureSampleType::Float { filterable: true },
			view_dimension: wgpu::TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

/// A layout entry for a non-filterable 2D texture seen by `stages`: what
/// the frame's own color and depth load as, unfiltered.
pub fn raw_texture_entry(binding: u32, stages: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: stages,
		ty: wgpu::BindingType::Texture {
			sample_type: wgpu::TextureSampleType::Float { filterable: false },
			view_dimension: wgpu::TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

/// A layout entry for a filtering sampler seen by `stages`.
pub fn sampler_entry(binding: u32, stages: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: stages,
		ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
		count: None,
	}
}

pub const READ: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };
pub const WRITE: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };
pub const UNIFORM: wgpu::BufferBindingType = wgpu::BufferBindingType::Uniform;

/// A zeroed buffer of `size` bytes.
pub fn buffer(
	device: &wgpu::Device,
	label: &str,
	size: u64,
	usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
	device.create_buffer(&wgpu::BufferDescriptor {
		label: Some(label),
		size,
		usage,
		mapped_at_creation: false,
	})
}

/// A storage buffer holding `contents`.
pub fn storage(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
	device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some(label),
		contents,
		usage: wgpu::BufferUsages::STORAGE,
	})
}

/// A half-float RGBA texture of `width` × `height` from `rgba`, four
/// little-endian f16s per texel, filterable.
pub fn half_image(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	label: &str,
	width: u32,
	height: u32,
	rgba: &[u8],
) -> wgpu::TextureView {
	device
		.create_texture_with_data(
			queue,
			&wgpu::TextureDescriptor {
				label: Some(label),
				size: wgpu::Extent3d {
					width,
					height,
					depth_or_array_layers: 1,
				},
				mip_level_count: 1,
				sample_count: 1,
				dimension: wgpu::TextureDimension::D2,
				format: wgpu::TextureFormat::Rgba16Float,
				usage: wgpu::TextureUsages::TEXTURE_BINDING,
				view_formats: &[],
			},
			wgpu::util::TextureDataOrder::LayerMajor,
			rgba,
		)
		.create_view(&Default::default())
}

/// A uniform buffer holding `contents`.
pub fn uniform(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
	device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some(label),
		contents,
		usage: wgpu::BufferUsages::UNIFORM,
	})
}

/// A vertex buffer holding `contents`.
pub fn vertices(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
	device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some(label),
		contents,
		usage: wgpu::BufferUsages::VERTEX,
	})
}

/// A render or compute pipeline layout over `groups`.
pub fn layout(
	device: &wgpu::Device,
	label: &str,
	groups: &[&wgpu::BindGroupLayout],
) -> wgpu::PipelineLayout {
	device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
		label: Some(label),
		bind_group_layouts: groups,
		push_constant_ranges: &[],
	})
}

/// A WGSL module from sources joined in order: shared pieces first.
pub fn shader(device: &wgpu::Device, label: &str, sources: &[&str]) -> wgpu::ShaderModule {
	device.create_shader_module(wgpu::ShaderModuleDescriptor {
		label: Some(label),
		source: wgpu::ShaderSource::Wgsl(sources.concat().into()),
	})
}

/// An RGBA8 texture over `rgba`, `width` by `height` texels, seen two ways:
/// as stored, and through an sRGB view that the hardware decodes to linear.
pub fn image(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	label: &str,
	width: u32,
	height: u32,
	rgba: &[u8],
) -> (wgpu::TextureView, wgpu::TextureView) {
	let texture = device.create_texture_with_data(
		queue,
		&wgpu::TextureDescriptor {
			label: Some(label),
			size: wgpu::Extent3d {
				width,
				height,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Rgba8Unorm,
			usage: wgpu::TextureUsages::TEXTURE_BINDING,
			view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
		},
		wgpu::util::TextureDataOrder::LayerMajor,
		rgba,
	);
	(
		texture.create_view(&Default::default()),
		texture.create_view(&wgpu::TextureViewDescriptor {
			format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
			..Default::default()
		}),
	)
}
