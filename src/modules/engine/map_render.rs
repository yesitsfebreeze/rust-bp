//! BSP map rendering: the whole map through one material. Faces carry their
//! texture's atlas rectangle in the vertex color and their lightmap block in
//! the vertex normal, so one albedo atlas and one lightmap atlas serve every
//! face; `shaders/map.frag` holds the shading model.

use crate::bsp::{Atlas, MapRender};
use macroquad::miniquad::{Comparison, PipelineParams, ShaderSource, UniformDesc, UniformType};
use macroquad::prelude::*;

/// macroquad's default index capacity per draw call; bigger meshes are split.
const DRAW_CALL_INDICES: usize = 5000;
const VERTEX: &str = include_str!("shaders/map.vert");
const FRAGMENT: &str = include_str!("shaders/map.frag");

/// The sky shows through the `sky` faces the loader skips.
pub const SKY: Color = Color::new(0.62, 0.71, 0.82, 1.0);

pub struct MapRenderer {
	material: Material,
	meshes: Vec<Mesh>,
	wireframe: Vec<(Vec3, Vec3)>,
}

impl MapRenderer {
	/// Uploads `render`'s atlases and builds draw-call-sized meshes.
	pub fn new(render: &MapRender) -> Result<MapRenderer, String> {
		let material = load_material(
			ShaderSource::Glsl {
				vertex: VERTEX,
				fragment: FRAGMENT,
			},
			MaterialParams {
				pipeline_params: PipelineParams {
					depth_test: Comparison::LessOrEqual,
					depth_write: true,
					..Default::default()
				},
				uniforms: vec![
					UniformDesc::new("LmSide", UniformType::Float1),
					UniformDesc::new("Exposure", UniformType::Float1),
				],
				textures: vec!["Lightmap".to_string()],
			},
		)
		.map_err(|e| format!("map material: {e:?}"))?;
		material.set_uniform("LmSide", render.lightmap.side as f32);
		// Nearest: vertices name exact albedo texels, no border padding.
		let albedo = texture(&render.albedo);
		albedo.set_filter(FilterMode::Nearest);
		// Linear: one unused texel around each block stops the bleed.
		let lightmap = texture(&render.lightmap);
		lightmap.set_filter(FilterMode::Linear);
		material.set_texture("Lightmap", lightmap);
		Ok(MapRenderer {
			material,
			meshes: meshes(render, &albedo),
			wireframe: render
				.faces
				.iter()
				.flat_map(|face| {
					(0..face.verts.len()).map(|i| {
						(
							face.verts[i].pos.into(),
							face.verts[(i + 1) % face.verts.len()].pos.into(),
						)
					})
				})
				.collect(),
		})
	}

	/// Draws the map under the current 3D camera. `exposure` scales the light,
	/// the same cvar as the arena renderer's.
	pub fn draw(&self, exposure: f32) {
		self.material.set_uniform("Exposure", exposure.max(0.0));
		gl_use_material(&self.material);
		for mesh in &self.meshes {
			draw_mesh(mesh);
		}
		gl_use_default_material();
	}

	/// Draws BSP face boundaries over the map under the current 3D camera.
	pub fn draw_wireframe(&self) {
		gl_use_default_material();
		for &(start, end) in &self.wireframe {
			draw_line_3d(start, end, Color::new(0.95, 0.85, 0.35, 1.0));
		}
	}
}

/// The atlas as a texture: no mipmaps, they would average the map's texels.
fn texture(atlas: &Atlas) -> Texture2D {
	let t = Texture2D::from_rgba8(atlas.side as u16, atlas.side as u16, &atlas.rgba);
	t.set_filter(FilterMode::Nearest);
	t
}

/// The map's faces as draw-call-sized meshes. One vertex per face corner; the
/// fan triangles are regenerated from each face's vertex range per mesh, since
/// indices restart at 0 in every draw call.
fn meshes(render: &MapRender, albedo: &Texture2D) -> Vec<Mesh> {
	let mut out = Vec::new();
	let mut vertices: Vec<Vertex> = Vec::new();
	let mut spans: Vec<(usize, usize)> = Vec::new();
	let mut indices = 0;
	for face in &render.faces {
		// Chunk by indices, not vertices: big n-gons fan at three indices per
		// corner, over macroquad's draw call limit of `DRAW_CALL_INDICES`.
		let fan = face.verts.len().saturating_sub(2) * 3;
		if indices + fan > DRAW_CALL_INDICES - 2 {
			out.push(mesh(
				std::mem::take(&mut vertices),
				std::mem::take(&mut spans),
				albedo,
			));
			indices = 0;
		}
		indices += fan;
		spans.push((vertices.len(), face.verts.len()));
		for v in &face.verts {
			let t = face.texture;
			let l = face.lightmap;
			vertices.push(Vertex {
				position: v.pos.into(),
				uv: macroquad::math::Vec2::new(v.uv.0, v.uv.1),
				// The texture rectangle as bytes: x/8, y/8, w/8, h/8.
				color: Color::new(
					(t.x / 8) as f32 / 255.0,
					(t.y / 8) as f32 / 255.0,
					(t.w / 8) as f32 / 255.0,
					(t.h / 8) as f32 / 255.0,
				)
				.into(),
				// The lightmap block: xy the vertex's texel, z the packed
				// origin, w the packed size.
				normal: macroquad::math::Vec4::new(
					v.lm.0,
					v.lm.1,
					(l.x + l.y * 4096) as f32,
					(l.w + l.h * 256) as f32,
				),
			});
		}
	}
	if !spans.is_empty() {
		out.push(mesh(vertices, spans, albedo));
	}
	out
}

/// One mesh: the face corners plus the fan indices, the texture on the slot
/// macroquad binds to the shader's `Texture` sampler.
fn mesh(vertices: Vec<Vertex>, spans: Vec<(usize, usize)>, albedo: &Texture2D) -> Mesh {
	let mut indices = Vec::with_capacity(spans.iter().map(|(_, n)| n.saturating_sub(2) * 3).sum());
	for (first, len) in spans {
		for k in 1..len - 1 {
			indices.extend([first as u16, (first + k) as u16, (first + k + 1) as u16]);
		}
	}
	Mesh {
		indices,
		vertices,
		texture: Some(albedo.clone()),
	}
}
