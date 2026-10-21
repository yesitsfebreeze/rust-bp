//! Physically based arena rendering with subsurface scattering.
//!
//! Each [`Surface`] is one pipeline holding its [`textures`]: macroquad records
//! uniforms per draw call but binds a pipeline's extra textures once per frame.
//! Faces are UV mapped by projecting world position on the plane of the normal's
//! dominant axis, so textures tile at a fixed world size and line up across faces.
//! `shaders/pbr.frag` holds the shading model.

use crate::surface::Surface;
use crate::textures::{self, Texels};
use linalg::Vec3;
use macroquad::math as mq;
use macroquad::miniquad::{Comparison, PipelineParams, ShaderSource, UniformDesc, UniformType};
use macroquad::prelude::*;

/// macroquad's default index capacity per draw call; bigger meshes are clamped.
const DRAW_CALL_INDICES: usize = 5000;
const VERTEX: &str = include_str!("shaders/pbr.vert");
const FRAGMENT: &str = include_str!("shaders/pbr.frag");

/// Linear radiance of the sun and the sky and ground hemisphere, balanced for
/// the flat Valorant and CS look: soft sun, bright ambient. Public for the
/// lightmap bake, which must agree with the shader's light.
pub const SUN_COLOR: mq::Vec3 = mq::vec3(2.6, 2.5, 2.4);
pub const SKY_COLOR: mq::Vec3 = mq::vec3(0.40, 0.47, 0.58);
pub const GROUND_COLOR: mq::Vec3 = mq::vec3(0.24, 0.23, 0.22);
/// Toward the sun; normalized when the renderer is built.
pub const SUN: mq::Vec3 = mq::vec3(0.4, 1.0, 0.25);

/// How light scatters inside a material; see the uniforms of `shaders/pbr.frag`.
struct Scatter {
	color: mq::Vec3,
	wrap: f32,
	distortion: f32,
	power: f32,
	scale: f32,
}

impl Surface {
	/// World units one texture covers.
	fn texture_size(self) -> f32 {
		match self {
			Surface::Tile | Surface::Marble | Surface::Steel => 128.0,
			Surface::Concrete => 256.0,
			Surface::Jade => 64.0,
		}
	}

	fn scatter(self) -> Scatter {
		match self {
			Surface::Marble => Scatter {
				color: mq::vec3(1.0, 0.82, 0.68),
				wrap: 0.5,
				distortion: 0.3,
				power: 4.0,
				scale: 0.6,
			},
			Surface::Jade => Scatter {
				color: mq::vec3(0.35, 1.0, 0.55),
				wrap: 0.8,
				distortion: 0.2,
				power: 3.0,
				scale: 1.5,
			},
			// Opaque: the subsurface map is zero, these only keep the uniforms sane.
			Surface::Tile | Surface::Concrete | Surface::Steel => Scatter {
				color: mq::Vec3::ONE,
				wrap: 0.0,
				distortion: 0.0,
				power: 1.0,
				scale: 0.0,
			},
		}
	}
}

struct Pass {
	material: Material,
	scatter: mq::Vec3,
	meshes: Vec<Mesh>,
}

pub struct Renderer {
	passes: Vec<Pass>,
	sun: mq::Vec3,
}

impl Renderer {
	/// Uploads `world`'s triangles, drawn with `surfaces[i]` for triangle `i`.
	pub fn new(world: &physics::Mesh, surfaces: &[Surface]) -> Result<Renderer, String> {
		assert_eq!(
			world.tris().len(),
			surfaces.len(),
			"one surface per triangle"
		);
		let passes = Surface::ALL
			.into_iter()
			.map(|s| {
				let tris = world
					.tris()
					.iter()
					.zip(surfaces)
					.filter(|(_, t)| **t == s)
					.map(|(t, _)| t);
				let texels = textures::texels(s);
				Ok(Pass {
					material: material(s)?,
					scatter: s.scatter().color,
					// One atlas per surface rides the reserved `Texture` sampler;
					// macroquad binds no other texture slot reliably.
					meshes: meshes(tris, s.texture_size(), atlas_texture(&texels)),
				})
			})
			.collect::<Result<_, String>>()?;
		Ok(Renderer {
			passes,
			sun: SUN.normalize(),
		})
	}

	/// The direction toward the sun, normalized; the lightmap bakes with it.
	pub fn sun(&self) -> mq::Vec3 {
		self.sun
	}

	/// Draws the arena from `eye` under the current 3D camera. `subsurface` scales
	/// every material's scattering, 0 turning it off; `exposure` scales the light.
	pub fn draw(&self, eye: Vec3, subsurface: f32, exposure: f32) {
		for pass in &self.passes {
			let m = &pass.material;
			m.set_uniform("CameraPosition", mq::Vec3::from(eye));
			m.set_uniform("SunDirection", self.sun);
			m.set_uniform("ScatterColor", pass.scatter.extend(subsurface.max(0.0)));
			m.set_uniform("Exposure", exposure.max(0.0));
			gl_use_material(m);
			for mesh in &pass.meshes {
				draw_mesh(mesh);
			}
		}
		gl_use_default_material();
	}
}

fn material(surface: Surface) -> Result<Material, String> {
	let uniform = |name: &str, kind| UniformDesc::new(name, kind);
	let m = load_material(
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
				uniform("CameraPosition", UniformType::Float3),
				uniform("SunDirection", UniformType::Float3),
				uniform("SunColor", UniformType::Float3),
				uniform("SkyColor", UniformType::Float3),
				uniform("GroundColor", UniformType::Float3),
				uniform("ScatterColor", UniformType::Float4),
				uniform("Scatter", UniformType::Float4),
				uniform("Exposure", UniformType::Float1),
			],
			..Default::default()
		},
	)
	.map_err(|e| format!("{surface:?} material: {e:?}"))?;
	let s = surface.scatter();
	m.set_uniform("SunColor", SUN_COLOR);
	m.set_uniform("SkyColor", SKY_COLOR);
	m.set_uniform("GroundColor", GROUND_COLOR);
	m.set_uniform("Scatter", vec4(s.wrap, s.distortion, s.power, s.scale));
	Ok(m)
}

/// The atlas of `texels` as a texture: linear filtering, no mipmaps, which
/// would average neighboring tiles.
fn atlas_texture(texels: &Texels) -> Texture2D {
	Texture2D::from_rgba8(
		textures::PITCH as u16,
		(textures::PITCH * textures::TILES) as u16,
		&textures::atlas(texels),
	)
}

/// Draw-call-sized meshes of `tris`, `size` world units per texture repeat. Each
/// vertex carries the face normal in `normal.xyz`, the bitangent's handedness in
/// `normal.w` and the tangent in `color`.
fn meshes<'a>(
	tris: impl Iterator<Item = &'a physics::Triangle>,
	size: f32,
	albedo: Texture2D,
) -> Vec<Mesh> {
	let mut out = Vec::new();
	let mut vertices = Vec::new();
	for t in tris {
		let n = physics::triangle_normal(t);
		let (ua, va) = projection(n);
		let tangent = (ua - n * n.dot(ua)).normalize();
		let handedness = if n.cross(tangent).dot(va) >= 0.0 {
			1.0
		} else {
			-1.0
		};
		let packed = Color::new(
			tangent.x * 0.5 + 0.5,
			tangent.y * 0.5 + 0.5,
			tangent.z * 0.5 + 0.5,
			1.0,
		);
		for p in t {
			vertices.push(Vertex {
				position: (*p).into(),
				uv: mq::vec2(p.dot(ua), p.dot(va)) / size,
				color: packed.into(),
				normal: mq::Vec3::from(n).extend(handedness),
			});
		}
		if vertices.len() + 3 > DRAW_CALL_INDICES {
			out.push(mesh(std::mem::take(&mut vertices), &albedo));
		}
	}
	if !vertices.is_empty() {
		out.push(mesh(vertices, &albedo));
	}
	out
}

/// The world axes along u and v for a face with normal `n`.
fn projection(n: Vec3) -> (Vec3, Vec3) {
	let a = n.map(f32::abs);
	if a.y >= a.x && a.y >= a.z {
		(Vec3::X, Vec3::Z)
	} else if a.x >= a.z {
		(Vec3::Z, Vec3::Y)
	} else {
		(Vec3::X, Vec3::Y)
	}
}

fn mesh(vertices: Vec<Vertex>, albedo: &Texture2D) -> Mesh {
	Mesh {
		indices: (0..vertices.len() as u16).collect(),
		vertices,
		texture: Some(albedo.clone()),
	}
}
