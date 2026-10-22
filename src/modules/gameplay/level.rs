//! A level: what the in-game editor places over a BSP map, saved beside it
//! as `<map>.level.toml`. Static geometry — boxes, ramps, cylinders,
//! stairs and imported models — joins the map itself: drawn as its faces,
//! walked on as its collision, lit by its bake. Lights shine every frame as
//! dynamic lights; props are physics bodies the world pushes around.
//!
//! Every object stands on its `pos`, the middle of its base, turned `yaw`
//! degrees about the up axis; `size` is its full extent along its own x, y
//! (up) and z.

use engine::bsp::{Face, Map, Rect, Vert};
use engine::linalg::{vec3, Quat, Vec3};
use engine::physics;
use engine::surface::Finish;
use engine::Surface;

/// What an object is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
	Box,
	Ramp,
	Cylinder,
	Stairs,
	Model,
	Light,
	Crate,
	Ball,
}

impl Kind {
	pub const ALL: [Kind; 8] = [
		Kind::Box,
		Kind::Ramp,
		Kind::Cylinder,
		Kind::Stairs,
		Kind::Model,
		Kind::Light,
		Kind::Crate,
		Kind::Ball,
	];

	/// Its name, in the level file and the editor.
	pub fn name(self) -> &'static str {
		match self {
			Kind::Box => "box",
			Kind::Ramp => "ramp",
			Kind::Cylinder => "cylinder",
			Kind::Stairs => "stairs",
			Kind::Model => "model",
			Kind::Light => "light",
			Kind::Crate => "crate",
			Kind::Ball => "ball",
		}
	}

	pub fn from_name(name: &str) -> Option<Kind> {
		Kind::ALL.into_iter().find(|k| k.name() == name)
	}

	/// Whether it joins the map: drawn, collided with and baked as the
	/// map's own geometry. Lights and props are not.
	pub fn is_static(self) -> bool {
		matches!(
			self,
			Kind::Box | Kind::Ramp | Kind::Cylinder | Kind::Stairs | Kind::Model
		)
	}

	/// Whether it is a physics body.
	pub fn is_prop(self) -> bool {
		matches!(self, Kind::Crate | Kind::Ball)
	}

	/// The size a new one starts at.
	pub fn default_size(self) -> Vec3 {
		match self {
			Kind::Box => vec3(64.0, 64.0, 64.0),
			Kind::Ramp => vec3(64.0, 48.0, 128.0),
			Kind::Cylinder => vec3(48.0, 128.0, 48.0),
			Kind::Stairs => vec3(64.0, 64.0, 128.0),
			Kind::Model => vec3(64.0, 128.0, 64.0),
			Kind::Light => vec3(8.0, 8.0, 8.0),
			Kind::Crate => vec3(32.0, 32.0, 32.0),
			Kind::Ball => vec3(16.0, 16.0, 16.0),
		}
	}
}

/// One placed thing.
#[derive(Clone, Debug, PartialEq)]
pub struct Object {
	pub kind: Kind,
	/// The middle of its base; a light's own position.
	pub pos: Vec3,
	pub size: Vec3,
	/// Degrees about the up axis.
	pub yaw: f32,
	/// The map texture it is dressed in.
	pub texture: String,
	/// A light's color, linear, and how bright and how far it shines.
	pub color: [f32; 3],
	pub intensity: f32,
	pub reach: f32,
	/// A model's file under `assets/models`.
	pub model: String,
}

impl Object {
	pub fn new(kind: Kind, pos: Vec3, texture: &str) -> Object {
		Object {
			kind,
			pos,
			size: kind.default_size(),
			yaw: 0.0,
			texture: texture.to_string(),
			color: [1.0, 0.8, 0.55],
			intensity: 4.0,
			reach: 400.0,
			model: String::new(),
		}
	}

	fn turn(&self) -> Quat {
		Quat::from_axis_angle(Vec3::Y, self.yaw.to_radians())
	}

	/// Its bounds' middle, world space.
	pub fn center(&self) -> Vec3 {
		if self.kind == Kind::Light {
			self.pos
		} else {
			self.pos + Vec3::Y * (self.size.y * 0.5)
		}
	}

	/// The radius of a ball that holds it, about [`Object::center`].
	pub fn radius(&self) -> f32 {
		self.size.length() * 0.5
	}

	/// Its static geometry's polygons, world space, wound so the cross of
	/// any fan triangle points out of the solid; none for lights and
	/// props. `models` loads a model by name.
	pub fn polygons(&self, models: &dyn Fn(&str) -> Option<Vec<[Vec3; 3]>>) -> Vec<Vec<Vec3>> {
		let half = self.size * 0.5;
		let local: Vec<Vec<Vec3>> = match self.kind {
			Kind::Box => cuboid(vec3(-half.x, 0.0, -half.z), vec3(half.x, self.size.y, half.z)),
			Kind::Ramp => {
				// Rising toward +z: high at the back, flush with the floor at
				// the front.
				let (x0, x1, z0, z1, h) = (-half.x, half.x, -half.z, half.z, self.size.y);
				convex(vec![
					vec![vec3(x0, 0.0, z0), vec3(x1, 0.0, z0), vec3(x1, 0.0, z1), vec3(x0, 0.0, z1)],
					vec![vec3(x0, 0.0, z1), vec3(x1, 0.0, z1), vec3(x1, h, z1), vec3(x0, h, z1)],
					vec![vec3(x0, 0.0, z0), vec3(x0, h, z1), vec3(x1, h, z1), vec3(x1, 0.0, z0)],
					vec![vec3(x0, 0.0, z0), vec3(x0, 0.0, z1), vec3(x0, h, z1)],
					vec![vec3(x1, 0.0, z0), vec3(x1, h, z1), vec3(x1, 0.0, z1)],
				])
			}
			Kind::Cylinder => {
				const SIDES: usize = 16;
				let ring = |y: f32| -> Vec<Vec3> {
					(0..SIDES)
						.map(|k| {
							let a = std::f32::consts::TAU * k as f32 / SIDES as f32;
							vec3(a.cos() * half.x, y, a.sin() * half.z)
						})
						.collect()
				};
				let (bottom, top) = (ring(0.0), ring(self.size.y));
				let mut polys = vec![bottom.clone(), top.clone()];
				for k in 0..SIDES {
					let j = (k + 1) % SIDES;
					polys.push(vec![bottom[k], bottom[j], top[j], top[k]]);
				}
				convex(polys)
			}
			Kind::Stairs => {
				// Steps of at most 16 units, climbing toward +z.
				let steps = (self.size.y / 16.0).ceil().max(1.0) as usize;
				let (rise, run) = (self.size.y / steps as f32, self.size.z / steps as f32);
				(0..steps)
					.flat_map(|k| {
						cuboid(
							vec3(-half.x, 0.0, -half.z + run * k as f32),
							vec3(half.x, rise * (k + 1) as f32, half.z),
						)
					})
					.collect()
			}
			Kind::Model => {
				let Some(tris) = models(&self.model) else {
					return Vec::new();
				};
				fit(&tris, self.size)
					.into_iter()
					.map(|t| t.to_vec())
					.collect()
			}
			Kind::Light | Kind::Crate | Kind::Ball => Vec::new(),
		};
		let turn = self.turn();
		local
			.into_iter()
			.map(|poly| poly.into_iter().map(|p| self.pos + turn.rotate(p)).collect())
			.collect()
	}

	/// A prop's physics body, at rest where it stands.
	pub fn body(&self) -> Option<physics::Body> {
		let mut body = match self.kind {
			Kind::Crate => {
				let half = self.size * 0.5;
				let tris: Vec<[Vec3; 3]> = cuboid(-half, half)
					.into_iter()
					.flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
					.collect();
				physics::Body::from_hull(physics::Hull::from_tris(tris))
			}
			Kind::Ball => {
				let mut ball = physics::Body::default();
				ball.radius = self.size.x * 0.5;
				ball
			}
			_ => return None,
		};
		body.pos = self.center();
		body.orient = self.turn();
		body.mass = (self.size.x * self.size.y * self.size.z).max(1.0) / 1000.0;
		body.rotates = true;
		body.friction = 0.6;
		Some(body)
	}
}

/// A crate prop's faces where its body stands, dressed in `texture`.
pub fn crate_faces(body: &physics::Body, size: Vec3, texture: (Rect, Finish)) -> Vec<Face> {
	let half = size * 0.5;
	cuboid(-half, half)
		.into_iter()
		.map(|q| face(q.iter().map(|&p| body.pos + body.orient.rotate(p)).collect(), texture))
		.collect()
}

/// The six faces of the box from `lo` to `hi`, each wound outward.
fn cuboid(lo: Vec3, hi: Vec3) -> Vec<Vec<Vec3>> {
	let c = |x: bool, y: bool, z: bool| {
		vec3(
			if x { hi.x } else { lo.x },
			if y { hi.y } else { lo.y },
			if z { hi.z } else { lo.z },
		)
	};
	convex(vec![
		vec![c(false, false, false), c(true, false, false), c(true, false, true), c(false, false, true)],
		vec![c(false, true, false), c(false, true, true), c(true, true, true), c(true, true, false)],
		vec![c(false, false, false), c(false, true, false), c(true, true, false), c(true, false, false)],
		vec![c(false, false, true), c(true, false, true), c(true, true, true), c(false, true, true)],
		vec![c(false, false, false), c(false, false, true), c(false, true, true), c(false, true, false)],
		vec![c(true, false, false), c(true, true, false), c(true, true, true), c(true, false, true)],
	])
}

/// The polygons of one convex solid, each turned to face away from the
/// solid's middle: however a polygon was listed, it ends wound outward.
fn convex(polys: Vec<Vec<Vec3>>) -> Vec<Vec<Vec3>> {
	let all: Vec<Vec3> = polys.iter().flatten().copied().collect();
	let middle = all.iter().fold(Vec3::ZERO, |s, &p| s + p) * (1.0 / all.len().max(1) as f32);
	polys
		.into_iter()
		.map(|mut poly| {
			let n = (poly[1] - poly[0]).cross(poly[2] - poly[0]);
			let at = poly.iter().fold(Vec3::ZERO, |s, &p| s + p) * (1.0 / poly.len() as f32);
			if n.dot(at - middle) < 0.0 {
				poly.reverse();
			}
			poly
		})
		.collect()
}

/// A model's triangles scaled and moved to fill `size`, its base's middle
/// at the origin.
fn fit(tris: &[[Vec3; 3]], size: Vec3) -> Vec<[Vec3; 3]> {
	let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
	for t in tris {
		for &p in t {
			lo = lo.min(p);
			hi = hi.max(p);
		}
	}
	let span = (hi - lo).max(Vec3::splat(1e-6));
	let base = vec3((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
	let scale = vec3(size.x / span.x, size.y / span.y, size.z / span.z);
	tris.iter()
		.map(|t| {
			t.map(|p| {
				let q = p - base;
				vec3(q.x * scale.x, q.y * scale.y, q.z * scale.z)
			})
		})
		.collect()
}

/// A face over `poly`, wound outward, dressed in `texture`: its texture
/// laid across the plane its normal faces most, one texel a unit, as
/// GoldSrc maps lay theirs.
fn face(poly: Vec<Vec3>, (rect, finish): (Rect, Finish)) -> Face {
	let n = (poly[1] - poly[0]).cross(poly[2] - poly[0]);
	let (tw, th) = (rect.w.max(1) as f32, rect.h.max(1) as f32);
	let uv = |p: Vec3| -> (f32, f32) {
		if n.y.abs() >= n.x.abs() && n.y.abs() >= n.z.abs() {
			(p.x / tw, p.z / th)
		} else if n.x.abs() >= n.z.abs() {
			(p.z / tw, -p.y / th)
		} else {
			(p.x / tw, -p.y / th)
		}
	};
	Face {
		texture: rect,
		lightmap: Rect {
			x: 0,
			y: 0,
			w: 0,
			h: 0,
		},
		solid: true,
		finish,
		verts: poly
			.into_iter()
			.map(|pos| Vert {
				pos,
				uv: uv(pos),
				lm: (0.0, 0.0),
			})
			.collect(),
	}
}

/// The rectangle and finish of texture `name` among the map's, or its
/// first texture when the name is not among them.
pub fn dress(textures: &[(String, Rect)], name: &str) -> (Rect, Finish) {
	textures
		.iter()
		.find(|(n, _)| n.eq_ignore_ascii_case(name))
		.or_else(|| textures.first())
		.map(|(n, r)| (*r, Finish::of(n)))
		.unwrap_or((
			Rect {
				x: 0,
				y: 0,
				w: 8,
				h: 8,
			},
			Finish::default(),
		))
}

/// A Wavefront OBJ's triangles: its `v` lines and its `f` polygons, fanned,
/// in the file's own counterclockwise winding and Y-up axes. Texture and
/// normal indices are ignored.
pub fn load_obj(path: &std::path::Path) -> Result<Vec<[Vec3; 3]>, String> {
	let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
	let mut verts = Vec::new();
	let mut tris = Vec::new();
	for (n, line) in text.lines().enumerate() {
		let mut words = line.split_whitespace();
		match words.next() {
			Some("v") => {
				let v: Vec<f32> = words.take(3).filter_map(|w| w.parse().ok()).collect();
				if v.len() != 3 {
					return Err(format!("{}:{}: a vertex needs three numbers", path.display(), n + 1));
				}
				verts.push(vec3(v[0], v[1], v[2]));
			}
			Some("f") => {
				let idx: Vec<usize> = words
					.map(|w| {
						let i: i64 = w.split('/').next().unwrap_or("").parse().unwrap_or(0);
						if i < 0 {
							(verts.len() as i64 + i) as usize
						} else {
							(i - 1).max(0) as usize
						}
					})
					.collect();
				if idx.iter().any(|&i| i >= verts.len()) {
					return Err(format!("{}:{}: a face names a missing vertex", path.display(), n + 1));
				}
				for k in 1..idx.len().saturating_sub(1) {
					tris.push([verts[idx[0]], verts[idx[k]], verts[idx[k + 1]]]);
				}
			}
			_ => {}
		}
	}
	if tris.is_empty() {
		return Err(format!("{}: no faces", path.display()));
	}
	Ok(tris)
}

/// Everything placed over one map.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Level {
	pub objects: Vec<Object>,
}

impl Level {
	/// Where the level for the map at `map` lives.
	pub fn path(map: &std::path::Path) -> std::path::PathBuf {
		map.with_extension("level.toml")
	}

	/// The level saved at `path`; an empty one when there is none.
	pub fn load(path: &std::path::Path) -> Result<Level, String> {
		match std::fs::read_to_string(path) {
			Ok(text) => Level::parse(&text).map_err(|e| format!("{}: {e}", path.display())),
			Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Level::default()),
			Err(e) => Err(format!("{}: {e}", path.display())),
		}
	}

	pub fn parse(text: &str) -> Result<Level, String> {
		let table: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
		let mut objects = Vec::new();
		let list = match table.get("object") {
			Some(toml::Value::Array(list)) => list.as_slice(),
			None => &[],
			Some(_) => return Err("`object` must be an array of tables".into()),
		};
		for (i, value) in list.iter().enumerate() {
			let t = value.as_table().ok_or(format!("object {i} is not a table"))?;
			let num = |v: &toml::Value| v.as_float().or_else(|| v.as_integer().map(|i| i as f32 as f64));
			let float = |key: &str, default: f32| -> Result<f32, String> {
				match t.get(key) {
					None => Ok(default),
					Some(v) => num(v).map(|f| f as f32).ok_or(format!("object {i}: `{key}` is not a number")),
				}
			};
			let triple = |key: &str, default: [f32; 3]| -> Result<[f32; 3], String> {
				match t.get(key) {
					None => Ok(default),
					Some(v) => {
						let a = v.as_array().ok_or(format!("object {i}: `{key}` is not a list"))?;
						let f: Vec<f32> = a.iter().filter_map(num).map(|f| f as f32).collect();
						if f.len() != 3 {
							return Err(format!("object {i}: `{key}` needs three numbers"));
						}
						Ok([f[0], f[1], f[2]])
					}
				}
			};
			let text = |key: &str| t.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string();
			let kind_name = text("kind");
			let kind = Kind::from_name(&kind_name).ok_or(format!("object {i}: unknown kind `{kind_name}`"))?;
			let [px, py, pz] = triple("pos", [0.0; 3])?;
			let d = kind.default_size();
			let [sx, sy, sz] = triple("size", [d.x, d.y, d.z])?;
			let defaults = Object::new(kind, Vec3::ZERO, "");
			objects.push(Object {
				kind,
				pos: vec3(px, py, pz),
				size: vec3(sx, sy, sz),
				yaw: float("yaw", 0.0)?,
				texture: text("texture"),
				color: triple("color", defaults.color)?,
				intensity: float("intensity", defaults.intensity)?,
				reach: float("reach", defaults.reach)?,
				model: text("model"),
			});
		}
		Ok(Level { objects })
	}

	pub fn to_toml(&self) -> String {
		let list: Vec<toml::Value> = self
			.objects
			.iter()
			.map(|o| {
				let triple = |v: [f32; 3]| {
					toml::Value::Array(v.iter().map(|&f| toml::Value::Float(f as f64)).collect())
				};
				let mut t = toml::Table::new();
				t.insert("kind".into(), o.kind.name().into());
				t.insert("pos".into(), triple([o.pos.x, o.pos.y, o.pos.z]));
				t.insert("size".into(), triple([o.size.x, o.size.y, o.size.z]));
				t.insert("yaw".into(), toml::Value::Float(o.yaw as f64));
				if !o.texture.is_empty() {
					t.insert("texture".into(), o.texture.clone().into());
				}
				if o.kind == Kind::Light {
					t.insert("color".into(), triple(o.color));
					t.insert("intensity".into(), toml::Value::Float(o.intensity as f64));
					t.insert("reach".into(), toml::Value::Float(o.reach as f64));
				}
				if o.kind == Kind::Model {
					t.insert("model".into(), o.model.clone().into());
				}
				toml::Value::Table(t)
			})
			.collect();
		let mut root = toml::Table::new();
		root.insert("object".into(), toml::Value::Array(list));
		format!(
			"# Placed by the in-game level editor (F2) over the map of the same name.\n\n{}",
			toml::to_string(&root).unwrap_or_default()
		)
	}

	pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
		std::fs::write(path, self.to_toml()).map_err(|e| format!("{}: {e}", path.display()))
	}

	/// Adds the level's static geometry to `map`: faces to draw and bake,
	/// triangles to collide with. `models` loads a model by name.
	pub fn apply(&self, map: &mut Map, models: &dyn Fn(&str) -> Option<Vec<[Vec3; 3]>>) {
		for o in self.objects.iter().filter(|o| o.kind.is_static()) {
			let dressed = dress(&map.render.textures, &o.texture);
			for poly in o.polygons(models) {
				if poly.len() < 3 {
					continue;
				}
				for k in 1..poly.len() - 1 {
					map.world.push_triangle([poly[0], poly[k], poly[k + 1]]);
					map.surfaces.push(Surface::Concrete);
				}
				map.render.faces.push(face(poly, dressed));
			}
		}
	}
}

/// The level's props as physics bodies: a world of their own over the
/// map's collision, its bodies in step with the props they stand for.
pub struct Props {
	pub world: physics::World,
	/// The object each body is, by its index in the level.
	pub objects: Vec<usize>,
}

impl Props {
	/// Every prop of `level` at rest where it was placed, over `mesh`.
	pub fn new(level: &Level, mesh: &physics::Mesh) -> Props {
		let mut world = physics::World::new(mesh.clone());
		// GoldSrc gravity, as the player and the playground's spheres fall.
		world.gravity = vec3(0.0, -800.0, 0.0);
		let mut objects = Vec::new();
		for (i, o) in level.objects.iter().enumerate() {
			if let Some(body) = o.body() {
				world.bodies.push(body);
				objects.push(i);
			}
		}
		Props { world, objects }
	}

	/// Each prop's body with the object it is.
	pub fn bodies<'a>(&'a self, level: &'a Level) -> impl Iterator<Item = (&'a physics::Body, &'a Object)> {
		self.world
			.bodies
			.iter()
			.zip(&self.objects)
			.filter_map(|(b, &i)| level.objects.get(i).map(|o| (b, o)))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn no_models(_: &str) -> Option<Vec<[Vec3; 3]>> {
		None
	}

	#[test]
	fn levels_round_trip_through_their_file() {
		let mut light = Object::new(Kind::Light, vec3(1.0, 2.0, 3.0), "");
		light.color = [0.25, 0.5, 1.0];
		let mut model = Object::new(Kind::Model, vec3(-8.0, 0.0, 16.0), "DUSTWALL01");
		model.model = "pillar.obj".into();
		model.yaw = 45.0;
		let level = Level {
			objects: vec![Object::new(Kind::Box, vec3(64.0, 0.0, -32.0), "CRATE"), light, model],
		};
		assert_eq!(Level::parse(&level.to_toml()), Ok(level));
	}

	#[test]
	fn solids_wind_every_face_outward() {
		for kind in [Kind::Box, Kind::Ramp, Kind::Cylinder, Kind::Stairs] {
			let mut o = Object::new(kind, vec3(100.0, 50.0, -20.0), "");
			o.yaw = 30.0;
			let polys = o.polygons(&no_models);
			assert!(!polys.is_empty(), "{kind:?} made nothing");
			if kind == Kind::Stairs {
				continue;
			}
			// Inside every convex solid: its corners' mean. A wedge's
			// bounds' middle lies on its slope.
			let all: Vec<Vec3> = polys.iter().flatten().copied().collect();
			let center = all.iter().fold(Vec3::ZERO, |s, &p| s + p) * (1.0 / all.len() as f32);
			for poly in polys {
				let n = (poly[1] - poly[0]).cross(poly[2] - poly[0]);
				let at = poly.iter().fold(Vec3::ZERO, |s, &p| s + p) * (1.0 / poly.len() as f32);
				assert!(n.dot(at - center) > 0.0, "{kind:?} face wound inward");
			}
		}
	}

	#[test]
	fn applied_geometry_collides_and_draws() {
		let mut map = Map {
			world: physics::Mesh::default(),
			surfaces: Vec::new(),
			render: engine::bsp::MapRender {
				albedo: engine::bsp::Atlas { side: 8, rgba: vec![0; 256] },
				lightmap: engine::bsp::Atlas { side: 1, rgba: vec![0; 4] },
				faces: Vec::new(),
				textures: vec![("CRATE".into(), Rect { x: 0, y: 0, w: 64, h: 64 })],
			},
			spawn: Vec3::ZERO,
			spawn_yaw: 0.0,
			spawn_pitch: 0.0,
			kill_y: -1000.0,
		};
		let level = Level {
			objects: vec![
				Object::new(Kind::Box, Vec3::ZERO, "CRATE"),
				Object::new(Kind::Light, Vec3::ZERO, ""),
			],
		};
		level.apply(&mut map, &no_models);
		assert_eq!(map.render.faces.len(), 6);
		assert_eq!(map.world.tris().len(), 12);
		assert_eq!(map.surfaces.len(), 12);
	}

	#[test]
	fn obj_faces_fan_into_triangles() {
		let dir = std::env::temp_dir().join("cs-level-obj-test");
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("quad.obj");
		std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n").unwrap();
		let tris = load_obj(&path).unwrap();
		assert_eq!(tris.len(), 2);
		assert_eq!(tris[1][2], vec3(0.0, 1.0, 0.0));
	}
}
