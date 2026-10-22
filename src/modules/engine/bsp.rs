//! GoldSrc BSP30 map loading: parses a compiled `.bsp` into this game's pieces.
//! World and brush-model faces become the physics mesh, the renderer gets one
//! textured face per BSP face with the map's own baked colored lightmaps, and
//! the entities supply the spawn point. No WAD files are needed: GoldSrc maps
//! embed every texture they use as WAD3 miptex.
//!
//! Coordinates: GoldSrc has Z up; this game has Y up. Every point maps
//! `(x, y, z)` → `(x, z, -y)`, a rotation, so the map keeps its layout and
//! handedness and the movement port runs on it unchanged.
//!
//! Deliberate simplifications:
//! - No BSP tree walk: every face renders and collides; the map is small
//!   enough that its visibility data is not worth porting yet.
//! - Lightmaps take style 0 only. CS 1.6 faces carry at most one style.
//! - Faces whose texture marks them invisible (`sky`, `clip`, `aaatrigger`, …)
//!   neither render nor collide; the sky shows through as the clear color.
//! - Brush models render and collide as static geometry where the file puts
//!   them: breakable crates are solid, doors frozen shut. `func_illusionary`
//!   and trigger volumes render but never collide.

use crate::surface::{Finish, Surface};
use linalg::{vec3, Vec3};
use physics::Mesh;

/// GoldSrc lightmap samples sit on a 16 unit grid.
const LM_STEP: f32 = 16.0;
/// Slack in texture-space coordinates before they snap onto the sample grid:
/// vertex · texinfo accumulates float error around the true integer values.
const LM_EPS: f32 = 1.0 / 128.0;
/// Textures that never render or collide.
const SKIP_TEXTURES: [&str; 9] = [
	"sky",
	"clip",
	"aaatrigger",
	"null",
	"hint",
	"skip",
	"origin",
	"trigger",
	"bevel",
];
/// Entities whose brush model is not solid.
const NON_SOLID: [&str; 3] = ["func_illusionary", "func_bomb_target", "func_buyzone"];
/// Albedo atlas side. Origins and sizes are multiples of 8 texels so a whole
/// rectangle fits in the vertex color's four bytes.
const ALBEDO_SIDE: u32 = 2048;
/// The placeable width inside the atlas: the encoding tops out at 2040.
const ALBEDO_USABLE: u32 = 2040;
/// One lightmap block, waiting to be packed: which face, its size, its texels.
type Block = (usize, u32, u32, Vec<u8>);
type EntityInfo = (
	Option<Vec3>,
	f32,
	f32,
	std::collections::BTreeMap<usize, bool>,
);

/// A loaded map: everything the game needs after the file is parsed.
pub struct Map {
	/// The world the player walks and bullets fly through.
	pub world: Mesh,
	/// The [`Surface`] of every triangle in `world`, in order, for bullets.
	pub surfaces: Vec<Surface>,
	/// What the map renderer draws.
	pub render: MapRender,
	/// Spawn point, feet on it, in game coordinates.
	pub spawn: Vec3,
	/// Spawn view angles in game coordinates.
	pub spawn_yaw: f32,
	pub spawn_pitch: f32,
	/// Falling below this respawns the player.
	pub kill_y: f32,
}

/// Textured faces and their two atlases.
pub struct MapRender {
	/// WAD3 textures packed in one RGBA atlas.
	pub albedo: Atlas,
	/// The map's lightmaps packed in one RGBA atlas, one rectangle per face.
	pub lightmap: Atlas,
	pub faces: Vec<Face>,
	/// Every texture the faces use, by name, and its rectangle in
	/// `albedo`, sorted by name: what new geometry can be dressed in.
	pub textures: Vec<(String, Rect)>,
}

/// One RGBA atlas. Row major, row 0 at v = 0.
pub struct Atlas {
	pub side: u32,
	pub rgba: Vec<u8>,
}

/// A rectangle of atlas texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
	pub x: u32,
	pub y: u32,
	pub w: u32,
	pub h: u32,
}

/// One BSP face: a convex polygon with texture and lightmap coordinates.
pub struct Face {
	/// The face's texture in the albedo atlas.
	pub texture: Rect,
	/// The face's lightmap block in the lightmap atlas.
	pub lightmap: Rect,
	/// Whether the face's fan triangles went into the physics mesh.
	pub solid: bool,
	/// How the face reflects light, from its texture's name.
	pub finish: Finish,
	/// Wound so the cross product of any fan triangle points out of the wall.
	pub verts: Vec<Vert>,
}

impl Face {
	/// The face's outward unit normal, from its fan winding: the first fan
	/// triangle with any area — a long thin face's first three corners may
	/// sit on one line, so the fan is scanned for a real triangle.
	pub fn normal(&self) -> Vec3 {
		let p = &self.verts;
		for k in 1..p.len().saturating_sub(1) {
			let n = (p[k].pos - p[0].pos).cross(p[k + 1].pos - p[0].pos);
			if n.length2() > 1e-12 {
				return n.normalize();
			}
		}
		// Every fan triangle flat: the face has no area, so no lighting
		// depends on the answer.
		Vec3::Y
	}
}

/// One vertex of a [`Face`].
pub struct Vert {
	pub pos: Vec3,
	/// Texture coordinates in whole-texture units; the shader tiles them.
	pub uv: (f32, f32),
	/// Lightmap coordinates in texels of the face's block, unclamped at the
	/// corners; samplers clamp on read.
	pub lm: (f32, f32),
}

/// Parses the BSP30 file at `path`.
pub fn load(path: &std::path::Path) -> Result<Map, String> {
	let name = path
		.file_stem()
		.and_then(|s| s.to_str())
		.unwrap_or("map")
		.to_owned();
	let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
	parse(&data, &name, Some(path))
}

/// Parses an in-memory BSP with no WAD search path; also how the tests build
/// their maps.
#[cfg(test)]
pub fn load_bytes(data: &[u8], name: &str) -> Result<Map, String> {
	parse(data, name, None)
}

/// Parses the BSP. `map` names the file the data came from, for the WAD
/// search when the map's textures live outside it.
fn parse(data: &[u8], name: &str, map: Option<&std::path::Path>) -> Result<Map, String> {
	if data.len() < 124 {
		return Err(format!(
			"{name}: {} bytes, too short for a BSP header",
			data.len()
		));
	}
	if i32_at(data, 0)? != 30 {
		return Err(format!(
			"{name}: BSP version {}, this loader reads 30",
			i32_at(data, 0)?
		));
	}
	let bsp = Lumps::new(data)?;
	let (spawn, spawn_yaw, spawn_pitch, solids) = entities(&bsp)?;
	let face_model = models(&bsp)?;
	// Sorted lightmap offsets, so a face's block ends where the next begins.
	let mut lit: Vec<i32> = (0..bsp.face_count())
		.filter_map(|i| bsp.face(i).ok())
		.filter(|f| f.lightofs >= 0)
		.map(|f| f.lightofs)
		.collect();
	lit.sort_unstable();
	lit.dedup();
	// Texture atlas first: every face needs its miptex's rectangle. External
	// textures load from the WADs the worldspawn names, beside the map.
	let wads = Wads::new(&bsp, map);
	let albedo = TextureAtlas::new(&bsp, &wads)?;
	// The world under construction.
	let mut world = Mesh::default();
	let mut surfaces = Vec::new();
	let mut faces = Vec::new();
	let mut textures = std::collections::BTreeMap::new();
	let mut blocks: Vec<Block> = Vec::new();
	let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
	for i in 0..bsp.face_count() {
		let f = bsp.face(i)?;
		let tex = bsp.texinfo(f.texinfo)?;
		if tex.miptex < 0 {
			continue;
		}
		let (tw, th) = match bsp.miptex_size(tex.miptex as usize) {
			Ok(size) if size.0 > 0 && size.1 > 0 => size,
			_ => continue,
		};
		let tex_name = bsp.miptex_name(tex.miptex as usize)?;
		if SKIP_TEXTURES.contains(&tex_name.to_ascii_lowercase().as_str()) {
			continue;
		}
		// The polygon: transformed positions plus texture and lightmap space.
		let mut verts = Vec::with_capacity(f.edges);
		let (mut s_lo, mut s_hi, mut t_lo, mut t_hi) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
		for e in 0..f.edges {
			let v = bsp.vertex(bsp.surfedge_vertex(f.firstedge + e)?)?;
			let pos = to_our(v);
			let s = dot3(v, &tex.s) + tex.s_shift;
			let t = dot3(v, &tex.t) + tex.t_shift;
			verts.push(Vert {
				pos,
				uv: (s / tw as f32, t / th as f32),
				lm: (0.0, 0.0),
			});
			s_lo = s_lo.min(s);
			s_hi = s_hi.max(s);
			t_lo = t_lo.min(t);
			t_hi = t_hi.max(t);
			lo = lo.min(pos);
			hi = hi.max(pos);
		}
		if verts.len() < 3 {
			continue;
		}
		// Wound outward: the fan's cross product points along the face's
		// plane normal, flipped to its side. The file's edge order is
		// arbitrary, and the light bake reads the winding as the surface's
		// normal — flipped faces bake lit walls black and shaded walls lit.
		// The whole polygon's area vector, summed over its fan: its first
		// three corners alone often stand on one line — a T-junction's
		// extra vertex — and their cross product's sign is then noise.
		let normal = to_our(bsp.plane_normal(f.plane, f.side)?);
		let p0 = verts[0].pos;
		let cross = verts.windows(2).skip(1).fold(Vec3::ZERO, |sum, w| {
			sum + (w[0].pos - p0).cross(w[1].pos - p0)
		});
		if cross.dot(normal) < 0.0 {
			verts.reverse();
		}
		// One lightmap block covering the face's extent on the sample grid.
		let (s0, t0) = (
			(s_lo / LM_STEP + LM_EPS).floor(),
			(t_lo / LM_STEP + LM_EPS).floor(),
		);
		let mut w = (s_hi / LM_STEP - LM_EPS).ceil() - s0 + 1.0;
		let mut h = (t_hi / LM_STEP - LM_EPS).ceil() - t0 + 1.0;
		let mut block = vec![128u8, 128, 128, 255]; // neutral until real samples land
		if f.lightofs >= 0 {
			// qrad's own rounding decides the last row, so the bytes between
			// this face's samples and the next face's settle the size.
			let span = light_span(&bsp, &lit, f.lightofs) / 3;
			for (dw, dh) in [
				(0, 0),
				(1, 0),
				(0, 1),
				(1, 1),
				(2, 0),
				(0, 2),
				(-1, 0),
				(0, -1),
			]
			.map(|(a, b): (i32, i32)| (a as f32, b as f32))
			{
				if w + dw > 0.0 && h + dh > 0.0 && ((w + dw) * (h + dh)) as usize == span {
					w += dw;
					h += dh;
					break;
				}
			}
			block = bsp.light_block(f.lightofs, w.max(1.0) as u32, h.max(1.0) as u32)?;
		}
		let (w, h) = (w.max(1.0) as u32, h.max(1.0) as u32);
		if w > 255 || h > 255 {
			return Err(format!(
				"{name}: face {i} lightmap {w}x{h} overflows the vertex encoding"
			));
		}
		for v in &mut verts {
			let (s, t) = (v.uv.0 * tw as f32, v.uv.1 * th as f32);
			// Unclamped: the corners of a face's extent sit outside its
			// block's sample centers, and clamping them would hide the
			// affine map its light grid runs on. Samplers clamp on read.
			v.lm = (
				(s - (s0 * LM_STEP + LM_STEP / 2.0)) / LM_STEP,
				(t - (t0 * LM_STEP + LM_STEP / 2.0)) / LM_STEP,
			);
		}
		// Physics: fan triangles wound outward already, above.
		let model = face_model.get(i).copied().unwrap_or(0);
		if solids.get(&model).copied().unwrap_or(true) {
			for k in 1..verts.len() - 1 {
				world.push_triangle([verts[0].pos, verts[k].pos, verts[k + 1].pos]);
				surfaces.push(bullet_surface(tex_name));
			}
		}
		faces.push(Face {
			texture: {
				let rect = albedo.rect(tex.miptex)?;
				textures.insert(tex_name.to_string(), rect);
				rect
			},
			lightmap: Rect {
				x: 0,
				y: 0,
				w: 0,
				h: 0,
			},
			solid: solids.get(&model).copied().unwrap_or(true),
			finish: Finish::of(tex_name),
			verts,
		});
		blocks.push((faces.len() - 1, w, h, block));
	}
	// Pack the lightmap blocks; 2048 texels fit every shipped CS map.
	let (lightmap, rects) = pack_lightmaps(&blocks)?;
	for (i, r) in rects {
		faces[i].lightmap = r;
	}
	world.prepare();
	let center = (lo + hi) * 0.5;
	let spawn = spawn.unwrap_or(vec3(center.x, lo.y, center.z));
	Ok(Map {
		world,
		surfaces,
		render: MapRender {
			albedo: albedo.atlas,
			lightmap,
			faces,
			textures: textures.into_iter().collect(),
		},
		spawn,
		spawn_yaw,
		spawn_pitch,
		kill_y: lo.y - 256.0,
	})
}

/// GoldSrc `(x, y, z)` with Z up to this game's Y up.
pub(crate) fn to_our(p: [f32; 3]) -> Vec3 {
	vec3(p[0], p[2], -p[1])
}

fn dot3(p: [f32; 3], v: &[f32; 3]) -> f32 {
	p[0] * v[0] + p[1] * v[1] + p[2] * v[2]
}

/// The bullet material a texture name stands for.
fn bullet_surface(tex_name: &str) -> Surface {
	let name = tex_name.to_ascii_lowercase();
	if name.contains("crt") {
		Surface::Jade
	} else if name.contains("rock") {
		Surface::Marble
	} else {
		Surface::Concrete
	}
}

/// Bytes one face's samples may use: up to the next face's samples, or to the
/// end of the lump for the last.
fn light_span(bsp: &Lumps, lit: &[i32], lightofs: i32) -> usize {
	let next = lit
		.iter()
		.find(|&&o| o > lightofs)
		.map(|&o| o as usize)
		.unwrap_or(bsp.lightdata.len());
	(next - lightofs.max(0) as usize).min(bsp.lightdata.len())
}

/// One raw BSP face as the file stores it.
struct RawFace {
	plane: usize,
	side: bool,
	firstedge: usize,
	edges: usize,
	texinfo: usize,
	lightofs: i32,
}

/// Texture axes of one texinfo entry.
struct Texinfo {
	s: [f32; 3],
	s_shift: f32,
	t: [f32; 3],
	t_shift: f32,
	miptex: i32,
}

/// The lumps a map is read from. Lump order: entities, planes, miptex,
/// vertices, visibility, nodes, texinfo, faces, lightdata, clipnodes, leaves,
/// marksurfaces, edges, surfedges, models.
struct Lumps<'a> {
	entities: &'a [u8],
	planes: &'a [u8],
	miptex: &'a [u8],
	vertices: &'a [u8],
	texinfo: &'a [u8],
	faces: &'a [u8],
	lightdata: &'a [u8],
	edges: &'a [u8],
	surfedges: &'a [u8],
	models: &'a [u8],
}

impl<'a> Lumps<'a> {
	fn new(data: &'a [u8]) -> Result<Lumps<'a>, String> {
		let mut lumps = [
			&[][..],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
			&[],
		];
		for (i, slot) in lumps.iter_mut().enumerate() {
			let off = i32_at(data, 4 + 8 * i)?.max(0) as usize;
			let len = i32_at(data, 4 + 8 * i + 4)?.max(0) as usize;
			if off + len > data.len() {
				return Err(format!("lump {i} runs past the file"));
			}
			*slot = &data[off..off + len];
		}
		Ok(Lumps {
			entities: lumps[0],
			planes: lumps[1],
			miptex: lumps[2],
			vertices: lumps[3],
			texinfo: lumps[6],
			faces: lumps[7],
			lightdata: lumps[8],
			edges: lumps[12],
			surfedges: lumps[13],
			models: lumps[14],
		})
	}

	fn face_count(&self) -> usize {
		self.faces.len() / 20
	}

	/// Face `i`'s plane, edges and lightmap offset.
	fn face(&self, i: usize) -> Result<RawFace, String> {
		let at = 20 * i;
		if at + 20 > self.faces.len() {
			return Err(format!("face {i} runs past the lump"));
		}
		Ok(RawFace {
			plane: u16_at(self.faces, at)? as usize,
			side: self.faces[at + 2] != 0,
			firstedge: i32_at(self.faces, at + 4)?.max(0) as usize,
			edges: u16_at(self.faces, at + 8)? as usize,
			texinfo: u16_at(self.faces, at + 10)? as usize,
			lightofs: i32_at(self.faces, at + 16)?,
		})
	}

	/// Vertex `i` of the map, GoldSrc coordinates.
	fn vertex(&self, i: usize) -> Result<[f32; 3], String> {
		let at = 12 * i;
		if at + 12 > self.vertices.len() {
			return Err(format!("vertex {i} runs past the lump"));
		}
		Ok([
			f32_at(self.vertices, at)?,
			f32_at(self.vertices, at + 4)?,
			f32_at(self.vertices, at + 8)?,
		])
	}

	/// The vertex surfedge `i` points at, following the edge's direction.
	fn surfedge_vertex(&self, i: usize) -> Result<usize, String> {
		if 4 * i + 4 > self.surfedges.len() {
			return Err(format!("surfedge {i} runs past the lump"));
		}
		let se = i32_at(self.surfedges, 4 * i)?;
		let edge = (se.unsigned_abs() as usize) * 4;
		if edge + 4 > self.edges.len() {
			return Err(format!("edge of surfedge {i} runs past the lump"));
		}
		let (a, b) = (u16_at(self.edges, edge)?, u16_at(self.edges, edge + 2)?);
		Ok(if se >= 0 { a } else { b } as usize)
	}

	/// The plane's normal, flipped to the face's side, GoldSrc coordinates.
	fn plane_normal(&self, plane: usize, side: bool) -> Result<[f32; 3], String> {
		let at = 20 * plane;
		if at + 12 > self.planes.len() {
			return Err(format!("plane {plane} runs past the lump"));
		}
		let flip = if side { -1.0 } else { 1.0 };
		Ok([
			f32_at(self.planes, at)? * flip,
			f32_at(self.planes, at + 4)? * flip,
			f32_at(self.planes, at + 8)? * flip,
		])
	}

	/// Texture axes of texinfo `i`.
	fn texinfo(&self, i: usize) -> Result<Texinfo, String> {
		let at = 40 * i;
		if at + 36 > self.texinfo.len() {
			return Err(format!("texinfo {i} runs past the lump"));
		}
		Ok(Texinfo {
			s: [
				f32_at(self.texinfo, at)?,
				f32_at(self.texinfo, at + 4)?,
				f32_at(self.texinfo, at + 8)?,
			],
			s_shift: f32_at(self.texinfo, at + 12)?,
			t: [
				f32_at(self.texinfo, at + 16)?,
				f32_at(self.texinfo, at + 20)?,
				f32_at(self.texinfo, at + 24)?,
			],
			t_shift: f32_at(self.texinfo, at + 28)?,
			miptex: i32_at(self.texinfo, at + 32)?,
		})
	}

	fn miptex_count(&self) -> Result<usize, String> {
		if self.miptex.len() < 4 {
			return Ok(0);
		}
		Ok(i32_at(self.miptex, 0)?.max(0) as usize)
	}

	/// Where miptex `i` starts inside the miptex lump, if embedded.
	fn miptex_at(&self, i: usize) -> Result<Option<usize>, String> {
		if i >= self.miptex_count()? {
			return Err(format!("miptex {i} of {}", self.miptex_count()?));
		}
		let off = i32_at(self.miptex, 4 + 4 * i)?;
		Ok((off >= 0).then_some(off as usize))
	}

	fn miptex_name(&self, i: usize) -> Result<&'a str, String> {
		let at = self.miptex_at(i)?.ok_or("miptex not embedded")?;
		let end = self.miptex[at..]
			.iter()
			.position(|&b| b == 0)
			.map(|p| at + p)
			.unwrap_or(self.miptex.len())
			.min(at + 16);
		Ok(std::str::from_utf8(&self.miptex[at..end]).unwrap_or("texture"))
	}

	fn miptex_size(&self, i: usize) -> Result<(u32, u32), String> {
		let at = self.miptex_at(i)?.ok_or("miptex not embedded")? + 16;
		if at + 8 > self.miptex.len() {
			return Err("miptex header runs past the lump".into());
		}
		Ok((
			i32_at(self.miptex, at)?.max(0) as u32,
			i32_at(self.miptex, at + 4)?.max(0) as u32,
		))
	}

	/// The miptex record of texture `i`: name, size, and its decoded texels.
	/// External textures — the header embedded, all four mip offsets zero —
	/// resolve from `wads`; a texture no source can decode renders neutral.
	fn texture(&self, i: usize, wads: &Wads) -> Result<(String, u32, u32, Vec<u8>), String> {
		let name = self.miptex_name(i)?.to_owned();
		let (w, h) = self.miptex_size(i)?;
		let m = self.miptex_at(i)?.ok_or("miptex header missing")?;
		if m + 40 > self.miptex.len() {
			return Err("miptex header runs past the lump".into());
		}
		let mip0 = i32_at(self.miptex, m + 24)?;
		let rgba = if mip0 > 0 {
			decode_miptex(&self.miptex[m..], &name)?.2
		} else {
			match wads.texture(&name) {
				Some(t) => t.to_vec(),
				None => neutral(w, h),
			}
		};
		Ok((name, w, h, rgba))
	}

	/// The face's `w` × `h` RGB samples as RGBA.
	fn light_block(&self, lightofs: i32, w: u32, h: u32) -> Result<Vec<u8>, String> {
		let at = lightofs.max(0) as usize;
		let samples = (w * h) as usize;
		if at + samples * 3 > self.lightdata.len() {
			return Err(format!("lightmap at {lightofs} runs past the lump"));
		}
		let mut out = Vec::with_capacity(samples * 4);
		for s in 0..samples {
			let at = at + s * 3;
			out.extend([
				self.lightdata[at],
				self.lightdata[at + 1],
				self.lightdata[at + 2],
				255,
			]);
		}
		Ok(out)
	}
}

/// Spawns and brush-model solidity from the entity list. Returns the first
/// `info_player_start` and the solid flag of every brush model.
fn entities(bsp: &Lumps) -> Result<EntityInfo, String> {
	let text = std::str::from_utf8(bsp.entities).map_err(|_| "entities are not UTF-8".to_string())?;
	let mut spawn = None;
	let mut spawn_yaw = 0.0;
	let mut spawn_pitch = 0.0;
	let mut solids = std::collections::BTreeMap::new();
	for block in text.split('{').skip(1) {
		let body = block.split('}').next().unwrap_or("");
		let Some(classname) = field(body, "classname") else {
			continue;
		};
		if spawn.is_none() && classname == "info_player_start" {
			if let Some(o) = field(body, "origin") {
				spawn = Some(to_our(origin(&o)?));
				if let Some(angles) = field(body, "angles") {
					let mut values = angles
						.split_whitespace()
						.map(str::parse::<f32>)
						.map(|v| v.map_err(|e| e.to_string()));
					spawn_pitch = values
						.next()
						.transpose()?
						.ok_or("spawn angles need pitch and yaw")?;
					let yaw = values
						.next()
						.transpose()?
						.ok_or("spawn angles need pitch and yaw")?;
					spawn_yaw = (-yaw).rem_euclid(360.0);
				} else if let Some(angle) = field(body, "angle") {
					spawn_yaw = -angle.parse::<f32>().map_err(|e| e.to_string())?;
					spawn_yaw = spawn_yaw.rem_euclid(360.0);
				}
			}
		}
		if let Some(model) = field(body, "model") {
			if let Ok(i) = model.trim_start_matches('*').parse::<usize>() {
				let solid = !NON_SOLID.contains(&classname.as_str()) && !classname.starts_with("trigger");
				solids.insert(i, solid);
			}
		}
	}
	Ok((spawn, spawn_yaw, spawn_pitch, solids))
}

/// The value of `"key" "value"` in one entity body; `lightmap` reads lights
/// and suns from the same entity text, so the parse lives here once.
pub(crate) fn field(body: &str, key: &str) -> Option<String> {
	let rest = body.split_once(&format!("\"{key}\" \""))?.1;
	Some(rest.split_once('"')?.0.to_owned())
}

/// The entities lump of a BSP30 file, the same text [`parse`] splits into
/// blocks; `lightmap`'s sun and light reads share it instead of re-parsing.
pub(crate) fn entity_text(bytes: &[u8]) -> Result<&str, String> {
	if bytes.len() < 12 || i32_at(bytes, 0)? != 30 {
		return Err("not a BSP30".to_string());
	}
	// The entities lump is index 0.
	let off = i32_at(bytes, 4)? as usize;
	let len = i32_at(bytes, 8)? as usize;
	let lump = bytes
		.get(off..off + len)
		.ok_or_else(|| "entities lump past the end".to_string())?;
	std::str::from_utf8(lump).map_err(|_| "entities are not UTF-8".to_string())
}

/// The model owning each face: world model 0 for faces no model claims.
fn models(bsp: &Lumps) -> Result<Vec<usize>, String> {
	let count = bsp.models.len() / 64;
	let mut face_model = vec![0usize; bsp.face_count()];
	for i in 0..count {
		let at = 64 * i + 56;
		let (first, num) = (
			i32_at(bsp.models, at)?.max(0) as usize,
			i32_at(bsp.models, at + 4)?.max(0) as usize,
		);
		if first + num > bsp.face_count() {
			return Err(format!("model {i} faces run past the face list"));
		}
		for f in &mut face_model[first..first + num] {
			*f = i;
		}
	}
	Ok(face_model)
}

/// `"x y z"` to GoldSrc coordinates.
fn origin(s: &str) -> Result<[f32; 3], String> {
	let mut it = s
		.split_whitespace()
		.map(str::parse::<f32>)
		.map(|n| n.map_err(|e| e.to_string()));
	let mut next = || {
		it.next()
			.ok_or_else(|| "origin needs 3 numbers".to_string())?
	};
	Ok([next()?, next()?, next()?])
}

/// The WAD3 archives a map's external textures live in: the miptex lump
/// embeds only each external texture's 40-byte header, with all four mip
/// offsets zero, and the worldspawn's `wad` key names the WADs to read.
struct Wads {
	textures: std::collections::BTreeMap<String, Vec<u8>>,
}

impl Wads {
	/// Loads worldspawn WADs from project assets and the local CS install.
	fn new(bsp: &Lumps, map: Option<&std::path::Path>) -> Wads {
		let mut wads = Wads {
			textures: std::collections::BTreeMap::new(),
		};
		let Some(map) = map else { return wads };
		let dir = map.parent().unwrap_or(map);
		let mut dirs = vec![dir.to_path_buf(), dir.parent().unwrap_or(dir).to_path_buf()];
		if let Some(assets) = dir.parent() {
			dirs.push(assets.join("wads"));
		}
		let mut installs = std::env::var_os("CS_CSTRIKE")
			.map(std::path::PathBuf::from)
			.into_iter()
			.collect::<Vec<_>>();
		installs.push(std::path::PathBuf::from(
			r"C:\Program Files (x86)\Steam\steamapps\common\Half-Life\cstrike",
		));
		for install in installs {
			dirs.push(install.clone());
			if let Some(parent) = install.parent() {
				dirs.push(parent.join("valve"));
			}
		}
		for name in wad_names(bsp) {
			for base in &dirs {
				let path = base.join(&name);
				if path.is_file() {
					wads.load_wad(&path);
					break;
				}
			}
		}
		wads
	}

	/// Reads one WAD3's miptex entries. Entries that cannot be decoded are
	/// skipped: a missing texture renders neutral, it never blocks the map.
	fn load_wad(&mut self, path: &std::path::Path) {
		let Ok(data) = std::fs::read(path) else {
			return;
		};
		if data.get(0..4) != Some(b"WAD3") {
			return;
		}
		let (count, dir_at) = (i32_at(&data, 4), i32_at(&data, 8));
		let (Ok(count), Ok(dir_at)) = (count, dir_at) else {
			return;
		};
		for e in 0..count.max(0) as usize {
			let at = dir_at as usize + 32 * e;
			let Ok(filepos) = i32_at(&data, at) else {
				break;
			};
			if at + 32 > data.len() || data[at + 12] != 0x43 {
				continue; // miptex entries only
			}
			let end = data[at + 16..]
				.iter()
				.position(|&b| b == 0)
				.map(|p| at + 16 + p)
				.unwrap_or(data.len())
				.min(at + 32);
			let Some(name) = std::str::from_utf8(&data[at + 16..end]).ok() else {
				continue;
			};
			if filepos >= 0 {
				if let Ok((_, _, rgba)) = decode_miptex(&data[filepos as usize..], name) {
					self.textures.insert(name.to_ascii_lowercase(), rgba);
				}
			}
		}
	}

	fn texture(&self, name: &str) -> Option<&[u8]> {
		self
			.textures
			.get(&name.to_ascii_lowercase())
			.map(Vec::as_slice)
	}
}

/// The WAD basenames the worldspawn's `wad` key names.
fn wad_names(bsp: &Lumps) -> Vec<String> {
	let Some(text) = std::str::from_utf8(bsp.entities).ok() else {
		return Vec::new();
	};
	let Some(list) = text
		.split('{')
		.nth(1)
		.and_then(|body| field(body.split('}').next().unwrap_or(""), "wad"))
	else {
		return Vec::new();
	};
	list
		.split(';')
		.filter_map(|p| {
			std::path::Path::new(p)
				.file_name()
				.and_then(|n| n.to_str())
				.map(str::to_owned)
		})
		.collect()
}

/// Decodes one miptex record — 40-byte header, four mips and its 256-color
/// palette — to RGBA. Palette index 255 is transparent for `{` textures, as in
/// GoldSrc.
fn decode_miptex(m: &[u8], name: &str) -> Result<(u32, u32, Vec<u8>), String> {
	if m.len() < 40 {
		return Err("miptex header truncated".into());
	}
	let (w, h) = (i32_at(m, 16)?.max(0) as u32, i32_at(m, 20)?.max(0) as u32);
	if w == 0 || h == 0 {
		return Err(format!("{name}: no texels"));
	}
	let mip0 = i32_at(m, 24)?.max(0) as usize;
	let mip3 = i32_at(m, 36)?.max(0) as usize;
	if mip0 == 0 {
		return Err(format!("{name}: mips not embedded"));
	}
	if mip0 + (w * h) as usize > m.len() {
		return Err(format!("{name}: mip 0 truncated"));
	}
	let pal_at = mip3 + (w * h / 64) as usize + 2;
	if pal_at + 768 > m.len() {
		return Err(format!("{name}: palette truncated"));
	}
	let transparent = name.starts_with('{');
	let mut out = Vec::with_capacity((w * h) as usize * 4);
	for &idx in &m[mip0..mip0 + (w * h) as usize] {
		let p = pal_at + 3 * idx as usize;
		let a = if transparent && idx == 255 { 0 } else { 255 };
		out.extend([m[p], m[p + 1], m[p + 2], a]);
	}
	Ok((w, h, out))
}

/// Flat gray texels for a texture no source provides.
fn neutral(w: u32, h: u32) -> Vec<u8> {
	vec![[96, 96, 96, 255].as_slice(); (w * h) as usize].concat()
}

/// The texture atlas: every texture the faces use, packed in shelves.
struct TextureAtlas {
	atlas: Atlas,
	rects: Vec<Rect>,
}

impl TextureAtlas {
	fn new(bsp: &Lumps, wads: &Wads) -> Result<TextureAtlas, String> {
		let count = bsp.miptex_count()?;
		let mut used = Vec::new();
		for i in 0..count {
			if bsp.miptex_at(i)?.is_none() {
				continue;
			}
			let (w, h) = bsp.miptex_size(i)?;
			if w > 0 && h > 0 {
				used.push((i, w, h));
			}
		}
		// Tallest first keeps the shelf rows shallow.
		used.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));
		let mut atlas = Atlas {
			side: ALBEDO_SIDE,
			rgba: vec![0; (ALBEDO_SIDE * ALBEDO_SIDE * 4) as usize],
		};
		let mut rects = vec![
			Rect {
				x: 0,
				y: 0,
				w: 8,
				h: 8
			};
			count
		];
		let mut packer = Packer::new(ALBEDO_USABLE, 8);
		for &(i, w, h) in &used {
			let rect = packer
				.push(w, h)
				.ok_or_else(|| format!("miptex {i} overflows the {ALBEDO_SIDE} atlas"))?;
			let rgba = match bsp.texture(i, wads) {
				Ok((_, _, _, rgba)) => rgba,
				Err(_) => neutral(w, h),
			};
			blit(&mut atlas, &rgba, (w, h), rect)?;
			rects[i] = rect;
		}
		Ok(TextureAtlas { atlas, rects })
	}

	fn rect(&self, miptex: i32) -> Result<Rect, String> {
		self
			.rects
			.get(miptex.max(0) as usize)
			.copied()
			.ok_or_else(|| format!("miptex {miptex} not in the atlas"))
	}
}

/// Copies `rgba` into the atlas, repeating the last row and column of
/// textures that grew to the atlas's 8-texel alignment.
fn blit(atlas: &mut Atlas, rgba: &[u8], (w, h): (u32, u32), rect: Rect) -> Result<(), String> {
	let (w, h) = (w as usize, h as usize);
	if rgba.len() < w * h * 4 {
		return Err("texture decode short of texels".into());
	}
	for row in 0..rect.h as usize {
		for col in 0..rect.w as usize {
			let at = (row.min(h - 1) * w + col.min(w - 1)) * 4;
			let to = ((rect.y as usize + row) * atlas.side as usize + rect.x as usize + col) * 4;
			atlas.rgba[to..to + 4].copy_from_slice(&rgba[at..at + 4]);
		}
	}
	Ok(())
}

/// Packs the lightmap blocks into one atlas with a one-texel gap between
/// neighbors, so linear filtering never blends two faces' light.
fn pack_lightmaps(blocks: &[Block]) -> Result<(Atlas, Vec<(usize, Rect)>), String> {
	for side in [2048u32, 4096] {
		let mut atlas = Atlas {
			side,
			rgba: vec![0; (side * side * 4) as usize],
		};
		let mut packer = lightbake::pack::Packer::new(side);
		let mut rects = Vec::with_capacity(blocks.len());
		let mut full = true;
		for &(face, w, h, ref block) in blocks {
			let Some((x, y)) = packer.place(w, h) else {
				full = false;
				break;
			};
			let rect = Rect { x, y, w, h };
			for row in 0..h as usize {
				for col in 0..w as usize {
					let at = (row * w as usize + col) * 4;
					let to = ((rect.y as usize + row) * side as usize + rect.x as usize + col) * 4;
					atlas.rgba[to..to + 4].copy_from_slice(&block[at..at + 4]);
				}
			}
			rects.push((face, rect));
		}
		if full {
			return Ok((atlas, rects));
		}
	}
	Err("lightmap atlas overflowed 4096 texels".into())
}

/// Shelf packer for the albedo atlas: rectangles go left to right on a row,
/// the next row starts below the tallest. Sizes and origins are multiples of
/// `align`. Lightmaps pack through `lightbake::pack::Packer` instead, whose
/// one-texel gutter serves the same no-bleed purpose at align 1.
struct Packer {
	side: u32,
	align: u32,
	x: u32,
	y: u32,
	row_h: u32,
}

impl Packer {
	fn new(side: u32, align: u32) -> Packer {
		Packer {
			side,
			align,
			x: 0,
			y: 0,
			row_h: 0,
		}
	}

	fn push(&mut self, w: u32, h: u32) -> Option<Rect> {
		let (w, h) = (
			w.div_ceil(self.align) * self.align,
			h.div_ceil(self.align) * self.align,
		);
		if w > self.side {
			return None;
		}
		if self.x > 0 && self.x + w > self.side {
			self.y += self.row_h + self.align;
			self.x = 0;
			self.row_h = 0;
		}
		if self.y + h > self.side {
			return None;
		}
		let rect = Rect {
			x: self.x,
			y: self.y,
			w,
			h,
		};
		self.x += w + self.align;
		self.row_h = self.row_h.max(h);
		Some(rect)
	}
}

fn f32_at(d: &[u8], at: usize) -> Result<f32, String> {
	d.get(at..at + 4)
		.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
		.ok_or_else(|| "read past the file".into())
}

fn i32_at(d: &[u8], at: usize) -> Result<i32, String> {
	d.get(at..at + 4)
		.map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
		.ok_or_else(|| "read past the file".into())
}

fn u16_at(d: &[u8], at: usize) -> Result<u16, String> {
	d.get(at..at + 2)
		.map(|b| u16::from_le_bytes([b[0], b[1]]))
		.ok_or_else(|| "read past the file".into())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A minimal BSP30 in memory: one floor quad with one texture and a
	/// complete 2 × 2 lightmap block, plus the entities that spawn the player.
	fn tiny_bsp() -> Vec<u8> {
		// One quad on z = 0, wound 0-1-2-3, 16 units on a side.
		let vertices: Vec<f32> = vec![
			0.0, 0.0, 0.0, // 0
			16.0, 0.0, 0.0, // 1
			16.0, 16.0, 0.0, // 2
			0.0, 16.0, 0.0, // 3
		];
		let texinfo: Vec<f32> = vec![
			1.0, 0.0, 0.0, // s = x
			0.0, // s shift
			0.0, 1.0, 0.0, // t = y
			0.0, // t shift
		];
		let mut texinfo = texinfo
			.iter()
			.flat_map(|v| v.to_le_bytes())
			.collect::<Vec<_>>();
		texinfo.extend(0i32.to_le_bytes()); // miptex 0
		texinfo.extend(0i32.to_le_bytes()); // flags
		let mut edges = Vec::new();
		for (a, b) in [(0u16, 1u16), (1, 2), (2, 3), (3, 0)] {
			edges.extend(a.to_le_bytes());
			edges.extend(b.to_le_bytes());
		}
		let surfedges: Vec<i32> = vec![0, 1, 2, 3];
		// Miptex "floor", 8 × 8, mip 0 plus the three smaller mips and a palette.
		let mut mip = Vec::new();
		mip.extend_from_slice(b"floor\0\0\0\0\0\0\0\0\0\0\0");
		mip.extend(8i32.to_le_bytes());
		mip.extend(8i32.to_le_bytes());
		for off in [40i32, 104, 120, 124] {
			mip.extend(off.to_le_bytes());
		}
		for i in 0..64 {
			mip.push(i as u8); // mip 0
		}
		mip.extend((64i32 / 4).to_le_bytes()); // mip 1
		mip.extend((64i32 / 16).to_le_bytes()); // mip 2: 16 texels
		mip.extend(1i32.to_le_bytes()); // mip 3: 64 / 64 texels
		mip.extend(256u16.to_le_bytes()); // palette count
		for c in 0..256u16 {
			mip.extend([(c % 256) as u8, c as u8, 255]);
		}
		let mut miptex = Vec::new();
		miptex.extend(1i32.to_le_bytes()); // one texture
		miptex.extend(8i32.to_le_bytes()); // it starts 8 bytes into the lump
		miptex.extend_from_slice(&mip);
		// Faces: one lit quad.
		let mut faces = Vec::new();
		faces.extend(0u16.to_le_bytes()); // planenum
		faces.push(0); // side
		faces.push(0); // pad
		faces.extend(0i32.to_le_bytes()); // firstedge
		faces.extend(4u16.to_le_bytes()); // numedges
		faces.extend(0u16.to_le_bytes()); // texinfo
		faces.extend([0u8, 255, 255, 255]); // styles: only style 0
		faces.extend(0i32.to_le_bytes()); // lightofs
																		// Models: the world owns the one face.
		let mut models = Vec::new();
		models.extend([0f32; 3].iter().flat_map(|v| v.to_le_bytes())); // origin
		models.extend([0u8; 44]); // headnodes and padding
		models.extend(0i32.to_le_bytes()); // firstface at 56
		models.extend(1i32.to_le_bytes()); // numfaces at 60
																		 // The header: version 30 and 15 empty lumps, filled in as data lands.
		let mut b = vec![0u8; 124];
		b[0..4].copy_from_slice(&30i32.to_le_bytes());
		let lump = |b: &mut Vec<u8>, i: usize, data: &[u8]| {
			let at = 4 + 8 * i;
			let (off, len) = (b.len() as i32, data.len() as i32);
			b[at..at + 4].copy_from_slice(&off.to_le_bytes());
			b[at + 4..at + 8].copy_from_slice(&len.to_le_bytes());
			b.extend_from_slice(data);
		};
		lump(
			&mut b,
			0,
			br#"{"classname" "info_player_start" "origin" "8 8 16" "angles" "0 90 0"}"#,
		);
		let mut plane = Vec::new();
		for v in [0.0f32, 0.0, 1.0, 0.0] {
			plane.extend(v.to_le_bytes());
		}
		lump(&mut b, 1, &plane);
		lump(&mut b, 2, &miptex);
		lump(
			&mut b,
			3,
			&vertices
				.iter()
				.flat_map(|v| v.to_le_bytes())
				.collect::<Vec<_>>(),
		);
		lump(
			&mut b,
			6,
			&texinfo
				.iter()
				.flat_map(|v| v.to_le_bytes())
				.collect::<Vec<_>>(),
		);
		lump(&mut b, 7, &faces);
		// Lightdata: 2 × 2 RGB samples, a gradient.
		lump(
			&mut b,
			8,
			&[10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
		);
		lump(&mut b, 12, &edges);
		lump(
			&mut b,
			13,
			&surfedges
				.iter()
				.flat_map(|v| v.to_le_bytes())
				.collect::<Vec<_>>(),
		);
		lump(&mut b, 14, &models);
		b
	}

	#[test]
	fn one_quad_loads_with_its_lightmap() {
		let map = load_bytes(&tiny_bsp(), "tiny").expect("parse");
		// One quad: two triangles, both concrete, wound to point up.
		assert_eq!(map.world.tris().len(), 2);
		assert_eq!(map.surfaces, vec![Surface::Concrete, Surface::Concrete]);
		for t in map.world.tris() {
			assert!(physics::triangle_normal(t).y > 0.9, "{t:?}");
		}
		// The spawn moves from GoldSrc Z up to this game's Y.
		assert_eq!(map.spawn, vec3(8.0, 16.0, -8.0));
		assert_eq!(map.spawn_yaw, 270.0);
		assert_eq!(map.spawn_pitch, 0.0);
		// One face, one texture, a 2 × 2 lightmap block straight from the file.
		assert_eq!(map.render.faces.len(), 1);
		let f = &map.render.faces[0];
		assert_eq!(f.verts.len(), 4);
		assert_eq!(f.verts[0].uv, (0.0, 0.0));
		assert_eq!(f.verts[2].uv, (2.0, 2.0));
		assert_eq!(f.lightmap.w, 2);
		assert_eq!(f.lightmap.h, 2);
		let lm = &map.render.lightmap;
		let at = |x: u32, y: u32| {
			let i = (((f.lightmap.y + y) * lm.side + f.lightmap.x + x) as usize) * 4;
			&lm.rgba[i..i + 4]
		};
		assert_eq!(at(0, 0), &[10, 20, 30, 255]);
		assert_eq!(at(1, 1), &[100, 110, 120, 255]);
		// The quad's corners sit halfway between the block's sample centers:
		// the first outside them, the third between the inner two.
		assert_eq!(f.verts[0].lm, (-0.5, -0.5));
		assert_eq!(f.verts[2].lm, (0.5, 0.5));
	}

	#[test]
	fn loads_external_dust2_textures_from_project_wads() {
		let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.join("..")
			.join("..")
			.join("..")
			.join("assets")
			.join("maps")
			.join("de_dust2.bsp");
		if !path.is_file() {
			return;
		}
		let data = std::fs::read(&path).expect("read bundled Dust2");
		let bsp = Lumps::new(&data).expect("parse bundled Dust2 header");
		let wads = Wads::new(&bsp, Some(&path));
		let sandroad = wads
			.texture("SandRoad")
			.expect("project WADs should supply Dust2's SandRoad texture");
		let miptex = (0..bsp.miptex_count().expect("miptex count"))
			.find(|&i| bsp.miptex_name(i).ok() == Some("SandRoad"))
			.expect("Dust2 references SandRoad");
		let atlas = TextureAtlas::new(&bsp, &wads).expect("build Dust2 texture atlas");
		let rect = atlas.rect(miptex as i32).expect("SandRoad atlas rectangle");
		let at = ((rect.y * atlas.atlas.side + rect.x) * 4) as usize;
		assert_eq!(&atlas.atlas.rgba[at..at + 4], &sandroad[..4]);
	}

	#[test]
	fn non_bsp30_files_are_rejected() {
		let mut b = tiny_bsp();
		b[0..4].copy_from_slice(&29i32.to_le_bytes());
		assert!(load_bytes(&b, "old").is_err());
		assert!(load_bytes(&[], "empty").is_err());
	}

	#[test]
	fn shelf_packer_aligns_and_wraps() {
		let mut p = Packer::new(24, 8);
		assert_eq!(
			p.push(8, 8).unwrap(),
			Rect {
				x: 0,
				y: 0,
				w: 8,
				h: 8
			}
		);
		// 8 more texels fit to the sheet's edge: 8 + 8 + 8 = 24.
		assert_eq!(
			p.push(8, 8).unwrap(),
			Rect {
				x: 16,
				y: 0,
				w: 8,
				h: 8
			}
		);
		// 16 no longer fit: the next row starts below the tallest, 8-aligned.
		assert_eq!(
			p.push(16, 8).unwrap(),
			Rect {
				x: 0,
				y: 16,
				w: 16,
				h: 8
			}
		);
		assert!(p.push(32, 8).is_none()); // too wide for the sheet
	}
}
