//! Tileable procedural texture sets for the arena's PBR materials: base color,
//! tangent-space normal, and a packed occlusion, roughness, metallic and subsurface
//! map. Pure CPU work; the renderer only uploads the bytes.

use crate::surface::Surface;
use linalg::{vec3, Vec3};

/// Width and height of every texture tile, texels.
pub const SIZE: usize = 256;
/// Tile side including the one-texel duplicated border: `(SIZE, SIZE)` texels
/// plus one wrap-around texel on each side, so linear filtering never crosses
/// tiles. `shaders/pbr.frag` hardcodes these numbers.
pub const PITCH: usize = SIZE + 2;
/// Tiles stacked vertically in an atlas: albedo, normal, ORM.
pub const TILES: usize = 3;

/// One surface's textures, RGBA8, `SIZE` × `SIZE`, row major, row 0 at v = 0.
pub struct Texels {
	/// sRGB base color; alpha unused.
	pub albedo: Vec<u8>,
	/// Tangent-space normal as `n * 0.5 + 0.5`, +x along u and +y along v; alpha unused.
	pub normal: Vec<u8>,
	/// Linear R occlusion, G roughness, B metallic, A subsurface amount.
	pub orm: Vec<u8>,
}

/// Packs this surface's three maps into one atlas the shader samples through
/// the single mesh texture slot: `TILES` tiles stacked vertically, each `PITCH`
/// texels with a one-texel wrap-around border, albedo first, then normal, then
/// ORM. The shader addresses tiles by `fract(uv)`, so the atlas needs no repeat
/// wrapping or mipmaps; mips would average neighboring tiles.
pub fn atlas(t: &Texels) -> Vec<u8> {
	let mut out = vec![0u8; PITCH * PITCH * TILES * 4];
	for tile in 0..TILES {
		let src = [t.albedo.as_slice(), t.normal.as_slice(), t.orm.as_slice()][tile];
		let base = tile * PITCH * PITCH * 4;
		for y in 0..PITCH {
			for x in 0..PITCH {
				// The border texels duplicate the far edge, making each tile wrap.
				let (sx, sy) = ((x + SIZE - 1) % SIZE, (y + SIZE - 1) % SIZE);
				let i = base + (y * PITCH + x) * 4;
				out[i..i + 4].copy_from_slice(&src[(sy * SIZE + sx) * 4..(sy * SIZE + sx) * 4 + 4]);
			}
		}
	}
	out
}

/// One texel before packing. Colors are sRGB, everything else 0 to 1.
struct Texel {
	height: f32,
	albedo: [f32; 3],
	occlusion: f32,
	roughness: f32,
	metallic: f32,
	subsurface: f32,
}

/// Mean linear albedo of `surface`'s base color: what the lightmap's indirect
/// rays reflect off it, one color standing in for the whole tile.
pub fn mean_albedo(surface: Surface) -> Vec3 {
	let t = texels(surface);
	let mut sum = Vec3::ZERO;
	let count = t.albedo.len() / 4;
	for i in 0..count {
		let at = i * 4;
		// sRGB bytes to linear, matching the shader's `to_linear`.
		sum += vec3(
			(t.albedo[at] as f32 / 255.0).powf(2.2),
			(t.albedo[at + 1] as f32 / 255.0).powf(2.2),
			(t.albedo[at + 2] as f32 / 255.0).powf(2.2),
		);
	}
	sum / count as f32
}

/// Builds the textures of `surface`. Every map wraps at its edges.
pub fn texels(surface: Surface) -> Texels {
	let (texel, bump): (fn(f32, f32) -> Texel, f32) = match surface {
		Surface::Tile => (tile, 3.0),
		Surface::Concrete => (concrete, 2.0),
		Surface::Marble => (marble, 1.0),
		Surface::Jade => (jade, 0.5),
		Surface::Steel => (steel, 0.5),
	};
	let at = |i: usize| (i as f32 + 0.5) / SIZE as f32;
	let grid: Vec<Texel> = (0..SIZE * SIZE)
		.map(|i| texel(at(i % SIZE), at(i / SIZE)))
		.collect();
	let height = |x: usize, y: usize| grid[(y % SIZE) * SIZE + x % SIZE].height;
	let byte = |f: f32| (f.clamp(0.0, 1.0) * 255.0).round() as u8;
	let mut out = Texels {
		albedo: Vec::with_capacity(SIZE * SIZE * 4),
		normal: Vec::with_capacity(SIZE * SIZE * 4),
		orm: Vec::with_capacity(SIZE * SIZE * 4),
	};
	for (i, t) in grid.iter().enumerate() {
		let (x, y) = (i % SIZE + SIZE, i / SIZE + SIZE);
		// Central differences in texels, wrapping; `bump` is the depth of a unit height.
		let du = (height(x + 1, y) - height(x - 1, y)) * 0.5 * bump;
		let dv = (height(x, y + 1) - height(x, y - 1)) * 0.5 * bump;
		let len = (du * du + dv * dv + 1.0).sqrt();
		let n = [-du / len, -dv / len, 1.0 / len];
		out
			.albedo
			.extend(t.albedo.map(byte).into_iter().chain([255]));
		out
			.normal
			.extend(n.map(|c| byte(c * 0.5 + 0.5)).into_iter().chain([255]));
		out
			.orm
			.extend([t.occlusion, t.roughness, t.metallic, t.subsurface].map(byte));
	}
	out
}

/// Floor: 2 × 2 flat glazed tiles in two tones, a crisp grout seam and a
/// narrow bevel, in the clean-panel style of Valorant and CS maps.
fn tile(u: f32, v: f32) -> Texel {
	let (tu, tv) = ((u * 2.0).fract(), (v * 2.0).fract());
	let edge = tu.min(1.0 - tu).min(tv.min(1.0 - tv));
	let grout = 1.0 - smoothstep(0.008, 0.02, edge);
	let bevel = (1.0 - smoothstep(0.02, 0.06, edge)) * (1.0 - grout);
	let base = if ((u * 2.0) as i32 + (v * 2.0) as i32) & 1 == 0 {
		[0.82, 0.79, 0.72]
	} else {
		[0.72, 0.70, 0.64]
	};
	Texel {
		height: 0.9 * bevel + 0.8 * grout,
		albedo: mix3(base, [0.42, 0.41, 0.38], grout),
		occlusion: 1.0 - 0.25 * grout,
		roughness: mix(0.35, 0.85, grout),
		metallic: 0.0,
		subsurface: 0.0,
	}
}

/// Walls: flat cast panels separated by recessed seams.
fn concrete(u: f32, v: f32) -> Texel {
	let (pu, pv) = ((u * 2.0).fract(), (v * 2.0).fract());
	let edge = pu.min(1.0 - pu).min(pv.min(1.0 - pv));
	let seam = 1.0 - smoothstep(0.01, 0.025, edge);
	let bevel = (1.0 - smoothstep(0.025, 0.07, edge)) * (1.0 - seam);
	Texel {
		height: 0.6 * bevel + 0.5 * seam,
		albedo: mix3([0.60, 0.62, 0.66], [0.40, 0.41, 0.43], seam),
		occlusion: 1.0 - 0.3 * seam,
		roughness: 0.85,
		metallic: 0.0,
		subsurface: 0.0,
	}
}

/// Stairs and ramps: flat white panels, translucent, with thin grey veins.
fn marble(u: f32, v: f32) -> Texel {
	let w = fbm(u, v, 4, 5, 4);
	// Integer frequencies in u and v keep the veins tileable; the high power
	// keeps them thin.
	let vein = smoothstep(
		0.75,
		0.95,
		(1.0
			- (std::f32::consts::TAU * (u + 2.0 * v + 1.5 * w))
				.sin()
				.abs())
		.powi(16),
	);
	Texel {
		height: 0.5 - 0.1 * vein,
		albedo: mix3([0.94, 0.92, 0.88], [0.55, 0.54, 0.55], vein),
		occlusion: 1.0,
		roughness: mix(0.15, 0.35, vein),
		metallic: 0.0,
		subsurface: 0.7 * (1.0 - vein),
	}
}

/// Crates and the surf wedge: flat green jade, strongly translucent.
fn jade(u: f32, v: f32) -> Texel {
	let c = fbm(u, v, 4, 3, 5);
	Texel {
		height: 0.05 * c,
		albedo: mix3([0.16, 0.46, 0.27], [0.28, 0.58, 0.36], c),
		occlusion: 1.0,
		roughness: 0.18,
		metallic: 0.0,
		subsurface: 0.95,
	}
}

/// Duck tunnel: flat steel panels with a light brushed grain along u.
fn steel(u: f32, v: f32) -> Texel {
	let streak = noise(u, v, 4, 256, 7);
	let (pu, pv) = ((u * 2.0).fract(), (v * 2.0).fract());
	let edge = pu.min(1.0 - pu).min(pv.min(1.0 - pv));
	let seam = 1.0 - smoothstep(0.008, 0.02, edge);
	Texel {
		height: 0.08 * streak,
		albedo: mix3([0.74, 0.75, 0.77], [0.45, 0.46, 0.47], seam),
		occlusion: 1.0 - 0.2 * seam,
		roughness: mix(0.25, 0.6, seam),
		metallic: 1.0,
		subsurface: 0.0,
	}
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
	a + (b - a) * t
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
	[mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
	let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
	t * t * (3.0 - 2.0 * t)
}

/// Lattice value in 0 to 1.
fn hash(x: i32, y: i32, seed: u32) -> f32 {
	let mut h = (x as u32).wrapping_mul(0x8da6_b343)
		^ (y as u32).wrapping_mul(0xd816_3841)
		^ seed.wrapping_mul(0xcb1a_b31f);
	h ^= h >> 13;
	h = h.wrapping_mul(0x5bd1_e995);
	h ^= h >> 15;
	(h & 0xff_ffff) as f32 / 0xff_ffff as f32
}

/// Smooth value noise over the unit square with `cx` × `cy` lattice cells, wrapping.
fn noise(u: f32, v: f32, cx: i32, cy: i32, seed: u32) -> f32 {
	let (x, y) = (u * cx as f32, v * cy as f32);
	let (x0, y0) = (x.floor(), y.floor());
	let s = |t: f32| t * t * (3.0 - 2.0 * t);
	let (sx, sy) = (s(x - x0), s(y - y0));
	let at = |i: i32, j: i32| {
		hash(
			(x0 as i32 + i).rem_euclid(cx),
			(y0 as i32 + j).rem_euclid(cy),
			seed,
		)
	};
	mix(mix(at(0, 0), at(1, 0), sx), mix(at(0, 1), at(1, 1), sx), sy)
}

/// `octaves` of [`noise`] from `cells` per side, each twice as fine and half as strong.
fn fbm(u: f32, v: f32, cells: i32, octaves: u32, seed: u32) -> f32 {
	let (mut sum, mut total, mut amp) = (0.0, 0.0, 1.0);
	for o in 0..octaves {
		let c = cells << o;
		sum += amp * noise(u, v, c, c, seed * 16 + o);
		total += amp;
		amp *= 0.5;
	}
	sum / total
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn noise_wraps_at_the_edges() {
		for k in 0..32 {
			let t = k as f32 / 32.0;
			assert!((noise(0.0, t, 8, 8, 1) - noise(1.0, t, 8, 8, 1)).abs() < 1e-5);
			assert!((fbm(t, 0.0, 4, 5, 2) - fbm(t, 1.0, 4, 5, 2)).abs() < 1e-5);
		}
	}

	#[test]
	fn every_surface_fills_its_maps() {
		for s in Surface::ALL {
			let t = texels(s);
			for map in [&t.albedo, &t.normal, &t.orm] {
				assert_eq!(map.len(), SIZE * SIZE * 4, "{s:?}");
			}
			// Normals face out of the surface.
			assert!(t.normal.chunks(4).all(|n| n[2] >= 128), "{s:?}");
		}
		assert!(texels(Surface::Jade).orm.chunks(4).all(|p| p[3] > 100));
		assert!(texels(Surface::Steel)
			.orm
			.chunks(4)
			.all(|p| p[2] == 255 && p[3] == 0));
	}
}
