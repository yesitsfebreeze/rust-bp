//! The bake on the GPU: the same light model, charts and cache as the CPU
//! bake, with the texel work in two wgpu compute passes.
//!
//! The CPU keeps everything deterministic and cheap — chart packing, the
//! BVH build, occluder sets, the cache format — and the GPU computes texel
//! values: direct pass (sun, point lights, shadow tables) then one gather
//! pass per configured bounce — the first also bringing the sky its rays
//! escape to — each adding to the atlas only what it gathers from the
//! light the pass before it added, ping-ponged between two compact
//! buffers with the same manual bilinear chart sampling the CPU gathers
//! with. The gathered light is smoothed on the CPU after, as the CPU bake
//! smooths it.
//!
//! Sun, light and shadow values are the same arithmetic as the CPU
//! path, to rounding. Only the gather rays differ: the CPU walks one seeded
//! sequence across all texels, the GPU hashes a seed per texel and ray, so
//! sky and bounced light agree within Monte-Carlo noise, not bitwise.
//!
//! The atlas and worklist buffers are sized for the configured atlas, so a
//! GPU with too small a storage binding reports it instead of baking a
//! partial map.

use crate::{bvh, ChartSpec, Config, Triangle};
use crate::{packed, ray_span, tri_chart_ranges};
use linalg::Vec3;

fn push3(out: &mut Vec<u8>, v: Vec3) {
	for c in [v.x, v.y, v.z] {
		out.extend_from_slice(&c.to_bits().to_le_bytes());
	}
}

fn push4(out: &mut Vec<u8>, v: Vec3, w: f32) {
	push3(out, v);
	out.extend_from_slice(&w.to_bits().to_le_bytes());
}

/// WGSL shared by both passes: bindings, geometry records and the ray math
/// ported from `bvh.rs` and the CPU bake.
const COMMON: &str = r#"
const PI: f32 = 3.14159265;
const OFFSET: f32 = 0.1;
// A sixteenth of a texel, as the CPU bake's `EDGE_INSET`.
const EDGE_INSET: f32 = 1.0 / 16.0;
const NO_SHADOW: u32 = 0xFFFFFFFFu;

struct Cfg {
	counts: vec4<u32>,  // atlas side, packed charts, texels, lights
	floats: vec4<f32>,  // units per texel, ray span, seed, rays
	sun_dir: vec4<f32>,  // w 1 when gather hits carry light on, 0 with no bounces
	sun_color: vec4<f32>,
	sky: vec4<f32>,
	ground: vec4<f32>,
};
struct Tri {
	a: vec4<f32>,  // xyz, w bitcasts the packed chart index
	b: vec4<f32>,
	c: vec4<f32>,
	alb: vec4<f32>,
};
struct Node { lo: vec3<f32>, first: u32, hi: vec3<f32>, count: u32 }
struct Chart {
	rx: u32, ry: u32, rw: u32, rh: u32,
	origin: vec4<f32>,
	u: vec4<f32>,
	v: vec4<f32>,
	n: vec4<f32>,
	first: u32, count: u32, off: u32, pad: u32,
};
struct Light {
	pos: vec3<f32>, fade: f32,
	color: vec3<f32>, cone: f32,
	dir: vec3<f32>, deg: f32,
};
struct RayHit { tri: u32, at: f32, hit: u32 }

@group(0) @binding(0) var<storage, read> tris: array<Tri>;
@group(0) @binding(1) var<storage, read> bvh: array<Node>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read> charts: array<Chart>;
@group(0) @binding(4) var<storage, read> lights: array<Light>;
@group(0) @binding(5) var<storage, read_write> atlas: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> cfg: Cfg;

fn enters(o: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>,
	tmin: f32, tmax: f32) -> bool {
	var tt = vec2<f32>(tmin, tmax);
	var a = (lo.x - o.x) * inv.x;
	var b = (hi.x - o.x) * inv.x;
	tt = vec2(max(tt.x, min(a, b)), min(tt.y, max(a, b)));
	a = (lo.y - o.y) * inv.y;
	b = (hi.y - o.y) * inv.y;
	tt = vec2(max(tt.x, min(a, b)), min(tt.y, max(a, b)));
	a = (lo.z - o.z) * inv.z;
	b = (hi.z - o.z) * inv.z;
	tt = vec2(max(tt.x, min(a, b)), min(tt.y, max(a, b)));
	return tt.x <= tt.y;
}

fn hit_tri(o: vec3<f32>, d: vec3<f32>, i: u32) -> f32 {
	let t = tris[i];
	let e1 = t.b.xyz - t.a.xyz;
	let e2 = t.c.xyz - t.a.xyz;
	let p = cross(d, e2);
	let det = dot(e1, p);
	if abs(det) < 1e-9 {
		return -1.0;
	}
	let inv = 1.0 / det;
	let s = o - t.a.xyz;
	let u = dot(s, p) * inv;
	if u < 0.0 || u > 1.0 {
		return -1.0;
	}
	let q = cross(s, e1);
	let v = dot(d, q) * inv;
	if v < 0.0 || u + v > 1.0 {
		return -1.0;
	}
	let at = dot(e2, q) * inv;
	return select(-1.0, at, at > 1e-4);
}

fn ray_cast(o: vec3<f32>, d: vec3<f32>, max_t: f32) -> RayHit {
	var inv = vec3<f32>(1e30, 1e30, 1e30);
	inv = select(inv, vec3<f32>(1.0) / d, abs(d) >= vec3<f32>(1e-12));
	var stack: array<u32, 64>;
	stack[0] = 0u;
	var top = 1u;
	var best = max_t;
	var found = 0u;
	var hit_idx = 0xFFFFFFFFu;
	loop {
		if top == 0u {
			break;
		}
		top = top - 1u;
		let at = stack[top];
		let node = bvh[at];
		if !enters(o, inv, node.lo, node.hi, 0.0, best) {
			continue;
		}
		if node.count > 0u {
			var k = node.first;
			loop {
				if k >= node.first + node.count {
					break;
				}
				let i = order[k];
				let t = hit_tri(o, d, i);
				if t >= 0.0 && t < best {
					best = t;
					hit_idx = i;
					found = 1u;
				}
				k = k + 1u;
			}
		} else {
			if top + 2u <= 64u {
				stack[top] = node.first;
				stack[top + 1u] = at + 1u;
				top = top + 2u;
			}
		}
	}
	return RayHit(hit_idx, best, found);
}

fn closest_point(point: vec3<f32>, i: u32) -> vec3<f32> {
	let t = tris[i];
	let a = t.a.xyz;
	let ab = t.b.xyz - a;
	let ac = t.c.xyz - a;
	let ap = point - a;
	let d1 = dot(ab, ap);
	let d2 = dot(ac, ap);
	if d1 <= 0.0 && d2 <= 0.0 {
		return a;
	}
	let bp = point - t.b.xyz;
	let d3 = dot(ab, bp);
	let d4 = dot(ac, bp);
	if d3 >= 0.0 && d4 <= d3 {
		return t.b.xyz;
	}
	let vc = d1 * d4 - d3 * d2;
	if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
		return a + ab * (d1 / (d1 - d3));
	}
	let cp = point - t.c.xyz;
	let d5 = dot(ab, cp);
	let d6 = dot(ac, cp);
	if d6 >= 0.0 && d5 <= d6 {
		return t.c.xyz;
	}
	let vb = d5 * d2 - d1 * d6;
	if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
		return a + ac * (d2 / (d2 - d6));
	}
	let va = d3 * d6 - d5 * d4;
	if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
		return t.b.xyz + (t.c.xyz - t.b.xyz) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
	}
	let denom = 1.0 / (va + vb + vc);
	return a + ab * (vb * denom) + ac * (vc * denom);
}

fn chart_of(i: u32) -> u32 {
	return bitcast<u32>(tris[i].a.w);
}

fn albedo_of(i: u32) -> vec3<f32> {
	return tris[i].alb.xyz;
}

// The nearest point to `p` on the chart's own triangles, so texels the
// chart's rectangle places outside the surface clamp onto it.
fn face_point(p: vec3<f32>, first: u32, count: u32) -> vec3<f32> {
	var best = p;
	var best_d = 1e30;
	var centroid = p;
	var k = first;
	loop {
		if k >= first + count {
			break;
		}
		let q = closest_point(p, k);
		let d = dot(q - p, q - p);
		if d < best_d {
			best_d = d;
			best = q;
			centroid = (tris[k].a.xyz + tris[k].b.xyz + tris[k].c.xyz) / 3.0;
		}
		k = k + 1u;
	}
	// A point clamped onto the face's edge moves `EDGE_INSET` of a texel
	// on into its
	// triangle, as the CPU bake's `inset`: on the edge it lies in the
	// plane of the face meeting it there, and rays along that face slip
	// through it.
	let to = centroid - best;
	let len = length(to);
	if best_d < 1e-6 || len < 1e-6 {
		return best;
	}
	return best + to * (min(EDGE_INSET * cfg.floats.x, len * 0.5) / len);
}

fn pcg(v: u32) -> u32 {
	let s = v * 747796405u + 2891336453u;
	let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (w >> 22u) ^ w;
}

fn rand01(seed: u32, id: u32, r: u32, k: u32) -> f32 {
	return f32(pcg(seed ^ pcg(id * 2654435761u + r * 40503u + k)) & 0xFFFFFFu)
		/ 16777216.0;
}

fn cosine(n: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
	let phi = 2.0 * PI * r1;
	let cos = sqrt(1.0 - r2);
	let sin = sqrt(r2);
	let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
		abs(n.y) < 0.9);
	let t = normalize(a - n * dot(a, n));
	let b = cross(n, t);
	return t * (cos(phi) * sin) + n * cos + b * (sin(phi) * sin);
}

fn chart_of_texel(id: u32) -> u32 {
	var lo = 0u;
	var hi = cfg.counts.y;
	while hi - lo > 1u {
		let mid = (lo + hi) / 2u;
		if charts[mid].off <= id {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return lo;
}
"#;

/// The direct pass: sun and lights per texel — the sky too when no gather
/// ray will bring it — plus the sun shadow table, and the same light into
/// the compact buffer the first gather samples. One workgroup item per
/// chart texel.
const DIRECT: &str = r#"
@group(0) @binding(6) var<storage, read_write> shadow: array<u32>;
@group(0) @binding(8) var<storage, read_write> first: array<vec4<f32>>;

@compute @workgroup_size(64)
fn direct(@builtin(global_invocation_id) gid: vec3<u32>) {
	// The dispatch folds past 65535 workgroups into grid Y.
	let id = gid.x + gid.y * 65535u;
	if id >= cfg.counts.z {
		return;
	}
	let c = charts[chart_of_texel(id)];
	let local = id - c.off;
	let i = local % c.rw;
	let j = local / c.rw;
	let n = c.n.xyz;
	let world = c.origin.xyz + c.u.xyz * f32(i) + c.v.xyz * f32(j);
	let p = face_point(world, c.first, c.count);

	var lum = vec3<f32>(0.0);
	if cfg.floats.w <= 0.0 {
		lum = mix(cfg.ground.xyz, cfg.sky.xyz, n.y * 0.5 + 0.5);
	}
	var vis = 1.0;
	var blocker = NO_SHADOW;
	let n_l = max(dot(n, cfg.sun_dir.xyz), 0.0);
	if n_l > 0.0 {
		let hit = ray_cast(p + n * OFFSET, cfg.sun_dir.xyz, cfg.floats.y);
		if hit.hit == 1u {
			vis = 0.0;
			blocker = hit.tri;
		} else {
			lum += cfg.sun_color.xyz * (n_l / PI);
		}
	}
	var k = 0u;
	loop {
		if k >= cfg.counts.w {
			break;
		}
		let light = lights[k];
		let to = light.pos - p;
		let d = length(to);
		if d >= light.fade || d < 1e-4 {
			k = k + 1u;
			continue;
		}
		let l = to / d;
		let nl = max(dot(n, l), 0.0);
		if nl == 0.0 {
			k = k + 1u;
			continue;
		}
		if light.cone > 0.5 {
			if dot(-l, light.dir) < cos(light.deg * PI / 180.0) {
				k = k + 1u;
				continue;
			}
		}
		let hit = ray_cast(p + n * OFFSET, l, max(d - OFFSET, 0.0));
		if hit.hit == 1u {
			k = k + 1u;
			continue;
		}
		let fade = 1.0 - d / light.fade;
		lum += light.color * (nl * fade * fade / PI);
		k = k + 1u;
	}
	atlas[(c.ry + j) * cfg.counts.x + (c.rx + i)] = vec4<f32>(lum, vis);
	first[id] = vec4<f32>(lum, vis);
	shadow[id] = blocker;
}
"#;

/// One gather pass: per texel, the light cosine-distributed rays bring
/// back — the albedo times what the pass before added where they hit, and
/// in the first pass the sky where they escape — added to the atlas and
/// written compact for the next pass to gather from.
const BOUNCE: &str = r#"
@group(0) @binding(8) var<storage, read_write> prev: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> next: array<vec4<f32>>;

fn prev_at(c: Chart, x: f32, y: f32) -> vec4<f32> {
	return prev[c.off + u32(y) * c.rw + u32(x)];
}

// What the pass before added, bilinear at world point `p` of chart `c`.
fn chart_sample(c: Chart, p: vec3<f32>) -> vec3<f32> {
	// The dual basis mapping world points to texel coordinates, the
	// inverse of the spec's `u` and `v` steps.
	let d = dot(cross(c.u.xyz, c.v.xyz), c.n.xyz);
	let g_u = cross(c.v.xyz, c.n.xyz) / d;
	let g_v = cross(c.n.xyz, c.u.xyz) / d;
	let rel = p - c.origin.xyz;
	let x = clamp(dot(g_u, rel), 0.0, f32(c.rw) - 1.0);
	let y = clamp(dot(g_v, rel), 0.0, f32(c.rh) - 1.0);
	let x0 = floor(x);
	let y0 = floor(y);
	let tx = x - x0;
	let ty = y - y0;
	let x1 = min(x0 + 1.0, f32(c.rw) - 1.0);
	let y1 = min(y0 + 1.0, f32(c.rh) - 1.0);
	let r0 = mix(prev_at(c, x0, y0), prev_at(c, x1, y0), vec4<f32>(tx));
	let r1 = mix(prev_at(c, x0, y1), prev_at(c, x1, y1), vec4<f32>(tx));
	return mix(r0, r1, vec4<f32>(ty)).xyz;
}

fn gather(gid: vec3<u32>, sky: bool) {
	// The dispatch folds past 65535 workgroups into grid Y.
	let id = gid.x + gid.y * 65535u;
	if id >= cfg.counts.z {
		return;
	}
	let c = charts[chart_of_texel(id)];
	let local = id - c.off;
	let i = local % c.rw;
	let j = local / c.rw;
	let at = (c.ry + j) * cfg.counts.x + (c.rx + i);
	let seed = bitcast<u32>(cfg.floats.z);
	let n = c.n.xyz;
	let world = c.origin.xyz + c.u.xyz * f32(i) + c.v.xyz * f32(j);
	let p = face_point(world, c.first, c.count);
	// The first pass and the later ones draw different rays.
	let k = select(2u, 0u, sky);
	var sum = vec3<f32>(0.0);
	var r = 0u;
	loop {
		if f32(r) >= cfg.floats.w {
			break;
		}
		let d = cosine(n, rand01(seed, id, r, k), rand01(seed, id, r, k + 1u));
		let o = p + n * OFFSET;
		let hit = ray_cast(o, d, cfg.floats.y);
		if hit.hit == 1u {
			let q = o + d * hit.at;
			let chart = charts[chart_of(hit.tri)];
			sum += albedo_of(hit.tri) * chart_sample(chart, q) * cfg.sun_dir.w;
		} else if sky {
			sum += mix(cfg.ground.xyz, cfg.sky.xyz, d.y * 0.5 + 0.5);
		}
		r = r + 1u;
	}
	let mean = sum / cfg.floats.w;
	next[id] = vec4<f32>(mean, 0.0);
	atlas[at] = atlas[at] + vec4<f32>(mean, 0.0);
}

@compute @workgroup_size(64)
fn first_bounce(@builtin(global_invocation_id) gid: vec3<u32>) {
	gather(gid, true);
}

@compute @workgroup_size(64)
fn bounce(@builtin(global_invocation_id) gid: vec3<u32>) {
	gather(gid, false);
}
"#;

/// Bakes `tris` on the GPU: the same charts, atlas and cache as `bake`,
/// with gather rays hashed per texel instead of one shared sequence.
///
/// The CPU still packs the charts, builds the BVH and, after the passes,
/// fills the shadow tables, the occluder sets (a swept-box
/// over-approximation of what the light rays can hit) and the gutters —
/// the same `commit` the CPU bake writes through, so a GPU-baked cache
/// loads everywhere a CPU one does.
pub fn bake_gpu(
	tris: &[Triangle],
	albedo: &[Vec3],
	charts: &[ChartSpec],
	tri_chart: &[u32],
	config: &Config,
) -> Result<crate::Baked, String> {
	if tris.len() != albedo.len() {
		return Err(format!(
			"{} triangles but {} albedos",
			tris.len(),
			albedo.len()
		));
	}
	if tris.len() != tri_chart.len() {
		return Err(format!(
			"{} triangles but {} chart indices",
			tris.len(),
			tri_chart.len()
		));
	}
	if config.atlas_size.count_ones() != 1 {
		return Err(format!(
			"atlas_size {} is not a power of two",
			config.atlas_size
		));
	}
	let (mut baked, _) = packed(tris, charts, tri_chart, config)?;
	let side = config.atlas_size;
	let chart_tris = tri_chart_ranges(tri_chart, charts.len())?;

	// The packed chart records, in packed order so their compact texel
	// offsets ascend for the shader's binary search, plus each triangle's
	// packed index, u32::MAX when its chart holds none. The WGSL struct is
	// four u32s, then four vec4s, then four u32s: 96 bytes.
	let mut chart_bytes = Vec::new();
	let mut chart_of_tri = vec![u32::MAX; tris.len()];
	let mut packed_charts = Vec::new();
	let mut offset = 0u32;
	for (ci, chart) in baked.charts.iter().enumerate() {
		let Some(c) = chart else {
			continue;
		};
		let (first, count) = chart_tris[ci];
		for t in first..first + count {
			chart_of_tri[t as usize] = packed_charts.len() as u32;
		}
		packed_charts.push(ci);
		for v in [c.rect.x, c.rect.y, c.rect.w, c.rect.h] {
			chart_bytes.extend_from_slice(&v.to_le_bytes());
		}
		push4(&mut chart_bytes, c.spec.origin, 0.0);
		push4(&mut chart_bytes, c.spec.u, 0.0);
		push4(&mut chart_bytes, c.spec.v, 0.0);
		push4(&mut chart_bytes, c.spec.n, 0.0);
		for v in [first, count, offset, 0] {
			chart_bytes.extend_from_slice(&v.to_le_bytes());
		}
		offset += c.rect.w * c.rect.h;
	}
	let texels = offset;
	if texels == 0 {
		return Ok(baked);
	}

	let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
	let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
		power_preference: wgpu::PowerPreference::HighPerformance,
		compatible_surface: None,
		force_fallback_adapter: false,
	}))
	.map_err(|e| format!("no GPU adapter for the bake: {e}"))?;
	let atlas_bytes = side as u64 * side as u64 * 16;
	let limits = adapter.limits();
	if atlas_bytes > limits.max_buffer_size
		|| atlas_bytes > limits.max_storage_buffer_binding_size as u64
	{
		return Err(format!(
			"the {side}x{side} atlas needs a {atlas_bytes}-byte storage buffer; this adapter tops out at {}",
			limits.max_storage_buffer_binding_size
		));
	}
	let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
		label: Some("lightbake"),
		required_features: wgpu::Features::empty(),
		required_limits: limits,
		..Default::default()
	}))
	.map_err(|e| format!("no GPU device for the bake: {e}"))?;

	// Each triangle: three vec4s of position, the first carrying the
	// packed chart index in its w, then the albedo.
	let mut tri_bytes = Vec::with_capacity(tris.len() * 64);
	for (i, t) in tris.iter().enumerate() {
		push4(&mut tri_bytes, t[0], f32::from_bits(chart_of_tri[i]));
		push4(&mut tri_bytes, t[1], 0.0);
		push4(&mut tri_bytes, t[2], 0.0);
		push4(&mut tri_bytes, albedo[i], 0.0);
	}
	let mut light_bytes = Vec::with_capacity(config.lights.len() * 48);
	for light in &config.lights {
		push4(&mut light_bytes, light.pos, light.fade);
		push4(
			&mut light_bytes,
			light.color,
			light.cone.is_some() as u32 as f32,
		);
		let (dir, deg) = light.cone.unwrap_or((Vec3::Y, 0.0));
		push4(&mut light_bytes, dir, deg);
	}
	// A bind group cannot bind a zero-sized buffer; a lightless scene keeps
	// one dummy record the entries never dereference.
	if light_bytes.is_empty() {
		light_bytes.resize(48, 0);
	}
	let mut cfg_bytes = Vec::with_capacity(96);
	for v in [
		side,
		packed_charts.len() as u32,
		texels,
		config.lights.len() as u32,
	] {
		cfg_bytes.extend_from_slice(&v.to_le_bytes());
	}
	for v in [
		config.units_per_texel,
		ray_span(tris),
		f32::from_bits(config.seed),
		config.rays as f32,
	] {
		cfg_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
	}
	push4(
		&mut cfg_bytes,
		config.sun_dir,
		(config.bounces > 0) as u32 as f32,
	);
	push4(&mut cfg_bytes, config.sun_color, 0.0);
	push4(&mut cfg_bytes, config.sky_color, 0.0);
	push4(&mut cfg_bytes, config.ground_color, 0.0);

	let bvh = bvh::Bvh::build(tris);
	let (bvh_bytes, order) = bvh.gpu();
	let mut order_bytes = Vec::with_capacity(order.len() * 4);
	for i in order {
		order_bytes.extend_from_slice(&i.to_le_bytes());
	}

	let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
		device.create_buffer(&wgpu::BufferDescriptor {
			label: Some(label),
			size,
			usage: wgpu::BufferUsages::STORAGE | extra,
			mapped_at_creation: false,
		})
	};
	let upload = wgpu::BufferUsages::COPY_DST;
	let tris_buf = storage("tris", tri_bytes.len() as u64, upload);
	let bvh_buf = storage("bvh", bvh_bytes.len() as u64, upload);
	let order_buf = storage("order", order_bytes.len() as u64, upload);
	let charts_buf = storage("charts", chart_bytes.len() as u64, upload);
	let lights_buf = storage("lights", light_bytes.len() as u64, upload);
	let copy = wgpu::BufferUsages::COPY_SRC;
	let atlas = storage("atlas", atlas_bytes, copy);
	// What each pass added, compact in texel order, ping-ponged: the one a
	// pass gathers from and the one it writes.
	let added = [
		storage("added a", texels as u64 * 16, copy),
		storage("added b", texels as u64 * 16, copy),
	];
	let shadow_buf = storage("shadow", texels as u64 * 4, copy);
	let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("cfg"),
		size: cfg_bytes.len() as u64,
		usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});
	// The passes write the atlas and the shadow table; everything else is
	// ours to upload.
	for (buf, bytes) in [
		(&tris_buf, &tri_bytes),
		(&bvh_buf, &bvh_bytes),
		(&order_buf, &order_bytes),
		(&charts_buf, &chart_bytes),
		(&lights_buf, &light_bytes),
		(&cfg_buf, &cfg_bytes),
	] {
		queue.write_buffer(buf, 0, bytes);
	}

	let buffer = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
		binding,
		visibility: wgpu::ShaderStages::COMPUTE,
		ty: wgpu::BindingType::Buffer {
			ty,
			has_dynamic_offset: false,
			min_binding_size: None,
		},
		count: None,
	};
	let read = |b: u32| buffer(b, wgpu::BufferBindingType::Storage { read_only: true });
	let write = |b: u32| buffer(b, wgpu::BufferBindingType::Storage { read_only: false });
	let uniform = |b: u32| buffer(b, wgpu::BufferBindingType::Uniform);
	// Every pass binds the same ten slots: five read-only geometry and
	// light sources, the atlas at 5, the shadow table at 6, the config
	// uniform at 7, and the two compact buffers at 8 and 9 — the direct
	// pass writes 8, each gather pass reads 8 and writes 9.
	let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
		label: None,
		entries: &[
			read(0),
			read(1),
			read(2),
			read(3),
			read(4),
			write(5),
			write(6),
			uniform(7),
			write(8),
			write(9),
		],
	});
	let bind = |bgl: &wgpu::BindGroupLayout, bufs: [&wgpu::Buffer; 10]| {
		let entries: Vec<wgpu::BindGroupEntry> = bufs
			.iter()
			.enumerate()
			.map(|(binding, buf)| wgpu::BindGroupEntry {
				binding: binding as u32,
				resource: buf.as_entire_binding(),
			})
			.collect();
		device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: bgl,
			entries: &entries,
		})
	};
	let binds = |from: usize| {
		bind(
			&layout,
			[
				&tris_buf,
				&bvh_buf,
				&order_buf,
				&charts_buf,
				&lights_buf,
				&atlas,
				&shadow_buf,
				&cfg_buf,
				&added[from],
				&added[1 - from],
			],
		)
	};
	let pass_binds = [binds(0), binds(1)];

	let module = |extra: &str| {
		device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("lightbake"),
			source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{extra}").into()),
		})
	};
	let pipeline = |entry: &str, layout: &wgpu::BindGroupLayout, extra: &str| {
		device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: Some("lightbake"),
			layout: Some(
				&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
					label: None,
					bind_group_layouts: &[layout],
					push_constant_ranges: &[],
				}),
			),
			module: &module(extra),
			entry_point: Some(entry),
			compilation_options: Default::default(),
			cache: None,
		})
	};
	let direct = pipeline("direct", &layout, DIRECT);
	let first_bounce = pipeline("first_bounce", &layout, BOUNCE);
	let bounce = pipeline("bounce", &layout, BOUNCE);

	let groups = texels.div_ceil(64);
	// Past 65535 workgroups the dispatch continues along grid Y, which the
	// entries fold back into their texel id.
	let (gx, gy) = (groups.min(65535), groups.div_ceil(65535).max(1));
	let mut encoder = device.create_command_encoder(&Default::default());
	{
		let mut pass = encoder.begin_compute_pass(&Default::default());
		pass.set_pipeline(&direct);
		pass.set_bind_group(0, &pass_binds[0], &[]);
		pass.dispatch_workgroups(gx, gy, 1);
	}
	let staging = |size: u64| {
		device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("lightbake readback"),
			size,
			usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		})
	};
	// The direct light, before the first gather pass writes over its
	// buffer, for the smoothing to tell the gathered light from it.
	let direct_out = staging(texels as u64 * 16);
	encoder.copy_buffer_to_buffer(&added[0], 0, &direct_out, 0, texels as u64 * 16);
	// Zero bounces still runs the first gather, for the sky; zero rays
	// leaves the direct pass's open hemisphere.
	let passes = if config.rays > 0 {
		config.bounces.max(1)
	} else {
		0
	};
	for k in 0..passes {
		let mut pass = encoder.begin_compute_pass(&Default::default());
		pass.set_pipeline(if k == 0 { &first_bounce } else { &bounce });
		pass.set_bind_group(0, &pass_binds[k % 2], &[]);
		pass.dispatch_workgroups(gx, gy, 1);
	}
	let atlas_out = staging(atlas_bytes);
	let shadow_out = staging(texels as u64 * 4);
	encoder.copy_buffer_to_buffer(&atlas, 0, &atlas_out, 0, atlas_bytes);
	encoder.copy_buffer_to_buffer(&shadow_buf, 0, &shadow_out, 0, texels as u64 * 4);
	queue.submit(Some(encoder.finish()));
	read_back(&device, &atlas_out)?;
	let atlas_read = atlas_out.slice(..).get_mapped_range().to_vec();
	atlas_out.unmap();
	read_back(&device, &direct_out)?;
	let direct_read = direct_out.slice(..).get_mapped_range().to_vec();
	direct_out.unmap();
	read_back(&device, &shadow_out)?;
	let shadow_read = shadow_out.slice(..).get_mapped_range().to_vec();
	shadow_out.unmap();
	let shadow_u32: Vec<u32> = shadow_read
		.as_chunks::<4>()
		.0
		.iter()
		.map(|c| u32::from_le_bytes(*c))
		.collect();
	let f32_at = |bytes: &[u8], at: usize| {
		f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
	};

	// The compact arrays are in packed order, their offsets ascending.
	let mut off = 0u32;
	for &ci in &packed_charts {
		let (values, w, h, lo, hi) = {
			let c = baked.charts[ci].as_ref().unwrap();
			let mut values = vec![[0.0f32; 4]; (c.rect.w * c.rect.h) as usize];
			for j in 0..c.rect.h {
				for i in 0..c.rect.w {
					let at = (((c.rect.y + j) * side + c.rect.x + i) * 16) as usize;
					values[(j * c.rect.w + i) as usize] = [
						f32_at(&atlas_read, at),
						f32_at(&atlas_read, at + 4),
						f32_at(&atlas_read, at + 8),
						f32_at(&atlas_read, at + 12),
					];
				}
			}
			let (lo, hi) = c.bounds();
			(values, c.rect.w, c.rect.h, lo, hi)
		};
		let shadow = shadow_u32[off as usize..(off + w * h) as usize].to_vec();
		off += w * h;
		// Every triangle that can block this chart's light rays: those
		// whose bounds meet the box swept from the chart to each light, a
		// superset of what the rays actually hit, so an edit never misses
		// a blocker.
		let mut occluders = Vec::new();
		for light in &config.lights {
			let q = lo.max(light.pos.min(hi));
			if (q - light.pos).length2() > light.fade * light.fade {
				continue;
			}
			occluders.extend(bvh.in_box(tris, lo.min(light.pos), hi.max(light.pos)));
		}
		occluders.sort_unstable();
		occluders.dedup();
		baked.commit(ci, &values, &shadow, &occluders);
	}
	if passes > 0 {
		// Each chart's direct values, compact in the same packed order.
		let mut direct = vec![None; baked.charts.len()];
		let mut off = 0usize;
		for &ci in &packed_charts {
			let rect = baked.charts[ci].as_ref().unwrap().rect;
			let n = (rect.w * rect.h) as usize;
			direct[ci] = Some(
				(off..off + n)
					.map(|t| {
						let at = t * 16;
						[
							f32_at(&direct_read, at),
							f32_at(&direct_read, at + 4),
							f32_at(&direct_read, at + 8),
							f32_at(&direct_read, at + 12),
						]
					})
					.collect(),
			);
			off += n;
		}
		baked.smooth(&direct);
	}
	Ok(baked)
}

/// Waits for `buffer`'s pending map to finish; the caller then reads the
/// mapped range and unmaps.
fn read_back(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<(), String> {
	let (tx, rx) = std::sync::mpsc::channel();
	buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
		let _ = tx.send(r);
	});
	device
		.poll(wgpu::PollType::Wait)
		.map_err(|e| format!("GPU bake timed out: {e}"))?;
	rx.recv()
		.map_err(|_| "the GPU bake readback never arrived".to_string())?
		.map_err(|e| format!("GPU bake readback failed: {e}"))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::Light;
	use linalg::vec3;

	fn quad(tris: &mut Vec<Triangle>, a: Vec3, b: Vec3, c: Vec3, d: Vec3, outward: Vec3) {
		for t in [[a, b, c], [a, c, d]] {
			let n = (t[1] - t[0]).cross(t[2] - t[0]);
			tris.push(if n.dot(outward) >= 0.0 {
				t
			} else {
				[t[0], t[2], t[1]]
			});
		}
	}

	/// An 8×8 floor of 1-unit tiles with a 2-high slab on it, mirroring the
	/// CPU tests' shadow scene.
	fn shadow_scene() -> Vec<Triangle> {
		let mut tris = Vec::new();
		for i in 0..8 {
			for k in 0..8 {
				let (x, z) = (i as f32, k as f32);
				quad(
					&mut tris,
					vec3(x, 0.0, z),
					vec3(x + 1.0, 0.0, z),
					vec3(x + 1.0, 0.0, z + 1.0),
					vec3(x, 0.0, z + 1.0),
					vec3(0.0, 1.0, 0.0),
				);
			}
		}
		quad(
			&mut tris,
			vec3(4.0, 0.0, 3.0),
			vec3(4.0, 2.0, 3.0),
			vec3(4.0, 2.0, 4.0),
			vec3(4.0, 0.0, 4.0),
			vec3(1.0, 0.0, 0.0),
		);
		tris
	}

	fn config() -> Config {
		Config {
			atlas_size: 128,
			units_per_texel: 1.0,
			min_chart: 2,
			sun_dir: vec3(1.0, 1.0, 0.0).normalize(),
			sun_color: vec3(1.0, 1.0, 1.0),
			sky_color: Vec3::ZERO,
			ground_color: Vec3::ZERO,
			bounces: 0,
			rays: 4,
			lights: Vec::new(),
			seed: 1,
		}
	}

	fn white(tris: &[Triangle]) -> Vec<Vec3> {
		vec![vec3(0.8, 0.8, 0.8); tris.len()]
	}

	/// One chart per triangle, the same placement the CPU tests bake.
	fn specs(tris: &[Triangle], config: &Config) -> Vec<ChartSpec> {
		tris
			.iter()
			.map(|t| ChartSpec::fit_triangle(t, config))
			.collect()
	}

	/// Every triangle its own chart, in soup order.
	fn identity(n: usize) -> Vec<u32> {
		(0..n as u32).collect()
	}

	/// The CPU and GPU bakes, chart by chart, texel by texel.
	fn diff(cpu: &crate::Baked, gpu: &crate::Baked, tri: usize) -> f32 {
		let (side, texels) = cpu.atlas();
		let (gpu_side, gpu_texels) = gpu.atlas();
		assert_eq!(side, gpu_side);
		let a = cpu.chart(tri).unwrap();
		let b = gpu.chart(tri).unwrap();
		assert_eq!(a.rect, b.rect);
		let mut worst = 0.0f32;
		for j in 0..a.rect.h {
			for i in 0..a.rect.w {
				let at = ((a.rect.y + j) * side + a.rect.x + i) as usize;
				for t in 0..4 {
					let (x, y) = (texels[at][t], gpu_texels[at][t]);
					let d = (x - y).abs();
					assert!(
						d <= x.abs() * 0.01 + 1e-3,
						"triangle {tri} texel {at} channel {t}: CPU {x}, GPU {y}"
					);
					worst = worst.max(d);
				}
			}
		}
		worst
	}

	#[test]
	fn gpu_direct_matches_cpu() {
		let tris = shadow_scene();
		let cpu = crate::bake(
			&tris,
			&white(&tris),
			&specs(&tris, &config()),
			&identity(tris.len()),
			&config(),
		)
		.unwrap();
		let gpu = bake_gpu(
			&tris,
			&white(&tris),
			&specs(&tris, &config()),
			&identity(tris.len()),
			&config(),
		)
		.unwrap();
		assert_eq!(cpu.charts(), gpu.charts());
		let mut worst = 0.0f32;
		for tri in 0..tris.len() {
			if cpu.chart(tri).is_some() {
				worst = worst.max(diff(&cpu, &gpu, tri));
			}
		}
		// Zero bounces and no lights: the two bakes agree on every texel
		// except rounding, which is the whole point of the shared setup.
		assert!(worst < 1e-4, "CPU and GPU differ by {worst}");
	}

	#[test]
	fn gpu_lights_match_cpu() {
		let tris = shadow_scene();
		let mut config = config();
		config.sun_color = Vec3::ZERO;
		config.lights = vec![
			Light {
				// Not on the slab's diagonal: the CPU tests' light at
				// (3.5, 2, 3.5) sits exactly on it, so the rays from the
				// tiles behind the slab graze its shared edge and hit or
				// miss on rounding alone.
				pos: vec3(3.5, 1.5, 3.2),
				color: vec3(1.0, 1.0, 1.0),
				fade: 10.0,
				cone: None,
			},
			Light {
				pos: vec3(2.0, 4.0, 2.0),
				color: vec3(0.5, 0.5, 0.25),
				fade: 8.0,
				cone: Some((vec3(0.0, -1.0, 0.0), 30.0)),
			},
		];
		let cpu = crate::bake(
			&tris,
			&white(&tris),
			&specs(&tris, &config),
			&identity(tris.len()),
			&config,
		)
		.unwrap();
		let gpu = bake_gpu(
			&tris,
			&white(&tris),
			&specs(&tris, &config),
			&identity(tris.len()),
			&config,
		)
		.unwrap();
		for tri in 0..tris.len() {
			if cpu.chart(tri).is_some() {
				diff(&cpu, &gpu, tri);
			}
		}
	}

	#[test]
	fn gpu_bounces_agree_within_noise() {
		let tris = shadow_scene();
		let mut config = config();
		config.sky_color = vec3(0.30, 0.38, 0.50);
		config.ground_color = vec3(0.16, 0.14, 0.12);
		config.bounces = 2;
		config.rays = 64;
		let cpu = crate::bake(
			&tris,
			&white(&tris),
			&specs(&tris, &config),
			&identity(tris.len()),
			&config,
		)
		.unwrap();
		let gpu = bake_gpu(
			&tris,
			&white(&tris),
			&specs(&tris, &config),
			&identity(tris.len()),
			&config,
		)
		.unwrap();
		// Gather rays are hashed per texel on the GPU, so indirect light
		// carries different Monte-Carlo noise: same model, same energy,
		// not the same samples. The mean over each chart stays close.
		let (_, cpu_texels) = cpu.atlas();
		let (_, gpu_texels) = gpu.atlas();
		for tri in 0..tris.len() {
			let (Some(a), Some(b)) = (cpu.chart(tri), gpu.chart(tri)) else {
				continue;
			};
			assert_eq!(a.rect, b.rect);
			let mut mean = (Vec3::ZERO, Vec3::ZERO);
			for j in 0..a.rect.h {
				for i in 0..a.rect.w {
					let at = ((a.rect.y + j) * cpu.atlas().0 + a.rect.x + i) as usize;
					mean.0 += vec3(cpu_texels[at][0], cpu_texels[at][1], cpu_texels[at][2]);
					mean.1 += vec3(gpu_texels[at][0], gpu_texels[at][1], gpu_texels[at][2]);
				}
			}
			let n = (a.rect.w * a.rect.h) as f32;
			let (cpu_mean, gpu_mean) = (mean.0 / n, mean.1 / n);
			let d = (cpu_mean - gpu_mean).length();
			assert!(
				d <= cpu_mean.length() * 0.15 + 0.01,
				"triangle {tri}: CPU mean {cpu_mean:?}, GPU mean {gpu_mean:?}"
			);
		}
	}
}
