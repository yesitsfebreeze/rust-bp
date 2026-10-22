// The map's surfaces drawn as themselves: the triangles the BSP wound,
// in a vertex buffer at load, rasterized at the screen's native resolution
// into the hardware's own depth test. bvh.wgsl is prepended: each lit
// pixel casts one ray toward the sun through the map's hierarchy, so the
// sun's shadow lands exactly where the geometry puts it, as sharp as the
// screen.
// Each fragment takes two hardware-filtered taps: the albedo atlas at the
// corner's whole-tile uv, and the baked light at the corner's lightmap
// texel — the bake's direct sun, sky and bounces, so shadows and ambient
// light come out as sharp as its texels. The straight sun is taken back
// out of the baked light — its color and angle are the bake's own, its
// visibility the texel's alpha — and lit again through the pixel's own
// shadow ray, so the bake's coarse luxels never shape a sun shadow; the
// rest, the ambient sky, bounces and lamps, scales on its own, and its
// share of the lit color rides out in alpha for the ambient occlusion pass
// to darken (see ao.wgsl).
// See-through texels discard, showing what lies behind.
//
// Light lands physically: each face carries a finish — roughness and
// metalness — and the sun and the frame's dynamic lights reflect off it by
// GGX microfacets with Schlick's Fresnel, the diffuse part what the
// specular leaves, none for metal. The albedo's own light and dark make a
// relief the light catches: its brightness read as height, its slope
// across the screen bending the normal. The frame's moving things —
// capsules and spheres — stand in the light of the sun and every dynamic
// light, so the player and the playground's spheres cast shadows over the
// baked and traced world shadows alike. The spheres themselves are drawn
// here too, instanced from the playground's buffer (`vs_body`).
//
// The view uniform is `View` as the trace's passes declare it, and the
// depth is reversed and infinite, near / view depth, as the overlays draw:
// lines and boxes over the world hide behind it exactly where the map
// stands in front of them.

struct View {
	// xyz eye; w the angle one pixel spans, radians.
	eye: vec4f,
	// xyz camera right; w tan(fovy / 2) * aspect.
	right: vec4f,
	// xyz camera up; w tan(fovy / 2).
	up: vec4f,
	// xyz view direction; w pixels per voxel, `K`.
	forward: vec4f,
	// rgb linear sky radiance behind everything; w the smallest voxel,
	// world units.
	sky: vec4f,
	// x exposure; y 1 when this pass must sRGB-encode; z the near plane;
	// w the view mode: 0 lit, else a debug view.
	grade: vec4f,
	// xy viewport pixels; z rays averaged per pixel: 1, 2 or 4; w 1 when
	// the map's triangles draw as pink wireframe.
	size: vec4f,
	// x the frame number so far. y how many times the dents changed.
	frame: vec4u,
	// x scale on the baked ambient light, all but the straight sun; y the
	// ambient occlusion's strength, 0 off to 1 full; z how far around a
	// pixel it looks, world units; w 1 when the sun's shadow is traced per
	// pixel, 0 when the bake's is read. Only the raster and the occlusion
	// pass declare it: the other passes' `View` ends before it.
	light: vec4f,
}

@group(0) @binding(0) var<uniform> view: View;

// The map's light, as `Lighting` in material.wgsl: the sun, to part the
// straight sun from the baked light, and the sky and ground, for faces
// without a baked chart.
struct Lighting {
	// xyz toward the sun.
	sun: vec4f,
	// rgb the sun's radiance.
	sun_color: vec4f,
	// rgb the sky's radiance overhead, and the ground's below.
	sky: vec4f,
	ground: vec4f,
}

// A light that moves or flickers, lit fresh every frame.
struct DynLight {
	// xyz where it is; w how far it reaches, world units.
	pos: vec4f,
	// rgb its radiance a reach's quarter away.
	color: vec4f,
}

// A moving thing's shape, for shadows: the segment `a` to `b` swept by a
// ball of radius `a.w` — a sphere when the two ends meet.
struct Capsule {
	a: vec4f,
	b: vec4f,
}

// What changes every frame: the dynamic lights and the moving things'
// shapes, and switches: x how many lights, y how many capsules, z 1 for
// physically based surfaces (specular and relief), w 1 for the moving
// things' shadows.
struct Dynamic {
	counts: vec4u,
	lights: array<DynLight, 16>,
	capsules: array<Capsule, 32>,
}

@group(0) @binding(3) var atlas: sampler;
// The albedo atlas through its sRGB view, so the hardware decodes the
// bytes to linear as it filters.
@group(0) @binding(5) var albedo_srgb: texture_2d<f32>;
// The bake's light atlas: linear radiance in half floats, one chart
// per triangle, its edges duplicated into the chart's gutter.
@group(0) @binding(6) var baked_light: texture_2d<f32>;
@group(0) @binding(7) var<uniform> dynamic: Dynamic;
@group(0) @binding(11) var<uniform> lighting: Lighting;

const PI: f32 = 3.14159265;
// How far off the surface shadow rays start, world units, so they do not
// hit the triangle they leave.
const SHADOW_OFFSET: f32 = 0.125;
// The depth view's far end, world units, as in trace.wgsl.
const DEPTH_RANGE: f32 = 4096.0;
// How high the albedo's brightest texel stands over its darkest, world
// units, for the relief.
const RELIEF: f32 = 1.5;
// Reflectance of every non-metal seen straight on.
const DIELECTRIC: f32 = 0.04;
// The playground spheres' matter, as linear albedo, as in trace.wgsl, and
// their finish.
const BODY: vec3f = vec3f(0.56, 0.38, 0.19);
const BODY_ROUGHNESS: f32 = 0.5;

// The debug views F5 cycles through, after the lit one, as in trace.wgsl.
const VIEW_DEPTH: u32 = 1u;
const VIEW_NORMAL: u32 = 2u;
const VIEW_ALBEDO: u32 = 3u;
const VIEW_CUBES: u32 = 4u;
const VIEW_SIZE: u32 = 5u;

fn view_mode() -> u32 {
	return u32(view.grade.w);
}

fn aces(x: vec3f) -> vec3f {
	return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

// One corner, as scene.rs packed it: where it is with its face's roughness
// in w, its whole-tile texture coordinates with its lightmap texel in `zw`,
// the face's unit normal with its metalness in w, its rectangle of the
// albedo atlas, and its triangle's rectangle of the baked light atlas.
struct Corner {
	@location(0) pos: vec4f,
	@location(1) uv: vec4f,
	@location(2) n: vec4f,
	@location(3) tex: vec4f,
	@location(4) lm: vec4f,
}

struct Varying {
	@builtin(position) clip: vec4f,
	@location(0) pos: vec3f,
	@location(1) uv: vec2f,
	@location(2) n: vec3f,
	@location(3) tex: vec4f,
	// The corner's lightmap texel, continuous, and its chart's rectangle in
	// the baked atlas — or, a height of -1 marking a loose face, the light
	// around it in rgb.
	@location(4) lmuv: vec2f,
	@location(5) lm: vec4f,
	// The face's roughness and metalness.
	@location(6) finish: vec2f,
}

// The clip position of world point `p`: reversed and infinite, as the
// overlays draw — z the near plane, w the view depth, so z / w is
// near / view depth.
fn clip_of(p: vec3f) -> vec4f {
	let rel = p - view.eye.xyz;
	return vec4f(
		dot(rel, view.right.xyz) / view.right.w,
		dot(rel, view.up.xyz) / view.up.w,
		view.grade.z,
		dot(rel, view.forward.xyz),
	);
}

@vertex
fn vs(c: Corner) -> Varying {
	var out: Varying;
	out.clip = clip_of(c.pos.xyz);
	out.pos = c.pos.xyz;
	out.uv = c.uv.xy;
	out.n = c.n.xyz;
	out.tex = c.tex;
	out.lmuv = c.uv.zw;
	out.lm = c.lm;
	out.finish = vec2f(c.pos.w, c.n.w);
	return out;
}

// The texel at whole-tile `uv`, within the corner's atlas tile: the tile's
// own texels only, held half a texel in from its edges so the hardware's
// linear filter never blends across a neighboring tile. `textureSample`
// takes the coordinate in the whole atlas's units, texels over its side.
fn texel_at(tex: vec4f, uv: vec2f) -> vec4f {
	let at = tex.xy + clamp(fract(uv) * tex.zw, vec2f(0.5), tex.zw - vec2f(0.5));
	let side = vec2f(textureDimensions(albedo_srgb));
	return textureSample(albedo_srgb, atlas, at / side);
}

// The baked light at the corner's lightmap texel: within its chart, held
// between its edge texels' centers — the gutter duplicates them, so the
// hardware's linear filter stays inside the chart either way. `lmuv`
// holds texel centers at whole numbers, as `FaceLm` maps them and
// material.wgsl rounds them; the atlas coordinate holds them half a texel
// in.
fn light_at(lm: vec4f, lmuv: vec2f) -> vec4f {
	let hi = max(lm.zw - vec2f(1.0), vec2f(0.0));
	let at = lm.xy + clamp(lmuv, vec2f(0.0), hi) + vec2f(0.5);
	let side = vec2f(textureDimensions(baked_light));
	return textureSample(baked_light, atlas, at / side);
}

// Rec. 709 luminance, to weigh one light's share of another.
fn luminance(c: vec3f) -> f32 {
	return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Whether the ray from `o` along unit `d` meets the capsule `c` before
// `far`: the swept segment's side, then whichever end cap it passes.
fn hits_capsule(o: vec3f, d: vec3f, far: f32, c: Capsule) -> bool {
	let r = c.a.w;
	let ba = c.b.xyz - c.a.xyz;
	let oa = o - c.a.xyz;
	let baba = dot(ba, ba);
	if (baba > 1e-6) {
		let bard = dot(ba, d);
		let baoa = dot(ba, oa);
		let rdoa = dot(d, oa);
		let oaoa = dot(oa, oa);
		let a = baba - bard * bard;
		let b = baba * rdoa - baoa * bard;
		let k = baba * oaoa - baoa * baoa - r * r * baba;
		let h = b * b - a * k;
		if (h < 0.0) {
			return false;
		}
		if (a > 1e-6) {
			let t = (-b - sqrt(h)) / a;
			let y = baoa + t * bard;
			if (y > 0.0 && y < baba) {
				return t > 0.0 && t < far;
			}
		}
	}
	// An end cap, or a sphere when the ends meet.
	let center = select(c.a.xyz, c.b.xyz, baba > 1e-6 && dot(o + d * dot(c.b.xyz - o, d) - c.a.xyz, ba) > baba * 0.5);
	let oc = o - center;
	let b = dot(d, oc);
	let h = b * b - (dot(oc, oc) - r * r);
	if (h < 0.0) {
		return false;
	}
	let t = -b - sqrt(h);
	return t > 0.0 && t < far;
}

// Whether a moving thing stands on the ray from `o` along unit `d` before
// `far`. Things holding the ray's far end — a light carried inside one —
// never shade their own light.
fn moving_blocks(o: vec3f, d: vec3f, far: f32) -> bool {
	if (dynamic.counts.w == 0u) {
		return false;
	}
	let end = o + d * far;
	for (var i = 0u; i < dynamic.counts.y; i++) {
		let c = dynamic.capsules[i];
		let ba = c.b.xyz - c.a.xyz;
		let along = clamp(dot(end - c.a.xyz, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
		if (far < MISS && length(end - (c.a.xyz + ba * along)) < c.a.w) {
			continue;
		}
		if (hits_capsule(o, d, far, c)) {
			return true;
		}
	}
	return false;
}

// What a point needs to be lit: where it is, its outward face normal (for
// shadow rays), the normal it reflects with (bent by relief, facing the
// eye), the unit direction toward the eye, its albedo and finish, the
// ambient light reaching it — sky, bounces and baked lamps, as the bake's
// `S` — and how much of the sun reaches it.
struct Point {
	pos: vec3f,
	face: vec3f,
	n: vec3f,
	v: vec3f,
	albedo: vec3f,
	roughness: f32,
	metal: f32,
	ambient: vec3f,
	sun: f32,
}

// Microfacet reflection of radiance `radiance` arriving along unit `l` at
// `p`: GGX distribution, Smith-Schlick geometry, Schlick Fresnel, and the
// diffuse share the specular leaves — Lambert alone when the surfaces are
// not physically based, as the bake lights them.
fn reflect_light(p: Point, l: vec3f, radiance: vec3f) -> vec3f {
	let n_l = max(dot(p.n, l), 0.0);
	if (n_l <= 0.0) {
		return vec3f(0.0);
	}
	if (dynamic.counts.z == 0u) {
		return p.albedo / PI * radiance * n_l;
	}
	let f0 = mix(vec3f(DIELECTRIC), p.albedo, p.metal);
	let h = normalize(p.v + l);
	let n_v = max(dot(p.n, p.v), 1e-3);
	let n_h = max(dot(p.n, h), 0.0);
	let a = p.roughness * p.roughness;
	let a2 = a * a;
	let q = n_h * n_h * (a2 - 1.0) + 1.0;
	let distribution = a2 / (PI * q * q);
	let k = (p.roughness + 1.0) * (p.roughness + 1.0) / 8.0;
	let geometry = n_v / (n_v * (1.0 - k) + k) * (n_l / (n_l * (1.0 - k) + k));
	let fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(p.v, h), 0.0), 5.0);
	let specular = distribution * geometry * fresnel / max(4.0 * n_v * n_l, 1e-4);
	let diffuse = (1.0 - fresnel) * (1.0 - p.metal) * p.albedo / PI;
	return (diffuse + specular) * radiance * n_l;
}

// The ambient light reflected toward the eye: diffuse off the albedo, and
// with physically based surfaces the specular share Fresnel gives at the
// surface's roughness, of the same light — the room's own brightness
// standing in for what the reflection would show.
fn reflect_ambient(p: Point) -> vec3f {
	if (dynamic.counts.z == 0u) {
		return p.albedo * p.ambient;
	}
	let f0 = mix(vec3f(DIELECTRIC), p.albedo, p.metal);
	let n_v = max(dot(p.n, p.v), 0.0);
	let fresnel = f0 + (max(vec3f(1.0 - p.roughness), f0) - f0) * pow(1.0 - n_v, 5.0);
	return (1.0 - fresnel) * (1.0 - p.metal) * p.albedo * p.ambient + fresnel * p.ambient;
}

// The point's lit color before exposure, and the ambient light's share of
// it for the occlusion pass.
fn shade(p: Point) -> vec4f {
	let l = lighting.sun.xyz;
	var direct = vec3f(0.0);
	if (p.sun > 0.0 && dot(p.face, l) > 0.0
		&& !moving_blocks(p.pos + p.face * SHADOW_OFFSET, l, MISS)) {
		direct = reflect_light(p, l, lighting.sun_color.rgb * p.sun);
	}
	for (var i = 0u; i < dynamic.counts.x; i++) {
		let light = dynamic.lights[i];
		let to = light.pos.xyz - p.pos;
		let dist = length(to);
		let reach = light.pos.w;
		if (dist >= reach || dist < 1e-3) {
			continue;
		}
		let dir = to / dist;
		if (dot(p.face, dir) <= 0.0) {
			continue;
		}
		let o = p.pos + p.face * SHADOW_OFFSET;
		let far = max(dist - SHADOW_OFFSET, 0.0);
		if (occluded(o, dir, far) || moving_blocks(o, dir, far)) {
			continue;
		}
		// Inverse square, windowed to nothing at the reach.
		let window = saturate(1.0 - pow(dist / reach, 4.0));
		let falloff = window * window / (1.0 + 16.0 * (dist / reach) * (dist / reach));
		direct += reflect_light(p, dir, light.color.rgb * falloff);
	}
	let ambient = reflect_ambient(p);
	let color = direct + ambient;
	return vec4f(color, saturate(luminance(ambient) / max(luminance(color), 1e-4)));
}

// The normal `n` bent by the relief of height field `h` over the surface
// through `pos`: the height's slope across the screen against the
// surface's own (Mikkelsen's surface gradient).
fn relief(n: vec3f, pos: vec3f, h: f32) -> vec3f {
	let dpx = dpdx(pos);
	let dpy = dpdy(pos);
	let r1 = cross(dpy, n);
	let r2 = cross(n, dpx);
	let det = dot(dpx, r1);
	let grad = sign(det) * (dpdx(h) * r1 + dpdy(h) * r2);
	let bent = abs(det) * n - grad;
	return select(n, normalize(bent), dot(bent, bent) > 1e-12);
}

// Half a byte of noise either way at pixel `frag`, interleaved so no two
// neighbors share it: added to a color about to land in bytes, it spreads
// the rounding so a smooth gradient stays smooth instead of stepping
// into bands.
fn dither(frag: vec2f) -> f32 {
	return (fract(52.9829189 * fract(dot(frag, vec2f(0.06711056, 0.00583715)))) - 0.5) / 255.0;
}

// Exposure, tone mapping and the swapchain's encoding, on the lit view
// only, dithered at pixel `frag` for the bytes it lands in; the debug views
// show their values as they are.
fn grade(c: vec3f, frag: vec2f) -> vec3f {
	var color = c;
	if (view_mode() == 0u) {
		color = aces(color * view.grade.x);
	}
	if (view.grade.y > 0.5) {
		color = pow(color, vec3f(1.0 / 2.2));
	}
	return color + dither(frag);
}

@fragment
fn fs(in: Varying) -> @location(0) vec4f {
	let texel = texel_at(in.tex, in.uv);
	// The relief's height, read before any pixel discards so the slopes
	// across the screen see their neighbors.
	let height = luminance(texel.rgb) * RELIEF;
	let d = normalize(in.pos - view.eye.xyz);
	let n = select(in.n, -in.n, dot(in.n, d) > 0.0);
	let bent = relief(n, in.pos, height);
	// See-through texels show what lies behind, as the trace's hit does.
	if (texel.a < 0.5) {
		discard;
	}
	// The baked light parted into the straight sun — as the bake lit the
	// face, by its own normal, through the filtered visibility — and the
	// ambient rest; a face without a chart keeps the hemisphere of sky
	// around its normal, as the trace's shade does, all of it ambient.
	let sky = mix(lighting.ground.rgb, lighting.sky.rgb, n.y * 0.5 + 0.5) * 0.5;
	let n_l = max(dot(in.n, lighting.sun.xyz), 0.0);
	let sun_full = lighting.sun_color.rgb * (n_l / PI);
	var ambient = sky;
	// The bake's own visibility of the sun: the shadow when it is not
	// traced.
	var sun = 0.0;
	if (in.lm.w >= 1.0) {
		let baked = light_at(in.lm, in.lmuv);
		ambient = baked.rgb - min(sun_full * baked.a, baked.rgb);
		sun = baked.a;
	} else if (in.lm.w < 0.0) {
		// A loose face — a prop — carries the light around it instead of
		// a chart, and sees the sun by its own ray.
		ambient = in.lm.rgb;
		sun = 1.0;
	}
	if (view.light.w > 0.5 || in.lm.w < 0.0) {
		sun = select(0.0, 1.0, n_l > 0.0
			&& !occluded(in.pos + in.n * SHADOW_OFFSET, lighting.sun.xyz, MISS));
	}
	var p: Point;
	p.pos = in.pos;
	p.face = in.n;
	p.n = select(n, bent, dynamic.counts.z != 0u);
	p.v = -d;
	p.albedo = texel.rgb;
	p.roughness = clamp(in.finish.x, 0.05, 1.0);
	p.metal = in.finish.y;
	p.ambient = ambient * view.light.x;
	p.sun = sun;
	let lit = shade(p);
	var color = lit.rgb;
	// The debug views F5 cycles through, as in trace.wgsl.
	switch view_mode() {
		case VIEW_DEPTH: {
			let z = dot(in.pos - view.eye.xyz, view.forward.xyz);
			color = vec3f(1.0 - clamp(log2(1.0 + z) / log2(1.0 + DEPTH_RANGE), 0.0, 1.0));
		}
		case VIEW_NORMAL: {
			color = p.n * 0.5 + 0.5;
		}
		case VIEW_ALBEDO: {
			color = texel.rgb;
		}
		case VIEW_CUBES, VIEW_SIZE: {
			// The cubes and cube-size views, of voxelized matter: whole
			// surfaces, one color for both.
			color = vec3f(0.9);
		}
		default: {}
	}
	return vec4f(grade(color, in.clip.xy), select(1.0, lit.a, view_mode() == 0u));
}

// One corner of the unit sphere every body is drawn from, and the body's
// center and radius, per instance: radius 0 for an empty slot.
struct BodyCorner {
	@location(0) unit: vec3f,
	@location(1) body: vec4f,
}

struct BodyVarying {
	@builtin(position) clip: vec4f,
	@location(0) pos: vec3f,
	@location(1) n: vec3f,
}

@vertex
fn vs_body(c: BodyCorner) -> BodyVarying {
	var out: BodyVarying;
	let pos = c.body.xyz + c.unit * c.body.w;
	out.clip = clip_of(pos);
	// An empty slot folds to a point behind the eye, drawing nothing.
	if (c.body.w <= 0.0) {
		out.clip = vec4f(0.0, 0.0, 0.0, -1.0);
	}
	out.pos = pos;
	out.n = c.unit;
	return out;
}

@fragment
fn fs_body(in: BodyVarying) -> @location(0) vec4f {
	let n = normalize(in.n);
	var p: Point;
	p.pos = in.pos;
	p.face = n;
	p.n = n;
	p.v = normalize(view.eye.xyz - in.pos);
	p.albedo = BODY;
	p.roughness = BODY_ROUGHNESS;
	p.metal = 0.0;
	// No chart: the hemisphere of sky around its normal, as a face without
	// one keeps.
	p.ambient = mix(lighting.ground.rgb, lighting.sky.rgb, n.y * 0.5 + 0.5) * 0.5 * view.light.x;
	p.sun = select(0.0, 1.0, dot(n, lighting.sun.xyz) > 0.0
		&& !occluded(in.pos + n * SHADOW_OFFSET, lighting.sun.xyz, MISS));
	let lit = shade(p);
	var color = lit.rgb;
	if (view_mode() == VIEW_NORMAL) {
		color = n * 0.5 + 0.5;
	} else if (view_mode() != 0u) {
		color = vec3f(0.9);
	}
	return vec4f(grade(color, in.clip.xy), select(1.0, lit.a, view_mode() == 0u));
}
