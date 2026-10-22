// How the map's surfaces look and how they are lit, shared by every pass
// that casts rays against it. Prepended with bvh.wgsl.
//
// Light comes two ways. Every map surface shades by its baked chart: the
// direct sun, the sky and the bounces `lightbake` traced, stored per face
// on the map's own luxel grid, the same grid qrad sampled. The bake rides
// in each triangle, with its face's lightmap mapping; `baked_light_at`
// samples it at a world point. What the bake does not cover — the
// playground's spheres, the dents' bowls — falls back to the live model:
// the sun through a shadow ray, the sky and the ground as a hemisphere of
// ambient radiance.

struct Lighting {
	// xyz toward the sun.
	sun: vec4f,
	// rgb the sun's radiance.
	sun_color: vec4f,
	// rgb the sky's radiance overhead, and the ground's below.
	sky: vec4f,
	ground: vec4f,
}

// The albedo atlas as stored, sRGB bytes read raw; and through an sRGB
// view, so the hardware decodes it to linear.
@group(0) @binding(4) var albedo_atlas: texture_2d<f32>;
@group(0) @binding(5) var albedo_srgb: texture_2d<f32>;
// The bake's light atlas: linear radiance in half floats, one chart per
// face, read unfiltered at its texels.
@group(0) @binding(6) var baked_light: texture_2d<f32>;
@group(0) @binding(11) var<uniform> lighting: Lighting;


const PI: f32 = 3.14159265;
// Shadow rays start this far off the surface.
const OFFSET: f32 = 0.5;

// Where in the atlas the triangle's texel at `p` lies, nearest.
fn texel(t: Tri, p: vec3f) -> vec2i {
	let uv = fract(vec2f(dot(t.su.xyz, p) + t.su.w, dot(t.sv.xyz, p) + t.sv.w));
	let at = min(vec2u(uv * vec2f(t.tex.zw)), t.tex.zw - 1u);
	return vec2i(t.tex.xy + at);
}

// The triangle's texel at `p` as stored: rgb sRGB-encoded, a coverage.
fn albedo(t: Tri, p: vec3f) -> vec4f {
	return textureLoad(albedo_atlas, texel(t, p), 0);
}

// The triangle's texel color at `p`, linear.
fn albedo_linear(t: Tri, p: vec3f) -> vec3f {
	return textureLoad(albedo_srgb, texel(t, p), 0).rgb;
}

// The triangle's color at `p`, linear, blended between the four nearest
// texels, so a voxel smaller than a texel takes the texture's gradient,
// not its texel's square. Wraps within the texture's tile.
fn albedo_smooth(t: Tri, p: vec3f) -> vec3f {
	let size = vec2f(t.tex.zw);
	let uv = fract(vec2f(dot(t.su.xyz, p) + t.su.w, dot(t.sv.xyz, p) + t.sv.w)) * size - 0.5;
	let base = floor(uv);
	let f = uv - base;
	var corner: array<vec3f, 4>;
	for (var i = 0; i < 4; i += 1) {
		let at = (base + vec2f(f32(i & 1), f32(i >> 1u)) + size) % size;
		corner[i] = textureLoad(albedo_srgb, vec2i(t.tex.xy) + vec2i(at), 0).rgb;
	}
	return mix(mix(corner[0], corner[1], f.x), mix(corner[2], corner[3], f.x), f.y);
}

fn luma(c: vec3f) -> f32 {
	return dot(c, vec3f(0.299, 0.587, 0.114));
}

// Sky above, ground below, blended across the horizon.
fn sky(d: vec3f) -> vec3f {
	return mix(lighting.ground.rgb, lighting.sky.rgb, d.y * 0.5 + 0.5);
}

// The global light at `p` facing `n`, per unit albedo: the sun through a
// shadow ray, and the hemisphere of sky around it. In the sun's shadow a
// surface keeps `shaded` of its sunlight, so shadowed matter dims rather
// than blacks out.
fn global_light(p: vec3f, n: vec3f, shaded: f32) -> vec3f {
	return sun_and_sky(n, occluded(p + n * OFFSET, lighting.sun.xyz, 1e30), shaded);
}

// The global light facing `n`, per unit albedo, the sun `blocked` or not.
fn sun_and_sky(n: vec3f, blocked: bool, shaded: f32) -> vec3f {
	let n_l = dot(n, lighting.sun.xyz);
	let sun = select(0.0, n_l / PI, n_l > 0.0) * select(1.0, shaded, blocked);
	return lighting.sun_color.rgb * sun + sky(n) * 0.5;
}

// The baked light on triangle `t` at world point `p` on its plane: the
// face's chart texel whose luxel `p` falls in, clamped into the chart —
// the bake's direct sun, sky and bounces. `p` must lie on the triangle's
// plane: the chart's grid is the plane's own.
fn baked_light_at(t: Tri, p: vec3f) -> vec3f {
	let lm = vec2f(dot(t.lmu.xyz, p) + t.lmu.w, dot(t.lmv.xyz, p) + t.lmv.w);
	let at = clamp(vec2i(round(lm)), vec2i(0), vec2i(t.lmr.zw) - vec2i(1));
	return textureLoad(baked_light, vec2i(t.lmr.xy) + at, 0).rgb;
}

// Whether triangle `t`'s face carries a baked chart.
fn is_baked(t: Tri) -> bool {
	return t.lmr.z >= 1u;
}