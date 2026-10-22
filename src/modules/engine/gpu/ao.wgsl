// The frame's ambient occlusion: after the world pass draws the map into
// the frame's own color and depth, two full-screen passes shade every
// drawn pixel darker where the depth shows matter standing near it — the
// crevice between two surfaces, the corner they meet in, the underside of
// a thing — and carry the color to the window. No ray is cast: the
// pixel's own depth comes back as its world position (the projection is
// reversed, so the stored value is the near plane over the view depth),
// its normal from its neighbors' positions, and a spiral of taps around it
// — each projected back to the screen — weighs the matter the depth shows
// standing over the surface's tangent plane, within `view.light.z`.
//
// `occlusion` turns each pixel's spiral by its place in a 4×4 tile, so the
// sixteen pixels of any tile together take sixteen different spirals, and
// writes the openness alone; `composite` averages it over that 4×4 tile
// among the pixels whose depth agrees — the same surface, not one behind
// it — which leaves sixteen spirals' worth of taps per pixel, smooth into
// every corner and sharp along every silhouette.
//
// Only ambient light is occluded — sky, bounces and lamps; the straight
// sun either reaches a point or its shadow ray already holds it back — so
// the darkening lands on the share of the color the world pass wrote in
// alpha, at the strength `view.light.y`. The traced view draws its whole
// screen at depth 0 and the sky never draws, so both pass through
// unchanged, as does every debug view.
//
// Last the pixel is fogged by the air between the eye and what it shows:
// fog `view.fog.x` thick at height 0, thinning by e every `view.fog.y`
// units up, integrated along the pixel's ray in closed form — out to its
// depth, or for the sky, out past the map — so a ray climbing out of the
// fog clears and the horizon sinks into it. The air glows with the sky's
// light all around and the sun's ahead of it, scattered mostly forward,
// tone mapped as the world pass maps its colors, and laid over the pixel
// by how much of it the air hides. No shadow reaches the fog: it glows the
// same in the shade. Only the surface view sets `view.fog.x`.
//
// First of all, with `view.fog.z` set, the color itself is read smoothed
// along the edges it shows (FXAA): where the brightness steps sharply
// across the pixel, the edge is walked both ways to its ends, and the
// pixel takes the color a step toward its brighter or darker neighbor by
// how far along the edge it sits, so a stair of whole pixels blends into
// the slope it stands for. Only the lit view smooths; debug views stay
// exact.

// The view uniform is `View` as the trace's passes declare it.
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
	// x the fog's extinction per world unit at height 0, 0 clear; y the
	// height it thins by e over, world units; z 1 when the color is
	// smoothed along its edges. Only this pass declares it.
	fog: vec4f,
	// rgb the tint on the fog's glow, white as the sky and sun light it.
	fog_color: vec4f,
}

// The map's light, as `Lighting` in material.wgsl: what the fog glows with.
struct Lighting {
	// xyz toward the sun.
	sun: vec4f,
	// rgb the sun's radiance.
	sun_color: vec4f,
	// rgb the sky's radiance overhead, and the ground's below.
	sky: vec4f,
	ground: vec4f,
}

@group(0) @binding(0) var<uniform> view: View;
// The world pass's depth, reversed as the raster wrote it: the near plane
// over the view depth, 0 where nothing drew.
@group(0) @binding(1) var depth: texture_2d<f32>;
// The world pass's color, what `composite` shades and carries over.
@group(0) @binding(2) var scene: texture_2d<f32>;
// The openness `occlusion` wrote, 1 under nothing.
@group(0) @binding(3) var open: texture_2d<f32>;
@group(0) @binding(4) var<uniform> lighting: Lighting;

// Taps each pixel's spiral takes over the hemisphere above the surface.
const TAPS: u32 = 12u;
// How far apart two depths may lie, as a share of the nearer, and still
// be one surface for the blur.
const SAME_SURFACE: f32 = 0.04;
// How far the sky's pixels look through the fog, world units: past any
// map.
const FOG_FAR: f32 = 32768.0;
// How strongly the fog scatters the sun forward, Henyey-Greenstein's g:
// 0 every way alike, toward 1 a tight glow around the sun.
const FOG_FORWARD: f32 = 0.6;
const PI: f32 = 3.14159265;
// The least brightness step across a pixel, absolute and as a share of
// its brightest neighbor, the edge smoothing takes for an edge.
const EDGE_MIN: f32 = 0.0312;
const EDGE_SHARE: f32 = 0.125;

fn view_mode() -> u32 {
	return u32(view.grade.w);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
	return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

// The world position the pixel `uv` shows: the ray through its center,
// walked out to the view depth its hardware depth holds.
fn world_at(uv: vec2u, z: f32) -> vec3f {
	let ray = ray_at(uv);
	// The stored value is the near plane over the view depth along
	// `forward`, so the walk is that view depth over the ray's own depth
	// along it.
	return view.eye.xyz + ray * (view.grade.z / z / dot(ray, view.forward.xyz));
}

// The world position the pixel `o` steps from `uv` shows, held on screen.
fn neighbor(uv: vec2u, o: vec2i) -> vec3f {
	let side = vec2i(textureDimensions(depth)) - vec2i(1);
	let at = vec2u(clamp(vec2i(uv) + o, vec2i(0), side));
	return world_at(at, textureLoad(depth, at, 0).r);
}

// The unit normal of the surface the pixel at `uv` shows, facing the eye:
// the cross of the depth's slopes across the screen, each taken on the
// side nearer the pixel itself, so a pixel on a corner's edge reads the
// slope of its own face, not the jump to the face beside it.
fn normal_at(uv: vec2u, p: vec3f) -> vec3f {
	let l = neighbor(uv, vec2i(-1, 0));
	let r = neighbor(uv, vec2i(1, 0));
	let d = neighbor(uv, vec2i(0, -1));
	let u = neighbor(uv, vec2i(0, 1));
	let dx = select(r - p, p - l, length(p - l) < length(r - p));
	let dy = select(u - p, p - d, length(p - d) < length(u - p));
	let raw = cross(dx, dy);
	let n = normalize(select(vec3f(0.0, 0.0, 1.0), raw, dot(raw, raw) > 1e-12));
	return select(n, -n, dot(n, p - view.eye.xyz) > 0.0);
}

// The pixel the world position `s` projects to, as the raster projected
// its corners.
fn pixel_of(s: vec3f) -> vec2u {
	let rel = s - view.eye.xyz;
	let ahead = max(dot(rel, view.forward.xyz), 1e-3);
	let ndc = vec2f(dot(rel, view.right.xyz) / view.right.w,
		dot(rel, view.up.xyz) / view.up.w) / ahead;
	let pix = vec2f((ndc.x + 1.0) * 0.5 * view.size.x,
		(1.0 - ndc.y) * 0.5 * view.size.y);
	return vec2u(clamp(pix, vec2f(0.0), vec2f(textureDimensions(depth)) - vec2f(1.0)));
}

// The pixel's place in its 4×4 tile, ordered so neighbors sit far apart
// in the sequence: a Bayer matrix, 0 to 15.
fn bayer(uv: vec2u) -> f32 {
	let x = uv.x & 3u;
	let y = uv.y & 3u;
	let m = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
	return f32(m[y * 4u + x]);
}

// The `i`th of `TAPS` points spread over the hemisphere around `n`: a
// spiral from straight up the normal to its rim, turned by `turn` and
// reaching out to a share of the radius that grows along it, so near
// matter weighs in as well as far.
fn tap(i: u32, turn: f32, n: vec3f, t: vec3f, b: vec3f) -> vec3f {
	let u = (f32(i) + 0.5) / f32(TAPS);
	let a = f32(i) * 2.39996323 + turn * 6.28318531;
	let dir = n * sqrt(1.0 - u) + (t * cos(a) + b * sin(a)) * sqrt(u);
	return dir * mix(0.2, 1.0, fract(u + turn * 0.618034));
}

// How open the hemisphere over the surface at `p`, facing `n`, stands:
// 1 under nothing, toward 0 walled in.
fn openness(p: vec3f, n: vec3f, turn: f32) -> f32 {
	// The hemisphere's frame: `t` and `b` any two directions across it.
	let axis = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.5);
	let t = normalize(cross(n, axis));
	let b = cross(n, t);
	let radius = view.light.z;
	var occluded = 0.0;
	for (var i = 0u; i < TAPS; i++) {
		let s = p + tap(i, turn, n, t, b) * radius;
		let at = pixel_of(s);
		let z = textureLoad(depth, at, 0).r;
		// The sky occludes nothing.
		if (z <= 0.0) {
			continue;
		}
		let q = world_at(at, z);
		let v = q - p;
		let dist = length(v);
		if (dist < 1e-3 || dist >= radius) {
			continue;
		}
		// Only matter standing above the surface's tangent plane occludes:
		// the surface continued past `p` lies on the plane, so it never
		// shades itself. The straighter over the surface the matter
		// stands, and the nearer, the more it occludes.
		let fall = 1.0 - dist / radius;
		occluded += saturate(dot(n, v / dist) - 0.05) * fall * fall;
	}
	return saturate(1.0 - 1.6 * occluded / f32(TAPS));
}

fn aces(x: vec3f) -> vec3f {
	return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

// The ray through the pixel `uv`'s center, from the eye.
fn ray_at(uv: vec2u) -> vec3f {
	let ndc = vec2f(2.0 * (f32(uv.x) + 0.5) / view.size.x - 1.0,
		1.0 - 2.0 * (f32(uv.y) + 0.5) / view.size.y);
	return normalize(view.forward.xyz
		+ view.right.xyz * (ndc.x * view.right.w)
		+ view.up.xyz * (ndc.y * view.up.w));
}

// The fog's optical depth along `len` units of the unit ray `d` from the
// eye: its density, falling off with height, integrated in closed form.
fn fog_depth(d: vec3f, len: f32) -> f32 {
	let k = 1.0 / max(view.fog.y, 1.0);
	let start = view.fog.x * exp(-view.eye.y * k);
	// The density's log change over the ray, held where a ray diving deep
	// below height 0 would overflow it.
	let fall = max(d.y * k * len, -60.0);
	if (abs(fall) < 1e-4) {
		return start * len;
	}
	return start * len * (1.0 - exp(-fall)) / fall;
}

// The light the fog scatters toward the eye along `d`: the sky's and the
// ground's all around, and the sun's, mostly forward, tinted by the fog's
// color.
fn fog_glow(d: vec3f) -> vec3f {
	let g = FOG_FORWARD;
	let c = dot(d, lighting.sun.xyz);
	let phase = (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * c, 1.5));
	let glow = mix(lighting.ground.rgb, lighting.sky.rgb, 0.5) + lighting.sun_color.rgb * phase;
	return glow * view.fog_color.rgb;
}

// The scene's color at `p`, pixels with centers on the halves, blended
// between the four pixels around it.
fn scene_at(p: vec2f) -> vec4f {
	let side = vec2i(textureDimensions(scene)) - vec2i(1);
	let q = p - 0.5;
	let i = vec2i(floor(q));
	let f = q - floor(q);
	let a = textureLoad(scene, clamp(i, vec2i(0), side), 0);
	let b = textureLoad(scene, clamp(i + vec2i(1, 0), vec2i(0), side), 0);
	let c = textureLoad(scene, clamp(i + vec2i(0, 1), vec2i(0), side), 0);
	let d = textureLoad(scene, clamp(i + vec2i(1, 1), vec2i(0), side), 0);
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// The brightness the eye sees in `rgb`, as the scene holds it: encoded
// already, or linear and encoded roughly here, so steps in the darks
// weigh as the eye weighs them.
fn luma(rgb: vec3f) -> f32 {
	let l = dot(rgb, vec3f(0.299, 0.587, 0.114));
	return select(sqrt(l), l, view.grade.y > 0.5);
}

fn luma_at(p: vec2f) -> f32 {
	return luma(scene_at(p).rgb);
}

// The brightness of the pixel `o` steps from `uv`, held on screen.
fn luma_by(uv: vec2u, o: vec2i) -> f32 {
	let side = vec2i(textureDimensions(scene)) - vec2i(1);
	return luma(textureLoad(scene, clamp(vec2i(uv) + o, vec2i(0), side), 0).rgb);
}

// The scene's color at the pixel `frag`, `own` its own, smoothed along
// the edge through it, if one runs there.
fn smoothed(frag: vec2f, own: vec3f) -> vec3f {
	let uv = vec2u(frag);
	let c = luma(own);
	let n = luma_by(uv, vec2i(0, -1));
	let s = luma_by(uv, vec2i(0, 1));
	let w = luma_by(uv, vec2i(-1, 0));
	let e = luma_by(uv, vec2i(1, 0));
	let hi = max(c, max(max(n, s), max(w, e)));
	let range = hi - min(c, min(min(n, s), min(w, e)));
	if (range < max(EDGE_MIN, hi * EDGE_SHARE)) {
		return own;
	}
	let nw = luma_by(uv, vec2i(-1, -1));
	let ne = luma_by(uv, vec2i(1, -1));
	let sw = luma_by(uv, vec2i(-1, 1));
	let se = luma_by(uv, vec2i(1, 1));
	// Whether the edge runs across the screen, its step up or down it,
	// weighing the pixel's own row and column twice.
	let across = abs(nw + ne - 2.0 * n) + 2.0 * abs(w + e - 2.0 * c) + abs(sw + se - 2.0 * s);
	let down = abs(nw + sw - 2.0 * w) + 2.0 * abs(n + s - 2.0 * c) + abs(ne + se - 2.0 * e);
	let flat = across >= down;
	// The neighbors on either side of the edge; it lies toward the one
	// the brightness steps to most.
	let before = select(w, n, flat);
	let after = select(e, s, flat);
	let to_before = abs(before - c) >= abs(after - c);
	let normal = select(vec2f(1.0, 0.0), vec2f(0.0, 1.0), flat) * select(1.0, -1.0, to_before);
	let middle = 0.5 * (c + select(after, before, to_before));
	let grad = 0.25 * max(abs(before - c), abs(after - c));
	// Walk the edge's seam both ways until the brightness along it leaves
	// the seam's own: the edge's ends.
	let along = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), flat);
	let seam = frag + normal * 0.5;
	var p1 = seam - along;
	var p2 = seam + along;
	var e1 = luma_at(p1) - middle;
	var e2 = luma_at(p2) - middle;
	// How far each step of the walk reaches, pixels: short near the pixel,
	// where most edges end, longer past it.
	var steps = array<f32, 10>(1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 4.0, 8.0);
	for (var i = 0u; i < 10u; i++) {
		let done1 = abs(e1) >= grad;
		let done2 = abs(e2) >= grad;
		if (done1 && done2) {
			break;
		}
		if (!done1) {
			p1 -= along * steps[i];
			e1 = luma_at(p1) - middle;
		}
		if (!done2) {
			p2 += along * steps[i];
			e2 = luma_at(p2) - middle;
		}
	}
	let d1 = dot(frag - p1, along);
	let d2 = dot(p2 - frag, along);
	// Toward the nearer end the pixel moves across the edge by how near
	// it sits, if that end steps the way the pixel itself does from the
	// seam; else it sits on the edge's far side and keeps its color.
	let near = select(e2, e1, d1 < d2);
	let shift = select(0.0, 0.5 - min(d1, d2) / (d1 + d2), (near < 0.0) != (c < middle));
	// A lone pixel unlike all around it, an edge too short to walk,
	// blends by how far it stands from its neighbors' mean.
	let mean = (2.0 * (n + s + w + e) + nw + ne + sw + se) / 12.0;
	let lone = saturate(abs(mean - c) / range);
	let blend = (3.0 - 2.0 * lone) * lone * lone;
	return scene_at(frag + normal * max(shift, blend * blend * 0.75)).rgb;
}

@fragment
fn occlusion(@builtin(position) frag: vec4f) -> @location(0) vec4f {
	let uv = vec2u(frag.xy);
	let z = textureLoad(depth, uv, 0).r;
	if (z <= 0.0 || view_mode() != 0u || view.light.y <= 0.0) {
		return vec4f(1.0);
	}
	let p = world_at(uv, z);
	return vec4f(openness(p, normal_at(uv, p), bayer(uv) / 16.0));
}

@fragment
fn composite(@builtin(position) frag: vec4f) -> @location(0) vec4f {
	let uv = vec2u(frag.xy);
	var color = textureLoad(scene, uv, 0);
	if (view.fog.z > 0.5 && view_mode() == 0u) {
		color = vec4f(smoothed(frag.xy, color.rgb), color.a);
	}
	let z = textureLoad(depth, uv, 0).r;
	if (z > 0.0 && view_mode() == 0u && view.light.y > 0.0) {
		// The openness over the pixel's 4×4 tile, among the pixels on its
		// own surface: their view depths, near over z, within
		// `SAME_SURFACE` of its own.
		let side = vec2i(textureDimensions(depth)) - vec2i(1);
		let own = 1.0 / z;
		var sum = 0.0;
		var weight = 0.0;
		for (var y = -2; y < 2; y++) {
			for (var x = -2; x < 2; x++) {
				let at = vec2u(clamp(vec2i(uv) + vec2i(x, y), vec2i(0), side));
				let d = textureLoad(depth, at, 0).r;
				if (d <= 0.0 || abs(1.0 / d - own) > SAME_SURFACE * own) {
					continue;
				}
				sum += textureLoad(open, at, 0).r;
				weight += 1.0;
			}
		}
		let ao = select(1.0, sum / weight, weight > 0.0);
		// Darken only the ambient share the world pass left in alpha.
		let shade = 1.0 - (1.0 - ao) * view.light.y * color.a;
		// The scene came sRGB-encoded when the swapchain takes it that
		// way: darken it in linear, or the multiply crushes the darks it
		// lands on.
		var rgb = select(color.rgb, pow(color.rgb, vec3f(2.2)), view.grade.y > 0.5);
		rgb *= shade;
		color = vec4f(select(rgb, pow(rgb, vec3f(1.0 / 2.2)), view.grade.y > 0.5), color.a);
	}
	if (view.fog.x > 0.0 && view_mode() == 0u) {
		// The ray out to the pixel's depth, or past the map for the sky.
		let d = ray_at(uv);
		let len = select(FOG_FAR, view.grade.z / z / dot(d, view.forward.xyz), z > 0.0);
		let hidden = 1.0 - exp(-fog_depth(d, len));
		// Laid over in the tone-mapped color the pixel already holds, so
		// the sky the pass cleared to and the surfaces fog alike.
		let air = aces(fog_glow(d) * view.grade.x);
		var rgb = select(color.rgb, pow(color.rgb, vec3f(2.2)), view.grade.y > 0.5);
		rgb = mix(rgb, air, hidden);
		color = vec4f(select(rgb, pow(rgb, vec3f(1.0 / 2.2)), view.grade.y > 0.5), color.a);
	}
	// Half a byte of interleaved noise either way, so the shading and fog
	// laid over the frame land in bytes without stepping into bands.
	let noise = (fract(52.9829189 * fract(dot(frag.xy, vec2f(0.06711056, 0.00583715)))) - 0.5) / 255.0;
	return vec4f(color.rgb + noise, 1.0);
}
