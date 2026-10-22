// Everything drawn over the traced world: lines and boxes in the world,
// the textured viewmodel, and the flat HUD. World geometry projects like
// the trace's depth — reversed and infinite, near / view depth — so it
// hides behind the map exactly where the map stands in front of it.

struct Camera {
	// xyz eye; w the near plane.
	eye: vec4f,
	// xyz camera right; w tan(fovy / 2) * aspect.
	right: vec4f,
	// xyz camera up; w tan(fovy / 2).
	up: vec4f,
	// xyz view direction; w unused.
	forward: vec4f,
	// xy the viewport in pixels.
	viewport: vec4f,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var image: texture_2d<f32>;
@group(1) @binding(1) var image_sampler: sampler;

struct Vertex {
	@location(0) pos: vec3f,
	@location(1) uv: vec2f,
	@location(2) color: vec4f,
}

struct Varying {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) color: vec4f,
}

@vertex
fn vs_world(v: Vertex) -> Varying {
	let rel = v.pos - camera.eye.xyz;
	let x = dot(rel, camera.right.xyz);
	let y = dot(rel, camera.up.xyz);
	let z = dot(rel, camera.forward.xyz);
	var out: Varying;
	out.clip = vec4f(x / camera.right.w, y / camera.up.w, camera.eye.w, z);
	out.uv = v.uv;
	out.color = v.color;
	return out;
}

// HUD positions are pixels from the top left.
@vertex
fn vs_screen(v: Vertex) -> Varying {
	var out: Varying;
	let size = camera.viewport.xy;
	out.clip = vec4f(v.pos.x / size.x * 2.0 - 1.0, 1.0 - v.pos.y / size.y * 2.0, 0.0, 1.0);
	out.uv = v.uv;
	out.color = v.color;
	return out;
}

@fragment
fn fs_color(in: Varying) -> @location(0) vec4f {
	return in.color;
}

@fragment
fn fs_textured(in: Varying) -> @location(0) vec4f {
	return textureSample(image, image_sampler, in.uv) * in.color;
}

// The HUD's image is the glyph atlas: red is coverage, and one solid texel
// fills rectangles and lines.
@fragment
fn fs_screen(in: Varying) -> @location(0) vec4f {
	let coverage = textureSample(image, image_sampler, in.uv).r;
	return vec4f(in.color.rgb, in.color.a * coverage);
}
