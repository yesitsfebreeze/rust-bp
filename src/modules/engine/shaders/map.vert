#version 100
// World position and the per-vertex map data for map.frag. The renderer packs
// the albedo atlas rectangle in `color0` as bytes over 255: x/8, y/8, w/8, h/8
// texels of the atlas, and the lightmap coordinates in `normal`: xy in texels,
// z the block's packed origin x + y * 4096, w its packed size w + h * 256.
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
attribute vec4 normal;

uniform mat4 Model;
uniform mat4 Projection;

varying highp vec2 uv;
varying highp vec4 v_rect;
varying highp vec2 v_lm;
varying highp vec4 v_lmrect;

void main() {
	uv = texcoord;
	v_rect = vec4(color0.xy, color0.zw) * 255.0 * 8.0;
	v_lm = normal.xy;
	v_lmrect = vec4(mod(normal.z, 4096.0), floor(normal.z / 4096.0),
		mod(normal.w, 256.0), floor(normal.w / 256.0));
	gl_Position = Projection * (Model * vec4(position, 1.0));
}