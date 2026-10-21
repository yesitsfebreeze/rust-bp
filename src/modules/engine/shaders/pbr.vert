#version 100
// World-space position, UV and tangent frame for pbr.frag. The renderer packs
// the face normal in `normal.xyz`, the bitangent's handedness in `normal.w` and
// the tangent in `color0` as `t * 0.5 + 0.5`.
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
attribute vec4 normal;

uniform mat4 Model;
uniform mat4 Projection;

varying highp vec3 world;
varying highp vec2 uv;
varying mediump vec3 v_normal;
varying mediump vec3 v_tangent;
varying mediump float v_handedness;

void main() {
	vec4 w = Model * vec4(position, 1.0);
	world = w.xyz;
	uv = texcoord;
	v_normal = (Model * vec4(normal.xyz, 0.0)).xyz;
	v_tangent = (Model * vec4(color0.xyz * 2.0 - 1.0, 0.0)).xyz;
	v_handedness = normal.w;
	gl_Position = Projection * w;
}
