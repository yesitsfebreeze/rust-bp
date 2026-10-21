#version 100
// BSP map shading: one material for the whole map, two samplers. The albedo
// atlas rides the mesh's `Texture` slot; every vertex carries its texture's
// atlas rectangle in `color0` and tiling UVs, so one atlas serves every face.
// The map's own baked lightmaps ride the `Lightmap` slot, one block per face;
// the vertices carry the block's rectangle and texel coordinates. Nearest
// sampling picks exact albedo texels, so no padding is needed; one unused
// texel around each lightmap block keeps linear filtering inside one face.
// GoldSrc stores light at double brightness for its old gamma ramp; halving
// here would only wash the map out. Output is ACES-tone-mapped and
// sRGB-encoded like pbr.frag, and `Exposure` is the same cvar as the arena's.
precision highp float;

varying highp vec2 uv;
varying highp vec4 v_rect;
varying highp vec2 v_lm;
varying highp vec4 v_lmrect;

uniform sampler2D Texture;
uniform sampler2D Lightmap;
uniform float LmSide;
uniform float Exposure;

const float ALBEDO_SIDE = 2048.0;

vec3 aces(vec3 x) {
	return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
	// Albedo: exact texel of the face's texture, tiled by whole textures.
	vec2 t = floor(fract(uv) * v_rect.zw);
	vec2 at = (v_rect.xy + t + 0.5) / ALBEDO_SIDE;
	vec4 base = texture2D(Texture, at);
	if (base.a < 0.5) {
		discard;
	}
	// Lightmap: the face's block, clamped so edges never read a neighbor.
	vec2 lm = (v_lmrect.xy + clamp(v_lm, vec2(0.0), v_lmrect.zw - 1.0) + 0.5) / LmSide;
	vec3 light = texture2D(Lightmap, lm).rgb * 2.0;

	vec3 color = aces(pow(base.rgb, vec3(2.2)) * light * Exposure);
	gl_FragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}