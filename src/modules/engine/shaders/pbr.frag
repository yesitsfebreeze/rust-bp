#version 100
// Metallic-roughness PBR with subsurface scattering, one sun plus a sky/ground
// hemisphere. Specular is GGX with a height-correlated Smith term and Schlick
// Fresnel. Subsurface light has two parts, both tinted by `ScatterColor` and
// weighted by the material's subsurface map:
// - diffuse wrapped past the terminator, as light diffusing under the surface;
// - transmission from the sun through thin parts toward the eye.
// Output is ACES-tone-mapped and sRGB-encoded, as there is no post pass.
//
// All three maps come from one atlas bound to the mesh's `Texture` slot:
// albedo, tangent-space normal and ORM tiles stacked vertically, each tile
// `TILE` texels inside a one-texel duplicated border (see textures.rs). The
// per-vertex `uv` tiles the atlas by whole world units; `fract` picks the
// spot inside the tile and the border keeps filtering inside one tile. There
// are no mipmaps: they would average neighboring tiles.
precision highp float;

varying highp vec3 world;
varying highp vec2 uv;
varying mediump vec3 v_normal;
varying mediump vec3 v_tangent;
varying mediump float v_handedness;

// sRGB base color, tangent-space normal `n * 0.5 + 0.5`, and
// R occlusion, G roughness, B metallic, A subsurface amount, per atlas tile.
uniform sampler2D Texture;

uniform vec3 CameraPosition;
// Toward the sun, normalized.
uniform vec3 SunDirection;
uniform vec3 SunColor;
uniform vec3 SkyColor;
uniform vec3 GroundColor;
// rgb: color light takes on inside the material; a: overall subsurface strength.
uniform vec4 ScatterColor;
// x: diffuse wrap, y: normal distortion of transmitted light, z: transmission
// falloff power, w: transmission scale.
uniform vec4 Scatter;
uniform float Exposure;

// Atlas geometry: tiles of 256 texels with a 1 texel border each side.
const float TILE = 256.0;
const float PITCH = 258.0;
const float TILES = 3.0;

const float PI = 3.14159265;

// Sample atlas tile `tile` (0 albedo, 1 normal, 2 ORM) at the tiled `uv`.
vec4 tile(int tile, vec2 uv) {
	vec2 f = fract(uv) * TILE + 1.0;
	return texture2D(Texture, vec2(f.x / PITCH, (float(tile) * PITCH + f.y) / (TILES * PITCH)));
}

vec3 to_linear(vec3 c) {
	return pow(c, vec3(2.2));
}

float d_ggx(float n_h, float a) {
	float a2 = a * a;
	float d = n_h * n_h * (a2 - 1.0) + 1.0;
	return a2 / (PI * d * d);
}

// Height-correlated Smith visibility, including the 1 / (4 n·l n·v) of the BRDF.
float v_smith(float n_v, float n_l, float a) {
	float a2 = a * a;
	float gv = n_l * sqrt(n_v * n_v * (1.0 - a2) + a2);
	float gl = n_v * sqrt(n_l * n_l * (1.0 - a2) + a2);
	return 0.5 / max(gv + gl, 1e-5);
}

vec3 f_schlick(float v_h, vec3 f0) {
	return f0 + (1.0 - f0) * pow(1.0 - v_h, 5.0);
}

// Karis' analytic fit of the split-sum environment BRDF.
vec3 env_brdf(vec3 f0, float roughness, float n_v) {
	vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
	vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
	vec4 r = roughness * c0 + c1;
	float a004 = min(r.x * r.x, exp2(-9.28 * n_v)) * r.x + r.y;
	vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
	return f0 * ab.x + ab.y;
}

vec3 aces(vec3 x) {
	return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
	vec3 albedo = to_linear(tile(0, uv).rgb);
	vec4 orm = tile(2, uv);
	float occlusion = orm.r;
	float roughness = clamp(orm.g, 0.045, 1.0);
	float metallic = orm.b;
	float subsurface = orm.a * ScatterColor.a;

	vec3 ng = normalize(v_normal);
	vec3 t = normalize(v_tangent - ng * dot(ng, v_tangent));
	vec3 b = cross(ng, t) * v_handedness;
	vec3 tn = tile(1, uv).xyz * 2.0 - 1.0;
	vec3 n = normalize(t * tn.x + b * tn.y + ng * tn.z);

	vec3 v = normalize(CameraPosition - world);
	vec3 l = SunDirection;
	vec3 h = normalize(l + v);
	float n_v = max(dot(n, v), 1e-4);
	float n_l_raw = dot(n, l);
	float n_l = max(n_l_raw, 0.0);
	float n_h = max(dot(n, h), 0.0);
	float v_h = max(dot(v, h), 0.0);
	float a = roughness * roughness;

	vec3 f0 = mix(vec3(0.04), albedo, metallic);
	vec3 f = f_schlick(v_h, f0);
	vec3 specular = d_ggx(n_h, a) * v_smith(n_v, n_l, a) * f;
	vec3 kd = (1.0 - f) * (1.0 - metallic);

	// Diffuse: Lambert, blended toward wrapped light tinted by the scatter color.
	float wrap = Scatter.x;
	float wrapped = max((n_l_raw + wrap) / ((1.0 + wrap) * (1.0 + wrap)), 0.0);
	vec3 diffuse_term = mix(vec3(n_l), wrapped * ScatterColor.rgb, subsurface);
	vec3 diffuse = kd * albedo / PI * diffuse_term;

	// Transmission (Barré-Brisebois and Bouchard): sun light leaving the far side,
	// strongest looking toward the sun, bent by the surface normal.
	vec3 lt = normalize(l + n * Scatter.y);
	float through = pow(clamp(dot(v, -lt), 0.0, 1.0), Scatter.z) * Scatter.w;
	vec3 transmitted = (1.0 - metallic) * subsurface * through * albedo * ScatterColor.rgb;

	vec3 sun = (diffuse + specular * n_l + transmitted) * SunColor;

	// Hemisphere ambient: diffuse by the normal, specular by the reflection.
	vec3 r = reflect(-v, n);
	vec3 sky_n = mix(GroundColor, SkyColor, n.y * 0.5 + 0.5);
	vec3 sky_r = mix(GroundColor, SkyColor, smoothstep(-0.2, 0.2, r.y));
	vec3 ambient_diffuse = kd * albedo * sky_n * mix(vec3(1.0), vec3(1.0) + 0.5 * ScatterColor.rgb, subsurface);
	vec3 ambient = (ambient_diffuse + env_brdf(f0, roughness, n_v) * sky_r) * occlusion;

	vec3 color = aces((sun + ambient) * Exposure);
	gl_FragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}