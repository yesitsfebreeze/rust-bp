// One ray per pixel against the map's triangles, drawn as whole voxels
// and lit as they stand. Prepended with bvh.wgsl, material.wgsl and
// voxel.wgsl.
//
// Around the eye lies a grid of world-aligned cells, finer toward the eye
// and coarser away from it — a cell covers about `K` pixels anywhere on
// screen — and a cell is matter exactly where a triangle overlaps it. The
// mesh decides only which cells are full and what color they are; it is
// never drawn. The ray finds its first triangle through the BVH; then the
// grid is walked from just before the hit to the first cell the triangle
// overlaps. That cell is the voxel: a cube in one flat color, drawn whole
// wherever the ray enters it. Grid sizes come in powers of two, nested from the
// world origin so each cell splits into eight of the next size down; each
// cell picks its size from how far its center lies from the eye, so a
// switch of size never splits a cube.
//
// A cube is matter wherever a triangle fills it, so it may stand out past
// the triangle's edge, where a ray misses that triangle: past a mesh's
// silhouette. A second pass through the BVH finds the triangle whose edge
// the ray passes within a cube of before its hit, and walks that one's
// cells too, so the cube shows whole there as well. Nothing is stored per
// voxel ahead of time, so meshes may move and deform freely.
//
// A voxel's look — the mean of its texels times its stored light — lives
// in the voxel field (voxel.wgsl): the first ray to land in a voxel casts
// its shadow ray and stores the result; later rays read it.
//
// The physics playground's spheres are matter the same way: a cell any of
// them overlaps is a cube of the spheres' color, lit by the sun and sky.
//
// Where a sphere struck the map hard it left a dent: a sphere of the map's
// matter carved away. Cells inside a dent hold nothing; the cells lining
// it, behind the surface, are matter — a bowl in the map's own texels, lit
// facing the dent's center, shaded by its rim where the sun cannot reach
// in through the opening.

// The playground's spheres: xyz a center, w its radius, 0 for none.
@group(0) @binding(12) var<storage, read> bodies: array<vec4f>;
// The spheres' matter, as linear albedo.
const BODY: vec3f = vec3f(0.56, 0.38, 0.19);
// The dents: xyz a center, w its radius, 0 past the last.
@group(0) @binding(14) var<storage, read> dents: array<vec4f>;
// Dents one hit weighs at most: those its cells can reach.
const NEAR_DENTS: u32 = 4u;

// How much a cube's box grows for the overlap test, in cube sides: past
// f32's rounding at the map's far corners for the smallest cubes, small
// enough to change no shape. As `SHELL_GROW` in shell.wgsl.
const GROW: f32 = 0.02;

// Cells walked at most from where a grazing ray comes within a cell of
// its triangle's plane.
const WALK: i32 = 256;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
	return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn aces(x: vec3f) -> vec3f {
	return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

// A walk through a grid of cells `s` wide from `lo`, along `d` from `o`:
// the cell it is in, when it crosses the next wall on each axis, when it
// entered the cell it is in, and the outward normal of the cell's face it
// entered through — the face of the cube the ray sees.
struct Walk {
	cell: vec3f,
	t_max: vec3f,
	t_delta: vec3f,
	step: vec3f,
	t: f32,
	face: vec3f,
}

fn walk_from(o: vec3f, d: vec3f, t0: f32, lo: vec3f, s: f32) -> Walk {
	let cell = floor((o + d * t0 - lo) / s);
	let ahead = d >= vec3f(0.0);
	let axisless = abs(d) < vec3f(1e-12);
	let next = lo + select(cell, cell + 1.0, ahead) * s;
	let back = lo + select(cell + 1.0, cell, ahead) * s;
	let t_back = select((back - o) / d, vec3f(-1e30), axisless);
	let step = select(vec3f(-1.0), vec3f(1.0), ahead);
	var axis = 2;
	if (t_back.x >= t_back.y && t_back.x >= t_back.z) {
		axis = 0;
	} else if (t_back.y >= t_back.z) {
		axis = 1;
	}
	var face = vec3f(0.0);
	face[axis] = -step[axis];
	return Walk(
		cell,
		select((next - o) / d, vec3f(1e30), axisless),
		select(s / abs(d), vec3f(1e30), axisless),
		step,
		max(max(t_back.x, max(t_back.y, t_back.z)), 0.0),
		face,
	);
}

// Steps into the next cell.
fn walk_next(w: ptr<function, Walk>) {
	let m = (*w).t_max;
	var axis = 2;
	if (m.x <= m.y && m.x <= m.z) {
		axis = 0;
	} else if (m.y <= m.z) {
		axis = 1;
	}
	(*w).t = m[axis];
	(*w).cell[axis] += (*w).step[axis];
	(*w).t_max[axis] += (*w).t_delta[axis];
	var face = vec3f(0.0);
	face[axis] = -(*w).step[axis];
	(*w).face = face;
}

// The nearest opaque triangle: see-through texels pass the ray on.
fn surface(o: vec3f, d: vec3f) -> Hit {
	var traveled = 0.0;
	for (var i = 0; i < 4; i += 1) {
		let hit = trace(o + d * traveled, d);
		if (hit.tri == NO_HIT) {
			return hit;
		}
		let at = traveled + hit.t;
		if (albedo(tris[hit.tri], o + d * at).a >= 0.5) {
			return Hit(hit.tri, at);
		}
		traveled = at + 1e-3;
	}
	return Hit(NO_HIT, 1e30);
}

// The side of the cubes drawn around `p`: each cell takes the size its
// center wants, `K` pixels wide, unless its parent's center already wants
// it whole. The decision runs down from a few levels above what `p`
// wants, so rays landing near each other pass through the same ancestors
// and settle on the same size; starting just one level up, two rays either
// side of a switch could each stop at their own and draw overlapping cubes.
fn cube_side(p: vec3f) -> f32 {
	let base = view.sky.w;
	var l = wanted_level(length(p - view.eye.xyz)) + 3u;
	loop {
		let s = base * exp2(f32(l));
		let center = (floor(p / s) + 0.5) * s;
		if (l == 0u || wanted_level(length(center - view.eye.xyz)) >= l) {
			return s;
		}
		l -= 1u;
	}
	return base;
}

// Closest approach of the ray from `o` along `d`, up to `far`, and the
// segment `a`..`b`: how far apart, and where along the ray.
fn ray_segment(o: vec3f, d: vec3f, far: f32, a: vec3f, b: vec3f) -> vec2f {
	let e = b - a;
	let r = o - a;
	let ee = dot(e, e);
	let de = dot(d, e);
	let dr = dot(d, r);
	let er = dot(e, r);
	let denom = ee - de * de;
	var t = 0.0;
	if (denom > 1e-12) {
		t = clamp((de * er - ee * dr) / denom, 0.0, far);
	}
	var u = clamp((er + de * t) / max(ee, 1e-12), 0.0, 1.0);
	t = clamp(de * u - dr, 0.0, far);
	u = clamp((er + de * t) / max(ee, 1e-12), 0.0, 1.0);
	return vec2f(length(o + d * t - (a + e * u)), t);
}

// The triangle, other than `skip`, whose edge the ray passes within
// `margin` of nearest the eye before `far`: its index and where along the
// ray, or NO_HIT. Its cubes may stand out past that edge into the ray.
fn graze(o: vec3f, d: vec3f, far: f32, margin: f32, skip: u32) -> Hit {
	var best = Hit(NO_HIT, far);
	let inv = select(vec3f(1e30), vec3f(1.0) / d, abs(d) >= vec3f(1e-12));
	var stack: array<u32, STACK>;
	stack[0] = 0u;
	var top = 1u;
	loop {
		if (top == 0u) {
			break;
		}
		top -= 1u;
		let at = stack[top];
		let node = nodes[at];
		let a = (node.lo - margin - o) * inv;
		let b = (node.hi + margin - o) * inv;
		let near = max(max(min(a.x, b.x), min(a.y, b.y)), max(min(a.z, b.z), 0.0));
		let exit = min(min(max(a.x, b.x), max(a.y, b.y)), min(max(a.z, b.z), best.t));
		if (near > exit) {
			continue;
		}
		if (node.count > 0u) {
			for (var i = node.first; i < node.first + node.count; i += 1u) {
				if (i == skip) {
					continue;
				}
				let t = tris[i];
				let corners = array<vec3f, 3>(t.a.xyz, t.b.xyz, t.c.xyz);
				for (var k = 0; k < 3; k += 1) {
					let close = ray_segment(o, d, best.t, corners[k], corners[(k + 1) % 3]);
					if (close.x <= margin && close.y < best.t) {
						best = Hit(i, close.y);
					}
				}
			}
		} else if (top + 2u <= STACK) {
			stack[top] = node.first;
			stack[top + 1u] = at + 1u;
			top += 2u;
		}
	}
	return best;
}

// The dents a hit's cells can reach, and how much farther than the hit
// its walk must go to find the bottom of a dent the hit lies in.
struct Near {
	dent: array<vec4f, NEAR_DENTS>,
	count: u32,
	deeper: f32,
}

fn near_dents(p: vec3f, margin: f32) -> Near {
	var near: Near;
	for (var i = 0u; i < arrayLength(&dents) && near.count < NEAR_DENTS; i += 1u) {
		let dent = dents[i];
		if (dent.w <= 0.0) {
			break;
		}
		let apart = length(p - dent.xyz);
		if (apart < dent.w + margin) {
			near.dent[near.count] = dent;
			near.count += 1u;
			if (apart < dent.w) {
				near.deeper = max(near.deeper, 2.0 * dent.w);
			}
		}
	}
	return near;
}

// Whether a dent carved the cell at `center` away.
fn carved(near: ptr<function, Near>, center: vec3f) -> bool {
	for (var i = 0u; i < (*near).count; i += 1u) {
		if (length(center - (*near).dent[i].xyz) < (*near).dent[i].w) {
			return true;
		}
	}
	return false;
}

// The dent whose bowl the uncarved cell at `center` lines, cells `s`
// wide, or -1.
fn bowl(near: ptr<function, Near>, center: vec3f, s: f32) -> i32 {
	for (var i = 0u; i < (*near).count; i += 1u) {
		if (length(center - (*near).dent[i].xyz) - (*near).dent[i].w <= s * 0.87) {
			return i32(i);
		}
	}
	return -1;
}

// The light of a bowl's voxel at `center` in `dent`, cut into triangle
// `tri` facing `n`, per unit albedo: lit facing the dent's center. The sun
// reaches it only when its ray leaves the dent through the opening, not
// the wall around.
fn bowl_light(tri: Tri, center: vec3f, n: vec3f, dent: vec4f) -> vec3f {
	let sun = lighting.sun.xyz;
	let oc = center - dent.xyz;
	let b = dot(oc, sun);
	let leaves = center + sun * (sqrt(max(b * b - dot(oc, oc) + dent.w * dent.w, 0.0)) - b);
	var blocked = dot(n, leaves - tri.a.xyz) <= 0.0;
	if (!blocked) {
		blocked = occluded(leaves + n * OFFSET, sun, 1e30);
	}
	return sun_and_sky(normalize(dent.xyz - center), blocked, SHADED);
}

// The first matter cube along the ray: where the ray enters it, its cell,
// the triangle that fills it, and the dent whose bowl it lines, or -1.
struct Cube {
	t: f32,
	cell: vec3f,
	tri: u32,
	dent: i32,
	found: bool,
	// The outward normal of the cube's face the ray enters.
	face: vec3f,
}

// The triangle a filled cube of side `s` at `center` shows as. A face
// fills the cube when it overlaps its box, from behind — the cube's center
// lies strictly on the solid side of the face — on an opaque texel: matter
// only grows inward, no cube stands out past the surface, save past a
// face's edge at a mesh's silhouette, where the face beyond the edge
// fills the cube that hangs off it. That cube shows as the first face it
// hangs in front of, its projection fallen on the face itself, so it
// lights facing out of the matter, not inside out. Among the rest, the
// first in the hierarchy's order owns the cube. Map faces lie on whole
// units, so on cube boundaries: the box grows by more than f32 rounds, so
// a face on a boundary still overlaps, and goes to the cube behind it
// alone, the same one for every pixel. Every ray asking about a cube gets
// the same answer, whatever triangle it hit. NO_HIT for an empty cube.
fn filler(center: vec3f, s: f32) -> u32 {
	let half = (0.5 + GROW) * s;
	let lo = center - half;
	let hi = center + half;
	var behind = NO_HIT;
	var front = NO_HIT;
	var stack: array<u32, STACK>;
	stack[0] = 0u;
	var top = 1u;
	loop {
		if (top == 0u) {
			break;
		}
		top -= 1u;
		let at_node = stack[top];
		let node = nodes[at_node];
		if (any(node.lo > hi) || any(node.hi < lo)) {
			continue;
		}
		if (node.count > 0u) {
			for (var i = node.first; i < node.first + node.count; i += 1u) {
				let tri = tris[i];
				if (!overlaps_box(tri, center, half) || albedo(tri, center).a < 0.5) {
					continue;
				}
				let off = dot(tri_normal(tri), center - tri.a.xyz);
				if (off < 0.0) {
					// Behind the face, inside the matter: the first in
					// the hierarchy's order fills the cube.
					if (behind == NO_HIT) {
						behind = i;
					}
				} else if (front == NO_HIT) {
					// In front of the face, its projection falls on the
					// face itself: its distance to the triangle is its
					// distance to the plane, not past an edge.
					if (length(closest_point(center, tri) - center) - off <= 1e-3 * s) {
						front = i;
					}
				}
			}
		} else if (top + 2u <= STACK) {
			stack[top] = node.first;
			stack[top + 1u] = at_node + 1u;
			top += 2u;
		}
	}
	// A cube hanging in front of a face is matter only where a face
	// behind it fills it.
	if (behind == NO_HIT) {
		return NO_HIT;
	}
	return select(behind, front, front != NO_HIT);
}

// Walks the grid of cells `s` wide from `start` to `end` along the ray:
// the first cube any triangle fills, or that lines a dent's bowl behind
// `tri_index`, the triangle the ray hit.
fn first_cube(o: vec3f, d: vec3f, tri_index: u32, start: f32, end: f32, s: f32, near: ptr<function, Near>) -> Cube {
	let tri = tris[tri_index];
	let n = facing(tri, d);
	let reach = 0.5 * s * (abs(n.x) + abs(n.y) + abs(n.z));
	var w = walk_from(o, d, max(start, 0.0), vec3f(0.0), s);
	for (var i = 0; i < WALK; i += 1) {
		let center = (w.cell + 0.5) * s;
		if (!carved(near, center)) {
			let fill = filler(center, s);
			if (fill != NO_HIT) {
				return Cube(w.t, w.cell, fill, -1, true, w.face);
			}
			// A dent's bowl lines it behind the surface.
			if (dot(n, center - tri.a.xyz) <= reach) {
				let dent = bowl(near, center, s);
				if (dent >= 0) {
					return Cube(w.t, w.cell, tri_index, dent, true, w.face);
				}
			}
		}
		if (min(w.t_max.x, min(w.t_max.y, w.t_max.z)) > end) {
			break;
		}
		walk_next(&w);
	}
	return Cube(end, vec3f(0.0), tri_index, -1, false, vec3f(0.0));
}

// A matter cube's look, one color whichever ray reaches it: stored in the
// voxel field, or worked out by the first ray to land in it from the
// plane that cube shows — the mean of its texels under the light on that
// plane, its own normal, never one turned toward the eye, or for a
// bowl's cube facing its dent's center — and stored. When the table has
// no room, it is worked out on the spot.
fn cube_color(cube: Cube, s: f32, near: ptr<function, Near>) -> vec3f {
	let stored = stored_color(voxel_key(cube.cell, s));
	if (stored.found) {
		return stored.color;
	}
	let center = (cube.cell + 0.5) * s;
	let tri = tris[cube.tri];
	let n = tri_normal(tri);
	var light = vec3f(0.0);
	if (cube.dent >= 0) {
		light = bowl_light(tri, center, n, (*near).dent[cube.dent]);
	} else {
		// The cube's light from the plane that fills it: its face's baked
		// chart, else the live sun and sky for what the bake left out.
		let p = center - n * dot(n, center - tri.a.xyz);
		light = select(global_light(p, n, SHADED), baked_light_at(tri, p), is_baked(tri));
	}
	let color = light * voxel_albedo(tri, center, n, s);
	store_color(stored, color);
	return color;
}

// The nearest playground sphere's cube along the ray before `far`: a cell
// of side `s` the sphere overlaps, where the ray enters it.
struct Body {
	t: f32,
	cell: vec3f,
	sphere: vec4f,
	found: bool,
	// The outward normal of the cube's face the ray enters.
	face: vec3f,
}

fn body(o: vec3f, d: vec3f, far: f32) -> Body {
	var best = Body(far, vec3f(0.0), vec4f(0.0), false, vec3f(0.0));
	for (var i = 0u; i < arrayLength(&bodies); i += 1u) {
		let sphere = bodies[i];
		if (sphere.w <= 0.0) {
			continue;
		}
		let s = cube_side(sphere.xyz);
		// The cubes reach past the sphere by up to a cube's diagonal.
		let r = sphere.w + s * 1.74;
		let oc = o - sphere.xyz;
		let half = dot(oc, d);
		let gap = half * half - (dot(oc, oc) - r * r);
		if (gap <= 0.0) {
			continue;
		}
		let enter = max(-half - sqrt(gap), 0.0);
		let leave = -half + sqrt(gap);
		if (leave < 0.0 || enter >= best.t) {
			continue;
		}
		var w = walk_from(o, d, enter, vec3f(0.0), s);
		for (var k = 0; k < WALK; k += 1) {
			let lo = w.cell * s;
			if (w.t >= best.t) {
				break;
			}
			if (length(clamp(sphere.xyz, lo, lo + s) - sphere.xyz) < sphere.w) {
				best = Body(w.t, w.cell, vec4f(sphere.xyz, s), true, w.face);
				break;
			}
			if (min(w.t_max.x, min(w.t_max.y, w.t_max.z)) > leave) {
				break;
			}
			walk_next(&w);
		}
	}
	return best;
}

// A sphere's cube: flat matter, lit facing out of the sphere from its
// center. `hit.sphere.w` holds the cube's side.
fn shade_body(hit: Body) -> vec3f {
	let center = (hit.cell + 0.5) * hit.sphere.w;
	let out = center - hit.sphere.xyz;
	let n = select(vec3f(0.0, 1.0, 0.0), normalize(out), length(out) > 1e-6);
	return BODY * global_light(center, n, SHADED);
}

// What a ray sees, and how far away: the depth the overlays test against.
// `lit` colors are linear radiance, tone mapped for the screen; the debug
// views' are shown as they are.
struct Seen {
	color: vec3f,
	t: f32,
	lit: bool,
}

// The debug views F5 cycles through, after the lit one: as `VIEW_MODES` in
// mod.rs.
const VIEW_DEPTH: u32 = 1u;
const VIEW_NORMAL: u32 = 2u;
const VIEW_ALBEDO: u32 = 3u;
const VIEW_CUBES: u32 = 4u;
const VIEW_SIZE: u32 = 5u;
// The depth view's far end, world units: about the widest map.
const DEPTH_RANGE: f32 = 4096.0;

fn view_mode() -> u32 {
	return u32(view.grade.w);
}

// A color per value, spread around the hue circle.
fn hue(x: f32) -> vec3f {
	return clamp(abs(fract(x + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

// One debug view of the cube in `cell` of side `s` the ray entered `t`
// along `d` through its face facing `face`, its albedo `albedo`: depth,
// white near to black far, logarithmic so near and far both read; the
// normal of the cube's face it sees, always along a world axis; the albedo;
// each cube its own color; each size its own color.
fn inspect(d: vec3f, t: f32, cell: vec3f, s: f32, face: vec3f, albedo: vec3f) -> Seen {
	switch view_mode() {
		case VIEW_DEPTH: {
			let z = t * dot(d, view.forward.xyz);
			return Seen(vec3f(1.0 - clamp(log2(1.0 + z) / log2(1.0 + DEPTH_RANGE), 0.0, 1.0)), t, false);
		}
		case VIEW_NORMAL: {
			return Seen(face * 0.5 + 0.5, t, false);
		}
		case VIEW_ALBEDO: {
			return Seen(albedo, t, true);
		}
		case VIEW_CUBES: {
			let c = vec3i(cell);
			let h = mix_hash(mix_hash(mix_hash(mix_hash(0x2545F491u, u32(c.x)), u32(c.y)), u32(c.z)), bitcast<u32>(s));
			return Seen(hue(f32(h & 0xFFFFu) / 65536.0) * (0.5 + 0.5 * f32((h >> 16u) & 0xFFu) / 255.0), t, false);
		}
		default: {
			return Seen(hue(log2(s / view.sky.w) / 7.0) * 0.9, t, false);
		}
	}
}

// The map along the ray as whole cubes: the hit triangle's, or a cube
// standing out past a grazed triangle's edge in front of it.
fn radiance(o: vec3f, d: vec3f) -> Seen {
	var map = Cube(1e30, vec3f(0.0), NO_HIT, -1, false, vec3f(0.0));
	var s = view.sky.w;
	var near: Near;
	// A hit no cube claims — its cell carved away, or its texel too dark to
	// rise into it — draws nothing of the mesh: the ray goes on past it.
	var beyond = 0.0;
	for (var tries = 0; tries < 3 && !map.found; tries += 1) {
		var hit = surface(o + d * beyond, d);
		if (hit.tri == NO_HIT) {
			break;
		}
		hit.t += beyond;
		let p = o + d * hit.t;
		s = cube_side(p);
		let tri = tris[hit.tri];
		let n = facing(tri, d);
		let reach = 0.5 * s * (abs(n.x) + abs(n.y) + abs(n.z));
		// Walk the grid from where the ray comes within reach of the cubes
		// the triangle fills: a cube's center lies within `reach` of the
		// plane and its box `reach` farther, so twice that. Starting any
		// later begins inside those cubes and meets the next one through
		// its side, not its face toward the eye. Only a walk too long for
		// its cells starts closer, where it can still reach past the hit.
		let layer = min(2.0 * reach / max(abs(dot(d, n)), 1e-3), f32(WALK) * s * 0.25);
		near = near_dents(p, layer + s * 2.0);
		let to = hit.t + layer + s * 2.0 + near.deeper;
		var start = hit.t - layer;
		// A neighbor's cube may stand in front of the hit's, past that
		// neighbor's edge: up to a whole cube's diagonal from the ray. The
		// walk then starts early enough to meet it, within the neighbor's
		// own layer.
		let margin = s * 1.74;
		let edge = graze(o, d, hit.t, margin, hit.tri);
		if (edge.tri != NO_HIT) {
			let ng = facing(tris[edge.tri], d);
			let reach_g = 0.5 * s * (abs(ng.x) + abs(ng.y) + abs(ng.z));
			let layer_g = min(2.0 * reach_g / max(abs(dot(d, ng)), 1e-3), f32(WALK) * s * 0.25);
			start = min(start, edge.t - max(layer_g, margin * 2.0));
		}
		map = first_cube(o, d, hit.tri, start, to, s, &near);
		beyond = to;
	}
	let ball = body(o, d, select(1e30, map.t, map.found));
	let debug = view_mode() != 0u;
	if (ball.found) {
		if (debug) {
			return inspect(d, ball.t, ball.cell, ball.sphere.w, ball.face, BODY);
		}
		return Seen(shade_body(ball), ball.t, true);
	}
	if (!map.found) {
		if (debug && view_mode() != VIEW_ALBEDO) {
			return Seen(vec3f(0.0), 1e30, false);
		}
		return Seen(view.sky.rgb, 1e30, true);
	}
	if (debug) {
		let tri = tris[map.tri];
		let n = facing(tri, d);
		let center = (map.cell + 0.5) * s;
		return inspect(d, map.t, map.cell, s, map.face, voxel_albedo(tri, center, n, s));
	}
	return Seen(cube_color(map, s, &near), map.t, true);
}

struct Pixel {
	@location(0) color: vec4f,
	// Reversed and infinite, as the overlays draw: near / view depth.
	@builtin(frag_depth) depth: f32,
}

// The ray through `at` on screen, in pixels: what it shows, as linear
// light on the screen — tone mapped when lit — and its view depth, reversed
// as the overlays draw.
fn sample(at: vec2f) -> vec4f {
	let ndc = vec2f(at.x / view.size.x * 2.0 - 1.0, 1.0 - at.y / view.size.y * 2.0);
	let d = normalize(view.forward.xyz + view.right.xyz * (ndc.x * view.right.w) + view.up.xyz * (ndc.y * view.up.w));
	let seen = radiance(view.eye.xyz, d);
	var color = seen.color;
	if (seen.lit) {
		if (view_mode() == 0u) {
			color = aces(color * view.grade.x);
		}
	} else {
		// The debug views' values are as shown: back to light, so they
		// average and encode like the rest.
		color = pow(color, vec3f(2.2));
	}
	var depth = 0.0;
	if (seen.t < 1e29) {
		depth = clamp(view.grade.z / max(seen.t * dot(d, view.forward.xyz), 1e-6), 0.0, 1.0);
	}
	return vec4f(color, depth);
}

// Where in the pixel each ray goes, for 2 and 4 rays: a diagonal pair, and
// a rotated grid, so edges at any angle land between samples.
const PAIR: array<vec2f, 2> = array<vec2f, 2>(vec2f(-0.25, -0.25), vec2f(0.25, 0.25));
const GRID: array<vec2f, 4> = array<vec2f, 4>(
	vec2f(-0.125, -0.375),
	vec2f(0.375, -0.125),
	vec2f(0.125, 0.375),
	vec2f(-0.375, 0.125),
);

@fragment
fn fs(@builtin(position) at: vec4f) -> Pixel {
	// Rays a pixel averages: one through its center, or `view.size.z`
	// spread across it, so cubes smaller than the gaps between pixels and
	// their edges blend instead of alias.
	let rays = u32(view.size.z);
	var sum = vec4f(0.0);
	var depth = 0.0;
	var count = 1u;
	if (rays >= 4u) {
		count = 4u;
		for (var i = 0u; i < 4u; i += 1u) {
			let one = sample(at.xy + GRID[i]);
			sum += one;
			depth = max(depth, one.w);
		}
	} else if (rays >= 2u) {
		count = 2u;
		for (var i = 0u; i < 2u; i += 1u) {
			let one = sample(at.xy + PAIR[i]);
			sum += one;
			depth = max(depth, one.w);
		}
	} else {
		sum = sample(at.xy);
		depth = sum.w;
	}
	var color = sum.rgb / f32(count);
	if (view.grade.y > 0.5) {
		color = pow(color, vec3f(1.0 / 2.2));
	}
	// The nearest of the rays: overlays hide behind any matter in the pixel.
	return Pixel(vec4f(color, 1.0), depth);
}
