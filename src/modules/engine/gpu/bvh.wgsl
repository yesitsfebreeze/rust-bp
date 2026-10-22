// The map's triangles and their hierarchy, shared by the one pass that
// casts rays: the view's trace. Bindings 1–2 of group 0 in the pipeline
// that includes it. The triangles sit in the hierarchy's leaf order, so a
// leaf's `first` indexes them directly.

struct Tri {
	// Corners; their w together are the unit normal.
	a: vec4f,
	b: vec4f,
	c: vec4f,
	// Texture tiling coordinates: dot(xyz, p) + w, in whole textures.
	su: vec4f,
	sv: vec4f,
	// The face's baked light mapping, shared by its whole fan: lm_u(p) =
	// dot(xyz, p) + w, in its chart's texels, and lm_v likewise.
	lmu: vec4f,
	lmv: vec4f,
	// Albedo atlas rectangle: x, y, w, h.
	tex: vec4u,
	// The face's baked chart in the light atlas: x, y, w, h; w 0 for none.
	lmr: vec4u,
}

struct Node {
	lo: vec3f,
	// Leaf: its first triangle. Inner: its far child; the near one follows
	// it directly.
	first: u32,
	hi: vec3f,
	count: u32,
}

@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;

const NO_HIT: u32 = 0xFFFFFFFFu;
// One entry a level: physics's `bvh::MAX_DEPTH` plus one.
const STACK: u32 = 32u;
const MISS: f32 = 3.4e38;

struct Hit {
	tri: u32,
	t: f32,
}

// Where the ray enters node `at`'s box before `far`; MISS when it does not.
fn enter_node(o: vec3f, inv: vec3f, at: u32, far: f32) -> f32 {
	let node = nodes[at];
	let a = (node.lo - o) * inv;
	let b = (node.hi - o) * inv;
	let near = max(max(min(a.x, b.x), min(a.y, b.y)), max(min(a.z, b.z), 0.0));
	let exit = min(min(max(a.x, b.x), max(a.y, b.y)), min(max(a.z, b.z), far));
	return select(MISS, near, near <= exit);
}

fn hit_tri(o: vec3f, d: vec3f, i: u32) -> f32 {
	let a = tris[i].a.xyz;
	let e1 = tris[i].b.xyz - a;
	let e2 = tris[i].c.xyz - a;
	let p = cross(d, e2);
	let det = dot(e1, p);
	if (abs(det) < 1e-9) {
		return -1.0;
	}
	let inv = 1.0 / det;
	let s = o - a;
	let u = dot(s, p) * inv;
	if (u < 0.0 || u > 1.0) {
		return -1.0;
	}
	let q = cross(s, e1);
	let v = dot(d, q) * inv;
	if (v < 0.0 || u + v > 1.0) {
		return -1.0;
	}
	let at = dot(e2, q) * inv;
	return select(-1.0, at, at > 1e-4);
}

// The nearest triangle along the ray before `far`, two-sided; with `any`,
// the first found instead. Children are visited nearer first, and a
// postponed one is dropped once a hit lies before it.
fn sweep(o: vec3f, d: vec3f, far: f32, any: bool) -> Hit {
	let inv = select(vec3f(1e30), vec3f(1.0) / d, abs(d) >= vec3f(1e-12));
	var hit = Hit(NO_HIT, far);
	if (enter_node(o, inv, 0u, far) == MISS) {
		return hit;
	}
	var stack: array<u32, STACK>;
	var entry: array<f32, STACK>;
	var top = 0u;
	var at = 0u;
	loop {
		let node = nodes[at];
		if (node.count > 0u) {
			for (var i = node.first; i < node.first + node.count; i += 1u) {
				let t = hit_tri(o, d, i);
				if (t >= 0.0 && t < hit.t) {
					hit = Hit(i, t);
					if (any) {
						return hit;
					}
				}
			}
		} else {
			let a = at + 1u;
			let b = node.first;
			let ta = enter_node(o, inv, a, hit.t);
			let tb = enter_node(o, inv, b, hit.t);
			if (ta != MISS && tb != MISS) {
				let a_first = ta <= tb;
				at = select(b, a, a_first);
				if (top < STACK) {
					stack[top] = select(a, b, a_first);
					entry[top] = max(ta, tb);
					top += 1u;
				}
				continue;
			}
			if (ta != MISS) {
				at = a;
				continue;
			}
			if (tb != MISS) {
				at = b;
				continue;
			}
		}
		// The next postponed node the ray still reaches before its hit.
		loop {
			if (top == 0u) {
				return hit;
			}
			top -= 1u;
			if (entry[top] <= hit.t) {
				break;
			}
		}
		at = stack[top];
	}
	return hit;
}

// The nearest triangle along the ray, two-sided.
fn trace(o: vec3f, d: vec3f) -> Hit {
	return sweep(o, d, 1e30, false);
}

// The nearest triangle along the ray before `far`; NO_HIT when none.
fn trace_to(o: vec3f, d: vec3f, far: f32) -> Hit {
	return sweep(o, d, far, false);
}

// Whether any triangle stands on the ray before `far`.
fn occluded(o: vec3f, d: vec3f, far: f32) -> bool {
	return sweep(o, d, far, true).tri != NO_HIT;
}

// The outward normal of a triangle: BSP faces wind so it faces open space.
fn tri_normal(t: Tri) -> vec3f {
	return vec3f(t.a.w, t.b.w, t.c.w);
}

// Whether triangle `t` overlaps the box `half` either way of `center`:
// no separating axis among the box's faces, the triangle's plane, and the
// crossings of their edges (Akenine-Möller).
fn overlaps_box(t: Tri, center: vec3f, half: f32) -> bool {
	let v = array<vec3f, 3>(t.a.xyz - center, t.b.xyz - center, t.c.xyz - center);
	if (any(min(min(v[0], v[1]), v[2]) > vec3f(half)) || any(max(max(v[0], v[1]), v[2]) < vec3f(-half))) {
		return false;
	}
	let n = cross(v[1] - v[0], v[2] - v[0]);
	if (abs(dot(n, v[0])) > half * (abs(n.x) + abs(n.y) + abs(n.z))) {
		return false;
	}
	for (var i = 0; i < 3; i += 1) {
		let e = v[(i + 1) % 3] - v[i];
		for (var k = 0; k < 3; k += 1) {
			var unit = vec3f(0.0);
			unit[k] = 1.0;
			let axis = cross(unit, e);
			let p = vec3f(dot(axis, v[0]), dot(axis, v[1]), dot(axis, v[2]));
			let r = half * (abs(axis.x) + abs(axis.y) + abs(axis.z));
			if (min(p.x, min(p.y, p.z)) > r || max(p.x, max(p.y, p.z)) < -r) {
				return false;
			}
		}
	}
	return true;
}

fn closest_point(p: vec3f, t: Tri) -> vec3f {
	let a = t.a.xyz;
	let ab = t.b.xyz - a;
	let ac = t.c.xyz - a;
	let ap = p - a;
	let d1 = dot(ab, ap);
	let d2 = dot(ac, ap);
	if (d1 <= 0.0 && d2 <= 0.0) {
		return a;
	}
	let bp = p - t.b.xyz;
	let d3 = dot(ab, bp);
	let d4 = dot(ac, bp);
	if (d3 >= 0.0 && d4 <= d3) {
		return t.b.xyz;
	}
	let vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
		return a + ab * (d1 / (d1 - d3));
	}
	let cp = p - t.c.xyz;
	let d5 = dot(ab, cp);
	let d6 = dot(ac, cp);
	if (d6 >= 0.0 && d5 <= d6) {
		return t.c.xyz;
	}
	let vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
		return a + ac * (d2 / (d2 - d6));
	}
	let va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
		return t.b.xyz + (t.c.xyz - t.b.xyz) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
	}
	let denom = 1.0 / (va + vb + vc);
	return a + ab * (vb * denom) + ac * (vc * denom);
}
