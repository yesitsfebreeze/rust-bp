// The shell field: the map voxelized ahead of the view into one sparse
// shell per cube size, and the full-screen pass that marches it instead of
// the trace (trace.wgsl). Prepended with bvh.wgsl, material.wgsl and
// voxel.wgsl.
//
// The trace tests each ray against the triangle it hit, so a cube another
// triangle fills can be missed: see-through corners, cubes that look
// inside out. Here every cell is voxelized ahead of time against every
// triangle that reaches it, and the ray marches the cells themselves.
//
// A cube `s` wide is only drawn between about R and 2R from the eye, with
// R = s / (pixel angle · K): nearer, it splits into eight; farther, its
// parent is drawn whole. So each size keeps only that shell, padded by how
// far the eye may move before the shell is brought along, and in it only
// the cells a surface fills: a hash of bricks, 4³ cells as 64 bits. Over
// it a dense pyramid of bits marks which blocks of 16³, 64³ and 256³
// cells hold any, so a ray crosses empty space the largest empty block at
// a time with one load per test, and stops where it leaves the map's
// bounds. As the eye moves a shell takes in only the cells
// that entered it; every few moves, or when the dents change, it is built
// again whole and drops the cells it left behind.
//
// A cell is filled by the rule the trace walks by: a triangle overlaps
// the cube, its texel at the cube's center is opaque, and no dent carved
// the cube away. Cells lining a dent's bowl behind the surface fill too. A cube's
// color lives in the voxel field (voxel.wgsl), keyed as the trace keys it.

// Shells, one per cube size from the smallest voxel up, as `SHELLS` in
// shell.rs.
const SHELLS: u32 = 12u;
// Entries one shell's table holds, a power of two, as `CAPACITY` in
// shell.rs.
const CAPACITY: u32 = 524288u;
// Entries a lookup tries before it gives up.
const SHELL_PROBES: u32 = 32u;
// Bricks a shell's keys count along each axis from its origin.
const KEY_SPAN: i32 = 1024;
// Words of one shell's pyramid: blocks of 16, 64 and 256 cells across the
// 4096 its keys reach, a bit each, finest first; as `PYRAMID_WORDS` in
// shell.rs.
const BLOCKS_16: u32 = 524288u;
const BLOCKS_64: u32 = 8192u;
const PYRAMID_WORDS: u32 = 532608u;
const NO_ENTRY: u32 = 0xFFFFFFFFu;
// The triangles one dispatch dimension holds; the workgroups that share
// one triangle's cells, as `ROW` and `SLICES` in shell.rs.
const ROW: u32 = 65535u;
const SLICES: u32 = 16u;
// Cells and bricks a ray visits at most.
const MARCH: u32 = 1024u;
// The spheres' matter, as linear albedo, as the trace draws it.
const BALL: vec3f = vec3f(0.56, 0.38, 0.19);
// How far a cube's box is nudged down, in sides, as `OWN` in trace.wgsl:
// it owns its low faces but not its high, so a map face on a cell
// boundary lands in exactly one cell.
const SHELL_OWN: f32 = 0.02;

// One brick's cells, 64 bits.
struct Entry {
	// 0 empty; bit 0 set when taken, then the brick 10 bits an axis from
	// its shell's origin.
	key: atomic<u32>,
	lo: atomic<u32>,
	hi: atomic<u32>,
	pad: u32,
}

struct Shells {
	// xyz the brick a shell's keys count from, a multiple of 64, so its
	// blocks count from a whole one; w 1 once built.
	origin: array<vec4i, SHELLS>,
}

// What one shell's dispatches take in.
struct Work {
	// xyz the eye the shell is brought to; w its cells' side.
	eye: vec4f,
	// xyz the eye it last took cells in for; w 1 when only the cells new
	// since then go in.
	last: vec4f,
	// x, y the shell's inner and outer radius about either eye.
	range: vec4f,
	// x the shell.
	shell: vec4u,
}

// The playground's spheres and the dents they left: xyz a center, w its
// radius, 0 for none.
@group(0) @binding(12) var<storage, read> bodies: array<vec4f>;
@group(0) @binding(14) var<storage, read> dents: array<vec4f>;
@group(0) @binding(15) var<storage, read_write> table: array<Entry>;
@group(0) @binding(16) var<uniform> shells: Shells;
@group(0) @binding(17) var<uniform> work: Work;
// Per shell, the pyramid of blocks holding any filled cell.
@group(0) @binding(18) var<storage, read_write> pyramid: array<atomic<u32>>;

// The key of brick `b` of shell `l`: 0 when the shell's keys do not reach
// it.
fn key_of(l: u32, b: vec3i) -> u32 {
	let r = b - shells.origin[l].xyz;
	if (any(r < vec3i(0)) || any(r >= vec3i(KEY_SPAN))) {
		return 0u;
	}
	let u = vec3u(r);
	return 1u | (u.x << 1u) | (u.y << 11u) | (u.z << 21u);
}

// The word and bit of shell `l`'s pyramid marking the block of grain `g`
// — 16, 64 or 256 cells wide for 0, 1, 2 — that holds cell `c`; x NO_ENTRY
// past the shell's keys.
fn block_bit(l: u32, c: vec3i, g: u32) -> vec2u {
	let shift = vec3u(4u + 2u * g);
	let n = 256 >> (2u * g);
	// The origin's cells, four a brick, whole blocks of every grain.
	let r = (c >> shift) - ((shells.origin[l].xyz * 4) >> shift);
	if (any(r < vec3i(0)) || any(r >= vec3i(n))) {
		return vec2u(NO_ENTRY, 0u);
	}
	let i = u32((r.z * n + r.y) * n + r.x);
	let first = select(select(0u, BLOCKS_16, g == 1u), BLOCKS_16 + BLOCKS_64, g == 2u);
	return vec2u(l * PYRAMID_WORDS + first + i / 32u, 1u << (i % 32u));
}

// Whether the block of grain `g` holding cell `c` of shell `l` holds no
// filled cell.
fn block_empty(l: u32, c: vec3i, g: u32) -> bool {
	let b = block_bit(l, c, g);
	return b.x == NO_ENTRY || (atomicLoad(&pyramid[b.x]) & b.y) == 0u;
}

fn home(key: u32) -> u32 {
	return mix_hash(key, 0x51ED270Bu) & (CAPACITY - 1u);
}

// The entry of shell `l` holding `key`, or NO_ENTRY.
fn find_entry(l: u32, key: u32) -> u32 {
	if (key == 0u) {
		return NO_ENTRY;
	}
	var at = home(key);
	for (var i = 0u; i < SHELL_PROBES; i += 1u) {
		let e = l * CAPACITY + at;
		let held = atomicLoad(&table[e].key);
		if (held == key) {
			return e;
		}
		if (held == 0u) {
			return NO_ENTRY;
		}
		at = (at + 1u) & (CAPACITY - 1u);
	}
	return NO_ENTRY;
}

// The entry of shell `l` holding `key`, taking an empty one when there is
// none; NO_ENTRY when the table is full there.
fn add_entry(l: u32, key: u32) -> u32 {
	if (key == 0u) {
		return NO_ENTRY;
	}
	var at = home(key);
	for (var i = 0u; i < SHELL_PROBES; i += 1u) {
		let e = l * CAPACITY + at;
		loop {
			let swap = atomicCompareExchangeWeak(&table[e].key, 0u, key);
			if (swap.exchanged || swap.old_value == key) {
				return e;
			}
			// A weak exchange may fail on an empty entry: try it again.
			if (swap.old_value != 0u) {
				break;
			}
		}
		at = (at + 1u) & (CAPACITY - 1u);
	}
	return NO_ENTRY;
}

// Cell `c`'s bit in its brick.
fn cell_bit(c: vec3i) -> u32 {
	let m = vec3u(c & vec3i(3));
	return m.x | (m.y << 2u) | (m.z << 4u);
}

fn fill_cell(l: u32, c: vec3i) {
	let e = add_entry(l, key_of(l, c >> vec3u(2u)));
	if (e == NO_ENTRY) {
		return;
	}
	let bit = cell_bit(c);
	if (bit < 32u) {
		atomicOr(&table[e].lo, 1u << bit);
	} else {
		atomicOr(&table[e].hi, 1u << (bit - 32u));
	}
	for (var g = 0u; g < 3u; g += 1u) {
		let b = block_bit(l, c, g);
		if (b.x != NO_ENTRY) {
			atomicOr(&pyramid[b.x], b.y);
		}
	}
}

// Whether cell `c` is filled in brick entry `e`.
fn bit_set(e: u32, c: vec3i) -> bool {
	let bit = cell_bit(c);
	if (bit < 32u) {
		return ((atomicLoad(&table[e].lo) >> bit) & 1u) != 0u;
	}
	return ((atomicLoad(&table[e].hi) >> (bit - 32u)) & 1u) != 0u;
}

// Whether a cell centered at `center` goes into the shell `work` fills:
// within its range of the eye and, when only new cells go in, not within
// it of the last eye.
fn takes_in(center: vec3f) -> bool {
	let near = work.range.x;
	let far = work.range.y;
	let d = length(center - work.eye.xyz);
	if (d < near || d > far) {
		return false;
	}
	if (work.last.w > 0.5) {
		let before = length(center - work.last.xyz);
		return before < near || before > far;
	}
	return true;
}

// Whether triangle `t` fills the cell of side `s` centered at `center`.
// The overlap test costs the most: tested last.
fn fills_cell(t: Tri, center: vec3f, s: f32) -> bool {
	return albedo(t, center).a >= 0.5 && overlaps_box(t, center - vec3f(SHELL_OWN * s), 0.5 * s);
}

// The first triangle, in the hierarchy's order, that fills the cube of
// side `s` at `center`, as the trace's `filler` picks it, so both passes
// color a cube off the same triangle; NO_HIT when none does.
fn shell_filler(center: vec3f, s: f32) -> u32 {
	let at = center - vec3f(SHELL_OWN * s);
	let lo = at - 0.5 * s;
	let hi = at + 0.5 * s;
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
				if (fills_cell(tris[i], center, s)) {
					return i;
				}
			}
		} else if (top + 2u <= STACK) {
			stack[top] = node.first;
			stack[top + 1u] = at_node + 1u;
			top += 2u;
		}
	}
	return NO_HIT;
}

// Whether a dent carved the point `p` away.
fn dented(p: vec3f) -> bool {
	for (var i = 0u; i < arrayLength(&dents); i += 1u) {
		let dent = dents[i];
		if (dent.w <= 0.0) {
			break;
		}
		if (length(p - dent.xyz) < dent.w) {
			return true;
		}
	}
	return false;
}

// The dent whose shell the cell of side `s` at `center` lies on, or -1.
fn dent_shell(center: vec3f, s: f32) -> i32 {
	for (var i = 0u; i < arrayLength(&dents); i += 1u) {
		let dent = dents[i];
		if (dent.w <= 0.0) {
			break;
		}
		if (length(center - dent.xyz) - dent.w <= s * 0.87) {
			return i32(i);
		}
	}
	return -1;
}

// The triangle nearest `p` among those within `reach` of it, or NO_HIT.
fn nearest_tri(p: vec3f, reach: f32) -> u32 {
	var best = NO_HIT;
	var best_dist = reach;
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
		if (length(clamp(p, node.lo, node.hi) - p) > best_dist) {
			continue;
		}
		if (node.count > 0u) {
			for (var i = node.first; i < node.first + node.count; i += 1u) {
				let dist = length(closest_point(p, tris[i]) - p);
				if (dist <= best_dist) {
					best_dist = dist;
					best = i;
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

// Whether the uncarved cell of side `s` at `center` lines a dent's bowl:
// on its shell, and behind the surface the dent was struck into.
fn lines_bowl(center: vec3f, s: f32) -> bool {
	let dent = dent_shell(center, s);
	if (dent < 0) {
		return false;
	}
	let tri = nearest_tri(center, dents[dent].w * 2.0 + s);
	if (tri == NO_HIT) {
		return false;
	}
	let t = tris[tri];
	let reach = 0.5 * s * dot(abs(tri_normal(t)), vec3f(1.0));
	return dot(tri_normal(t), center - closest_point(center, t)) <= reach;
}

// One triangle's cells in shell `work`, its columns shared by `SLICES`
// workgroups. The threads walk columns across the axis the triangle faces
// most, and in each only the few cells its plane can reach, so a
// large slanted triangle costs its area within the shell, not its box.
@compute @workgroup_size(64)
fn shell_fill(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) k: u32) {
	let i = group.x + group.z * ROW;
	if (i >= arrayLength(&tris)) {
		return;
	}
	let l = work.shell.x;
	let s = work.eye.w;
	let eye = work.eye.xyz;
	let t = tris[i];
	let pad = vec3f(s);
	let box_lo = min(min(t.a.xyz, t.b.xyz), t.c.xyz) - pad;
	let box_hi = max(max(t.a.xyz, t.b.xyz), t.c.xyz) + pad;
	// Triangles wholly outside the shell, or wholly inside its hole, fill
	// nothing in it.
	let far = work.range.y + s;
	if (length(clamp(eye, box_lo, box_hi) - eye) > far) {
		return;
	}
	if (length(max(abs(box_lo - eye), abs(box_hi - eye))) < work.range.x - s) {
		return;
	}
	let lo = floor(max(box_lo, eye - far) / s);
	let hi = floor(min(box_hi, eye + far) / s);
	if (any(hi < lo)) {
		return;
	}
	let n = tri_normal(t);
	let an = abs(n);
	var a = 2u;
	if (an.x >= an.y && an.x >= an.z) {
		a = 0u;
	} else if (an.y >= an.z) {
		a = 1u;
	}
	let u = (a + 1u) % 3u;
	let v = (a + 2u) % 3u;
	let span_u = u32(hi[u] - lo[u]) + 1u;
	let columns = span_u * (u32(hi[v] - lo[v]) + 1u);
	// Along the facing axis a cube, nudged as it owns its box, overlaps the
	// plane only while its center lies within this of it.
	let band = (0.5 + SHELL_OWN) * s * (an.x + an.y + an.z) / an[a];
	let plane = dot(n, t.a.xyz);
	for (var j = group.y * 64u + k; j < columns; j += 64u * SLICES) {
		var c = vec3f(0.0);
		c[u] = lo[u] + f32(j % span_u);
		c[v] = lo[v] + f32(j / span_u);
		let pu = (c[u] + 0.5) * s;
		let pv = (c[v] + 0.5) * s;
		let w = (plane - n[u] * pu - n[v] * pv) / n[a];
		let first = max(ceil((w - band) / s - 0.5), lo[a]);
		let last = min(floor((w + band) / s - 0.5), hi[a]);
		for (var cw = first; cw <= last; cw += 1.0) {
			c[a] = cw;
			let center = (c + 0.5) * s;
			if (takes_in(center) && fills_cell(t, center, s) && !dented(center)) {
				fill_cell(l, vec3i(c));
			}
		}
	}
}

// One dent's bowl in shell `work`: the uncarved cells on its sphere,
// behind the surface it was struck into, column by column across x and y,
// each column only where it crosses the sphere's skin.
@compute @workgroup_size(64)
fn shell_bowl(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) k: u32) {
	if (group.x >= arrayLength(&dents)) {
		return;
	}
	let dent = dents[group.x];
	if (dent.w <= 0.0) {
		return;
	}
	let l = work.shell.x;
	let s = work.eye.w;
	let outer = dent.w + s * 0.87;
	let apart = length(dent.xyz - work.eye.xyz);
	if (apart - outer > work.range.y || apart + outer < work.range.x) {
		return;
	}
	let lo = floor((dent.xyz - outer) / s);
	let hi = floor((dent.xyz + outer) / s);
	let span = vec2u(hi.xy - lo.xy) + 1u;
	let columns = span.x * span.y;
	for (var j = group.y * 64u + k; j < columns; j += 64u * SLICES) {
		let cx = lo.x + f32(j % span.x);
		let cy = lo.y + f32(j / span.x);
		let q = vec2f((cx + 0.5) * s, (cy + 0.5) * s) - dent.xy;
		let qq = dot(q, q);
		if (qq > outer * outer) {
			continue;
		}
		let skin_in = sqrt(max(dent.w * dent.w - qq, 0.0));
		let skin_out = sqrt(outer * outer - qq);
		// The skin crosses the column twice, once either side of the
		// center along z.
		for (var side = 0; side < 2; side += 1) {
			let sign = select(-1.0, 1.0, side == 1);
			let za = dent.z + sign * skin_in;
			let zb = dent.z + sign * skin_out;
			let first = ceil(min(za, zb) / s - 0.5);
			let last = floor(max(za, zb) / s - 0.5);
			for (var cz = first; cz <= last; cz += 1.0) {
				let center = (vec3f(cx, cy, cz) + 0.5) * s;
				if (takes_in(center) && !dented(center) && lines_bowl(center, s)) {
					fill_cell(l, vec3i(vec3f(cx, cy, cz)));
				}
			}
		}
	}
}

// The shell whose cubes are drawn around `p`: each cell takes the size its
// center wants unless its parent's already wants it whole, decided from a
// few sizes up as the trace's `cube_side` does, so both passes agree.
fn shell_level(p: vec3f) -> u32 {
	let base = view.sky.w;
	var l = wanted_level(length(p - view.eye.xyz)) + 3u;
	loop {
		let s = base * exp2(f32(l));
		let center = (floor(p / s) + 0.5) * s;
		if (l == 0u || wanted_level(length(center - view.eye.xyz)) >= l) {
			return l;
		}
		l -= 1u;
	}
	return 0u;
}

// Where along the ray from `o` it leaves the box `lo`..`lo + side`, `inv`
// its inverse direction.
fn exit_box(o: vec3f, inv: vec3f, lo: vec3f, side: f32) -> f32 {
	let far = max((lo - o) * inv, (lo + side - o) * inv);
	return min(far.x, min(far.y, far.z));
}

// The first filled cube along the ray: where it enters, its cell, its side.
struct Found {
	t: f32,
	cell: vec3f,
	s: f32,
	found: bool,
}

fn march(o: vec3f, d: vec3f) -> Found {
	let inv = select(vec3f(1e30), vec3f(1.0) / d, abs(d) >= vec3f(1e-12));
	// R of the smallest cubes: a shell's grows with its side.
	let unit = 1.0 / (view.eye.w * view.forward.w);
	// The map's bounds: no cube stands out of them by more than its side.
	let bound_lo = nodes[0].lo;
	let bound_hi = nodes[0].hi;
	var t = 0.0;
	var steps = 0u; // EXPERIMENT
	for (var i = 0u; i < MARCH; i += 1u) {
		steps = i; // EXPERIMENT
		let p = o + d * t;
		let l = shell_level(p);
		if (l >= SHELLS || shells.origin[l].w == 0) {
			break;
		}
		let s = view.sky.w * exp2(f32(l));
		if (any(p < bound_lo - s) || any(p > bound_hi + s)) {
			// Outside the bounds, only a ray heading back in meets matter.
			let far = max((bound_lo - s - o) * inv, (bound_hi + s - o) * inv);
			let near = min((bound_lo - s - o) * inv, (bound_hi + s - o) * inv);
			let enter = max(max(near.x, near.y), near.z);
			if (enter <= t || enter > min(min(far.x, far.y), far.z)) {
				break;
			}
			t = enter + 1e-3 * s;
			continue;
		}
		let c = vec3i(floor(p / s));
		let r = s * unit;
		// Well inside the band where every cube is this size, empty blocks
		// are crossed whole, the largest first, never past the band's end;
		// near a switch of size the ray goes a cell at a time.
		let band_end = select(2.0 * r - 4.0 * s, 1e30, l == SHELLS - 1u);
		let inside = (l == 0u || t >= r + s) && t < band_end;
		if (inside) {
			var skipped = false;
			for (var g = 2i; g >= 0; g -= 1) {
				if (block_empty(l, c, u32(g))) {
					let side = s * f32(16 << (2u * u32(g)));
					let lo = floor(p / side) * side;
					t = max(min(exit_box(o, inv, lo, side), band_end), t) + 1e-3 * s;
					skipped = true;
					break;
				}
			}
			if (skipped) {
				continue;
			}
		} else if (block_empty(l, c, 0u)) {
			t = max(exit_box(o, inv, vec3f(c) * s, s), t) + 1e-3 * s;
			continue;
		}
		let b = c >> vec3u(2u);
		let e = find_entry(l, key_of(l, b));
		if (e == NO_ENTRY && inside) {
			t = max(min(exit_box(o, inv, vec3f(b) * 4.0 * s, 4.0 * s), band_end), t) + 1e-3 * s;
			continue;
		}
		if (e != NO_ENTRY && bit_set(e, c)) {
			return Found(t, vec3f(c), view.sky.w * exp2(f32(steps) / 16.0), true); // EXPERIMENT
		}
		t = max(exit_box(o, inv, vec3f(c) * s, s), t) + 1e-3 * s;
	}
	return Found(1e30, vec3f(0.0), 0.0, false);
}

// The light of a bowl's cube at `center` in `dent`, cut into triangle
// `tri` facing `n`, per unit albedo, as the trace lights it: facing the
// dent's center, the sun reaching in only through the opening.
fn dent_light(tri: Tri, center: vec3f, n: vec3f, dent: vec4f) -> vec3f {
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

// A map cube's look, one color whichever ray reaches it: stored in the
// voxel field, or worked out from the triangle that fills it — its
// texels lit facing the eye off that surface, or for a bowl's cube facing
// its dent's center — and stored.
fn shell_color(d: vec3f, hit: Found) -> vec3f {
	let stored = stored_color(voxel_key(hit.cell, hit.s));
	if (stored.found) {
		return stored.color;
	}
	let s = hit.s;
	let center = (hit.cell + 0.5) * s;
	var color = vec3f(0.5);
	let near = shell_filler(center, s);
	if (near != NO_HIT) {
		let tri = tris[near];
		let n = facing(tri, d);
		let light = global_light(center - n * dot(n, center - tri.a.xyz), n, SHADED);
		color = light * voxel_albedo(tri, center, n, s);
	} else {
		let dent = dent_shell(center, s);
		if (dent >= 0) {
			let bowl = nearest_tri(center, dents[dent].w * 2.0 + s);
			if (bowl != NO_HIT) {
				let tri = tris[bowl];
				let n = facing(tri, d);
				color = dent_light(tri, center, n, dents[dent]) * voxel_albedo(tri, center, n, s);
			}
		}
	}
	store_color(stored, color);
	return color;
}

// The nearest playground sphere's cube along the ray before `far`: where
// the ray enters it, its cell, and the sphere with the cube's side in w.
struct Ball {
	t: f32,
	cell: vec3f,
	sphere: vec4f,
	found: bool,
}

fn ball(o: vec3f, d: vec3f, far: f32) -> Ball {
	var best = Ball(far, vec3f(0.0), vec4f(0.0), false);
	let inv = select(vec3f(1e30), vec3f(1.0) / d, abs(d) >= vec3f(1e-12));
	for (var i = 0u; i < arrayLength(&bodies); i += 1u) {
		let sphere = bodies[i];
		if (sphere.w <= 0.0) {
			continue;
		}
		let s = view.sky.w * exp2(f32(shell_level(sphere.xyz)));
		// The cubes reach past the sphere by up to a cube's diagonal.
		let r = sphere.w + s * 1.74;
		let oc = o - sphere.xyz;
		let half = dot(oc, d);
		let gap = half * half - (dot(oc, oc) - r * r);
		if (gap <= 0.0) {
			continue;
		}
		let leave = -half + sqrt(gap);
		var t = max(-half - sqrt(gap), 0.0);
		for (var k = 0; k < 256; k += 1) {
			if (t >= best.t || t > leave) {
				break;
			}
			let c = floor((o + d * t) / s);
			let lo = c * s;
			if (length(clamp(sphere.xyz, lo, lo + s) - sphere.xyz) < sphere.w) {
				best = Ball(t, c, vec4f(sphere.xyz, s), true);
				break;
			}
			t = max(exit_box(o, inv, lo, s), t) + 1e-3 * s;
		}
	}
	return best;
}

// A sphere's cube: flat matter, lit facing out of the sphere.
fn ball_color(hit: Ball) -> vec3f {
	let center = (hit.cell + 0.5) * hit.sphere.w;
	let out = center - hit.sphere.xyz;
	let n = select(vec3f(0.0, 1.0, 0.0), normalize(out), length(out) > 1e-6);
	return BALL * global_light(center, n, SHADED);
}

// The debug views F5 cycles through, after the lit one: as `VIEW_MODES`
// in mod.rs, shown as the trace shows them.
const SHELL_DEPTH: u32 = 1u;
const SHELL_NORMAL: u32 = 2u;
const SHELL_ALBEDO: u32 = 3u;
const SHELL_CUBES: u32 = 4u;
// The depth view's far end, world units.
const SHELL_DEPTH_RANGE: f32 = 4096.0;

fn shell_hue(x: f32) -> vec3f {
	return clamp(abs(fract(x + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

// One debug view of the cube in `cell` of side `s` the ray entered `t`
// along `d`, facing `n`, its albedo `albedo`.
fn shell_inspect(mode: u32, d: vec3f, t: f32, cell: vec3f, s: f32, n: vec3f, albedo: vec3f) -> vec3f {
	switch mode {
		case SHELL_DEPTH: {
			let z = t * dot(d, view.forward.xyz);
			return vec3f(1.0 - clamp(log2(1.0 + z) / log2(1.0 + SHELL_DEPTH_RANGE), 0.0, 1.0));
		}
		case SHELL_NORMAL: {
			return n * 0.5 + 0.5;
		}
		case SHELL_ALBEDO: {
			return albedo;
		}
		case SHELL_CUBES: {
			let c = vec3i(cell);
			let h = mix_hash(mix_hash(mix_hash(mix_hash(0x2545F491u, u32(c.x)), u32(c.y)), u32(c.z)), bitcast<u32>(s));
			return shell_hue(f32(h & 0xFFFFu) / 65536.0) * (0.5 + 0.5 * f32((h >> 16u) & 0xFFu) / 255.0);
		}
		default: {
			return shell_hue(log2(s / view.sky.w) / 7.0) * 0.9;
		}
	}
}

// A map cube's facing and albedo for the debug views: off the triangle
// that fills it, or flat grey for a bowl's cube.
fn shell_surface(d: vec3f, hit: Found) -> mat2x3f {
	let center = (hit.cell + 0.5) * hit.s;
	let near = shell_filler(center, hit.s);
	if (near == NO_HIT) {
		return mat2x3f(-d, vec3f(0.5));
	}
	let tri = tris[near];
	let n = facing(tri, d);
	return mat2x3f(n, voxel_albedo(tri, center, n, hit.s));
}

@vertex
fn shell_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
	return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn shell_aces(x: vec3f) -> vec3f {
	return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

struct ShellPixel {
	@location(0) color: vec4f,
	// Reversed and infinite, as the overlays draw: near / view depth.
	@builtin(frag_depth) depth: f32,
}

@fragment
fn shell_fs(@builtin(position) at: vec4f) -> ShellPixel {
	let ndc = vec2f(at.x / view.size.x * 2.0 - 1.0, 1.0 - at.y / view.size.y * 2.0);
	let d = normalize(view.forward.xyz + view.right.xyz * (ndc.x * view.right.w) + view.up.xyz * (ndc.y * view.up.w));
	let o = view.eye.xyz;
	let mode = u32(view.grade.w);
	let map = march(o, d);
	let hit = ball(o, d, map.t);
	// Lit colors and the albedo view tone-map and encode as the trace's
	// do; the other debug views are shown as they are.
	let lit = mode == 0u || mode == SHELL_ALBEDO;
	var seen = select(vec3f(0.0), view.sky.rgb, lit);
	var t = 1e30;
	if (hit.found) {
		t = hit.t;
		if (mode == 0u) {
			seen = ball_color(hit);
		} else {
			let out = (hit.cell + 0.5) * hit.sphere.w - hit.sphere.xyz;
			let n = select(vec3f(0.0, 1.0, 0.0), normalize(out), length(out) > 1e-6);
			seen = shell_inspect(mode, d, t, hit.cell, hit.sphere.w, n, BALL);
		}
	} else if (map.found) {
		t = map.t;
		if (mode == 0u) {
			seen = shell_inspect(5u, d, t, map.cell, map.s, vec3f(0.0), vec3f(0.0)); // EXPERIMENT
		} else {
			let surface = shell_surface(d, map);
			seen = shell_inspect(mode, d, t, map.cell, map.s, surface[0], surface[1]);
		}
	}
	var color = seen;
	if (lit) {
		if (mode == 0u) {
			color = shell_aces(color * view.grade.x);
		}
		if (view.grade.y > 0.5) {
			color = pow(color, vec3f(1.0 / 2.2));
		}
	} else if (view.grade.y < 0.5) {
		// The target encodes to sRGB itself: hand it the shown value as linear.
		color = pow(color, vec3f(2.2));
	}
	var depth = 0.0;
	if (t < 1e29) {
		depth = clamp(view.grade.z / max(t * dot(d, view.forward.xyz), 1e-6), 0.0, 1.0);
	}
	return ShellPixel(vec4f(color, 1.0), depth);
}
