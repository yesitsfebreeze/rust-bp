// The shell field: the map voxelized ahead of the view into one sparse
// shell per cube size, and the full-screen pass that marches it instead of
// the trace (trace.wgsl). Prepended with bvh.wgsl, material.wgsl and
// voxel.wgsl.
//
// No ray ever tests the mesh to draw: the voxelizer intersects every
// triangle with the grid of cells around the eye and fills the cells it
// crosses, ahead of time, ahead of the frame. Drawing is then a march
// through those cells from the eye along the view direction: one hash
// lookup a cell, empty blocks skipped whole.
//
// A cube `s` wide is drawn only between about R_l and R_{l+1} from the eye,
// R_l the LOD curve's radius for its size (`lod_r`), with an overlap either
// side where neighbouring sizes blend. Each size keeps only that shell,
// padded by how far the eye may move before the shell is brought along, and
// in it only the cells a surface fills: a hash of bricks, 4^3 cells as 64
// bits. Over it a dense pyramid of bits marks which blocks of 16, 64 and
// 256 cells hold any, so a ray crosses empty space the largest empty block
// at a time with one load per test, and stops where it leaves the map's
// bounds. As the eye moves a shell takes in only the cells that entered it;
// every few moves, or when the dents change, it is built again whole and
// drops the cells it left behind.
//
// A cell is filled by the rule the trace walked by: a triangle overlaps the
// cube, its texel at the cube's center is opaque, and no dent carved the
// cube away. Cells lining a dent's bowl behind the surface fill too. A
// cube is drawn one solid color: the light on the BSP plane that fills it
// — the plane's own normal, never one turned toward the eye, no thickness
// the mesh has not — stored per voxel, whichever ray reaches it. The
// playground's spheres march as cells of the size their center wants, lit
// per pixel. The map's triangles draw as a pink wireframe over the march
// (`r_wire`).
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
// Cells and bricks a ray visits at most: a ray grazing along matter
// crosses every cell of every band it runs in, about 1.75 near / s of
// them per size, so the budget must cover the longest wall at the
// largest `near` the fov may draw.
const MARCH: u32 = 8192u;
// The spheres and the dents their strikes leave, as the world's buffers
// hold them: uniform arrays hold a fixed count.
const BODIES: u32 = 64u;
const DENTS: u32 = 256u;
// The spheres' matter, as linear albedo, as the trace draws it.
const BALL: vec3f = vec3f(0.56, 0.38, 0.19);
// How much a cube's box grows for the overlap test, in cube sides, as
// `GROW` in trace.wgsl: past f32's rounding at the map's far corners for
// the smallest cubes, small enough to change no shape.
const SHELL_GROW: f32 = 0.02;

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
	// Where the sizes change over distance: x the R of the smallest cubes,
	// world units; y sizes up per doubling of distance; z half the overlap
	// of neighbouring sizes, a share of the R they switch at; w the most
	// pixels a cube spans where it is drawn nearest.
	lod: vec4f,
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
// radius, 0 for none. Uniforms, a fixed count each.
@group(0) @binding(12) var<uniform> bodies: array<vec4f, BODIES>;
@group(0) @binding(14) var<uniform> dents: array<vec4f, DENTS>;
@group(0) @binding(15) var<storage, read_write> table: array<Entry>;
@group(0) @binding(16) var<uniform> shells: Shells;
@group(0) @binding(17) var<uniform> work: Work;
// Per shell, the pyramid of blocks holding any filled cell.
@group(0) @binding(18) var<storage, read_write> pyramid: array<atomic<u32>>;
// Per pixel, how far its ray goes before it meets a triangle; then the
// least of that over the pixels around it, where its march starts. And the
// rows' least, between the two passes that take it.
@group(0) @binding(19) var<storage, read_write> reach: array<f32>;
@group(0) @binding(20) var<storage, read_write> reach_rows: array<f32>;
// Pixels `reach` holds, as `REACH_PIXELS` in shell.rs: a larger screen's
// rays past it march from the eye.
const REACH_PIXELS: u32 = 8294400u;

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

// Whether triangle `t` fills the cell of side `s` centered at `center`:
// the cell reaches the triangle and its center lies behind the face, in
// the matter — as the trace's `filler` counts matter, so no cube hangs
// out of the mesh past a surface, doubling it up at every edge. The
// overlap test costs the most: tested last.
fn fills_cell(t: Tri, center: vec3f, s: f32) -> bool {
	return albedo(t, center).a >= 0.5 && dot(tri_normal(t), center - t.a.xyz) < 0.0 && overlaps_box(t, center, 0.5 * s);
}

// The triangle a filled cube of side `s` at `center` shows as, as the
// trace's `filler` picks it, so both passes color a cube off the same
// plane: the face it hangs in front of, else the first behind it, in the
// hierarchy's order; NO_HIT when none does.
fn shell_filler(center: vec3f, s: f32) -> u32 {
	let half = (0.5 + SHELL_GROW) * s;
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
	// A cube inside matter — some face behind it — colors off the plane
	// behind it: the face in front of a cube in the wall at a floor joint
	// would paint the floor up it, a doubled plane at every joint. Only a
	// cube hanging in front of a face, none behind it, shows as that face;
	// the trace leaves those out.
	if (behind != NO_HIT) {
		return behind;
	}
	return select(NO_HIT, front, front != NO_HIT);
}

// Whether a dent carved the point `p` away.
fn dented(p: vec3f) -> bool {
	for (var i = 0u; i < DENTS; i += 1u) {
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
	for (var i = 0u; i < DENTS; i += 1u) {
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

// A bowl's cube facing: into the void the dent carved, toward its center;
// straight up when no dent lines the cube.
fn bowl_normal(center: vec3f, s: f32) -> vec3f {
	let dent = dent_shell(center, s);
	if (dent < 0) {
		return vec3f(0.0, 1.0, 0.0);
	}
	return normalize(dents[dent].xyz - center);
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
	// Along the facing axis a cube overlaps the plane only while its center
	// lies within this of it.
	let band = 0.5 * s * (an.x + an.y + an.z) / an[a];
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
	if (group.x >= DENTS) {
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

// Where size `l` begins, world units from the eye: the curve over
// distance the sizes follow.
fn lod_r(l: u32) -> f32 {
	return shells.lod.x * exp2(f32(l) / shells.lod.y);
}

fn side(l: u32) -> f32 {
	return view.sky.w * exp2(f32(l));
}

// Where in the overlap the cell `c` of side `s` cuts over to its own size,
// as a share of its R: within the overlap either side of 1, hashed per
// cell.
fn cut(c: vec3f, s: f32) -> f32 {
	let i = vec3i(c);
	let h = mix_hash(mix_hash(mix_hash(mix_hash(0x7F4A7C15u, u32(i.x)), u32(i.y)), u32(i.z)), bitcast<u32>(s));
	return 1.0 + shells.lod.z * (2.0 * f32(h & 0xFFFFu) / 65536.0 - 1.0);
}

// The shell whose cubes are drawn around `p`: from a few sizes up, each
// cell is drawn whole once its center lies past its cut, else splits; so
// every point falls in exactly one drawn cube, and neighbouring rays agree.
fn shell_level(p: vec3f) -> u32 {
	var l = min(level_hi(length(p - view.eye.xyz)) + 1u, SHELLS - 1u);
	loop {
		let s = side(l);
		let cell = floor(p / s);
		let center = (cell + 0.5) * s;
		if (l == 0u || length(center - view.eye.xyz) >= lod_r(l) * cut(cell, s)) {
			return l;
		}
		l -= 1u;
	}
	return 0u;
}

// The coarsest size any cube `dist` from the eye may be drawn at: its cut
// may lie as near as the overlap's start, a cell's corner either way.
fn level_hi(dist: f32) -> u32 {
	let a = shells.lod.z;
	let guess = floor(shells.lod.y * log2(max(dist / (shells.lod.x * (1.0 - a)), 1e-30))) + 1.0;
	var l = u32(clamp(guess, 0.0, f32(SHELLS - 1u)));
	loop {
		if (l == 0u || lod_r(l) * (1.0 - a) - 2.0 * side(l) <= dist) {
			return l;
		}
		l -= 1u;
	}
	return 0u;
}

// Where shell `l`'s cubes are drawn, from the eye that frame: from its
// cut, at the nearest the overlap's start before its R, to its parent's at
// the farthest, give or take a cell's corner — as `near` and `far` in
// shell.rs, without the pad the shell fills past them by.
fn band_lo(l: u32, s: f32) -> f32 {
	if (l == 0u) {
		return 0.0;
	}
	return lod_r(l) * (1.0 - shells.lod.z) - 2.87 * s;
}

fn band_hi(l: u32, s: f32) -> f32 {
	if (l >= SHELLS - 1u) {
		return 1e30;
	}
	return lod_r(l + 1u) * (1.0 + shells.lod.z) + 4.87 * s;
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

fn march(o: vec3f, d: vec3f, start: f32) -> Found {
	let inv = select(vec3f(1e30), vec3f(1.0) / d, abs(d) >= vec3f(1e-12));
	// The map's bounds: no cube stands out of them by more than its side.
	let bound_lo = nodes[0].lo;
	let bound_hi = nodes[0].hi;
	var t = start;
	for (var i = 0u; i < MARCH; i += 1u) {
		let p = o + d * t;
		// The sizes that may draw a cube here: from the finest this point's
		// cut picks to the coarsest whose band still holds its distance.
		let lo = shell_level(p);
		if (lo >= SHELLS) {
			break;
		}
		let hi = max(min(level_hi(length(p - view.eye.xyz)), SHELLS - 1u), lo);
		let coarse = side(hi);
		let fine = side(lo);
		if (any(p < bound_lo - coarse) || any(p > bound_hi + coarse)) {
			// Outside the bounds, only a ray heading back in meets matter.
			let far = max((bound_lo - coarse - o) * inv, (bound_hi + coarse - o) * inv);
			let near = min((bound_lo - coarse - o) * inv, (bound_hi + coarse - o) * inv);
			let enter = max(max(near.x, near.y), near.z);
			if (enter <= t || enter > min(min(far.x, far.y), far.z)) {
				break;
			}
			t = enter + 1e-3 * fine;
			continue;
		}
		let c = vec3i(floor(p / fine));
		let r = lod_r(lo);
		// Well inside the band where every legal size draws, empty blocks
		// are crossed whole, the largest first — a block empty at the
		// coarsest size is empty at every finer one — never past the band's
		// end; near a switch of size the ray goes a cell at a time.
		let band_end = band_hi(lo, fine);
		let inside = (lo == 0u || t >= r + fine) && t < band_end;
		if (inside) {
			var skipped = false;
			if (shells.origin[hi].w != 0) {
				// The blocks of the coarsest size are tested at its own
				// scale: a fine cell's index there would run past the
				// shell's keys and read empty, skipping matter whole.
				let ch = vec3i(floor(p / coarse));
				for (var g = 2i; g >= 0; g -= 1) {
					if (block_empty(hi, ch, u32(g))) {
						let span = coarse * f32(16 << (2u * u32(g)));
						let blo = floor(p / span) * span;
						t = max(min(exit_box(o, inv, blo, span), band_end), t) + 1e-3 * fine;
						skipped = true;
						break;
					}
				}
			}
			if (skipped) {
				continue;
			}
		} else if (shells.origin[lo].w != 0 && block_empty(lo, c, 0u)) {
			t = max(exit_box(o, inv, vec3f(c) * fine, fine), t) + 1e-3 * fine;
			continue;
		}
		// Matter near: every size that may draw a cube here, finest first,
		// and the first cell any of them fills is the cube. A coarse cell
		// may sit with its center in front of a face its fine children
		// behind it fill — matter only grows inward — so testing only the
		// one size the cut picks leaves them out: gaps and moire where the
		// sizes switch. Testing them all, each ray finds the matter
		// whichever size holds it, and across a switch every cube stays
		// whole.
		var hit = Found(1e30, vec3f(0.0), 0.0, false);
		var m = lo;
		loop {
			if (shells.origin[m].w != 0) {
				let sm = side(m);
				let cm = vec3i(floor(p / sm));
				let em = find_entry(m, key_of(m, cm >> vec3u(2u)));
				if (em != NO_ENTRY && bit_set(em, cm)) {
					hit = Found(t, vec3f(cm), sm, true);
					break;
				}
			}
			if (m >= hi) {
				break;
			}
			m += 1u;
		}
		if (hit.found) {
			return hit;
		}
		// Nothing here: on past the finest cell the ray is in, so no
		// filled cell it later crosses is stepped over.
		let b = c >> vec3u(2u);
		let e = find_entry(lo, key_of(lo, b));
		if (e == NO_ENTRY && inside) {
			t = max(min(exit_box(o, inv, vec3f(b) * 4.0 * fine, 4.0 * fine), band_end), t) + 1e-3 * fine;
			continue;
		}
		t = max(exit_box(o, inv, vec3f(c) * fine, fine), t) + 1e-3 * fine;
	}
	return Found(1e30, vec3f(0.0), 0.0, false);
}

// The light of a map cube on the plane of normal `n`, per unit albedo:
// what the bake left out — a bowl's cubes — lit by the sun where the
// plane faces it, the sky around. No ray to the mesh; stored per voxel.
fn cube_light(n: vec3f) -> vec3f {
	return sun_and_sky(n, dot(n, lighting.sun.xyz) <= 0.0, SHADED);
}

// A map cube's look, one color whichever ray reaches it: stored in the
// voxel field, or worked out from the plane that fills it — its face's
// baked chart, sampled where the cube's center falls on that plane; its
// texels under that light, its own normal, never one turned toward the
// eye — and stored.
fn shell_color(hit: Found) -> vec3f {
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
		let n = tri_normal(tri);
		// The cube's center falls on the filling plane: the chart's grid is
		// the plane's own.
		let p = center - n * dot(n, center - tri.a.xyz);
		let light = select(cube_light(n), baked_light_at(tri, p), is_baked(tri));
		color = light * voxel_albedo(tri, center, n, s);
	} else {
		// No plane behind it: a cube of a bowl's dent, lit facing into
		// the bowl.
		let n = bowl_normal(center, s);
		color = cube_light(n) * vec3f(0.5);
	}
	store_color(stored, color);
	return color;
}

// A map cube's lit color, blended toward its parent's across the sizes'
// overlap: from where the parent first may be drawn whole to where it
// must be, the fine cube's color thins out into the coarse one's, so a
// switch keeps every cube whole and the blend follows the eye's distance.
fn blended_color(hit: Found) -> vec3f {
	var color = shell_color(hit);
	let l = u32(round(log2(hit.s / view.sky.w)));
	if (l + 1u < SHELLS) {
		let r = lod_r(l + 1u);
		let a = shells.lod.z;
		let center = (hit.cell + 0.5) * hit.s;
		let f = clamp((length(center - view.eye.xyz) - r * (1.0 - a)) / (2.0 * a * r), 0.0, 1.0);
		if (f > 0.0) {
			let ps = 2.0 * hit.s;
			let pc = floor(hit.cell / 2.0);
			// Only a parent already stored thins the cube: none is filled
			// just for the blend, so a cube keeps its color the frames
			// before its parent's own rays store one.
			let stored = stored_color(voxel_key(pc, ps));
			if (stored.found) {
				color = mix(color, stored.color, f);
			}
		}
	}
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
	for (var i = 0u; i < BODIES; i += 1u) {
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

// A map cube's facing and albedo for the debug views: the plane that
// fills it, its own normal; into its bowl for a bowl's cube.
fn shell_surface(hit: Found) -> mat2x3f {
	let center = (hit.cell + 0.5) * hit.s;
	let near = shell_filler(center, hit.s);
	if (near == NO_HIT) {
		return mat2x3f(bowl_normal(center, hit.s), vec3f(0.5));
	}
	let tri = tris[near];
	let n = tri_normal(tri);
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

// The ray through pixel `at`, as the trace's ray through a pixel.
fn pixel_ray(at: vec2f) -> vec3f {
	let ndc = vec2f(at.x / view.size.x * 2.0 - 1.0, 1.0 - at.y / view.size.y * 2.0);
	return normalize(view.forward.xyz + view.right.xyz * (ndc.x * view.right.w) + view.up.xyz * (ndc.y * view.up.w));
}

// Pixels either way a march looks for a nearer triangle: a cube spans up
// to `K` of them, and a pixel to round.
fn reach_radius() -> i32 {
	return clamp(i32(ceil(view.forward.w)) + 1, 1, 8);
}

// One thread a pixel: how far its ray goes before any triangle.
@compute @workgroup_size(8, 8)
fn shell_reach(@builtin(global_invocation_id) id: vec3u) {
	let w = u32(view.size.x);
	let i = id.y * w + id.x;
	if (id.x >= w || id.y >= u32(view.size.y) || i >= REACH_PIXELS) {
		return;
	}
	let hit = trace(view.eye.xyz, pixel_ray(vec2f(id.xy) + 0.5));
	reach[i] = select(1e30, hit.t, hit.tri != NO_HIT);
}

// The least reach along each row within the radius.
@compute @workgroup_size(8, 8)
fn shell_reach_rows(@builtin(global_invocation_id) id: vec3u) {
	let w = i32(view.size.x);
	let i = id.y * u32(w) + id.x;
	if (i32(id.x) >= w || id.y >= u32(view.size.y) || i >= REACH_PIXELS) {
		return;
	}
	let r = reach_radius();
	var least = 1e30;
	for (var dx = -r; dx <= r; dx += 1) {
		let x = clamp(i32(id.x) + dx, 0, w - 1);
		least = min(least, reach[id.y * u32(w) + u32(x)]);
	}
	reach_rows[i] = least;
}

// The least of the rows' least within the radius up and down: back into
// `reach`, where the march reads it.
@compute @workgroup_size(8, 8)
fn shell_reach_columns(@builtin(global_invocation_id) id: vec3u) {
	let w = u32(view.size.x);
	let h = i32(view.size.y);
	let i = id.y * w + id.x;
	if (id.x >= w || i32(id.y) >= h || i >= REACH_PIXELS) {
		return;
	}
	let r = reach_radius();
	var least = 1e30;
	for (var dy = -r; dy <= r; dy += 1) {
		let y = u32(clamp(i32(id.y) + dy, 0, h - 1));
		if (y * w + id.x < REACH_PIXELS) {
			least = min(least, reach_rows[y * w + id.x]);
		}
	}
	reach[i] = least;
}

// The ray through `at` on screen, in pixels: what it shows, as linear
// light on the screen — tone mapped when lit — and its view depth,
// reversed as the overlays draw. The debug views' values are as shown:
// back to light, so they average and encode like the rest.
fn shell_sample(at: vec2f) -> vec4f {
	let d = pixel_ray(at);
	let o = view.eye.xyz;
	let mode = u32(view.grade.w);
	// No cube stands nearer than the nearest triangle around, by more than
	// a few of the cubes there.
	let i = u32(floor(at.y - 0.5)) * u32(view.size.x) + u32(floor(at.x - 0.5));
	var near = 0.0;
	if (i < REACH_PIXELS) {
		near = reach[i];
	}
	var map = Found(1e30, vec3f(0.0), 0.0, false);
	if (near < 1e29) {
		map = march(o, d, max(near - 3.0 * side(level_hi(near)), 0.0));
	}
	let hit = ball(o, d, map.t);
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
			seen = blended_color(map);
		} else {
			let surface = shell_surface(map);
			seen = shell_inspect(mode, d, t, map.cell, map.s, surface[0], surface[1]);
		}
	}
	var color = seen;
	if (lit) {
		if (mode == 0u) {
			color = shell_aces(color * view.grade.x);
		}
	} else {
		color = pow(color, vec3f(2.2));
	}
	var depth = 0.0;
	if (t < 1e29) {
		depth = clamp(view.grade.z / max(t * dot(d, view.forward.xyz), 1e-6), 0.0, 1.0);
	}
	return vec4f(color, depth);
}

// Where in the pixel each ray goes, for 2 and 4 rays: as the trace's — a
// diagonal pair, and a rotated grid, so edges at any angle land between
// samples.
const PAIR: array<vec2f, 2> = array<vec2f, 2>(vec2f(-0.25, -0.25), vec2f(0.25, 0.25));
const GRID: array<vec2f, 4> = array<vec2f, 4>(
	vec2f(-0.125, -0.375),
	vec2f(0.375, -0.125),
	vec2f(0.125, 0.375),
	vec2f(-0.375, 0.125),
);

@fragment
fn shell_fs(@builtin(position) at: vec4f) -> ShellPixel {
	// Rays a pixel averages: one through its center, or `view.size.z`
	// spread across it, so cubes smaller than the gaps between pixels
	// and their edges blend instead of alias.
	let rays = u32(view.size.z);
	var sum = vec4f(0.0);
	var depth = 0.0;
	var count = 1u;
	if (rays >= 4u) {
		count = 4u;
		for (var i = 0u; i < 4u; i += 1u) {
			let one = shell_sample(at.xy + GRID[i]);
			sum += one;
			depth = max(depth, one.w);
		}
	} else if (rays >= 2u) {
		count = 2u;
		for (var i = 0u; i < 2u; i += 1u) {
			let one = shell_sample(at.xy + PAIR[i]);
			sum += one;
			depth = max(depth, one.w);
		}
	} else {
		sum = shell_sample(at.xy);
		depth = sum.w;
	}
	var color = sum.rgb / f32(count);
	if (view.grade.y > 0.5) {
		color = pow(color, vec3f(1.0 / 2.2));
	}
	// The nearest of the rays: overlays hide behind any matter in the pixel.
	return ShellPixel(vec4f(color, 1.0), depth);
}

// Projects a world position, reversed and infinite as the overlays draw:
// near / view depth, the sky clearing to 0.
fn project(p: vec3f) -> vec4f {
	let rel = p - view.eye.xyz;
	let x = dot(rel, view.right.xyz);
	let y = dot(rel, view.up.xyz);
	let z = dot(rel, view.forward.xyz);
	return vec4f(x / view.right.w, y / view.up.w, view.grade.z, z);
}

// The map's triangles as pink wireframe over whatever the march drew: three
// edges a triangle, read straight from the trace's triangles. Not
// depth-tested: a cube fills its whole cell, so it hides the mesh inside it.
struct WireIn {
	@builtin(position) clip: vec4f,
}

@vertex
fn wire_vs(@builtin(vertex_index) i: u32) -> WireIn {
	let tri = tris[i / 6u];
	// v0 v1 v1 v2 v2 v0: three edges, two vertices each.
	let corner = array<vec4f, 6>(tri.a, tri.b, tri.b, tri.c, tri.c, tri.a)[i % 6u];
	var out: WireIn;
	out.clip = project(corner.xyz);
	return out;
}

@fragment
fn wire_fs(in: WireIn) -> @location(0) vec4f {
	return vec4f(1.0, 0.4, 0.75, 1.0);
}
