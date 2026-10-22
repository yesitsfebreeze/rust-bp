// The voxel field the view traces through: how big a voxel a hit draws
// as, its lit color, and where that color is stored. Prepended with
// bvh.wgsl and material.wgsl.
//
// Every voxel the view draws gets a slot in one hash table, keyed by its
// cell and size alone. The first ray to land in a voxel works out its
// color — the mean of its texels, lit by the sun through a shadow ray and
// the sky around it — and stores it; every later ray reads it, whichever
// triangle led it there, so a voxel is one color however many triangles
// cross it, and its shadow ray is cast once, not once per pixel. Slots a
// voxel has not read for a while are taken over by new ones, so the field
// follows the eye around the map.

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
	// w the view mode: 0 lit, else a debug view (trace.wgsl).
	grade: vec4f,
	// xy viewport pixels; z, w unused.
	size: vec4f,
	// x the frame number so far. Words, not floats: as f32 it would stop
	// growing past 2^24 and freeze the field's clock. y how many times the
	// dents changed.
	frame: vec4u,
}

// One stored voxel: its identity, when it was last read, and its color.
struct Slot {
	key: atomic<u32>,
	// A second hash confirming the key.
	check: u32,
	// The frame it was last read, counted from 1; 0 while it is written.
	seen: u32,
	// rgb its lit color, linear; w the voxel's check, as bits, once the
	// color is written.
	color: vec4f,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(13) var<storage, read_write> slots: array<Slot>;

// The share of its sunlight a voxel in shadow keeps.
const SHADED: f32 = 0.35;
// Slots a lookup tries; frames after which an unread voxel's slot may be
// taken by another.
const PROBES: u32 = 8u;
const EVICT: u32 = 600u;

fn frame_number() -> u32 {
	return view.frame.x + 1u;
}

// The triangle's normal, turned toward `d`'s origin.
fn facing(tri: Tri, d: vec3f) -> vec3f {
	let n = tri_normal(tri);
	return select(n, -n, dot(n, d) > 0.0);
}

fn mix_hash(h: u32, v: u32) -> u32 {
	var x = (h ^ v) * 0x9E3779B1u;
	x ^= x >> 15u;
	x *= 0x85EBCA77u;
	return x ^ (x >> 13u);
}

// The power-of-two level of the voxel a point `dist` from the eye wants:
// `K` pixels wide, counted in doublings of the smallest voxel.
fn wanted_level(dist: f32) -> u32 {
	return u32(floor(log2(max(dist * view.eye.w * view.forward.w / view.sky.w, 1.0))));
}

// A voxel's key and check: its cell and size, and how many times the
// dents changed, so a new dent recolors the matter it reshaped.
fn voxel_key(cell: vec3f, s: f32) -> vec2u {
	let c = vec3i(cell);
	let level = u32(log2(s / view.sky.w));
	let h = mix_hash(mix_hash(mix_hash(mix_hash(0x2545F491u, u32(c.x)), u32(c.y)), u32(c.z)), level);
	let key = mix_hash(h, view.frame.y) | 1u;
	return vec2u(key, mix_hash(key, 0x68E31DA4u));
}

// The slot holding a voxel, or -1.
fn find(id: vec2u) -> i32 {
	let mask = arrayLength(&slots) - 1u;
	var at = id.x & mask;
	for (var i = 0u; i < PROBES; i += 1u) {
		let held = atomicLoad(&slots[at].key);
		if (held == id.x && slots[at].check == id.y) {
			return i32(at);
		}
		if (held == 0u) {
			return -1;
		}
		at = (at + 1u) & mask;
	}
	return -1;
}

// The slot for a voxel, claiming one when it has none: x the slot or -1
// when the table is full there, y 1 when just claimed.
fn claim(id: vec2u) -> vec2i {
	let mask = arrayLength(&slots) - 1u;
	let frame = frame_number();
	var at = id.x & mask;
	for (var i = 0u; i < PROBES; i += 1u) {
		let held = atomicLoad(&slots[at].key);
		if (held == id.x && slots[at].check == id.y) {
			return vec2i(i32(at), 0);
		}
		let seen = slots[at].seen;
		// An empty slot, or one no voxel has read for a while.
		if (held == 0u || (seen != 0u && frame - seen > EVICT)) {
			let swap = atomicCompareExchangeWeak(&slots[at].key, held, id.x);
			if (swap.exchanged) {
				slots[at].seen = 0u;
				slots[at].check = id.y;
				return vec2i(i32(at), 1);
			}
		}
		at = (at + 1u) & mask;
	}
	return vec2i(-1, 0);
}

// The mean color across a voxel of side `s`, linear: four blended samples
// a quarter of the voxel off its center along the surface.
fn voxel_albedo(tri: Tri, center: vec3f, n: vec3f, s: f32) -> vec3f {
	let t1 = normalize(tri.b.xyz - tri.a.xyz) * (0.25 * s);
	let t2 = cross(n, t1);
	var sum = vec3f(0.0);
	for (var i = 0; i < 4; i += 1) {
		let sx = select(-1.0, 1.0, (i & 1) == 1);
		let sy = select(-1.0, 1.0, (i & 2) == 2);
		sum += albedo_smooth(tri, center + t1 * sx + t2 * sy);
	}
	return sum * 0.25;
}

// A voxel's stored color, when `found`; otherwise the slot to store it in
// once worked out, or -1 when the table has no room or another ray already
// claimed it and is still working it out. `check` is the voxel's second
// hash: stored beside the color, it says the color is this voxel's and
// finished, never an empty slot's or the one the slot held before.
struct Stored {
	color: vec3f,
	slot: i32,
	found: bool,
	check: u32,
}

fn stored_color(id: vec2u) -> Stored {
	let slot = claim(id);
	if (slot.x >= 0 && slot.y == 0) {
		let at = u32(slot.x);
		let held = slots[at].color;
		if (bitcast<u32>(held.w) != id.y) {
			return Stored(vec3f(0.0), -1, false, id.y);
		}
		slots[at].seen = frame_number();
		return Stored(held.rgb, slot.x, true, id.y);
	}
	return Stored(vec3f(0.0), slot.x, false, id.y);
}

// Keeps a voxel's color worked out in the slot `stored_color` gave.
fn store_color(stored: Stored, color: vec3f) {
	if (stored.slot >= 0) {
		let at = u32(stored.slot);
		slots[at].color = vec4f(color, bitcast<f32>(stored.check));
		slots[at].seen = frame_number();
	}
}
