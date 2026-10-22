//! GoldSrc `.mdl` studio models: bones, animation, bodypart meshes and
//! palette textures, parsed into `linalg` types. Every layout is Half-Life's
//! `studio.h` byte-for-byte, and the animation decoding follows the SDK's
//! `mdlviewer` (`CalcBonePosition` / `CalcBoneQuaternion`), including the
//! RLE-compressed anim values and frame-to-frame slerp.

use linalg::{vec3, Quat, Vec3};

/// `IDST` as the file's first four bytes.
const IDST: [u8; 4] = *b"IDST";
/// Every GoldSrc model is format version 10.
const VERSION: i32 = 10;

/// Header field offsets, from `studiohdr_t`'s field order.
mod hdr {
	pub const NUMBONES: usize = 140;
	pub const BONEINDEX: usize = 144;
	pub const NUMSEQ: usize = 164;
	pub const SEQINDEX: usize = 168;
	pub const NUMTEXTURES: usize = 180;
	pub const TEXTUREINDEX: usize = 184;
	pub const NUMSKINREF: usize = 192;
	pub const SKININDEX: usize = 200;
	pub const NUMBODYPARTS: usize = 204;
	pub const BODYPARTINDEX: usize = 208;
}

fn u16_at(b: &[u8], at: usize) -> u16 {
	u16::from_le_bytes([b[at], b[at + 1]])
}

fn i16_at(b: &[u8], at: usize) -> i16 {
	i16::from_le_bytes([b[at], b[at + 1]])
}

fn i32_at(b: &[u8], at: usize) -> i32 {
	i32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn f32_at(b: &[u8], at: usize) -> f32 {
	f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The NUL-terminated string at `at`, or the whole field if unterminated.
fn str_at(b: &[u8], at: usize) -> String {
	let end = b[at..]
		.iter()
		.position(|&c| c == 0)
		.map_or(b.len(), |n| at + n);
	String::from_utf8_lossy(&b[at..end]).into_owned()
}

/// One bone: its parent (or -1 for roots) and the default values and
/// animation scales for the six degrees of freedom, X Y Z XR YR ZR.
#[derive(Clone, Debug)]
pub struct Bone {
	pub parent: i32,
	pub value: [f32; 6],
	pub scale: [f32; 6],
}

/// One named animation: bones' DoF tracks, compressed.
#[derive(Clone, Debug)]
pub struct Sequence {
	pub label: String,
	pub fps: f32,
	pub numframes: i32,
	pub looping: bool,
	/// Offset of the bone animation blocks, `[bone][6 × u16]` each.
	pub anim: usize,
}

/// One palette skin. `data` and `palette` are offsets into the file:
/// `width × height` palette indices, then 256 RGB triples.
#[derive(Clone, Debug)]
pub struct Texture {
	/// The texture's own name; kept for debugging even though rendering goes
	/// by skinref.
	#[allow(dead_code)]
	pub name: String,
	pub flags: i32,
	pub width: u32,
	pub height: u32,
	pub data: usize,
	pub palette: usize,
}

/// A triangle corner in the mesh: which vertex and normal, and skin texel.
#[derive(Clone, Copy, Debug)]
pub struct Corner {
	pub vert: usize,
	pub norm: usize,
	pub s: f32,
	pub t: f32,
}

/// One textured mesh, its triangle commands expanded into triangles.
#[derive(Clone, Debug)]
pub struct Part {
	pub skinref: usize,
	pub tris: Vec<[Corner; 3]>,
}

/// One bodypart submodel: shared vertex arrays and its meshes.
#[derive(Clone, Debug)]
pub struct SubModel {
	pub vert_bones: Vec<u8>,
	pub verts: Vec<Vec3>,
	pub norms: Vec<Vec3>,
	pub parts: Vec<Part>,
}

/// A parsed model.
pub struct Mdl {
	b: Vec<u8>,
	pub bones: Vec<Bone>,
	pub sequences: Vec<Sequence>,
	pub textures: Vec<Texture>,
	/// `skinrefs[skinref]` names the texture a mesh draws with, skin family 0.
	pub skinrefs: Vec<i16>,
	pub submodels: Vec<SubModel>,
}

/// Walks the RLE-compressed anim value spans the way the SDK does, to the
/// span holding frame `k`; returns its byte offset with `k` made span-local.
fn span(b: &[u8], off: usize, k: &mut i32) -> usize {
	let mut p = off;
	while (b[p + 1] as i32) <= *k {
		*k -= b[p + 1] as i32;
		p += (b[p] as usize + 1) * 2;
	}
	p
}

/// The raw anim pair for frame `k` on one axis: this frame's value and the
/// next's, as the SDK's `CalcBonePosition`/`CalcBoneQuaternion` read them.
fn axis_pair(b: &[u8], off: usize, k: i32) -> (f32, f32) {
	let mut k = k;
	let p = span(b, off, &mut k);
	let (valid, total) = (b[p] as i32, b[p + 1] as i32);
	// The first sample follows the two-byte span header; the header itself is
	// animation value zero in GoldSrc's union-based indexing.
	let v = |i: i32| i16_at(b, p + 2 + i as usize * 2) as f32;
	if valid > k {
		let a1 = v(k);
		let a2 = if valid > k + 1 { v(k + 1) } else { a1 };
		(a1, a2)
	} else {
		let a1 = v(valid - 1);
		let a2 = if total > k + 1 { a1 } else { v(valid + 1) };
		(a1, a2)
	}
}

/// The SDK's `AngleQuaternion`: Euler (x, y, z) radians to a quaternion,
/// GoldSrc's own component order.
fn angle_quat(angles: [f32; 3]) -> Quat {
	let (sr, cr) = (angles[0] * 0.5).sin_cos();
	let (sp, cp) = (angles[1] * 0.5).sin_cos();
	let (sy, cy) = (angles[2] * 0.5).sin_cos();
	Quat {
		w: cr * cp * cy + sr * sp * sy,
		x: sr * cp * cy - cr * sp * sy,
		y: cr * sp * cy + sr * cp * sy,
		z: cr * cp * sy - sr * sp * cy,
	}
}

impl Mdl {
	/// Parses the file, or says what about it is not a GoldSrc model.
	pub fn parse(b: Vec<u8>) -> Result<Mdl, String> {
		if b.len() < hdr::NUMBODYPARTS + 4 || b[..4] != IDST {
			return Err("not a GoldSrc studio model (no IDST)".into());
		}
		if i32_at(&b, 4) != VERSION {
			return Err(format!("studio version {}", i32_at(&b, 4)));
		}
		let (nbones, boff) = (
			i32_at(&b, hdr::NUMBONES) as usize,
			i32_at(&b, hdr::BONEINDEX) as usize,
		);
		// mstudiobone_t: name[32], parent, flags, bonecontroller[6], value[6],
		// scale[6] — 112 bytes.
		let bones = (0..nbones)
			.map(|i| {
				let at = boff + i * 112;
				Bone {
					parent: i32_at(&b, at + 32),
					value: std::array::from_fn(|j| f32_at(&b, at + 64 + j * 4)),
					scale: std::array::from_fn(|j| f32_at(&b, at + 88 + j * 4)),
				}
			})
			.collect();
		let (nseq, soff) = (
			i32_at(&b, hdr::NUMSEQ) as usize,
			i32_at(&b, hdr::SEQINDEX) as usize,
		);
		// mstudioseqdesc_t, 176 bytes: label[32], fps, flags, ..., numframes,
		// ..., animindex at byte 124; seqgroup 0 models keep anims in-file.
		let sequences = (0..nseq)
			.map(|i| {
				let at = soff + i * 176;
				Sequence {
					label: str_at(&b, at),
					fps: f32_at(&b, at + 32),
					numframes: i32_at(&b, at + 56),
					looping: i32_at(&b, at + 36) & 0x0001 != 0,
					anim: i32_at(&b, at + 124) as usize,
				}
			})
			.collect();
		let (ntex, toff) = (
			i32_at(&b, hdr::NUMTEXTURES) as usize,
			i32_at(&b, hdr::TEXTUREINDEX) as usize,
		);
		// mstudiotexture_t: name[64], flags, width, height, index — 80 bytes.
		// `index` is a file-absolute offset to the pixels; the 256-entry RGB
		// palette follows the `width × height` indices.
		let textures = (0..ntex)
			.map(|i| {
				let at = toff + i * 80;
				let data = i32_at(&b, at + 76) as usize;
				Texture {
					name: str_at(&b, at),
					flags: i32_at(&b, at + 64),
					width: i32_at(&b, at + 68) as u32,
					height: i32_at(&b, at + 72) as u32,
					data,
					palette: data + i32_at(&b, at + 68) as usize * i32_at(&b, at + 72) as usize,
				}
			})
			.collect();
		let nskinref = i32_at(&b, hdr::NUMSKINREF) as usize;
		let skinrefs = (0..nskinref)
			.map(|i| i16_at(&b, i32_at(&b, hdr::SKININDEX) as usize + i * 2))
			.collect();
		// Bodyparts hold alternate submodels; the game draws one per part by
		// body value. Draw submodel 0 of each.
		let (nparts, poff) = (
			i32_at(&b, hdr::NUMBODYPARTS) as usize,
			i32_at(&b, hdr::BODYPARTINDEX) as usize,
		);
		let mut submodels = Vec::new();
		for i in 0..nparts {
			// mstudiobodyparts_t: name[64], nummodels, base, modelindex.
			let bp = poff + i * 76;
			let nmodels = i32_at(&b, bp + 64) as usize;
			if nmodels == 0 {
				continue;
			}
			let m = i32_at(&b, bp + 72) as usize;
			// mstudiomodel_t, 112 bytes: name[64], type, boundingradius,
			// nummesh, meshindex, numverts, vertinfoindex, vertindex,
			// numnorms, norminfoindex, normindex, numgroups, groupindex.
			// All offsets are file-absolute.
			let (nmesh, meshoff) = (i32_at(&b, m + 72) as usize, i32_at(&b, m + 76) as usize);
			let (nverts, vboff, voff) = (
				i32_at(&b, m + 80) as usize,
				i32_at(&b, m + 84) as usize,
				i32_at(&b, m + 88) as usize,
			);
			let (nnorms, noff) = (i32_at(&b, m + 92) as usize, i32_at(&b, m + 100) as usize);
			if nverts == 0 {
				continue;
			}
			let vert_bones = (0..nverts).map(|j| b[vboff + j]).collect();
			let verts = (0..nverts)
				.map(|j| {
					vec3(
						f32_at(&b, voff + j * 12),
						f32_at(&b, voff + j * 12 + 4),
						f32_at(&b, voff + j * 12 + 8),
					)
				})
				.collect();
			let norms = (0..nnorms)
				.map(|j| {
					vec3(
						f32_at(&b, noff + j * 12),
						f32_at(&b, noff + j * 12 + 4),
						f32_at(&b, noff + j * 12 + 8),
					)
				})
				.collect();
			// mstudiomesh_t: numtris, triindex, skinref, numnorms, normindex.
			let parts = (0..nmesh)
				.map(|j| {
					let at = meshoff + j * 20;
					Part {
						skinref: i32_at(&b, at + 8) as usize,
						tris: read_tris(&b, i32_at(&b, at + 4) as usize),
					}
				})
				.collect();
			submodels.push(SubModel {
				vert_bones,
				verts,
				norms,
				parts,
			});
		}
		Ok(Mdl {
			b,
			bones,
			sequences,
			textures,
			skinrefs,
			submodels,
		})
	}

	/// The first sequence whose label starts with `want`, as 1.6's own models
	/// name them (`idle01`, `shoot1`, `reload`, `draw`, ...).
	pub fn sequence(&self, want: &str) -> Option<usize> {
		self.sequence_exact(want).or_else(|| {
			self
				.sequences
				.iter()
				.position(|s| s.label.to_lowercase().starts_with(&want.to_lowercase()))
		})
	}

	/// The sequence whose label exactly matches `label`, ignoring ASCII case.
	pub fn sequence_exact(&self, label: &str) -> Option<usize> {
		self
			.sequences
			.iter()
			.position(|s| s.label.eq_ignore_ascii_case(label))
	}

	/// The animation's duration, seconds, or 0 for single-frame sequences.
	pub fn seq_duration(&self, seq: usize) -> f32 {
		let s = &self.sequences[seq];
		(s.numframes.max(1) as f32 - 1.0) / s.fps.max(0.001)
	}

	/// World-space bone transforms for frame `f` of `seq`, position and
	/// rotation per bone, the SDK's `SetUpBones`.
	pub fn pose(&self, seq: usize, f: f32) -> Vec<(Vec3, Quat)> {
		self.pose_with_looping(seq, f, self.sequences[seq].looping)
	}

	/// World-space bone transforms with the caller's loop policy. Weapon idles
	/// are replayed by GoldSrc even when their sequence descriptor lacks the
	/// studio-looping flag.
	pub fn pose_with_looping(&self, seq: usize, f: f32, looping: bool) -> Vec<(Vec3, Quat)> {
		let s = &self.sequences[seq];
		// Looping sequences wrap on the last frame, non-looping hold it.
		let span = (s.numframes.max(1) as f32 - 1.0).max(1.0);
		let f = if looping {
			f.rem_euclid(span)
		} else {
			f.clamp(0.0, span)
		};
		let frame = f.floor() as i32;
		let frac = f - frame as f32;
		let mut out = Vec::with_capacity(self.bones.len());
		for (i, bone) in self.bones.iter().enumerate() {
			// mstudioanim_t: six u16 axis offsets, per bone, from `anim`.
			let at = s.anim + i * 12;
			let offset = |j| u16_at(&self.b, at + j * 2) as usize;
			let mut pos = [0.0f32; 3];
			for (j, p) in pos.iter_mut().enumerate() {
				*p = bone.value[j];
				if offset(j) != 0 {
					// The SDK lerps adjacent frames inside a span.
					let (v1, v2) = axis_pair(&self.b, at + offset(j), frame);
					*p += (v1 * (1.0 - frac) + frac * v2) * bone.scale[j];
				}
			}
			let mut angles = [bone.value[3], bone.value[4], bone.value[5]];
			let mut next = angles;
			let mut delta = false;
			for j in 0..3 {
				if offset(j + 3) == 0 {
					continue;
				}
				delta = true;
				let (a1, a2) = axis_pair(&self.b, at + offset(j + 3), frame);
				angles[j] += a1 * bone.scale[j + 3];
				next[j] += a2 * bone.scale[j + 3];
			}
			let rot = if delta {
				angle_quat(angles).nlerp(angle_quat(next), frac)
			} else {
				angle_quat(angles)
			};
			// Fold up the hierarchy: the SDK concatenates 3×4 matrices, which
			// is the parent's rotation applied to this bone's translation.
			let (p, q) = if bone.parent < 0 {
				(vec3(pos[0], pos[1], pos[2]), rot)
			} else {
				let (pp, pq): (Vec3, Quat) = out[bone.parent as usize];
				(pp + pq.rotate(vec3(pos[0], pos[1], pos[2])), pq * rot)
			};
			out.push((p, q));
		}
		out
	}

	/// One texture as RGBA, with palette index 255 dropped where the flags
	/// mask or add alpha — GoldSrc's transparent texel.
	pub fn rgba(&self, tex: usize) -> (u32, u32, Vec<u8>) {
		let t = &self.textures[tex];
		let n = t.width as usize * t.height as usize;
		let mut out = vec![255u8; n * 4];
		let cut = t.flags & (0x0010 | 0x0040) != 0;
		for (i, px) in out.chunks_mut(4).enumerate() {
			let idx = self.b[t.data + i] as usize;
			px[..3].copy_from_slice(&self.b[t.palette + idx * 3..t.palette + idx * 3 + 3]);
			if cut && idx == 255 {
				px[3] = 0;
			}
		}
		(t.width, t.height, out)
	}
}

/// Expands a mesh's triangle commands: runs of corners, strips and fans.
fn read_tris(b: &[u8], at: usize) -> Vec<[Corner; 3]> {
	let mut p = at;
	let mut tris = Vec::new();
	let mut run = |fan: bool, corners: &[Corner]| {
		if fan {
			for i in 1..corners.len() - 1 {
				tris.push([corners[0], corners[i], corners[i + 1]]);
			}
		} else {
			for i in 0..corners.len() - 2 {
				// Strips flip winding every other triangle.
				if i % 2 == 0 {
					tris.push([corners[i], corners[i + 1], corners[i + 2]]);
				} else {
					tris.push([corners[i + 1], corners[i], corners[i + 2]]);
				}
			}
		}
	};
	loop {
		let n = i16_at(b, p);
		p += 2;
		if n == 0 {
			return tris;
		}
		let corners: Vec<Corner> = (0..n.unsigned_abs())
			.map(|i| Corner {
				vert: u16_at(b, p + i as usize * 8) as usize,
				norm: u16_at(b, p + i as usize * 8 + 2) as usize,
				s: i16_at(b, p + i as usize * 8 + 4) as f32,
				t: i16_at(b, p + i as usize * 8 + 6) as f32,
			})
			.collect();
		p += n.unsigned_abs() as usize * 8;
		run(n < 0, &corners);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn animation_samples_start_at_the_first_value_and_interpolate_neighbors() {
		let mut run = vec![3, 4];
		for value in [10i16, 20, 30] {
			run.extend(value.to_le_bytes());
		}

		assert_eq!(axis_pair(&run, 0, 0), (10.0, 20.0));
		assert_eq!(axis_pair(&run, 0, 1), (20.0, 30.0));
		assert_eq!(axis_pair(&run, 0, 2), (30.0, 30.0));
	}

	#[test]
	fn animation_samples_hold_through_a_run_and_continue_at_the_next_run() {
		let mut runs = vec![1, 3];
		runs.extend(5i16.to_le_bytes());
		runs.extend([1, 2]);
		runs.extend(9i16.to_le_bytes());

		assert_eq!(axis_pair(&runs, 0, 0), (5.0, 5.0));
		assert_eq!(axis_pair(&runs, 0, 1), (5.0, 5.0));
		assert_eq!(axis_pair(&runs, 0, 2), (5.0, 9.0));
		assert_eq!(axis_pair(&runs, 0, 3), (9.0, 9.0));
	}

	/// The local Counter-Strike install, when this machine has one; tests
	/// that need the real models skip without it.
	fn cstrike() -> Option<std::path::PathBuf> {
		if let Some(path) = std::env::var_os("CS_CSTRIKE") {
			return Some(path.into());
		}
		let assets = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../assets");
		if assets.join("models").is_dir() {
			return Some(assets);
		}
		Some(std::path::PathBuf::from(
			r"C:\Program Files (x86)\Steam\steamapps\common\Half-Life\cstrike",
		))
	}

	#[test]
	fn parses_v_awp_with_its_sequences() {
		let Some(dir) = cstrike() else {
			return;
		};
		let bytes = std::fs::read(dir.join("models/v_awp.mdl")).unwrap();
		let mdl = Mdl::parse(bytes).unwrap();
		assert!(mdl.bones.len() > 5, "{:?}", mdl.bones.len());
		assert!(mdl.sequence("idle").is_some());
		assert!(mdl.sequence("shoot").is_some());
		assert!(mdl.sequence("reload").is_some());
		assert!(mdl.sequence("draw").is_some());
		assert!(!mdl.submodels.is_empty());
		assert!(!mdl.textures.is_empty());
		for sub in &mdl.submodels {
			assert!(!sub.verts.is_empty());
			assert!(sub.parts.iter().any(|p| !p.tris.is_empty()));
		}
		// The idle pose is finite and within the model's bounding box scale.
		let pose = mdl.pose(mdl.sequence("idle").unwrap(), 0.0);
		assert_eq!(pose.len(), mdl.bones.len());
		assert!(pose.iter().all(|(p, _)| p.length() < 4096.0));
	}

	#[test]
	fn reload_sequences_take_real_time() {
		let Some(dir) = cstrike() else {
			return;
		};
		for name in ["v_ak47", "v_m4a1", "v_mp5", "v_deagle", "v_usp"] {
			let bytes = std::fs::read(dir.join(format!("models/{name}.mdl"))).unwrap();
			let mdl = Mdl::parse(bytes).unwrap();
			let reload = mdl.sequence("reload").expect(name);
			let d = mdl.seq_duration(reload);
			assert!((1.0..=8.0).contains(&d), "{name} reload {d}s");
		}
	}

	#[test]
	fn reads_sequence_animation_index_at_studio_header_offset() {
		let Some(dir) = cstrike() else {
			return;
		};
		let bytes = std::fs::read(dir.join("models/v_ak47.mdl")).unwrap();
		let mdl = Mdl::parse(bytes.clone()).unwrap();
		let seq_index = i32_at(&bytes, hdr::SEQINDEX) as usize;
		for (i, seq) in mdl.sequences.iter().enumerate() {
			assert_eq!(seq.anim, i32_at(&bytes, seq_index + i * 176 + 124) as usize);
			assert!(seq.anim < bytes.len(), "{} animation offset", seq.label);
		}
	}

	#[test]
	fn weapon_idle_can_loop_without_a_studio_loop_flag() {
		let Some(dir) = cstrike() else {
			return;
		};
		let bytes = std::fs::read(dir.join("models/v_ak47.mdl")).unwrap();
		let mdl = Mdl::parse(bytes).unwrap();
		let idle = mdl.sequence("idle").unwrap();
		assert!(!mdl.sequences[idle].looping);
		let period = (mdl.sequences[idle].numframes - 1) as f32;
		let first = mdl.pose_with_looping(idle, 0.0, true);
		let wrapped = mdl.pose_with_looping(idle, period, true);
		assert_eq!(first.len(), wrapped.len());
		for ((p1, q1), (p2, q2)) in first.iter().zip(wrapped.iter()) {
			assert!((*p1 - *p2).length() < 0.001);
			assert!((q1.w - q2.w).abs() < 0.001);
			assert!((q1.x - q2.x).abs() < 0.001);
			assert!((q1.y - q2.y).abs() < 0.001);
			assert!((q1.z - q2.z).abs() < 0.001);
		}
	}

	#[test]
	fn rejects_a_non_model() {
		assert!(Mdl::parse(vec![0; 512]).is_err());
	}
}
