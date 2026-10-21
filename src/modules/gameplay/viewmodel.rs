//! The 1.6 viewmodels: the GoldSrc `.mdl` from the local Counter-Strike
//! install, skinned onto the drawn bones and played sequence by sequence —
//! idle, shoot, reload, draw — with the install's `.wav` sounds.

use crate::weapons::Weapon;
use engine::goldsrc::Mdl;
use engine::linalg::Vec3;
use engine::macroquad;
use engine::macroquad::models::{Mesh, Vertex};
use engine::macroquad::prelude::*;

/// The game's own assets folder: the 1.6 models, sounds and maps copied from
/// the local install, laid out like the install itself.
pub fn assets() -> std::path::PathBuf {
	crate::root().join("assets")
}

/// Where the 1.6 assets live: `CS_CSTRIKE` when set, the `assets` folder when
/// it holds models, else the local Steam install.
pub fn cstrike() -> Option<std::path::PathBuf> {
	if let Some(p) = std::env::var_os("CS_CSTRIKE") {
		return Some(p.into());
	}
	let assets = assets();
	if assets.join("models").is_dir() {
		return Some(assets);
	}
	Some(std::path::PathBuf::from(
		r"C:\Program Files (x86)\Steam\steamapps\common\Half-Life\cstrike",
	))
}

/// Which of the model's sequences each event plays, found by 1.6's labels.
#[derive(Clone, Debug)]
pub struct Sequences {
	pub idle: Vec<usize>,
	pub draw: Vec<usize>,
	pub reload: Vec<usize>,
	pub shoot: Vec<usize>,
	pub last_shot: Option<usize>,
	pub slash: Vec<usize>,
	pub stab: Option<usize>,
}

impl Sequences {
	fn of(mdl: &Mdl) -> Sequences {
		let choices = |labels: &[&str]| {
			labels
				.iter()
				.filter_map(|label| mdl.sequence_exact(label))
				.collect::<Vec<_>>()
		};
		let idle = choices(&["idle_unsil", "idle1", "idle", "idle2", "idle3"]);
		let draw = choices(&["draw_unsil", "draw", "draw2"]);
		let reload = choices(&["reload_unsil", "reload", "reload2"]);
		let shoot = [
			"shoot1_unsil",
			"shoot2_unsil",
			"shoot3_unsil",
			"shoot1",
			"shoot2",
			"shoot3",
			"shoot",
		]
			.iter()
			.filter_map(|label| mdl.sequence_exact(label))
			.collect();
		Sequences {
			idle: if idle.is_empty() { vec![0] } else { idle },
			draw: if draw.is_empty() { vec![0] } else { draw },
			reload: if reload.is_empty() { vec![0] } else { reload },
			shoot,
			last_shot: [
				"shootlast_unsil",
				"shootlast",
				"shoot_empty",
			]
			.iter()
			.find_map(|label| mdl.sequence_exact(label)),
			slash: choices(&["slash1", "slash2"]),
			stab: mdl
				.sequence_exact("stab_miss")
				.or_else(|| mdl.sequence_exact("stab")),
		}
	}
}

fn choose(options: &[usize]) -> usize {
	match options {
		[] => 0,
		[only] => *only,
		_ => options[macroquad::rand::gen_range(0usize, options.len())],
	}
}

/// A loaded viewmodel: the model, its palette textures, and what is playing.
pub struct Viewmodel {
	mdl: Mdl,
	textures: Vec<Texture2D>,
	seqs: Sequences,
	/// The playing sequence, when it started and whether it loops.
	playing: usize,
	started: f32,
	looping: bool,
}

impl Viewmodel {
	/// Loads `models/v_<weapon>.mdl` from the install; None without the model.
	pub fn load(dir: &std::path::Path, w: &Weapon) -> Option<Viewmodel> {
		let name = w.name.strip_prefix("knife").unwrap_or(&w.name);
		let path = dir.join("models").join(format!("v_{name}.mdl"));
		let bytes = std::fs::read(path).ok()?;
		let mdl = Mdl::parse(bytes).ok()?;
		let textures = (0..mdl.textures.len())
			.map(|i| {
				let (width, height, rgba) = mdl.rgba(i);
				Texture2D::from_rgba8(width as u16, height as u16, &rgba)
			})
			.collect();
		let seqs = Sequences::of(&mdl);
		let playing = choose(&seqs.idle);
		Some(Viewmodel {
			mdl,
			textures,
			seqs,
			playing,
			started: 0.0,
			looping: true,
		})
	}

	/// The draw sequence's duration, for the deploy's end time.
	pub fn draw_duration(&self) -> f32 {
		self.seqs
			.draw
			.iter()
			.map(|&seq| self.mdl.seq_duration(seq))
			.fold(0.0, f32::max)
	}

	/// The reload sequence's duration, when the weapon's own is missing.
	pub fn reload_duration(&self) -> f32 {
		self.seqs
			.reload
			.iter()
			.map(|&seq| self.mdl.seq_duration(seq))
			.fold(0.0, f32::max)
	}

	/// Starts a sequence by label: `shoot`, `reload`, `draw`; a playing
	/// sequence restarts. `last_shot` selects the model's special final-round
	/// animation when it has one. `now` is the game clock the model animates against.
	pub fn start(&mut self, label: &str, now: f32, last_shot: bool) {
		let seq = match label {
			"shoot" if !self.seqs.slash.is_empty() => choose(&self.seqs.slash),
			"shoot" if last_shot => self
				.seqs
				.last_shot
				.unwrap_or_else(|| choose(&self.seqs.shoot)),
			"shoot" => choose(&self.seqs.shoot),
			"reload" => choose(&self.seqs.reload),
			"draw" => choose(&self.seqs.draw),
			"stab" => self.seqs.stab.unwrap_or_else(|| choose(&self.seqs.idle)),
			_ => choose(&self.seqs.idle),
		};
		self.playing = seq;
		self.started = now;
		self.looping = self.mdl.sequences[seq].looping || label == "idle";
	}

	/// Advances nothing but the end of one-shot sequences; the frame time
	/// drives the pose, so this only walks back to idle.
	pub fn tick(&mut self, now: f32) {
		if !self.looping && now - self.started >= self.mdl.seq_duration(self.playing) {
			self.start("idle", now);
		}
	}

	#[cfg(test)]
	mod tests {
		use super::*;

		fn model(dir: &std::path::Path, name: &str) -> Mdl {
			let bytes = std::fs::read(dir.join("models").join(format!("v_{name}.mdl"))).unwrap();
			Mdl::parse(bytes).unwrap()
		}

		fn label<'a>(mdl: &'a Mdl, seq: usize) -> &'a str {
			&mdl.sequences[seq].label
		}

		#[test]
		fn selects_model_clips_by_exact_label_and_default_weapon_state() {
			let Some(dir) = cstrike().filter(|dir| dir.is_dir()) else {
				return;
			};

			for name in ["m4a1", "usp"] {
				let mdl = model(&dir, name);
				let seqs = Sequences::of(&mdl);
				assert_eq!(label(&mdl, seqs.idle[0]), "idle_unsil");
				assert_eq!(label(&mdl, seqs.draw[0]), "draw_unsil");
				assert_eq!(label(&mdl, seqs.reload[0]), "reload_unsil");
				assert!(seqs
					.shoot
					.iter()
					.all(|&seq| label(&mdl, seq).ends_with("_unsil")));
			}

			let glock = model(&dir, "glock18");
			let seqs = Sequences::of(&glock);
			assert_eq!(seqs.idle.len(), 3);
			assert_eq!(seqs.draw.len(), 2);
			assert_eq!(seqs.reload.len(), 2);
			assert_eq!(seqs.shoot.len(), 3);
			assert_eq!(label(&glock, seqs.last_shot.unwrap()), "shoot_empty");

			let knife = model(&dir, "knife");
			let seqs = Sequences::of(&knife);
			assert_eq!(seqs.slash.len(), 2);
			assert_eq!(label(&knife, seqs.slash[0]), "slash1");
			assert_eq!(label(&knife, seqs.stab.unwrap()), "stab_miss");
		}
	}

	/// Draws the weapon in front of the eye: GoldSrc models live in view space
	/// (x forward, y left, z up), so the bones land where the eye looks, with
	/// the shot's `recoil` shoving the gun back.
	pub fn draw(&self, eye: Vec3, forward: Vec3, right: Vec3, up: Vec3, recoil: f32, now: f32) {
		let eye = eye + forward * (-recoil * 2.0) + up * (recoil * 1.0);
		let elapsed = now - self.started;
		let frame = elapsed * self.mdl.sequences[self.playing].fps;
		let pose = self.mdl.pose(self.playing, frame);
		let q = |v: Vec3| macroquad::math::Vec3::from(v);
		for sub in &self.mdl.submodels {
			for part in &sub.parts {
				let Some(&tex_index) = self.mdl.skinrefs.get(part.skinref) else {
					continue;
				};
				let Some(tex) = self.textures.get(tex_index as usize) else {
					continue;
				};
				let (width, height) = (
					self.mdl.textures[tex_index as usize].width as f32,
					self.mdl.textures[tex_index as usize].height as f32,
				);
				// The gun's own space is GoldSrc's: x forward, y left, z up.
				let place = |v: Vec3| eye + forward * v.x - right * v.y + up * v.z;
				let mut vertices = Vec::with_capacity(sub.verts.len());
				for tri in &part.tris {
					for c in tri {
						let (bp, bq) = pose[sub.vert_bones[c.vert] as usize];
						let p = place(bp + bq.rotate(sub.verts[c.vert]));
						let n = bq.rotate(sub.norms[c.norm]);
						vertices.push(Vertex {
							position: q(p),
							uv: Vec2::new((c.s + 0.5) / width, (c.t + 0.5) / height),
							color: WHITE.into(),
							normal: macroquad::math::Vec4::new(n.x, n.y, n.z, 0.0),
						});
					}
				}
				let indices: Vec<u16> = (0..part.tris.len() as u16 * 3).collect();
				draw_mesh(&Mesh {
					vertices,
					indices,
					texture: Some(tex.clone()),
				});
			}
		}
	}
}
