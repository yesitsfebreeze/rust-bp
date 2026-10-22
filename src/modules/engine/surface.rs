/// Material categories shared by engine rendering and gameplay systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
	Tile,
	Concrete,
	Marble,
	Jade,
	Steel,
}

impl Surface {
	pub const ALL: [Surface; 5] = [
		Surface::Tile,
		Surface::Concrete,
		Surface::Marble,
		Surface::Jade,
		Surface::Steel,
	];
}

/// How a surface reflects light, for physically based shading: how rough
/// its microsurface is, 0 a mirror to 1 chalk, and whether it is metal,
/// whose reflection takes its own color and which has no diffuse part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Finish {
	pub roughness: f32,
	pub metal: f32,
}

impl Default for Finish {
	/// Plaster and sandstone: rough, not metal.
	fn default() -> Finish {
		Finish {
			roughness: 0.85,
			metal: 0.0,
		}
	}
}

impl Finish {
	/// The finish a GoldSrc texture name stands for. The maps carry no
	/// material data, so the name's words decide: metal and grates are
	/// metal, tile and marble polished, wood and crates satin, sand and
	/// dirt chalky, everything else the default plaster and stone.
	pub fn of(tex_name: &str) -> Finish {
		let name = tex_name.to_ascii_lowercase();
		let has = |words: &[&str]| words.iter().any(|w| name.contains(w));
		let (roughness, metal) = if has(&[
			"metal", "steel", "pipe", "grate", "vent", "rail", "tin", "iron",
		]) {
			(0.35, 1.0)
		} else if has(&["water"]) {
			(0.05, 0.0)
		} else if has(&["glass", "window"]) {
			(0.1, 0.0)
		} else if has(&["tile", "marble", "lino"]) {
			(0.3, 0.0)
		} else if has(&["crt", "crate", "wood", "door", "box", "barrel"]) {
			(0.65, 0.0)
		} else if has(&["sand", "dirt", "ground", "grass", "mud"]) {
			(0.95, 0.0)
		} else {
			return Finish::default();
		};
		Finish { roughness, metal }
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn texture_names_pick_their_finish() {
		assert_eq!(Finish::of("METAL_DOOR02").metal, 1.0);
		assert_eq!(Finish::of("{grate1").metal, 1.0);
		assert!(Finish::of("DUSTCRATE").roughness < Finish::default().roughness);
		assert_eq!(Finish::of("DUSTWALL01"), Finish::default());
	}
}
