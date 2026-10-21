//! What bullets do to each surface, in the spirit of Counter-Strike 1.6's
//! penetration: a bullet punches through walls up to a material-dependent
//! thickness, paying damage for every wall it crosses.

use engine::Surface;

/// How a surface resists bullets.
#[derive(Clone, Copy, Debug)]
pub struct Material {
	pub name: &'static str,
	/// Thickest wall, in units, a bullet passes through; a thicker wall stops it.
	pub max_thickness: f32,
	/// Damage multiplied by this for every wall of the material crossed.
	pub damage_mod: f32,
}

pub trait SurfaceMaterial {
	fn material(self) -> Material;
}

impl SurfaceMaterial for Surface {
	/// 1.6-flavored resistance: jade crates punch like wood, concrete and glazed
	/// tile bleed half the damage, marble stone bites hardest, steel stops the
	/// thinnest walls.
	fn material(self) -> Material {
		match self {
			Surface::Tile => Material {
				name: "tile",
				max_thickness: 40.0,
				damage_mod: 0.5,
			},
			Surface::Concrete => Material {
				name: "concrete",
				max_thickness: 60.0,
				damage_mod: 0.5,
			},
			Surface::Marble => Material {
				name: "marble",
				max_thickness: 50.0,
				damage_mod: 0.4,
			},
			Surface::Jade => Material {
				name: "jade",
				max_thickness: 48.0,
				damage_mod: 0.75,
			},
			Surface::Steel => Material {
				name: "steel",
				max_thickness: 36.0,
				damage_mod: 0.5,
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn every_surface_resists_bullets() {
		for s in Surface::ALL {
			let m = s.material();
			assert!(m.max_thickness > 0.0, "{s:?}");
			assert!(
				m.damage_mod > 0.0 && m.damage_mod < 1.0,
				"{s:?} keeps some damage through a wall"
			);
		}
	}
}
