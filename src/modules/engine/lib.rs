//! Engine-owned math, collision, console, variables, map, asset, lighting,
//! material, and rendering systems.
//!
//! Gameplay depends on this crate. No engine package depends on gameplay.

pub mod bsp;
pub mod goldsrc;
pub mod lightmap;
pub mod map_render;
pub mod render;
pub mod surface;
pub mod textures;

pub use console;
pub use cvar;
pub use lightbake;
pub use linalg;
pub use macroquad;
pub use physics;
pub use surface::Surface;
