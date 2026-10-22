//! Engine-owned math, collision, console, variables, map, asset, lighting,
//! material, rendering, and menu widget systems.
//!
//! Gameplay depends on this crate. No engine package depends on gameplay.

pub mod bsp;
pub mod goldsrc;
pub mod gpu;
pub mod lightmap;
pub mod surface;
pub mod ui;

pub use console;
pub use cvar;
pub use lightbake;
pub use linalg;
pub use physics;
pub use surface::Surface;
