//! The math every crate here shares, written once.
//!
//! - [`Vec3`]: points, directions and component-wise triples.
//! - [`Quat`][]: rotations.
//! - [`Mat4`]: column-major transforms in OpenGL clip conventions.
//! - [`Camera`]: a perspective view, as the matrix that draws it.
//!
//! Right-handed, y up.

mod camera;
mod mat4;
mod quat;
mod vec3;

pub use camera::Camera;
pub use mat4::Mat4;
pub use quat::Quat;
pub use vec3::{vec3, Vec3};

/// `alpha` of the way from `a` to `b`.
pub fn lerp(a: f32, b: f32, alpha: f32) -> f32 {
	a + (b - a) * alpha
}
