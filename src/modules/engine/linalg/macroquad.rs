//! Hand-off to macroquad for drawing; no math happens here.

use crate::{Camera, Mat4, Quat, Vec3};
use macroquad::{camera, math, texture, window};

impl From<Vec3> for math::Vec3 {
	fn from(v: Vec3) -> math::Vec3 {
		math::vec3(v.x, v.y, v.z)
	}
}

impl From<math::Vec3> for Vec3 {
	fn from(v: math::Vec3) -> Vec3 {
		crate::vec3(v.x, v.y, v.z)
	}
}

impl From<Quat> for math::Quat {
	fn from(q: Quat) -> math::Quat {
		math::Quat::from_xyzw(q.x, q.y, q.z, q.w)
	}
}

impl From<Mat4> for math::Mat4 {
	fn from(m: Mat4) -> math::Mat4 {
		math::Mat4::from_cols_array_2d(&m.cols)
	}
}

/// Draws with the screen's aspect ratio, like macroquad's `Camera3D`.
impl camera::Camera for Camera {
	fn matrix(&self) -> math::Mat4 {
		self
			.view_projection(window::screen_width() / window::screen_height())
			.into()
	}

	fn depth_enabled(&self) -> bool {
		true
	}

	fn render_pass(&self) -> Option<texture::RenderPass> {
		None
	}

	fn viewport(&self) -> Option<(i32, i32, i32, i32)> {
		None
	}
}

#[cfg(test)]
mod tests {
	use crate::{vec3, Camera, Mat4, Quat};
	use macroquad::math;

	fn close(a: math::Mat4, b: math::Mat4) -> bool {
		a.to_cols_array()
			.iter()
			.zip(b.to_cols_array())
			.all(|(x, y)| (x - y).abs() < 1e-4)
	}

	#[test]
	fn matrices_match_macroquads() {
		let cam = Camera {
			fovy: 0.8,
			..Camera::orbit(vec3(1.0, -2.0, 0.5), -1.9, 0.55, 11.0)
		};
		let theirs = math::Mat4::perspective_rh_gl(cam.fovy, 1.6, cam.near, cam.far)
			* math::Mat4::look_at_rh(cam.eye.into(), cam.target.into(), cam.up.into());
		assert!(close(cam.view_projection(1.6).into(), theirs));

		let q = Quat::from_axis_angle(vec3(1.0, 2.0, -0.5).normalize(), 1.1);
		let t = vec3(3.0, -1.0, 2.0);
		let theirs = math::Mat4::from_rotation_translation(q.into(), t.into());
		assert!(close(Mat4::from_rotation_translation(q, t).into(), theirs));
	}
}
