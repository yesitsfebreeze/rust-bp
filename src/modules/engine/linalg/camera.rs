use crate::{Mat4, Vec3};

/// A perspective view from `eye` toward `target`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
	pub eye: Vec3,
	pub target: Vec3,
	pub up: Vec3,
	/// Vertical field of view, radians.
	pub fovy: f32,
	pub near: f32,
	pub far: f32,
}

impl Default for Camera {
	fn default() -> Camera {
		Camera {
			eye: Vec3::Z,
			target: Vec3::ZERO,
			up: Vec3::Y,
			fovy: 45f32.to_radians(),
			near: 0.01,
			far: 10000.0,
		}
	}
}

impl Camera {
	/// Looking at `target` from `distance` away: `yaw` radians around the y axis from
	/// +x toward +z, `pitch` radians above the horizon.
	pub fn orbit(target: Vec3, yaw: f32, pitch: f32, distance: f32) -> Camera {
		let (sy, cy) = yaw.sin_cos();
		let (sp, cp) = pitch.sin_cos();
		Camera {
			eye: target
				+ Vec3 {
					x: cp * cy,
					y: sp,
					z: cp * sy,
				} * distance,
			target,
			..Camera::default()
		}
	}

	/// Unit direction the camera looks along.
	pub fn forward(&self) -> Vec3 {
		(self.target - self.eye).normalize()
	}

	/// World to clip space for a viewport `aspect` wide per unit of height.
	pub fn view_projection(&self, aspect: f32) -> Mat4 {
		Mat4::perspective(self.fovy, aspect, self.near, self.far)
			* Mat4::look_at(self.eye, self.target, self.up)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::vec3;

	#[test]
	fn target_projects_to_screen_center_between_the_planes() {
		let cam = Camera::orbit(vec3(1.0, 2.0, 3.0), 0.7, 0.4, 10.0);
		let p = cam.view_projection(16.0 / 9.0).project_point(cam.target);
		assert!(
			p.x.abs() < 1e-5 && p.y.abs() < 1e-5 && p.z > -1.0 && p.z < 1.0,
			"{p:?}"
		);
		assert!(((cam.eye - cam.target).length() - 10.0).abs() < 1e-4);
	}

	#[test]
	fn near_and_far_planes_map_to_clip_bounds() {
		let cam = Camera {
			eye: Vec3::ZERO,
			target: -Vec3::Z,
			near: 0.5,
			far: 100.0,
			..Camera::default()
		};
		let m = cam.view_projection(1.0);
		assert!((m.project_point(vec3(0.0, 0.0, -0.5)).z + 1.0).abs() < 1e-5);
		assert!((m.project_point(vec3(0.0, 0.0, -100.0)).z - 1.0).abs() < 1e-4);
		// Up stays up and right stays right.
		assert!(m.project_point(vec3(0.0, 1.0, -5.0)).y > 0.0);
		assert!(m.project_point(vec3(1.0, 0.0, -5.0)).x > 0.0);
	}
}
