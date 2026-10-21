use crate::{Quat, Vec3};
use std::ops::Mul;

/// 4×4 matrix stored as four columns; it multiplies column vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
	pub cols: [[f32; 4]; 4],
}

impl Mat4 {
	pub const IDENTITY: Mat4 = Mat4 {
		cols: [
			[1.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
	};

	/// Rotates by `rotation`, then moves by `translation`.
	pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Mat4 {
		let [x, y, z] = [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| rotation.rotate(axis));
		Mat4 {
			cols: [
				[x.x, x.y, x.z, 0.0],
				[y.x, y.y, y.z, 0.0],
				[z.x, z.y, z.z, 0.0],
				[translation.x, translation.y, translation.z, 1.0],
			],
		}
	}

	/// World to view space for an eye at `eye` looking at `target`: the view looks down
	/// its -z with `up` toward its +y.
	pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
		let f = (target - eye).normalize();
		let s = f.cross(up).normalize();
		let u = s.cross(f);
		Mat4 {
			cols: [
				[s.x, u.x, -f.x, 0.0],
				[s.y, u.y, -f.y, 0.0],
				[s.z, u.z, -f.z, 0.0],
				[-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
			],
		}
	}

	/// View to clip space with a vertical field of view `fovy` in radians, depth from
	/// `near` to `far` mapped onto OpenGL's [-1, 1].
	pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
		let f = 1.0 / (fovy * 0.5).tan();
		let inv = 1.0 / (near - far);
		Mat4 {
			cols: [
				[f / aspect, 0.0, 0.0, 0.0],
				[0.0, f, 0.0, 0.0],
				[0.0, 0.0, (near + far) * inv, -1.0],
				[0.0, 0.0, 2.0 * near * far * inv, 0.0],
			],
		}
	}

	/// `p` with w = 1 transformed, divided by the resulting w.
	pub fn project_point(&self, p: Vec3) -> Vec3 {
		let c = &self.cols;
		let row = |i: usize| c[0][i] * p.x + c[1][i] * p.y + c[2][i] * p.z + c[3][i];
		Vec3 {
			x: row(0),
			y: row(1),
			z: row(2),
		} / row(3)
	}
}

impl Mul for Mat4 {
	type Output = Mat4;
	fn mul(self, o: Mat4) -> Mat4 {
		let mut cols = [[0.0; 4]; 4];
		for (j, col) in cols.iter_mut().enumerate() {
			for (i, out) in col.iter_mut().enumerate() {
				*out = (0..4).map(|k| self.cols[k][i] * o.cols[j][k]).sum();
			}
		}
		Mat4 { cols }
	}
}
