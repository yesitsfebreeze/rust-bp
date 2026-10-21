use crate::{vec3, Vec3};
use std::ops::Mul;

/// Unit quaternion: a rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
	pub w: f32,
	pub x: f32,
	pub y: f32,
	pub z: f32,
}

impl Quat {
	pub const IDENTITY: Quat = Quat {
		w: 1.0,
		x: 0.0,
		y: 0.0,
		z: 0.0,
	};

	/// Rotation by `angle` radians about the unit vector `axis`.
	pub fn from_axis_angle(axis: Vec3, angle: f32) -> Quat {
		let (s, c) = (angle * 0.5).sin_cos();
		Quat {
			w: c,
			x: axis.x * s,
			y: axis.y * s,
			z: axis.z * s,
		}
	}

	fn vector(self) -> Vec3 {
		vec3(self.x, self.y, self.z)
	}

	pub fn dot(self, o: Quat) -> f32 {
		self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z
	}

	pub fn normalize(self) -> Quat {
		let s = 1.0 / self.dot(self).sqrt();
		Quat {
			w: self.w * s,
			x: self.x * s,
			y: self.y * s,
			z: self.z * s,
		}
	}

	pub fn conjugate(self) -> Quat {
		Quat {
			w: self.w,
			x: -self.x,
			y: -self.y,
			z: -self.z,
		}
	}

	pub fn rotate(self, v: Vec3) -> Vec3 {
		let u = self.vector();
		let t = u.cross(v) * 2.0;
		v + t * self.w + u.cross(t)
	}

	/// This orientation after spinning at angular velocity `spin`, world space radians
	/// per second, for `dt`.
	pub fn integrate(self, spin: Vec3, dt: f32) -> Quat {
		let q = Quat {
			w: 0.0,
			x: spin.x,
			y: spin.y,
			z: spin.z,
		} * self;
		let h = 0.5 * dt;
		Quat {
			w: self.w + q.w * h,
			x: self.x + q.x * h,
			y: self.y + q.y * h,
			z: self.z + q.z * h,
		}
		.normalize()
	}

	/// Normalized blend `alpha` of the way toward `o`, along the shorter arc.
	pub fn nlerp(self, o: Quat, alpha: f32) -> Quat {
		let s = if self.dot(o) < 0.0 { -alpha } else { alpha };
		let a = 1.0 - alpha;
		Quat {
			w: self.w * a + o.w * s,
			x: self.x * a + o.x * s,
			y: self.y * a + o.y * s,
			z: self.z * a + o.z * s,
		}
		.normalize()
	}
}

impl Mul for Quat {
	type Output = Quat;
	fn mul(self, o: Quat) -> Quat {
		let (a, b) = (self.vector(), o.vector());
		let v = b * self.w + a * o.w + a.cross(b);
		Quat {
			w: self.w * o.w - a.dot(b),
			x: v.x,
			y: v.y,
			z: v.z,
		}
	}
}
