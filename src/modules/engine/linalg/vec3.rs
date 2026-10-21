use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
	pub x: f32,
	pub y: f32,
	pub z: f32,
}

pub const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
	Vec3 { x, y, z }
}

impl Vec3 {
	pub const ZERO: Vec3 = vec3(0.0, 0.0, 0.0);
	pub const X: Vec3 = vec3(1.0, 0.0, 0.0);
	pub const Y: Vec3 = vec3(0.0, 1.0, 0.0);
	pub const Z: Vec3 = vec3(0.0, 0.0, 1.0);

	pub const fn splat(s: f32) -> Vec3 {
		vec3(s, s, s)
	}

	pub fn dot(self, o: Vec3) -> f32 {
		self.x * o.x + self.y * o.y + self.z * o.z
	}

	pub fn cross(self, o: Vec3) -> Vec3 {
		vec3(
			self.y * o.z - self.z * o.y,
			self.z * o.x - self.x * o.z,
			self.x * o.y - self.y * o.x,
		)
	}

	pub fn length2(self) -> f32 {
		self.dot(self)
	}

	pub fn length(self) -> f32 {
		self.length2().sqrt()
	}

	pub fn normalize(self) -> Vec3 {
		self * (1.0 / self.length())
	}

	pub fn min(self, o: Vec3) -> Vec3 {
		vec3(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
	}

	pub fn max(self, o: Vec3) -> Vec3 {
		vec3(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
	}

	/// `alpha` of the way from `self` to `o`.
	pub fn lerp(self, o: Vec3, alpha: f32) -> Vec3 {
		self + (o - self) * alpha
	}

	/// `f` applied to each component.
	pub fn map(self, f: impl Fn(f32) -> f32) -> Vec3 {
		vec3(f(self.x), f(self.y), f(self.z))
	}
}

impl Add for Vec3 {
	type Output = Vec3;
	fn add(self, o: Vec3) -> Vec3 {
		vec3(self.x + o.x, self.y + o.y, self.z + o.z)
	}
}

impl Sub for Vec3 {
	type Output = Vec3;
	fn sub(self, o: Vec3) -> Vec3 {
		vec3(self.x - o.x, self.y - o.y, self.z - o.z)
	}
}

impl Mul<f32> for Vec3 {
	type Output = Vec3;
	fn mul(self, s: f32) -> Vec3 {
		vec3(self.x * s, self.y * s, self.z * s)
	}
}

impl Div<f32> for Vec3 {
	type Output = Vec3;
	fn div(self, s: f32) -> Vec3 {
		self * (1.0 / s)
	}
}

/// Component-wise.
impl Div for Vec3 {
	type Output = Vec3;
	fn div(self, o: Vec3) -> Vec3 {
		vec3(self.x / o.x, self.y / o.y, self.z / o.z)
	}
}

impl Neg for Vec3 {
	type Output = Vec3;
	fn neg(self) -> Vec3 {
		vec3(-self.x, -self.y, -self.z)
	}
}

impl AddAssign for Vec3 {
	fn add_assign(&mut self, o: Vec3) {
		*self = *self + o;
	}
}

impl SubAssign for Vec3 {
	fn sub_assign(&mut self, o: Vec3) {
		*self = *self - o;
	}
}

impl MulAssign<f32> for Vec3 {
	fn mul_assign(&mut self, s: f32) {
		*self = *self * s;
	}
}
