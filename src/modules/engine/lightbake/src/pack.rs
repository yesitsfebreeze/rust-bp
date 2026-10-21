//! Shelf rectangle packing into a fixed square: rows filled left to right, the
//! caller sorting tallest first keeps rows shallow. Every placement carries a
//! one-texel gutter on the right and below, so bilinear filtering at a chart
//! edge never reaches a neighbouring chart's texels.

pub struct Packer {
	size: u32,
	/// Next free x in the current row, and the row's bottom edge.
	x: u32,
	y: u32,
	/// Height of the current row: the tallest rectangle placed in it.
	row: u32,
}

impl Packer {
	pub fn new(size: u32) -> Packer {
		Packer {
			size,
			x: 0,
			y: 0,
			row: 0,
		}
	}

	/// Atlas position of the next `w` × `h` rectangle, `None` when it does not
	/// fit. The returned position is the rectangle's corner; the gutter sits
	/// outside it, so packing never returns overlapping rectangles.
	pub fn place(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
		let (w, h) = (w + 1, h + 1); // gutter
		if w > self.size || h > self.size {
			return None;
		}
		if self.x + w > self.size {
			self.y += self.row;
			(self.x, self.row) = (0, 0);
		}
		if self.y + h > self.size {
			return None;
		}
		let at = (self.x, self.y);
		self.x += w;
		self.row = self.row.max(h);
		Some(at)
	}
}

#[cfg(test)]
mod tests {
	use super::Packer;

	#[test]
	fn placed_rects_never_overlap() {
		let mut p = Packer::new(16);
		let mut placed = Vec::new();
		for (w, h) in [(4, 4), (7, 3), (2, 9), (5, 5), (16, 1), (3, 3)] {
			let at = p.place(w, h);
			if let Some((x, y)) = at {
				placed.push((x, y, w, h));
			}
		}
		for (i, &(x, y, w, h)) in placed.iter().enumerate() {
			assert!(x + w < 16 && y + h < 16, "{i} outside");
			for &(u, v, a, b) in placed.iter().take(i) {
				let overlap = x < u + a + 1 && u < x + w + 1 && y < v + b + 1 && v < y + h + 1;
				assert!(!overlap, "charts {i} and its earlier neighbor share texels");
			}
		}
	}

	#[test]
	fn taller_than_the_atlas_never_places() {
		assert!(Packer::new(8).place(8, 9).is_none());
	}
}
