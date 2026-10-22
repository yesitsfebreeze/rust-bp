//! HUD text: ProggyClean, macroquad's old default font, rasterized with
//! fontdue into one coverage atlas, a glyph set per pixel size the first
//! time that size draws. The atlas's top-left texels are solid, for
//! rectangles and lines.

use std::collections::HashMap;

/// The atlas side, texels.
pub const SIDE: u32 = 1024;
/// The solid block in the top-left corner.
pub const SOLID: u32 = 4;
/// Printable ASCII: space through tilde.
const FIRST: char = ' ';
const LAST: char = '~';

const FONT: &[u8] = include_bytes!("ProggyClean.ttf");

/// One rasterized glyph.
#[derive(Clone, Copy, Debug, Default)]
pub struct Glyph {
	/// Its texels in the atlas: x, y, width, height.
	pub rect: [u32; 4],
	/// Offset of its bitmap's left from the pen, and of its bottom above
	/// the baseline.
	pub xmin: f32,
	pub ymin: f32,
	pub advance: f32,
}

pub struct Glyphs {
	font: fontdue::Font,
	sets: HashMap<u16, Vec<Glyph>>,
	/// The shelf packer for the atlas, `lightbake`'s; it starts past the
	/// solid block in the top-left corner, which that first row clears.
	packer: lightbake::pack::Packer,
}

impl Glyphs {
	pub fn new() -> Result<Glyphs, String> {
		let font = fontdue::Font::from_bytes(FONT, fontdue::FontSettings::default())
			.map_err(|e| format!("font: {e}"))?;
		Ok(Glyphs {
			font,
			sets: HashMap::new(),
			packer: lightbake::pack::Packer::restore([SIDE, SOLID, 0, SOLID]),
		})
	}

	/// The glyphs at `px` pixels, rasterizing them the first time through
	/// `upload(x, y, width, height, coverage)`. A full atlas leaves glyphs
	/// empty rather than failing.
	pub fn set(&mut self, px: u16, mut upload: impl FnMut(u32, u32, u32, u32, &[u8])) -> &[Glyph] {
		if !self.sets.contains_key(&px) {
			let mut set = Vec::new();
			for c in FIRST..=LAST {
				let (metrics, coverage) = self.font.rasterize(c, px as f32);
				let (w, h) = (metrics.width as u32, metrics.height as u32);
				let mut glyph = Glyph {
					rect: [0, 0, 0, 0],
					xmin: metrics.xmin as f32,
					ymin: metrics.ymin as f32,
					advance: metrics.advance_width,
				};
				if w > 0 && h > 0 {
					if let Some((x, y)) = self.packer.place(w, h) {
						upload(x, y, w, h, &coverage);
						glyph.rect = [x, y, w, h];
					}
				}
				set.push(glyph);
			}
			self.sets.insert(px, set);
		}
		&self.sets[&px]
	}
}

/// The glyph for `c` in a set, or a space for characters the font lacks.
pub fn glyph(set: &[Glyph], c: char) -> Glyph {
	let at = (c as u32).wrapping_sub(FIRST as u32) as usize;
	set.get(at).copied().unwrap_or(set[0])
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn rasterizes_each_size_once_into_distinct_rects() {
		let mut glyphs = Glyphs::new().unwrap();
		let mut uploads = 0;
		let a = glyph(glyphs.set(20, |_, _, _, _, _| uploads += 1), 'A');
		let again = glyph(glyphs.set(20, |_, _, _, _, _| uploads += 1000), 'A');
		assert_eq!(a.rect, again.rect);
		assert!(uploads > 50 && uploads < 1000, "{uploads}");
		let b = glyph(glyphs.set(20, |_, _, _, _, _| {}), 'B');
		assert_ne!(a.rect, b.rect);
		assert!(a.rect[0] >= SOLID || a.rect[1] > 0);
		assert!(a.advance > 0.0);
	}

	#[test]
	fn unknown_characters_draw_as_spaces() {
		let mut glyphs = Glyphs::new().unwrap();
		let set = glyphs.set(16, |_, _, _, _, _| {});
		assert_eq!(glyph(set, 'é').advance, glyph(set, ' ').advance);
	}
}
