//! HUD text: ProggyClean, macroquad's old default font, rasterized with
//! fontdue into one coverage atlas, a glyph set per pixel size the first
//! time that size draws. The atlas's top-left texels are solid, for
//! rectangles and lines.

use std::collections::HashMap;

/// The atlas side, texels.
pub const SIDE: u32 = 1024;
/// The solid block in the top-left corner.
pub const SOLID: u32 = 4;

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
	sets: HashMap<u16, HashMap<char, Glyph>>,
	fallbacks: Vec<fontdue::Font>,
	fonts: Option<fontdb::Database>,
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
			fallbacks: Vec::new(),
			fonts: None,
			packer: lightbake::pack::Packer::restore([SIDE, SOLID, 0, SOLID]),
		})
	}

	/// Choose a system fallback lazily, so ordinary HUD text never scans fonts.
	fn font_for(&mut self, c: char) -> &fontdue::Font {
		if self.font.lookup_glyph_index(c) != 0 {
			return &self.font;
		}
		if let Some(at) = self
			.fallbacks
			.iter()
			.position(|font| font.lookup_glyph_index(c) != 0)
		{
			return &self.fallbacks[at];
		}
		let fonts = self.fonts.get_or_insert_with(|| {
			let mut fonts = fontdb::Database::new();
			fonts.load_system_fonts();
			fonts
		});
		let preferred = fonts.query(&fontdb::Query {
			families: &[fontdb::Family::Monospace, fontdb::Family::SansSerif],
			..Default::default()
		});
		let ids = preferred
			.into_iter()
			.chain(fonts.faces().map(|face| face.id));
		for id in ids {
			let font = fonts
				.with_face_data(id, |data, index| {
					let face = ttf_parser::Face::parse(data, index).ok()?;
					face.glyph_index(c)?;
					fontdue::Font::from_bytes(
						data,
						fontdue::FontSettings {
							collection_index: index,
							..Default::default()
						},
					)
					.ok()
				})
				.flatten();
			if let Some(font) = font {
				self.fallbacks.push(font);
				return self.fallbacks.last().unwrap();
			}
		}
		&self.font
	}

	/// Rasterize only missing characters. Unsupported characters get a visible
	/// replacement instead of silently turning into spaces in an editable line.
	pub fn set(
		&mut self,
		px: u16,
		text: &str,
		mut upload: impl FnMut(u32, u32, u32, u32, &[u8]),
	) -> &HashMap<char, Glyph> {
		for c in text.chars() {
			if self.sets.get(&px).is_some_and(|set| set.contains_key(&c)) {
				continue;
			}
			let font = self.font_for(c);
			let shown = if font.lookup_glyph_index(c) != 0 {
				c
			} else {
				'?'
			};
			let (metrics, coverage) = font.rasterize(shown, px as f32);
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
			self.sets.entry(px).or_default().insert(c, glyph);
		}
		self.sets.entry(px).or_default()
	}
}

/// `set` has been prepared for all characters in the line being measured/drawn.
pub fn glyph(set: &HashMap<char, Glyph>, c: char) -> Glyph {
	set[&c]
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn rasterizes_each_size_once_into_distinct_rects() {
		let mut glyphs = Glyphs::new().unwrap();
		let mut uploads = 0;
		let a = glyph(glyphs.set(20, "AB", |_, _, _, _, _| uploads += 1), 'A');
		let again = glyph(glyphs.set(20, "AB", |_, _, _, _, _| uploads += 1000), 'A');
		assert_eq!(a.rect, again.rect);
		assert!(uploads == 2, "{uploads}");
		let b = glyph(glyphs.set(20, "AB", |_, _, _, _, _| {}), 'B');
		assert_ne!(a.rect, b.rect);
		assert!(a.rect[0] >= SOLID || a.rect[1] > 0);
		assert!(a.advance > 0.0);
	}

	#[test]
	fn accented_input_is_visible_and_cached() {
		let mut glyphs = Glyphs::new().unwrap();
		let set = glyphs.set(16, "é ", |_, _, _, _, _| {});
		assert!(glyph(set, 'é').rect[2] > 0);
		assert!(glyph(set, 'é').advance > 0.0);
		assert_eq!(glyph(set, ' ').rect[2], 0);
		glyphs.set(16, "é", |_, _, _, _, _| {
			panic!("cached glyph uploaded twice")
		});
	}
}
