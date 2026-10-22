//! Immediate-mode widgets in the look of Counter-Strike 1.6's Steam VGUI:
//! olive panels with bevelled edges, a gold highlight. Each widget draws
//! through [`Gfx`] and answers from this frame's [`Input`]; whatever state
//! it edits lives with the caller, so menus are plain functions over their
//! own data.

use crate::gpu::{Color, Gfx, Input, MouseButton};

const fn rgb(r: u8, g: u8, b: u8) -> Color {
	Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}

/// The colors and text size widgets draw with.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
	pub panel: Color,
	/// Text fields, slider tracks and check boxes: sunk into the panel.
	pub field: Color,
	pub bright: Color,
	pub dark: Color,
	pub text: Color,
	pub dim: Color,
	pub highlight: Color,
	/// Pixels tall of widget text.
	pub size: f32,
}

impl Theme {
	/// 1.6's Steam scheme.
	pub const STEAM: Theme = Theme {
		panel: rgb(76, 88, 68),
		field: rgb(62, 70, 55),
		bright: rgb(136, 145, 128),
		dark: rgb(40, 46, 34),
		text: rgb(216, 222, 211),
		dim: rgb(160, 170, 149),
		highlight: rgb(196, 181, 80),
		size: 20.0,
	};
}

/// A screen rectangle in pixels from the top left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
	pub x: f32,
	pub y: f32,
	pub w: f32,
	pub h: f32,
}

impl Rect {
	pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Rect {
		Rect { x, y, w, h }
	}

	/// A `w` by `h` rectangle centered on the screen of `size`.
	pub fn centered(size: (f32, f32), w: f32, h: f32) -> Rect {
		Rect::new(
			((size.0 - w) * 0.5).round(),
			((size.1 - h) * 0.5).round(),
			w,
			h,
		)
	}

	pub fn contains(&self, (x, y): (f32, f32)) -> bool {
		x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
	}

	/// Shrunk by `by` on every side.
	pub fn inset(&self, by: f32) -> Rect {
		Rect::new(
			self.x + by,
			self.y + by,
			(self.w - 2.0 * by).max(0.0),
			(self.h - 2.0 * by).max(0.0),
		)
	}

	/// The top `h` pixels, and the rest below them.
	pub fn cut_top(&self, h: f32) -> (Rect, Rect) {
		let h = h.min(self.h);
		(
			Rect::new(self.x, self.y, self.w, h),
			Rect::new(self.x, self.y + h, self.w, self.h - h),
		)
	}

	/// The bottom `h` pixels, and the rest above them.
	pub fn cut_bottom(&self, h: f32) -> (Rect, Rect) {
		let h = h.min(self.h);
		(
			Rect::new(self.x, self.y + self.h - h, self.w, h),
			Rect::new(self.x, self.y, self.w, self.h - h),
		)
	}

	/// The right `w` pixels, and the rest left of them.
	pub fn cut_right(&self, w: f32) -> (Rect, Rect) {
		let w = w.min(self.w);
		(
			Rect::new(self.x + self.w - w, self.y, w, self.h),
			Rect::new(self.x, self.y, self.w - w, self.h),
		)
	}
}

/// The value at `x` along a track spanning `track`, clamped to its ends.
fn slide(track: Rect, x: f32, min: f32, max: f32) -> f32 {
	let t = ((x - track.x) / track.w.max(1.0)).clamp(0.0, 1.0);
	min + t * (max - min)
}

/// One frame of widgets over a screen.
pub struct Ui<'a> {
	pub gfx: &'a mut Gfx,
	pub theme: Theme,
	/// Whether widgets answer the mouse; false draws them inert, as the
	/// menu behind a dialog is.
	pub enabled: bool,
	cursor: (f32, f32),
	click: bool,
	held: bool,
}

impl<'a> Ui<'a> {
	pub fn new(gfx: &'a mut Gfx, input: &Input, theme: Theme) -> Ui<'a> {
		Ui {
			gfx,
			theme,
			enabled: true,
			cursor: input.cursor(),
			click: input.clicked(MouseButton::Left),
			held: input.button_down(MouseButton::Left),
		}
	}

	pub fn size(&self) -> (f32, f32) {
		self.gfx.size()
	}

	pub fn hovered(&self, rect: Rect) -> bool {
		self.enabled && rect.contains(self.cursor)
	}

	/// Whether this frame's click landed in `rect`; one widget takes it.
	pub fn clicked(&mut self, rect: Rect) -> bool {
		let hit = self.click && self.hovered(rect);
		self.click &= !hit;
		hit
	}

	pub fn fill(&mut self, rect: Rect, color: Color) {
		self.gfx.rect(rect.x, rect.y, rect.w, rect.h, color);
	}

	/// A one-pixel edge: bright top-left and dark bottom-right when raised,
	/// the other way when sunk.
	pub fn bevel(&mut self, rect: Rect, sunk: bool) {
		let (lit, shade) = if sunk {
			(self.theme.dark, self.theme.bright)
		} else {
			(self.theme.bright, self.theme.dark)
		};
		let Rect { x, y, w, h } = rect;
		self.gfx.rect(x, y, w, 1.0, lit);
		self.gfx.rect(x, y, 1.0, h, lit);
		self.gfx.rect(x, y + h - 1.0, w, 1.0, shade);
		self.gfx.rect(x + w - 1.0, y, 1.0, h, shade);
	}

	/// A raised panel.
	pub fn panel(&mut self, rect: Rect) {
		self.fill(rect, self.theme.panel);
		self.bevel(rect, false);
	}

	pub fn text_width(&mut self, text: &str, size: f32) -> f32 {
		self.gfx.text_width(text, size)
	}

	/// Text with its top-left at `x`, `y`.
	pub fn label(&mut self, text: &str, x: f32, y: f32, size: f32, color: Color) {
		// ProggyClean's baseline sits about four fifths down its line.
		self
			.gfx
			.text(text, x, (y + size * 0.8).round(), size, color);
	}

	/// Text centered vertically in `rect`, from its left or centered across it.
	fn caption(&mut self, rect: Rect, text: &str, centered: bool, color: Color) {
		let size = self.theme.size;
		let x = if centered {
			rect.x + ((rect.w - self.text_width(text, size)) * 0.5).round()
		} else {
			rect.x
		};
		self.label(
			text,
			x,
			rect.y + ((rect.h - size) * 0.5).round(),
			size,
			color,
		);
	}

	/// A bare line of text that brightens under the cursor: the main menu's
	/// items. Clicked returns true.
	pub fn link(&mut self, rect: Rect, text: &str, size: f32) -> bool {
		let color = if self.hovered(rect) {
			self.theme.highlight
		} else {
			self.theme.text
		};
		self.label(
			text,
			rect.x,
			rect.y + ((rect.h - size) * 0.5).round(),
			size,
			color,
		);
		self.clicked(rect)
	}

	/// A bevelled push button, sunk while pressed. Clicked returns true.
	pub fn button(&mut self, rect: Rect, text: &str) -> bool {
		let hovered = self.hovered(rect);
		self.fill(rect, self.theme.panel);
		self.bevel(rect, hovered && self.held);
		let color = if hovered {
			self.theme.highlight
		} else {
			self.theme.text
		};
		self.caption(rect, text, true, color);
		self.clicked(rect)
	}

	/// A dialog window: a panel with its title and a close box. Returns the
	/// area under the title for its contents, and whether the box was clicked.
	pub fn frame(&mut self, rect: Rect, title: &str) -> (Rect, bool) {
		self.panel(rect);
		let (bar, body) = rect.inset(1.0).cut_top(30.0);
		self.caption(
			Rect::new(bar.x + 10.0, bar.y, bar.w, bar.h),
			title,
			false,
			self.theme.text,
		);
		let close = Rect::new(bar.x + bar.w - 24.0, bar.y + 7.0, 16.0, 16.0);
		let hovered = self.hovered(close);
		self.bevel(close, hovered && self.held);
		let color = if hovered {
			self.theme.highlight
		} else {
			self.theme.text
		};
		let (cx, cy) = (close.x + 8.0, close.y + 8.0);
		self
			.gfx
			.line(cx - 4.0, cy - 4.0, cx + 4.0, cy + 4.0, 1.5, color);
		self
			.gfx
			.line(cx - 4.0, cy + 4.0, cx + 4.0, cy - 4.0, 1.5, color);
		let closed = self.clicked(close);
		(body.inset(10.0), closed)
	}

	/// A row of tabs over a sheet; clicking one selects it. Returns the
	/// sheet's area inside its edge.
	pub fn tabs(&mut self, rect: Rect, labels: &[&str], selected: &mut usize) -> Rect {
		let (row, sheet) = rect.cut_top(28.0);
		self.bevel(sheet, false);
		let mut x = row.x;
		for (i, label) in labels.iter().enumerate() {
			let w = self.text_width(label, self.theme.size) + 24.0;
			let current = i == *selected;
			// The selected tab rises a little and joins the sheet.
			let tab = if current {
				Rect::new(x, row.y, w, row.h + 1.0)
			} else {
				Rect::new(x, row.y + 3.0, w, row.h - 3.0)
			};
			if !current && self.clicked(tab) {
				*selected = i;
			}
			let color = if current || self.hovered(tab) {
				self.theme.text
			} else {
				self.theme.dim
			};
			self.fill(tab, self.theme.panel);
			self.bevel(tab, false);
			if current {
				self.fill(
					Rect::new(tab.x + 1.0, sheet.y, tab.w - 2.0, 1.0),
					self.theme.panel,
				);
			}
			self.caption(tab, label, true, color);
			x += w + 2.0;
		}
		sheet.inset(12.0)
	}

	/// A check box with its label on the right. Returns whether it toggled.
	pub fn checkbox(&mut self, rect: Rect, text: &str, value: &mut bool) -> bool {
		let side = 14.0;
		let bx = Rect::new(rect.x, rect.y + ((rect.h - side) * 0.5).round(), side, side);
		self.fill(bx, self.theme.field);
		self.bevel(bx, true);
		if *value {
			let c = self.theme.highlight;
			let (x, y) = (bx.x, bx.y);
			self.gfx.line(x + 3.0, y + 7.0, x + 6.0, y + 10.5, 2.0, c);
			self.gfx.line(x + 6.0, y + 10.5, x + 11.0, y + 3.5, 2.0, c);
		}
		let color = if self.hovered(rect) {
			self.theme.highlight
		} else {
			self.theme.text
		};
		self.caption(
			Rect::new(rect.x + side + 8.0, rect.y, rect.w - side - 8.0, rect.h),
			text,
			false,
			color,
		);
		let toggled = self.clicked(rect);
		*value ^= toggled;
		toggled
	}

	/// A labelled horizontal slider over `min..=max`, its value printed at the
	/// right; held down over its row, the thumb follows the cursor. Returns
	/// whether the value changed.
	pub fn slider(&mut self, rect: Rect, text: &str, value: &mut f32, min: f32, max: f32) -> bool {
		let (top, track_row) = rect.cut_top(self.theme.size + 4.0);
		let (number, name) = top.cut_right(80.0);
		self.caption(name, text, false, self.theme.text);
		let shown = format!("{:.2}", *value);
		let x = number.x + number.w - self.text_width(&shown, self.theme.size);
		self.label(&shown, x, number.y + 2.0, self.theme.size, self.theme.dim);

		let track = Rect::new(
			track_row.x + 6.0,
			track_row.y + track_row.h * 0.5 - 2.0,
			track_row.w - 12.0,
			4.0,
		);
		self.fill(track, self.theme.field);
		self.bevel(track, true);
		let before = *value;
		if self.held && self.hovered(track_row) {
			*value = slide(track, self.cursor.0, min, max);
		}
		let t = ((*value - min) / (max - min)).clamp(0.0, 1.0);
		let thumb = Rect::new(
			(track.x + t * track.w - 5.0).round(),
			track_row.y + track_row.h * 0.5 - 9.0,
			10.0,
			18.0,
		);
		self.fill(thumb, self.theme.panel);
		self.bevel(thumb, false);
		*value != before
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn rects_cut_into_parts_that_tile_them() {
		let r = Rect::new(10.0, 20.0, 100.0, 50.0);
		let (top, rest) = r.cut_top(15.0);
		assert_eq!(top, Rect::new(10.0, 20.0, 100.0, 15.0));
		assert_eq!(rest, Rect::new(10.0, 35.0, 100.0, 35.0));
		let (bottom, rest) = r.cut_bottom(60.0);
		assert_eq!(bottom, r);
		assert_eq!(rest.h, 0.0);
		let (right, rest) = r.cut_right(30.0);
		assert_eq!(right.x, 80.0);
		assert_eq!(rest.w, 70.0);
		assert_eq!(r.inset(5.0), Rect::new(15.0, 25.0, 90.0, 40.0));
		assert!(r.contains((10.0, 20.0)) && !r.contains((110.0, 20.0)));
		assert_eq!(
			Rect::centered((200.0, 100.0), 50.0, 20.0),
			Rect::new(75.0, 40.0, 50.0, 20.0)
		);
	}

	#[test]
	fn sliders_clamp_to_their_track() {
		let track = Rect::new(100.0, 0.0, 200.0, 4.0);
		assert_eq!(slide(track, 50.0, 1.0, 5.0), 1.0);
		assert_eq!(slide(track, 200.0, 1.0, 5.0), 3.0);
		assert_eq!(slide(track, 900.0, 1.0, 5.0), 5.0);
	}
}
