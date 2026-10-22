//! Immediate-mode widgets in the look of Counter-Strike 1.6's Steam VGUI:
//! olive panels with bevelled edges, a gold highlight. Each widget draws
//! through [`Gfx`] and answers from this frame's [`Input`]; whatever state
//! it edits lives with the caller, so menus are plain functions over their
//! own data — a scrolled area's [`Scroll`] and a color picker's open flag
//! included.

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

/// A scrolled area's state, kept by its caller from frame to frame: how
/// far down it is scrolled, how tall its content stood last frame, and
/// where on the thumb a drag holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scroll {
	pub offset: f32,
	content: f32,
	grab: Option<f32>,
}

/// Pixels one wheel notch scrolls.
const WHEEL_STEP: f32 = 48.0;
/// The scroll bar's width, and its thumb's least height.
const BAR: f32 = 10.0;
const THUMB_MIN: f32 = 24.0;
/// A color picker's rows: its label and swatch, and once open the
/// saturation-value square and hue bar below.
const COLOR_ROW: f32 = 24.0;
const PICKER: f32 = 132.0;

/// `rgb` as hue, saturation and value, each 0 to 1.
pub fn hsv_of(rgb: [f32; 3]) -> [f32; 3] {
	let [r, g, b] = rgb;
	let max = r.max(g).max(b);
	let min = r.min(g).min(b);
	let d = max - min;
	let h = if d <= 0.0 {
		0.0
	} else if max == r {
		((g - b) / d).rem_euclid(6.0) / 6.0
	} else if max == g {
		((b - r) / d + 2.0) / 6.0
	} else {
		((r - g) / d + 4.0) / 6.0
	};
	let s = if max <= 0.0 { 0.0 } else { d / max };
	[h, s, max]
}

/// Hue, saturation and value back as red, green and blue.
pub fn rgb_of([h, s, v]: [f32; 3]) -> [f32; 3] {
	let h6 = h.rem_euclid(1.0) * 6.0;
	let f = |n: f32| {
		let k = (n + h6).rem_euclid(6.0);
		v - v * s * k.min(4.0 - k).clamp(0.0, 1.0)
	};
	[f(5.0), f(3.0), f(1.0)]
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
	/// Wheel notches not yet taken by a scrolled area.
	wheel: f32,
	/// The scrolled area being laid out: its view, and the top of its
	/// content.
	scrolling: Option<(Rect, f32)>,
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
			wheel: input.wheel(),
			scrolling: None,
		}
	}

	pub fn size(&self) -> (f32, f32) {
		self.gfx.size()
	}

	/// Whether the cursor is over `rect`, and over the scrolled area's view
	/// when inside one: a widget scrolled out of sight answers nothing.
	pub fn hovered(&self, rect: Rect) -> bool {
		self.enabled
			&& rect.contains(self.cursor)
			&& self.scrolling.is_none_or(|(view, _)| view.contains(self.cursor))
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

	/// Begins a scrolled area over `view`: the wheel over it, or the bar
	/// down its right side, scrolls it by `scroll`. Returns where to lay its
	/// content out — as wide as the view less the bar, starting above the
	/// view by the offset, with no bottom: rows cut from it keep their
	/// heights past the view's, so [`Ui::end_scroll`] measures the whole
	/// content and the bar appears once it overflows. Everything drawn
	/// until [`Ui::end_scroll`] is cut to the view.
	pub fn scroll(&mut self, view: Rect, scroll: &mut Scroll) -> Rect {
		let over = scroll.content - view.h;
		let bar = over > 0.0;
		if self.enabled && view.contains(self.cursor) && self.wheel != 0.0 {
			scroll.offset -= self.wheel * WHEEL_STEP;
			self.wheel = 0.0;
		}
		if bar {
			let track = Rect::new(view.x + view.w - BAR, view.y, BAR, view.h);
			let thumb_h = (view.h * view.h / scroll.content).max(THUMB_MIN).min(view.h);
			let travel = (view.h - thumb_h).max(1.0);
			let thumb = |offset: f32| Rect::new(track.x, track.y + offset / over * travel, BAR, thumb_h);
			if !self.held {
				scroll.grab = None;
			}
			if self.clicked(track) {
				let t = thumb(scroll.offset);
				// On the thumb, hold it where it was taken; off it, jump the
				// thumb's middle to the cursor and hold it there.
				scroll.grab = Some(if t.contains(self.cursor) {
					self.cursor.1 - t.y
				} else {
					thumb_h * 0.5
				});
			}
			if let Some(grab) = scroll.grab {
				scroll.offset = (self.cursor.1 - grab - track.y) / travel * over;
			}
			scroll.offset = scroll.offset.clamp(0.0, over);
			self.fill(track, self.theme.field);
			self.bevel(track, true);
			let t = thumb(scroll.offset);
			self.fill(t, self.theme.panel);
			self.bevel(t, false);
		} else {
			scroll.offset = 0.0;
		}
		let top = view.y - scroll.offset;
		self.scrolling = Some((view, top));
		self.gfx.set_clip(Some([view.x, view.y, view.w, view.h]));
		let w = if bar { view.w - BAR - 6.0 } else { view.w };
		Rect::new(view.x, top, w, f32::INFINITY)
	}

	/// Ends the scrolled area [`Ui::scroll`] began, its content having
	/// reached down to `bottom`, so next frame's bar knows its height.
	pub fn end_scroll(&mut self, scroll: &mut Scroll, bottom: f32) {
		if let Some((_, top)) = self.scrolling.take() {
			scroll.content = bottom - top;
		}
		self.gfx.set_clip(None);
	}

	/// How tall [`Ui::color`] stands, open or shut.
	pub const fn color_height(open: bool) -> f32 {
		if open {
			COLOR_ROW + 8.0 + PICKER
		} else {
			COLOR_ROW
		}
	}

	/// A labelled color: a swatch at the right that opens a picker under it
	/// when clicked — a square of saturation across and value down at the
	/// color's hue, a hue bar beside it, each followed while held — and
	/// shuts it clicked again. `open` is the caller's, as is the room the
	/// picker takes: [`Ui::color_height`]. Returns whether the color changed.
	pub fn color(&mut self, rect: Rect, text: &str, rgb: &mut [f32; 3], open: &mut bool) -> bool {
		let (row, below) = rect.cut_top(COLOR_ROW);
		let (swatch_cell, name) = row.cut_right(64.0);
		let color = if self.hovered(row) {
			self.theme.highlight
		} else {
			self.theme.text
		};
		self.caption(name, text, false, color);
		let swatch = Rect::new(swatch_cell.x + 8.0, row.y + 3.0, 56.0, row.h - 6.0);
		let [r, g, b] = *rgb;
		self.fill(swatch, Color::new(r, g, b, 1.0));
		self.bevel(swatch, !*open);
		if self.clicked(row) {
			*open = !*open;
		}
		if !*open {
			return false;
		}
		let before = *rgb;
		let mut hsv = hsv_of(*rgb);
		let area = below.cut_top(8.0).1;
		let square = Rect::new(area.x, area.y, (area.w - 40.0).min(240.0), PICKER);
		let hue = Rect::new(square.x + square.w + 12.0, area.y, 20.0, PICKER);
		if self.held && self.hovered(square) {
			hsv[1] = ((self.cursor.0 - square.x) / square.w).clamp(0.0, 1.0);
			hsv[2] = 1.0 - ((self.cursor.1 - square.y) / square.h).clamp(0.0, 1.0);
			*rgb = rgb_of(hsv);
		}
		if self.held && self.hovered(hue) {
			hsv[0] = ((self.cursor.1 - hue.y) / hue.h).clamp(0.0, 0.9999);
			*rgb = rgb_of(hsv);
		}
		// The square in cells, each shaded between its corners' colors, so
		// saturation and value read true across it.
		const CELLS: u32 = 12;
		let (cw, ch) = (square.w / CELLS as f32, square.h / CELLS as f32);
		let at = |i: u32, j: u32| {
			let [r, g, b] = rgb_of([hsv[0], i as f32 / CELLS as f32, 1.0 - j as f32 / CELLS as f32]);
			Color::new(r, g, b, 1.0)
		};
		for j in 0..CELLS {
			for i in 0..CELLS {
				self.gfx.gradient(
					square.x + i as f32 * cw,
					square.y + j as f32 * ch,
					cw + 0.5,
					ch + 0.5,
					[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)],
				);
			}
		}
		self.bevel(square, true);
		// The hue bar, red through the spectrum back to red.
		for k in 0..6 {
			let top = rgb_of([k as f32 / 6.0, 1.0, 1.0]);
			let bottom = rgb_of([(k + 1) as f32 / 6.0, 1.0, 1.0]);
			let t = Color::new(top[0], top[1], top[2], 1.0);
			let b = Color::new(bottom[0], bottom[1], bottom[2], 1.0);
			self.gfx.gradient(
				hue.x,
				hue.y + k as f32 * hue.h / 6.0,
				hue.w,
				hue.h / 6.0 + 0.5,
				[t, t, b, b],
			);
		}
		self.bevel(hue, true);
		// Markers: a ring at the color in the square, a notch at its hue.
		let (mx, my) = (
			square.x + hsv[1] * square.w,
			square.y + (1.0 - hsv[2]) * square.h,
		);
		let dot = if hsv[2] > 0.5 {
			self.theme.dark
		} else {
			self.theme.text
		};
		self.bevel(Rect::new(mx - 4.0, my - 4.0, 8.0, 8.0), false);
		self.fill(Rect::new(mx - 1.0, my - 1.0, 2.0, 2.0), dot);
		let hy = hue.y + hsv[0] * hue.h;
		self.fill(Rect::new(hue.x - 3.0, hy - 1.0, hue.w + 6.0, 2.0), self.theme.text);
		*rgb != before
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
	fn hsv_round_trips_colors() {
		for rgb in [
			[1.0, 0.0, 0.0],
			[0.2, 0.6, 0.3],
			[0.9, 0.9, 0.1],
			[0.5, 0.5, 0.5],
			[0.0, 0.0, 0.0],
		] {
			let back = rgb_of(hsv_of(rgb));
			for k in 0..3 {
				assert!((back[k] - rgb[k]).abs() < 1e-5, "{rgb:?} -> {back:?}");
			}
		}
		assert_eq!(hsv_of([0.0, 1.0, 0.0])[0], 1.0 / 3.0);
	}

	#[test]
	fn sliders_clamp_to_their_track() {
		let track = Rect::new(100.0, 0.0, 200.0, 4.0);
		assert_eq!(slide(track, 50.0, 1.0, 5.0), 1.0);
		assert_eq!(slide(track, 200.0, 1.0, 5.0), 3.0);
		assert_eq!(slide(track, 900.0, 1.0, 5.0), 5.0);
	}
}
