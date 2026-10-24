//! Single-line field interaction and rendering. The console owns commands and
//! history; `LineEdit` owns text/selection invariants; winit owns repeat and IME.

use super::{Rect, Theme};
use crate::gpu::{Gfx, Input, ModifiersState, MouseButton, NamedKey, TextEvent, TextKey};
use console::{LineEdit, Movement};
use std::ops::Range;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextAction {
	Submit,
	Complete,
	Previous,
	Next,
	Close,
}

struct Drag {
	anchor: Range<usize>,
	words: bool,
}

pub struct TextField {
	offset: f32,
	drag: Option<Drag>,
	click: Option<(Instant, (f32, f32), u8)>,
	changed: Instant,
	preedit: String,
	preedit_cursor: Option<(usize, usize)>,
	// Keep the clipboard owner alive (required by X11).
	clipboard: Option<arboard::Clipboard>,
	menu: Option<Rect>,
}

impl Default for TextField {
	fn default() -> Self {
		Self {
			offset: 0.0,
			drag: None,
			click: None,
			changed: Instant::now(),
			preedit: String::new(),
			preedit_cursor: None,
			clipboard: None,
			menu: None,
		}
	}
}

impl TextField {
	/// Cancel transient interaction on close or focus loss, keeping the text.
	pub fn blur(&mut self) {
		self.drag = None;
		self.click = None;
		self.preedit.clear();
		self.preedit_cursor = None;
		self.menu = None;
		self.changed = Instant::now();
	}

	fn clipboard(&mut self) -> Result<&mut arboard::Clipboard, String> {
		if self.clipboard.is_none() {
			self.clipboard = Some(arboard::Clipboard::new().map_err(|e| format!("clipboard: {e}"))?);
		}
		Ok(self.clipboard.as_mut().unwrap())
	}

	fn copy(&mut self, line: &mut LineEdit, cut: bool) -> Result<(), String> {
		if !line.selection().is_empty() {
			self
				.clipboard()?
				.set_text(line.selected())
				.map_err(|e| format!("clipboard: {e}"))?;
			// A failed clipboard write must never destroy the selected text.
			if cut {
				line.replace_range(line.selection(), "");
			}
		}
		Ok(())
	}

	fn paste(&mut self, line: &mut LineEdit) -> Result<(), String> {
		match self.clipboard()?.get_text() {
			Ok(text) => line.insert(&text),
			Err(arboard::Error::ContentNotAvailable) => {}
			Err(e) => return Err(format!("clipboard: {e}")),
		}
		Ok(())
	}

	fn key(
		&mut self,
		line: &mut LineEdit,
		key: &TextKey,
		text: Option<&str>,
		modifiers: ModifiersState,
		repeat: bool,
	) -> Result<Option<TextAction>, String> {
		if !self.preedit.is_empty() {
			return Ok(None); // The native IME owns editing until it commits/cancels.
		}
		let shift = modifiers.shift_key();
		let command = if cfg!(target_os = "macos") {
			modifiers.super_key()
		} else {
			modifiers.control_key() && !modifiers.alt_key() // AltGr produces text.
		};
		let word = if cfg!(target_os = "macos") {
			modifiers.alt_key()
		} else {
			command
		};
		if command {
			if let TextKey::Character(c) = key {
				match c.to_lowercase().as_str() {
					"a" => line.select_all(),
					"c" => self.copy(line, false)?,
					"x" => self.copy(line, true)?,
					"v" => self.paste(line)?,
					"z" if shift => line.redo(),
					"z" => line.undo(),
					"y" => line.redo(),
					_ => {}
				}
				return Ok(None);
			}
		}
		let mac_line = cfg!(target_os = "macos") && command;
		match key {
			TextKey::Named(NamedKey::ArrowLeft) => line.move_cursor(
				if mac_line {
					Movement::Start
				} else if word {
					Movement::WordLeft
				} else {
					Movement::Left
				},
				shift,
			),
			TextKey::Named(NamedKey::ArrowRight) => line.move_cursor(
				if mac_line {
					Movement::End
				} else if word {
					Movement::WordRight
				} else {
					Movement::Right
				},
				shift,
			),
			TextKey::Named(NamedKey::Home) => line.move_cursor(Movement::Start, shift),
			TextKey::Named(NamedKey::End) => line.move_cursor(Movement::End, shift),
			TextKey::Named(NamedKey::Insert) if command => self.copy(line, false)?,
			TextKey::Named(NamedKey::Insert) if shift => self.paste(line)?,
			TextKey::Named(NamedKey::Delete) if shift && !command => self.copy(line, true)?,
			TextKey::Named(NamedKey::Backspace | NamedKey::Delete) => {
				let back = *key == TextKey::Named(NamedKey::Backspace);
				if mac_line && line.selection().is_empty() {
					line.move_cursor(if back { Movement::Start } else { Movement::End }, true);
				}
				line.delete(back, word);
			}
			TextKey::Named(NamedKey::Enter) if !repeat => return Ok(Some(TextAction::Submit)),
			TextKey::Named(NamedKey::Tab) if !repeat => return Ok(Some(TextAction::Complete)),
			TextKey::Named(NamedKey::Escape) if !repeat => return Ok(Some(TextAction::Close)),
			TextKey::Named(NamedKey::ArrowUp) => return Ok(Some(TextAction::Previous)),
			TextKey::Named(NamedKey::ArrowDown) => return Ok(Some(TextAction::Next)),
			_ if !command && !modifiers.super_key() => {
				if let Some(text) = text {
					// Control keys (notably repeated Enter/Tab) never become spaces.
					let text: String = text.chars().filter(|c| !c.is_control()).collect();
					line.type_text(&text);
				}
			}
			_ => {}
		}
		Ok(None)
	}

	/// Process events in order; the caller applies command/history actions before
	/// passing the next event, including when several commands arrive in one frame.
	pub fn event(
		&mut self,
		line: &mut LineEdit,
		event: &TextEvent,
		gfx: &mut Gfx,
		rect: Rect,
		size: f32,
	) -> Result<Option<TextAction>, String> {
		if !matches!(event, TextEvent::PointerMoved(_)) || self.drag.is_some() {
			self.changed = Instant::now();
		}
		if let Some(menu) = self.menu {
			match event {
				TextEvent::Key {
					key: TextKey::Named(NamedKey::Escape),
					..
				} => {
					self.menu = None;
					return Ok(None);
				}
				TextEvent::PointerButton {
					position,
					button: MouseButton::Left,
					pressed: true,
					..
				} => {
					self.menu = None;
					if menu.contains(*position) {
						match ((position.1 - menu.y) / MENU_ROW) as usize {
							0 => line.undo(),
							1 => line.redo(),
							2 => self.copy(line, true)?,
							3 => self.copy(line, false)?,
							4 => self.paste(line)?,
							5 => line.select_all(),
							_ => {}
						}
						return Ok(None);
					}
				}
				TextEvent::Key { .. } => self.menu = None,
				_ => {}
			}
		}
		match event {
			TextEvent::Key {
				key,
				text,
				modifiers,
				repeat,
			} => self.key(line, key, text.as_deref(), *modifiers, *repeat),
			TextEvent::Commit(text) => {
				self.preedit.clear();
				self.preedit_cursor = None;
				line.insert(text);
				Ok(None)
			}
			TextEvent::Preedit(text, cursor) => {
				self.preedit.clone_from(text);
				self.preedit_cursor = *cursor;
				Ok(None)
			}
			TextEvent::PointerButton {
				position,
				button: MouseButton::Right,
				pressed: true,
				..
			} if rect.contains(*position) && self.preedit.is_empty() => {
				let (w, h) = gfx.size();
				self.menu = Some(Rect::new(
					position.0.min((w - 180.0).max(0.0)),
					position.1.min((h - MENU_ROW * 6.0).max(0.0)),
					180.0,
					MENU_ROW * 6.0,
				));
				self.drag = None;
				Ok(None)
			}
			TextEvent::PointerButton {
				position,
				button: MouseButton::Left,
				pressed,
				modifiers,
			} => {
				if !pressed {
					self.drag = None;
				} else if rect.contains(*position) && self.preedit.is_empty() {
					let at = self.hit(line, gfx, rect, size, position.0);
					let now = Instant::now();
					let clicks = self
						.click
						.filter(|(time, last, _)| {
							now.duration_since(*time) < Duration::from_millis(500)
								&& (last.0 - position.0).abs() < 5.0
								&& (last.1 - position.1).abs() < 5.0
						})
						.map_or(1, |(_, _, count)| count % 3 + 1);
					self.click = Some((now, *position, clicks));
					let words = clicks == 2 && !modifiers.shift_key();
					if modifiers.shift_key() {
						line.place(at, true);
					} else if clicks == 3 {
						line.select_all();
					} else if words {
						let range = line.word_at(at);
						line.select(range.start, range.end);
					} else {
						line.place(at, false);
					}
					self.drag = Some(Drag {
						anchor: if words || clicks == 3 {
							line.selection()
						} else {
							line.anchor()..line.anchor()
						},
						words,
					});
				}
				Ok(None)
			}
			TextEvent::PointerMoved(position) => {
				if self.drag.is_some() {
					let at = self.hit(line, gfx, rect, size, position.0);
					self.drag_to(line, at);
				}
				Ok(None)
			}
			_ => Ok(None),
		}
	}

	fn drag_to(&self, line: &mut LineEdit, at: usize) {
		let Some(drag) = &self.drag else {
			return;
		};
		let range = if drag.words { line.word_at(at) } else { at..at };
		if at < drag.anchor.start {
			line.select(drag.anchor.end, range.start);
		} else {
			line.select(drag.anchor.start, range.end.max(drag.anchor.end));
		}
	}

	fn hit(&self, line: &LineEdit, gfx: &mut Gfx, rect: Rect, size: f32, x: f32) -> usize {
		let mut previous = 0;
		let mut width = 0.0;
		for at in line.boundaries().skip(1) {
			let next = width + gfx.text_width(&line.text()[previous..at], size);
			if x - rect.x + self.offset < (width + next) * 0.5 {
				return previous;
			}
			previous = at;
			width = next;
		}
		line.text().len()
	}

	/// Keep dragging beyond the edges scrolling even when the mouse is stationary.
	pub fn update(
		&mut self,
		line: &mut LineEdit,
		input: &Input,
		gfx: &mut Gfx,
		rect: Rect,
		size: f32,
		dt: f32,
	) {
		if !input.focused() {
			self.blur();
		}
		if !input.button_down(MouseButton::Left) {
			self.drag = None;
		}
		if self.drag.is_some() {
			let x = input.cursor().0;
			let outside = x - x.clamp(rect.x, rect.x + rect.w);
			if outside != 0.0 {
				let max = (gfx.text_width(line.text(), size) - rect.w + 2.0).max(0.0);
				self.offset = (self.offset + outside * dt.min(0.1) * 8.0).clamp(0.0, max);
				let at = self.hit(line, gfx, rect, size, x.clamp(rect.x, rect.x + rect.w));
				self.drag_to(line, at);
			}
		}
		gfx.text_cursor(rect.contains(input.cursor()));
	}

	pub fn draw(&mut self, line: &LineEdit, gfx: &mut Gfx, rect: Rect, theme: Theme, focused: bool) {
		let size = theme.size;
		let composing = !self.preedit.is_empty();
		let selection = line.selection();
		let mut shown = line.text().to_owned();
		let mut cursor = line.cursor();
		if composing {
			shown.replace_range(selection.clone(), &self.preedit);
			let mut local = self
				.preedit_cursor
				.map_or(self.preedit.len(), |(at, _)| at.min(self.preedit.len()));
			while !self.preedit.is_char_boundary(local) {
				local -= 1;
			}
			cursor = selection.start + local;
		}
		let caret = gfx.text_width(&shown[..cursor], size);
		let width = gfx.text_width(&shown, size);
		self.offset = visible_offset(self.offset, caret, width, rect.w);
		let x = rect.x - self.offset;
		let baseline = rect.y + size * 0.8;
		gfx.set_clip(Some([rect.x, rect.y, rect.w, rect.h]));
		if !selection.is_empty() && !composing {
			let left = gfx.text_width(&shown[..selection.start], size);
			let right = gfx.text_width(&shown[..selection.end], size);
			gfx.rect(x + left, rect.y, right - left, rect.h, theme.highlight);
		}
		gfx.text(&shown, x, baseline, size, theme.text);
		if !selection.is_empty() && !composing {
			let left = gfx.text_width(&shown[..selection.start], size);
			gfx.text(line.selected(), x + left, baseline, size, theme.field);
		}
		if composing {
			let left = gfx.text_width(&shown[..selection.start], size);
			let length = gfx.text_width(&self.preedit, size);
			gfx.rect(x + left, rect.y + rect.h - 2.0, length, 1.0, theme.text);
		}
		if focused
			&& (if composing {
				self.preedit_cursor.is_some()
			} else {
				self.changed.elapsed().as_secs_f32() % 1.0 < 0.5
			}) {
			gfx.rect(x + caret, rect.y, 1.0, rect.h, theme.text);
		}
		gfx.set_clip(None);
		gfx.ime_cursor((x + caret).clamp(rect.x, rect.x + rect.w), rect.y, rect.h);
		if let Some(menu) = self.menu {
			gfx.rect(menu.x, menu.y, menu.w, menu.h, theme.panel);
			for (i, label) in ["Undo", "Redo", "Cut", "Copy", "Paste", "Select all"]
				.iter()
				.enumerate()
			{
				gfx.text(
					label,
					menu.x + 8.0,
					menu.y + i as f32 * MENU_ROW + size,
					size,
					theme.text,
				);
			}
		}
	}
}

const MENU_ROW: f32 = 28.0;

fn visible_offset(offset: f32, caret: f32, width: f32, viewport: f32) -> f32 {
	let available = (viewport - 2.0).max(0.0);
	let offset = offset.min(caret).max(caret - available);
	offset.clamp(0.0, (width - available).max(0.0))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn repeated_keys_and_text_keep_their_order() {
		let mut field = TextField::default();
		let mut line = LineEdit::default();
		let plain = ModifiersState::empty();
		for (key, text, repeat) in [
			(TextKey::Character("abcd".into()), Some("abcd"), false),
			(TextKey::Named(NamedKey::ArrowLeft), None, false),
			(TextKey::Named(NamedKey::ArrowLeft), None, true),
			(TextKey::Character("X".into()), Some("X"), false),
			(TextKey::Named(NamedKey::Delete), None, false),
			(TextKey::Named(NamedKey::Delete), None, true),
		] {
			assert_eq!(
				field.key(&mut line, &key, text, plain, repeat).unwrap(),
				None
			);
		}
		assert_eq!(line.text(), "abX");
		assert_eq!(
			field
				.key(
					&mut line,
					&TextKey::Named(NamedKey::Enter),
					Some("\r"),
					plain,
					true
				)
				.unwrap(),
			None
		);
		assert_eq!(line.text(), "abX");
		assert_eq!(
			field
				.key(
					&mut line,
					&TextKey::Named(NamedKey::Enter),
					Some("\r"),
					plain,
					false
				)
				.unwrap(),
			Some(TextAction::Submit)
		);
	}

	#[test]
	fn shortcut_text_is_suppressed_but_altgr_and_literal_console_keys_work() {
		let mut field = TextField::default();
		let mut line = LineEdit::default();
		line.set("draft");
		let command = if cfg!(target_os = "macos") {
			ModifiersState::SUPER
		} else {
			ModifiersState::CONTROL
		};
		field
			.key(
				&mut line,
				&TextKey::Character("a".into()),
				Some("a"),
				command,
				false,
			)
			.unwrap();
		assert_eq!(line.selected(), "draft");
		field
			.key(
				&mut line,
				&TextKey::Character("@".into()),
				Some("@"),
				ModifiersState::CONTROL | ModifiersState::ALT,
				false,
			)
			.unwrap();
		field
			.key(
				&mut line,
				&TextKey::Character("`^".into()),
				Some("`^"),
				ModifiersState::empty(),
				false,
			)
			.unwrap();
		assert_eq!(line.text(), "@`^");
	}

	#[test]
	fn composition_keeps_edit_keys_and_enter_out_of_the_command_line() {
		let mut field = TextField {
			preedit: "你好".into(),
			..Default::default()
		};
		let mut line = LineEdit::default();
		line.set("echo ");
		for key in [NamedKey::Enter, NamedKey::Backspace, NamedKey::Escape] {
			assert_eq!(
				field
					.key(
						&mut line,
						&TextKey::Named(key),
						None,
						ModifiersState::empty(),
						false
					)
					.unwrap(),
				None
			);
		}
		assert_eq!(line.text(), "echo ");
		field.blur();
		assert!(field.preedit.is_empty());
	}

	#[test]
	fn scrolling_keeps_caret_visible_and_resets_for_shorter_lines() {
		assert_eq!(visible_offset(0.0, 300.0, 400.0, 100.0), 202.0);
		assert_eq!(visible_offset(202.0, 20.0, 400.0, 100.0), 20.0);
		assert_eq!(visible_offset(202.0, 20.0, 20.0, 100.0), 0.0);
		assert_eq!(visible_offset(10.0, 5.0, 20.0, 0.0), 5.0);
	}
}
