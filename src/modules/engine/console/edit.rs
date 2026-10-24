//! A single-line editor. Positions are UTF-8 byte offsets at grapheme boundaries;
//! the caret and selection anchor always belong to the current text.

use std::ops::Range;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct State {
	text: String,
	cursor: usize,
	anchor: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Movement {
	Left,
	Right,
	WordLeft,
	WordRight,
	Start,
	End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edit {
	Typing,
	Backspace,
	Delete,
}

/// Text, selection and undo history, independent of the window and renderer.
#[derive(Clone, Debug, Default)]
pub struct LineEdit {
	state: State,
	undo: Vec<State>,
	redo: Vec<State>,
	group: Option<Edit>,
}

impl LineEdit {
	pub fn text(&self) -> &str {
		&self.state.text
	}

	pub fn cursor(&self) -> usize {
		self.state.cursor
	}

	pub fn anchor(&self) -> usize {
		self.state.anchor
	}

	pub fn selection(&self) -> Range<usize> {
		self.state.anchor.min(self.cursor())..self.state.anchor.max(self.cursor())
	}

	pub fn selected(&self) -> &str {
		&self.text()[self.selection()]
	}

	/// Loads a new line, with its caret at the end and a fresh undo history.
	pub fn set(&mut self, text: &str) {
		*self = Self::default();
		self.state.text = single_line(text);
		self.state.cursor = self.text().len();
		self.state.anchor = self.cursor();
	}

	/// Every legal caret position, including the end of the line.
	pub fn boundaries(&self) -> impl DoubleEndedIterator<Item = usize> + '_ {
		self
			.text()
			.grapheme_indices(true)
			.map(|(i, _)| i)
			.chain(std::iter::once(self.text().len()))
	}

	fn boundary(&self, at: usize) -> usize {
		self
			.boundaries()
			.take_while(|&i| i <= at)
			.last()
			.unwrap_or(0)
	}

	pub fn select(&mut self, anchor: usize, cursor: usize) {
		self.state.anchor = self.boundary(anchor);
		self.state.cursor = self.boundary(cursor);
		self.group = None;
	}

	pub fn place(&mut self, at: usize, extend: bool) {
		self.select(if extend { self.state.anchor } else { at }, at);
	}

	pub fn select_all(&mut self) {
		self.select(0, self.text().len());
	}

	pub fn word_at(&self, at: usize) -> Range<usize> {
		let mut at = self.boundary(at);
		if at == self.text().len() {
			at = self.boundaries().rev().nth(1).unwrap_or(0);
		}
		self
			.text()
			.split_word_bound_indices()
			.find(|(i, word)| at < i + word.len())
			.map_or(at..at, |(i, word)| i..i + word.len())
	}

	fn destination(&self, movement: Movement) -> usize {
		let at = self.cursor();
		match movement {
			Movement::Start => 0,
			Movement::End => self.text().len(),
			Movement::Left => self
				.boundaries()
				.take_while(|&i| i < at)
				.last()
				.unwrap_or(0),
			Movement::Right => self
				.boundaries()
				.find(|&i| i > at)
				.unwrap_or(self.text().len()),
			Movement::WordLeft => self
				.text()
				.unicode_word_indices()
				.map(|(i, _)| i)
				.take_while(|&i| i < at)
				.last()
				.unwrap_or(0),
			Movement::WordRight => self
				.text()
				.unicode_word_indices()
				.map(|(i, _)| i)
				.find(|&i| i > at)
				.unwrap_or(self.text().len()),
		}
	}

	pub fn move_cursor(&mut self, movement: Movement, extend: bool) {
		let selection = self.selection();
		let at = if !extend && !selection.is_empty() {
			match movement {
				Movement::Left | Movement::WordLeft => selection.start,
				Movement::Right | Movement::WordRight => selection.end,
				_ => self.destination(movement),
			}
		} else {
			self.destination(movement)
		};
		self.place(at, extend);
	}

	/// Printable keyboard text, grouped until a selection or another edit kind.
	pub fn type_text(&mut self, text: &str) {
		let text = single_line(text);
		if !text.is_empty() {
			self.replace(self.selection(), &text, Some(Edit::Typing));
		}
	}

	/// Pasting is one undo step. Newlines become spaces; they never submit commands.
	pub fn insert(&mut self, text: &str) {
		let text = single_line(text);
		if !text.is_empty() {
			self.replace(self.selection(), &text, None);
		}
	}

	/// Completion and cut use the same replacement path as ordinary editing.
	pub fn replace_range(&mut self, range: Range<usize>, text: &str) {
		let range = self.boundary(range.start)..self.boundary(range.end.max(range.start));
		self.replace(range, &single_line(text), None);
	}

	fn replace(&mut self, range: Range<usize>, text: &str, group: Option<Edit>) {
		if range.is_empty() && text.is_empty() {
			return;
		}
		if group.is_none() || self.group != group || !self.selection().is_empty() {
			if self.undo.len() == 128 {
				self.undo.remove(0);
			}
			self.undo.push(self.state.clone());
		}
		self.redo.clear();
		self.state.text.replace_range(range.clone(), text);
		// Combining marks can join the surrounding graphemes after replacement.
		let cursor = self
			.boundaries()
			.find(|&i| i >= range.start + text.len())
			.unwrap();
		self.state.cursor = cursor;
		self.state.anchor = cursor;
		self.group = group;
	}

	pub fn delete(&mut self, backward: bool, word: bool) {
		let mut range = self.selection();
		if range.is_empty() {
			let movement = match (backward, word) {
				(true, true) => Movement::WordLeft,
				(false, true) => Movement::WordRight,
				(true, false) => Movement::Left,
				(false, false) => Movement::Right,
			};
			let to = self.destination(movement);
			range = to.min(self.cursor())..to.max(self.cursor());
		}
		self.replace(
			range,
			"",
			Some(if backward {
				Edit::Backspace
			} else {
				Edit::Delete
			}),
		);
	}

	pub fn undo(&mut self) {
		if let Some(state) = self.undo.pop() {
			self.redo.push(std::mem::replace(&mut self.state, state));
		}
		self.group = None;
	}

	pub fn redo(&mut self) {
		if let Some(state) = self.redo.pop() {
			self.undo.push(std::mem::replace(&mut self.state, state));
		}
		self.group = None;
	}
}

fn single_line(text: &str) -> String {
	text
		.replace("\r\n", " ")
		.chars()
		.filter_map(|c| match c {
			'\r' | '\n' | '\t' | '\u{2028}' | '\u{2029}' => Some(' '),
			c if c.is_control() => None,
			c => Some(c),
		})
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn inserts_replaces_and_deletes_at_the_caret() {
		let mut line = LineEdit::default();
		line.set("echo hllo");
		line.place(6, false);
		line.type_text("e");
		assert_eq!(line.text(), "echo hello");
		line.select(5, 10);
		line.type_text("world");
		assert_eq!(line.text(), "echo world");
		line.move_cursor(Movement::Left, false);
		line.delete(false, false);
		line.delete(true, false);
		assert_eq!(line.text(), "echo wor");
		line.select_all();
		line.delete(false, true);
		assert_eq!((line.text(), line.cursor()), ("", 0));
		line.delete(true, false);
		line.delete(false, false);
	}

	#[test]
	fn selection_collapses_and_extends_from_its_anchor() {
		let mut line = LineEdit::default();
		line.set("one two three");
		line.move_cursor(Movement::WordLeft, true);
		assert_eq!(line.selected(), "three");
		line.move_cursor(Movement::WordLeft, true);
		assert_eq!(line.selected(), "two three");
		line.move_cursor(Movement::Right, false);
		assert_eq!(line.cursor(), 13);
		assert!(line.selected().is_empty());
		line.move_cursor(Movement::Start, false);
		line.move_cursor(Movement::WordRight, true);
		assert_eq!(line.selected(), "one ");
		line.delete(true, true);
		assert_eq!(line.text(), "two three");
	}

	#[test]
	fn editing_never_splits_a_user_perceived_character() {
		let mut line = LineEdit::default();
		line.set("é e\u{301} 👩‍💻🇩🇪");
		line.delete(true, false);
		assert_eq!(line.text(), "é e\u{301} 👩‍💻");
		line.move_cursor(Movement::Left, true);
		assert_eq!(line.selected(), "👩‍💻");
		line.insert("你好");
		line.place(1, false); // Even external hit testing cannot split UTF-8.
		assert_eq!(line.cursor(), 0);
		line.delete(false, false);
		assert_eq!(line.text(), " e\u{301} 你好");
		line.place(1, false);
		line.delete(false, false);
		assert_eq!(line.text(), "  你好");
	}

	#[test]
	fn undo_restores_selection_and_redo_is_invalidated_by_new_edits() {
		let mut line = LineEdit::default();
		line.type_text("h");
		line.type_text("i");
		line.undo();
		assert_eq!(line.text(), "");
		line.redo();
		line.select_all();
		line.insert("bye\r\nworld\t!\0");
		assert_eq!(line.text(), "bye world !");
		line.undo();
		assert_eq!(line.selected(), "hi");
		line.type_text("new");
		line.redo();
		assert_eq!(line.text(), "new");
	}
}
