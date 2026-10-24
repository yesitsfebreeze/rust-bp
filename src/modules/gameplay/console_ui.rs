//! Console presentation; the shared text field handles ordinary editing.

use engine::console::{Action, Console};
use engine::gpu::{Color, Gfx, Input};
use engine::ui::{Rect, TextAction, TextField, Theme};

use crate::cvars::Cvars;

const SIZE: f32 = 20.0;

#[derive(Default)]
pub struct ConsoleUi {
	field: TextField,
}

fn layout(gfx: &mut Gfx) -> Rect {
	let (width, height) = gfx.size();
	let x = 12.0 + gfx.text_width("] ", SIZE);
	Rect::new(
		x,
		height * 0.45 - 12.0 - SIZE * 0.8,
		(width - x - 12.0).max(0.0),
		SIZE,
	)
}

impl ConsoleUi {
	pub fn blur(&mut self) {
		self.field.blur();
	}

	pub fn input(
		&mut self,
		gfx: &mut Gfx,
		console: &mut Console,
		input: &Input,
		cvars: &mut Cvars,
		dt: f32,
	) -> Vec<Action> {
		let rect = layout(gfx);
		let mut actions = Vec::new();
		if input.focused() {
			for event in input.text_events() {
				match self.field.event(&mut console.input, event, gfx, rect, SIZE) {
					Ok(Some(TextAction::Submit)) => actions.extend(console.submit(cvars)),
					Ok(Some(TextAction::Complete)) => console.complete::<Cvars>(),
					Ok(Some(TextAction::Previous)) => console.recall(true),
					Ok(Some(TextAction::Next)) => console.recall(false),
					Ok(Some(TextAction::Close)) => {
						console.open = false;
						self.blur();
						break;
					}
					Ok(None) => {}
					Err(error) => console.print(error),
				}
			}
		}
		self
			.field
			.update(&mut console.input, input, gfx, rect, SIZE, dt);
		actions
	}

	pub fn draw(&mut self, gfx: &mut Gfx, console: &Console, focused: bool) {
		let (width, height) = gfx.size();
		let h = height * 0.45;
		gfx.rect(0.0, 0.0, width, h, Color::new(0.05, 0.07, 0.09, 0.88));
		let rect = layout(gfx);
		let mut y = h - 12.0;
		gfx.text("] ", 12.0, y, SIZE, super::TEXT);
		for line in console.lines().iter().rev() {
			y -= SIZE;
			if y < SIZE {
				break;
			}
			gfx.text(line, 12.0, y, SIZE, super::MUTED);
		}
		self.field.draw(
			&console.input,
			gfx,
			rect,
			Theme {
				text: super::TEXT,
				..Theme::STEAM
			},
			focused,
		);
	}
}
