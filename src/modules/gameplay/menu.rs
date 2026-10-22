//! The escape menu, rebuilt from 1.6's in-game Steam menu: the title and
//! its items down the left over the paused view, the Options dialog with
//! its tabs, and the quit confirmation. Built from `engine::ui`'s widgets;
//! every option edits its cvar live, and Cancel puts them back.

use engine::gpu::{Color, Gfx, Input};
use engine::ui::{Rect, Scroll, Theme, Ui};

use crate::cvars::Cvars;

/// What a frame of the menu asks the game to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
	Resume,
	NewGame,
	Console,
	/// Options closed with OK: keep the cvars in the config.
	Save,
	Quit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
	Main,
	Options,
	Quit,
}

const TABS: [&str; 4] = ["Multiplayer", "Keyboard", "Mouse", "Video"];

/// The keys the game binds, as the Keyboard tab lists them.
const KEYS: [(&str, &str); 14] = [
	("Move forward", "W"),
	("Move back", "S"),
	("Move left", "A"),
	("Move right", "D"),
	("Jump", "SPACE / MWHEEL"),
	("Duck", "CTRL / C"),
	("Walk", "SHIFT"),
	("Primary attack", "MOUSE1"),
	("Secondary attack", "MOUSE2"),
	("Reload", "R"),
	("Weapon 1-8", "1-8"),
	("Buy menu", "B"),
	("Flycam", "F"),
	("Console", "` / F1"),
];

pub struct Menu {
	pub open: bool,
	page: Page,
	tab: usize,
	/// The cvars as the Options dialog found them, for Cancel.
	saved: Option<Cvars>,
	/// How far each tab's sheet is scrolled.
	scrolls: [Scroll; TABS.len()],
	/// Whether the fog color's picker is open.
	fog_color: bool,
}

impl Menu {
	pub fn new() -> Menu {
		Menu {
			open: false,
			page: Page::Main,
			tab: 0,
			saved: None,
			scrolls: Default::default(),
			fog_color: false,
		}
	}

	/// Escape: opens the menu, backs out of a dialog, or resumes.
	pub fn escape(&mut self, cvars: &mut Cvars) {
		match self.page {
			_ if !self.open => {
				self.open = true;
				self.page = Page::Main;
			}
			Page::Main => self.open = false,
			Page::Options => self.cancel(cvars),
			Page::Quit => self.page = Page::Main,
		}
	}

	fn cancel(&mut self, cvars: &mut Cvars) {
		if let Some(saved) = self.saved.take() {
			*cvars = saved;
		}
		self.page = Page::Main;
	}

	/// Draws the open menu and answers its clicks.
	pub fn frame(&mut self, gfx: &mut Gfx, input: &Input, cvars: &mut Cvars) -> Option<Choice> {
		if !self.open {
			return None;
		}
		let mut ui = Ui::new(gfx, input, Theme::STEAM);
		let size = ui.size();
		ui.fill(
			Rect::new(0.0, 0.0, size.0, size.1),
			Color::new(0.0, 0.0, 0.0, 0.45),
		);

		// Behind a dialog the items stay drawn but do not answer.
		ui.enabled = self.page == Page::Main;
		let mut choice = self.main(&mut ui);
		ui.enabled = true;
		match self.page {
			Page::Main => {}
			Page::Options => choice = choice.or(self.options(&mut ui, cvars)),
			Page::Quit => choice = choice.or(self.quit(&mut ui)),
		}
		if matches!(
			choice,
			Some(Choice::Resume | Choice::NewGame | Choice::Console)
		) {
			self.open = false;
		}
		choice
	}

	fn main(&mut self, ui: &mut Ui) -> Option<Choice> {
		let (_, height) = ui.size();
		let x = 72.0;
		let mut y = (height * 0.32).round();
		ui.label("COUNTER-STRIKE", x, y - 64.0, 48.0, ui.theme.text);
		let size = 26.0;
		let mut hit = None;
		// Each item with the gap above it, grouped as 1.6 groups them.
		for (gap, text) in [
			(0.0, "Resume Game"),
			(0.0, "New Game"),
			(16.0, "Console"),
			(0.0, "Options"),
			(16.0, "Quit"),
		] {
			y += gap;
			let w = ui.text_width(text, size);
			if ui.link(Rect::new(x, y, w, size + 8.0), text, size) {
				hit = Some(text);
			}
			y += size + 12.0;
		}
		match hit? {
			"Resume Game" => Some(Choice::Resume),
			"New Game" => Some(Choice::NewGame),
			"Console" => Some(Choice::Console),
			"Options" => {
				self.page = Page::Options;
				None
			}
			_ => {
				self.page = Page::Quit;
				None
			}
		}
	}

	fn options(&mut self, ui: &mut Ui, cvars: &mut Cvars) -> Option<Choice> {
		self.saved.get_or_insert_with(|| cvars.clone());
		// As tall as the screen allows; the sheets scroll what does not fit.
		let height = (ui.size().1 - 40.0).clamp(240.0, 560.0);
		let (body, closed) = ui.frame(Rect::centered(ui.size(), 600.0, height), "Options");
		let (buttons, body) = body.cut_bottom(34.0);
		let sheet = ui.tabs(body.cut_bottom(6.0).1, &TABS, &mut self.tab);
		let tab = self.tab;
		let mut row = ui.scroll(sheet, &mut self.scrolls[tab]);
		let mut next = |h: f32| {
			let (top, rest) = row.cut_top(h);
			row = rest.cut_top(8.0).1;
			top
		};
		match TABS[self.tab] {
			"Multiplayer" => {
				toggle(ui, next(24.0), "Draw crosshair", &mut cvars.cl_crosshair);
				toggle(ui, next(24.0), "Show frame rate", &mut cvars.cl_showfps);
				toggle(ui, next(24.0), "Show speed", &mut cvars.cl_showspeed);
				toggle(
					ui,
					next(24.0),
					"Bullet penetration",
					&mut cvars.sv_penetration,
				);
			}
			"Keyboard" => {
				let size = ui.theme.size;
				for (action, key) in KEYS {
					let line = next(size - 6.0);
					ui.label(action, line.x, line.y, size, ui.theme.text);
					ui.label(key, line.x + line.w * 0.55, line.y, size, ui.theme.dim);
				}
			}
			"Mouse" => {
				ui.slider(
					next(48.0),
					"Mouse sensitivity",
					&mut cvars.sensitivity,
					0.1,
					20.0,
				);
				let mut invert = cvars.m_pitch < 0.0;
				if ui.checkbox(next(24.0), "Reverse mouse", &mut invert) {
					cvars.m_pitch = -cvars.m_pitch;
				}
			}
			_ => {
				ui.slider(next(48.0), "Field of view", &mut cvars.fov, 60.0, 130.0);
				// Brightness moves only between the bounds the game sets.
				let (lo, hi) = (
					cvars.r_exposure_min,
					cvars.r_exposure_max.max(cvars.r_exposure_min),
				);
				ui.slider(next(48.0), "Brightness", &mut cvars.r_exposure, lo, hi);
				toggle(ui, next(24.0), "Auto exposure", &mut cvars.r_autoexposure);
				toggle(ui, next(24.0), "Traced sun shadows", &mut cvars.r_shadows);
				toggle(ui, next(24.0), "Wait for vertical sync", &mut cvars.r_vsync);
				toggle(ui, next(24.0), "Anti-aliasing (FXAA)", &mut cvars.r_fxaa);
				toggle(
					ui,
					next(24.0),
					"Physically based surfaces",
					&mut cvars.r_pbr,
				);
				toggle(ui, next(24.0), "Dynamic lights", &mut cvars.r_dlights);
				toggle(
					ui,
					next(24.0),
					"Player and object shadows",
					&mut cvars.r_dynshadows,
				);
				ui.slider(next(48.0), "Global illumination", &mut cvars.r_gi, 0.0, 2.0);
				ui.slider(next(48.0), "Ambient occlusion", &mut cvars.r_ao, 0.0, 1.0);
				ui.slider(
					next(48.0),
					"Occlusion radius (units)",
					&mut cvars.r_ao_radius,
					4.0,
					96.0,
				);
				ui.slider(next(48.0), "Fog", &mut cvars.r_fog, 0.0, 4.0);
				let mut fog = [cvars.r_fog_r, cvars.r_fog_g, cvars.r_fog_b];
				let room = Ui::color_height(self.fog_color);
				if ui.color(next(room), "Fog color", &mut fog, &mut self.fog_color) {
					[cvars.r_fog_r, cvars.r_fog_g, cvars.r_fog_b] = fog;
				}
			}
		}
		ui.end_scroll(&mut self.scrolls[tab], row.y);

		// OK, Cancel and Apply along the bottom right, as 1.6 sets them.
		let (apply, rest) = buttons.cut_right(96.0);
		let (cancel, rest) = rest.cut_right(104.0);
		let (ok, _) = rest.cut_right(104.0);
		let pad = |r: Rect| Rect::new(r.x, r.y + 4.0, 96.0, r.h - 4.0);
		if ui.button(pad(ok), "OK") {
			self.saved = None;
			self.page = Page::Main;
			return Some(Choice::Save);
		}
		if ui.button(pad(cancel), "Cancel") || closed {
			self.cancel(cvars);
		}
		if ui.button(pad(apply), "Apply") {
			self.saved = Some(cvars.clone());
			return Some(Choice::Save);
		}
		None
	}

	fn quit(&mut self, ui: &mut Ui) -> Option<Choice> {
		let (body, closed) = ui.frame(Rect::centered(ui.size(), 380.0, 150.0), "Quit game");
		ui.label(
			"Are you sure you want to quit?",
			body.x,
			body.y + 4.0,
			ui.theme.size,
			ui.theme.text,
		);
		let (buttons, _) = body.cut_bottom(30.0);
		let (cancel, rest) = buttons.cut_right(96.0);
		let (quit, _) = rest.cut_right(124.0);
		if ui.button(Rect::new(quit.x, quit.y, 116.0, quit.h), "Quit game") {
			return Some(Choice::Quit);
		}
		if ui.button(cancel, "Cancel") || closed {
			self.page = Page::Main;
		}
		None
	}
}

/// A check box over a 0/1 cvar.
fn toggle(ui: &mut Ui, rect: Rect, text: &str, value: &mut f32) {
	let mut on = *value != 0.0;
	if ui.checkbox(rect, text, &mut on) {
		*value = if on { 1.0 } else { 0.0 };
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn escape_opens_backs_out_and_resumes() {
		let mut cvars = Cvars::default();
		let mut menu = Menu::new();
		menu.escape(&mut cvars);
		assert!(menu.open && menu.page == Page::Main);
		menu.page = Page::Quit;
		menu.escape(&mut cvars);
		assert!(menu.open && menu.page == Page::Main);
		menu.escape(&mut cvars);
		assert!(!menu.open);
	}

	#[test]
	fn escaping_options_cancels_their_edits() {
		let mut cvars = Cvars::default();
		let mut menu = Menu::new();
		menu.escape(&mut cvars);
		menu.page = Page::Options;
		menu.saved = Some(cvars.clone());
		cvars.sensitivity = 9.0;
		menu.escape(&mut cvars);
		assert_eq!(cvars, Cvars::default());
		assert!(menu.open && menu.page == Page::Main && menu.saved.is_none());
	}
}
