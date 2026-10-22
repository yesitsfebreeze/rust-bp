//! The in-game level editor (F2): a panel down the left of the screen and
//! the map behind it. Hold the right mouse button to look and fly with
//! WASD; with it up the cursor is free. Pick a tool and click the map to
//! place a box, ramp, cylinder, stairs, model, light or prop where the
//! cursor points; with Select, click a thing to select it and drag it
//! across the ground. Arrows and Page Up/Down nudge the selection along the
//! grid, Q and E turn it, Delete removes it, Ctrl+D duplicates it, Ctrl+S
//! saves. Its properties — size, turn, texture, a light's color — edit in
//! the panel.
//!
//! The editor only edits the [`Level`]; what the game must do about an
//! edit — rebuild the map's geometry, save, rebake the light — it asks for
//! with [`Request`]s.

use engine::bsp::Rect as AtlasRect;
use engine::gpu::{Color, Gfx, Input, Key, MouseButton};
use engine::linalg::{vec3, Quat, Vec3};
use engine::physics;
use engine::ui::{Rect, Scroll, Theme, Ui};

use crate::level::{Kind, Level, Object};

/// Where the camera looks from, for turning the cursor into a ray.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
	pub eye: Vec3,
	pub forward: Vec3,
	pub right: Vec3,
	pub up: Vec3,
	/// Vertical field of view, radians.
	pub fovy: f32,
}

/// What an edit asks of the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
	/// The level's static geometry changed: rebuild the map with it.
	Rebuild,
	/// The props or lights changed: put the props back where they stand.
	Reset,
	Save,
	/// Save, then bake the map's light with the level's geometry in it.
	Bake,
}

/// The panel's width, pixels.
const PANEL: f32 = 330.0;
/// The grid placement and nudges snap to, world units.
const GRID: f32 = 8.0;
/// Degrees Q and E turn the selection.
const TURN: f32 = 15.0;

pub struct Editor {
	pub open: bool,
	/// The tool: a kind to place, or none to select.
	tool: Option<Kind>,
	pub selected: Option<usize>,
	/// The texture and model new things are made with, by index.
	texture: usize,
	model: usize,
	scroll: Scroll,
	/// Whether the selected light's color picker is open.
	color_open: bool,
	/// A drag in progress: where on the selection it was taken, and whether
	/// it has moved yet.
	drag: Option<(Vec3, bool)>,
	/// The last thing the editor has to say: saved, baked, a failure.
	pub status: String,
}

impl Editor {
	pub fn new() -> Editor {
		Editor {
			open: false,
			tool: None,
			selected: None,
			texture: 0,
			model: 0,
			scroll: Scroll::default(),
			color_open: false,
			drag: None,
			status: String::new(),
		}
	}

	/// Whether the cursor is over the panel, where clicks are its own.
	pub fn over_panel(&self, input: &Input) -> bool {
		self.open && input.cursor().0 < PANEL
	}

	/// One frame of the editor over `level`: the panel, picking and
	/// dragging in the map through `mesh`, the keys, and the selection and
	/// lights drawn in the world. `textures` are the map's, `models` the
	/// model files to place. Returns what the game must do about the edits.
	#[allow(clippy::too_many_arguments)]
	pub fn frame(
		&mut self,
		gfx: &mut Gfx,
		input: &Input,
		camera: &Camera,
		level: &mut Level,
		textures: &[(String, AtlasRect)],
		models: &[String],
		mesh: &physics::Mesh,
	) -> Vec<Request> {
		let mut requests = Vec::new();
		if self.selected.is_some_and(|i| i >= level.objects.len()) {
			self.selected = None;
		}
		self.panel(gfx, input, level, textures, models, &mut requests);
		let looking = input.button_down(MouseButton::Right);
		if !looking && !self.over_panel(input) {
			self.pointer(gfx, input, camera, level, textures, models, mesh, &mut requests);
		} else if !input.button_down(MouseButton::Left) {
			self.drop_drag(level, &mut requests);
		}
		self.keys(input, camera, level, &mut requests);
		self.draw_world(gfx, level);
		requests.dedup();
		requests
	}

	/// The panel down the left: tools, what new things are made of, the
	/// selection's properties, and saving and baking.
	fn panel(
		&mut self,
		gfx: &mut Gfx,
		input: &Input,
		level: &mut Level,
		textures: &[(String, AtlasRect)],
		models: &[String],
		requests: &mut Vec<Request>,
	) {
		let mut ui = Ui::new(gfx, input, Theme::STEAM);
		let size = ui.size();
		let panel = Rect::new(0.0, 0.0, PANEL, size.1);
		ui.panel(panel);
		let size_text = ui.theme.size;
		ui.label("Level editor", 12.0, 10.0, size_text + 4.0, ui.theme.highlight);
		let view = Rect::new(10.0, 44.0, PANEL - 20.0, size.1 - 54.0);
		let mut row = ui.scroll(view, &mut self.scroll);
		let mut next = |h: f32| {
			let (top, rest) = row.cut_top(h);
			row = rest.cut_top(6.0).1;
			top
		};
		let heading = |ui: &mut Ui, r: Rect, text: &str| {
			let c = ui.theme.dim;
			ui.label(text, r.x, r.y + 4.0, size_text, c);
		};

		heading(&mut ui, next(24.0), "Tools");
		let mut tools: Vec<(Option<Kind>, &str)> = vec![(None, "select")];
		tools.extend(Kind::ALL.iter().map(|&k| (Some(k), k.name())));
		for pair in tools.chunks(2) {
			let line = next(26.0);
			let (right, left) = line.cut_right(line.w * 0.5);
			for (cell, &(tool, name)) in [left, right].into_iter().zip(pair) {
				let label = if self.tool == tool {
					format!("> {name}")
				} else {
					name.to_string()
				};
				if ui.button(Rect::new(cell.x, cell.y, cell.w - 4.0, cell.h), &label) {
					self.tool = tool;
				}
			}
		}

		heading(&mut ui, next(24.0), "New things");
		if !textures.is_empty() {
			self.texture %= textures.len();
			if let Some(step) = stepper(&mut ui, next(26.0), &textures[self.texture].0) {
				self.texture = step_index(self.texture, step, textures.len());
			}
		}
		if models.is_empty() {
			heading(&mut ui, next(24.0), "no .obj models in assets/models");
		} else {
			self.model %= models.len();
			if let Some(step) = stepper(&mut ui, next(26.0), &models[self.model]) {
				self.model = step_index(self.model, step, models.len());
			}
		}

		if let Some(i) = self.selected {
			let o = &mut level.objects[i];
			heading(&mut ui, next(24.0), &format!("Selected: {} #{i}", o.kind.name()));
			heading(
				&mut ui,
				next(22.0),
				&format!("at {:.0} {:.0} {:.0}", o.pos.x, o.pos.y, o.pos.z),
			);
			let before = o.clone();
			if o.kind == Kind::Light {
				let room = Ui::color_height(self.color_open);
				ui.color(next(room), "Color", &mut o.color, &mut self.color_open);
				ui.slider(next(44.0), "Intensity", &mut o.intensity, 0.0, 20.0);
				ui.slider(next(44.0), "Reach", &mut o.reach, 32.0, 2000.0);
			} else {
				ui.slider(next(44.0), "Width", &mut o.size.x, GRID, 1024.0);
				ui.slider(next(44.0), "Height", &mut o.size.y, GRID, 1024.0);
				if o.kind == Kind::Ball {
					o.size = Vec3::splat(o.size.y);
				} else {
					ui.slider(next(44.0), "Depth", &mut o.size.z, GRID, 1024.0);
				}
				ui.slider(next(44.0), "Turn", &mut o.yaw, 0.0, 360.0);
				// Sizes snap to the grid as they slide.
				o.size = o.size.map(|v| (v / GRID).round().max(1.0) * GRID);
				if !textures.is_empty() && o.kind != Kind::Ball {
					let at = textures
						.iter()
						.position(|(n, _)| n.eq_ignore_ascii_case(&o.texture))
						.unwrap_or(0);
					if let Some(step) = stepper(&mut ui, next(26.0), &textures[at].0) {
						o.texture = textures[step_index(at, step, textures.len())].0.clone();
					}
				}
				if o.kind == Kind::Model && !models.is_empty() {
					let at = models.iter().position(|m| *m == o.model).unwrap_or(0);
					if let Some(step) = stepper(&mut ui, next(26.0), &models[at]) {
						o.model = models[step_index(at, step, models.len())].clone();
					}
				}
			}
			if *o != before {
				requests.push(changed(o.kind));
			}
			let line = next(26.0);
			let (right, left) = line.cut_right(line.w * 0.5);
			let kind = o.kind;
			if ui.button(Rect::new(left.x, left.y, left.w - 4.0, left.h), "Duplicate") {
				duplicate(level, &mut self.selected);
				requests.push(changed(kind));
			}
			if ui.button(right, "Delete") {
				level.objects.remove(i);
				self.selected = None;
				requests.push(changed(kind));
			}
		}

		heading(&mut ui, next(24.0), "Level");
		heading(&mut ui, next(22.0), &format!("{} things", level.objects.len()));
		let line = next(26.0);
		let (right, left) = line.cut_right(line.w * 0.5);
		if ui.button(Rect::new(left.x, left.y, left.w - 4.0, left.h), "Save") {
			requests.push(Request::Save);
		}
		if ui.button(right, "Bake lighting") {
			requests.push(Request::Bake);
		}
		if !self.status.is_empty() {
			heading(&mut ui, next(22.0), &self.status.clone());
		}
		for help in [
			"Right mouse: look, WASD fly",
			"Left click: place or select",
			"Drag: move over the ground",
			"Arrows, PgUp/PgDn: nudge",
			"Q/E turn, Del delete",
			"Ctrl+D duplicate, Ctrl+S save",
			"F2: leave the editor",
		] {
			heading(&mut ui, next(20.0), help);
		}
		let bottom = row.y;
		ui.end_scroll(&mut self.scroll, bottom);
	}

	/// The cursor in the map: placing with a kind's tool, selecting and
	/// dragging with Select, and a preview of where a placement would land.
	#[allow(clippy::too_many_arguments)]
	fn pointer(
		&mut self,
		gfx: &mut Gfx,
		input: &Input,
		camera: &Camera,
		level: &mut Level,
		textures: &[(String, AtlasRect)],
		models: &[String],
		mesh: &physics::Mesh,
		requests: &mut Vec<Request>,
	) {
		let (w, h) = gfx.size();
		let ray = cursor_ray(camera, input.cursor(), (w, h));
		let clicked = input.clicked(MouseButton::Left);
		let held = input.button_down(MouseButton::Left);
		// A drag carries the selection across the horizontal plane it
		// stands on.
		if let (Some(i), Some((grab, _))) = (self.selected, self.drag) {
			if held {
				let o = &mut level.objects[i];
				if ray.1.y.abs() > 1e-4 {
					let t = (o.pos.y - ray.0.y) / ray.1.y;
					if t > 0.0 {
						let at = snap(ray.0 + ray.1 * t - grab);
						let to = vec3(at.x, o.pos.y, at.z);
						if to != o.pos {
							o.pos = to;
							self.drag = Some((grab, true));
						}
					}
				}
				return;
			}
			self.drop_drag(level, requests);
		}
		let hit = crate::bullets::nearest_hit(mesh, ray.0, ray.1, None).map(|(t, tri, _)| {
			let n = physics::triangle_normal(&mesh.tris()[tri]);
			(ray.0 + ray.1 * t, n)
		});
		match self.tool {
			Some(kind) => {
				let Some((at, n)) = hit else {
					return;
				};
				let mut pos = snap(at);
				if kind == Kind::Light {
					pos = snap(at + n * 32.0);
				} else {
					pos.y = at.y;
				}
				let texture = textures.get(self.texture).map_or("", |(n, _)| n.as_str());
				let mut o = Object::new(kind, pos, texture);
				if kind == Kind::Model {
					o.model = models.get(self.model).cloned().unwrap_or_default();
				}
				outline(gfx, &o, Color::new(0.4, 0.9, 1.0, 0.8));
				if clicked {
					level.objects.push(o);
					self.selected = Some(level.objects.len() - 1);
					requests.push(changed(kind));
				}
			}
			None if clicked => {
				let picked = pick(level, ray);
				if picked.is_some() && picked == self.selected {
					let o = &level.objects[picked.unwrap()];
					// Take it where the ray meets its base's plane.
					let grab = if ray.1.y.abs() > 1e-4 {
						let t = (o.pos.y - ray.0.y) / ray.1.y;
						ray.0 + ray.1 * t - o.pos
					} else {
						Vec3::ZERO
					};
					self.drag = Some((vec3(grab.x, 0.0, grab.z), false));
				}
				self.selected = picked;
				self.color_open &= picked.is_some();
			}
			None => {}
		}
	}

	/// Ends a drag: a moved static thing asks for the map to be rebuilt.
	fn drop_drag(&mut self, level: &Level, requests: &mut Vec<Request>) {
		if let (Some((_, true)), Some(i)) = (self.drag, self.selected) {
			requests.push(changed(level.objects[i].kind));
		}
		self.drag = None;
	}

	/// Nudging, turning, deleting, duplicating and saving from the keys.
	fn keys(&mut self, input: &Input, camera: &Camera, level: &mut Level, requests: &mut Vec<Request>) {
		let ctrl = input.down(Key::ControlLeft) || input.down(Key::ControlRight);
		if ctrl && input.pressed(Key::KeyS) {
			requests.push(Request::Save);
		}
		let Some(i) = self.selected else {
			return;
		};
		let kind = level.objects[i].kind;
		if input.pressed(Key::Delete) {
			level.objects.remove(i);
			self.selected = None;
			requests.push(changed(kind));
			return;
		}
		if ctrl && input.pressed(Key::KeyD) {
			duplicate(level, &mut self.selected);
			requests.push(changed(kind));
			return;
		}
		// Nudges along the ground axis nearest the camera's own.
		let step = if input.down(Key::ShiftLeft) { GRID * 8.0 } else { GRID };
		let flat = |v: Vec3| {
			if v.x.abs() >= v.z.abs() {
				vec3(v.x.signum(), 0.0, 0.0)
			} else {
				vec3(0.0, 0.0, v.z.signum())
			}
		};
		let (ahead, side) = (flat(camera.forward), flat(camera.right));
		let mut moved = Vec3::ZERO;
		for (key, dir) in [
			(Key::ArrowUp, ahead),
			(Key::ArrowDown, -ahead),
			(Key::ArrowRight, side),
			(Key::ArrowLeft, -side),
			(Key::PageUp, Vec3::Y),
			(Key::PageDown, -Vec3::Y),
		] {
			if input.pressed(key) {
				moved += dir * step;
			}
		}
		let o = &mut level.objects[i];
		let before = o.clone();
		o.pos += moved;
		if input.pressed(Key::KeyQ) {
			o.yaw = (o.yaw - TURN).rem_euclid(360.0);
		}
		if input.pressed(Key::KeyE) {
			o.yaw = (o.yaw + TURN).rem_euclid(360.0);
		}
		if *o != before {
			requests.push(changed(kind));
		}
	}

	/// Every light as a star in its color, the selection's outline.
	fn draw_world(&self, gfx: &mut Gfx, level: &Level) {
		for (i, o) in level.objects.iter().enumerate() {
			if o.kind == Kind::Light {
				let [r, g, b] = o.color;
				let c = Color::new(r, g, b, 1.0);
				for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
					gfx.line_3d(o.pos - axis * 10.0, o.pos + axis * 10.0, c);
				}
			}
			if Some(i) == self.selected {
				outline(gfx, o, Color::new(1.0, 0.85, 0.2, 1.0));
			}
		}
	}
}

impl Default for Editor {
	fn default() -> Editor {
		Editor::new()
	}
}

/// What changing a thing of `kind` asks for.
fn changed(kind: Kind) -> Request {
	if kind.is_static() {
		Request::Rebuild
	} else {
		Request::Reset
	}
}

/// A copy of the selection one grid step aside, selected instead.
fn duplicate(level: &mut Level, selected: &mut Option<usize>) {
	if let Some(i) = *selected {
		let mut copy = level.objects[i].clone();
		copy.pos += vec3(copy.size.x.max(GRID), 0.0, 0.0);
		level.objects.push(copy);
		*selected = Some(level.objects.len() - 1);
	}
}

/// `p` on the grid.
fn snap(p: Vec3) -> Vec3 {
	p.map(|v| (v / GRID).round() * GRID)
}

/// `at` stepped by `step` among `len`, wrapping.
fn step_index(at: usize, step: i32, len: usize) -> usize {
	(at as i32 + step).rem_euclid(len as i32) as usize
}

/// A name between back and forward buttons. Returns the step clicked.
fn stepper(ui: &mut Ui, rect: Rect, name: &str) -> Option<i32> {
	let (right, rest) = rect.cut_right(28.0);
	let left = Rect::new(rest.x, rest.y, 28.0, rest.h);
	let mid = Rect::new(rest.x + 34.0, rest.y, rest.w - 40.0, rest.h);
	let c = ui.theme.text;
	ui.label(name, mid.x, mid.y + 3.0, ui.theme.size, c);
	if ui.button(left, "<") {
		return Some(-1);
	}
	if ui.button(right, ">") {
		return Some(1);
	}
	None
}

/// The ray from the eye through pixel `cursor` of a `size` screen: its
/// origin and unit direction.
pub fn cursor_ray(camera: &Camera, cursor: (f32, f32), size: (f32, f32)) -> (Vec3, Vec3) {
	let tan = (camera.fovy * 0.5).tan();
	let aspect = size.0 / size.1.max(1.0);
	let x = (2.0 * cursor.0 / size.0.max(1.0) - 1.0) * tan * aspect;
	let y = (1.0 - 2.0 * cursor.1 / size.1.max(1.0)) * tan;
	let dir = (camera.forward + camera.right * x + camera.up * y).normalize();
	(camera.eye, dir)
}

/// The thing the ray meets first: through its turned bounds.
fn pick(level: &Level, (origin, dir): (Vec3, Vec3)) -> Option<usize> {
	let mut best: Option<(f32, usize)> = None;
	for (i, o) in level.objects.iter().enumerate() {
		// Into the thing's own frame, where its bounds are a box.
		let turn = Quat::from_axis_angle(Vec3::Y, o.yaw.to_radians()).conjugate();
		let half = if o.kind == Kind::Light {
			Vec3::splat(12.0)
		} else {
			o.size * 0.5
		};
		let o_local = turn.rotate(origin - o.center());
		let d_local = turn.rotate(dir);
		let (mut near, mut far) = (0.0f32, f32::MAX);
		for (p, d, h) in [
			(o_local.x, d_local.x, half.x),
			(o_local.y, d_local.y, half.y),
			(o_local.z, d_local.z, half.z),
		] {
			if d.abs() < 1e-8 {
				if p.abs() > h {
					near = f32::MAX;
				}
				continue;
			}
			let (a, b) = ((-h - p) / d, (h - p) / d);
			near = near.max(a.min(b));
			far = far.min(a.max(b));
		}
		if near <= far && best.is_none_or(|(t, _)| near < t) {
			best = Some((near, i));
		}
	}
	best.map(|(_, i)| i)
}

/// The thing's turned bounds as twelve lines.
fn outline(gfx: &mut Gfx, o: &Object, color: Color) {
	let turn = Quat::from_axis_angle(Vec3::Y, o.yaw.to_radians());
	let half = if o.kind == Kind::Light {
		Vec3::splat(12.0)
	} else {
		o.size * 0.5
	};
	let c = o.center();
	let corner = |x: f32, y: f32, z: f32| c + turn.rotate(vec3(x * half.x, y * half.y, z * half.z));
	for (a, b) in [
		((-1., -1., -1.), (1., -1., -1.)),
		((1., -1., -1.), (1., -1., 1.)),
		((1., -1., 1.), (-1., -1., 1.)),
		((-1., -1., 1.), (-1., -1., -1.)),
		((-1., 1., -1.), (1., 1., -1.)),
		((1., 1., -1.), (1., 1., 1.)),
		((1., 1., 1.), (-1., 1., 1.)),
		((-1., 1., 1.), (-1., 1., -1.)),
		((-1., -1., -1.), (-1., 1., -1.)),
		((1., -1., -1.), (1., 1., -1.)),
		((1., -1., 1.), (1., 1., 1.)),
		((-1., -1., 1.), (-1., 1., 1.)),
	] {
		gfx.line_3d(corner(a.0, a.1, a.2), corner(b.0, b.1, b.2), color);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn camera() -> Camera {
		Camera {
			eye: Vec3::ZERO,
			forward: vec3(0.0, 0.0, -1.0),
			right: vec3(1.0, 0.0, 0.0),
			up: vec3(0.0, 1.0, 0.0),
			fovy: 90f32.to_radians(),
		}
	}

	#[test]
	fn the_screen_middle_looks_straight_ahead() {
		let (o, d) = cursor_ray(&camera(), (400.0, 300.0), (800.0, 600.0));
		assert_eq!(o, Vec3::ZERO);
		assert!((d - vec3(0.0, 0.0, -1.0)).length() < 1e-5);
		let (_, up) = cursor_ray(&camera(), (400.0, 0.0), (800.0, 600.0));
		assert!(up.y > 0.5, "the top edge looks up: {up:?}");
	}

	#[test]
	fn picking_finds_the_nearest_thing_on_the_ray() {
		let mut level = Level::default();
		level.objects.push(Object::new(Kind::Box, vec3(0.0, -32.0, -300.0), ""));
		level.objects.push(Object::new(Kind::Box, vec3(0.0, -32.0, -150.0), ""));
		level.objects.push(Object::new(Kind::Box, vec3(500.0, -32.0, -150.0), ""));
		let ray = (Vec3::ZERO, vec3(0.0, 0.0, -1.0));
		assert_eq!(pick(&level, ray), Some(1));
		assert_eq!(pick(&level, (Vec3::ZERO, vec3(0.0, 0.0, 1.0))), None);
	}
}
