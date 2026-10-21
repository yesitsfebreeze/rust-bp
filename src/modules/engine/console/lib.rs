//! In-game console: command parsing, history and completion over any set of
//! [`cvar::Registry`] cvars. Drawing and key handling belong to the game; this
//! holds the text and runs the commands.

use std::path::{Path, PathBuf};

use cvar::Registry;

/// What a command asks of the game beyond changing cvars.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
	Quit,
	Respawn,
	Noclip,
}

const COMMANDS: &[(&str, &str)] = &[
	("help", "list commands"),
	("cvarlist", "cvarlist [filter]: list cvars with values"),
	("reset", "reset <cvar|all>: restore defaults"),
	("toggle", "toggle <cvar>: switch between 0 and 1"),
	(
		"exec",
		"exec [file]: apply a TOML config (default: the loaded one)",
	),
	(
		"writeconfig",
		"writeconfig [file]: save cvars as TOML (default: the loaded one)",
	),
	("respawn", "move back to the spawn point"),
	("noclip", "fly through walls"),
	("echo", "echo <text>"),
	("clear", "clear the console"),
	("quit", "exit the game"),
];

/// Lines kept for scrollback.
const SCROLLBACK: usize = 512;

pub struct Console {
	pub open: bool,
	pub input: String,
	lines: Vec<String>,
	history: Vec<String>,
	/// Index into `history` while browsing it with up and down.
	recall: Option<usize>,
	/// The config `exec` and `writeconfig` use without an argument.
	config: PathBuf,
}

impl Console {
	pub fn new(config: impl Into<PathBuf>) -> Console {
		Console {
			open: false,
			input: String::new(),
			lines: Vec::new(),
			history: Vec::new(),
			recall: None,
			config: config.into(),
		}
	}

	pub fn config(&self) -> &Path {
		&self.config
	}

	pub fn lines(&self) -> &[String] {
		&self.lines
	}

	pub fn print(&mut self, line: impl Into<String>) {
		self.lines.push(line.into());
		if self.lines.len() > SCROLLBACK {
			self.lines.drain(..self.lines.len() - SCROLLBACK);
		}
	}

	/// Runs the input line, echoing it and remembering it in the history.
	pub fn submit<R: Registry>(&mut self, cvars: &mut R) -> Vec<Action> {
		let line = std::mem::take(&mut self.input);
		self.recall = None;
		self.print(format!("] {line}"));
		if !line.trim().is_empty() && self.history.last() != Some(&line) {
			self.history.push(line.clone());
		}
		self.execute(&line, cvars)
	}

	/// Runs `;`-separated commands.
	pub fn execute<R: Registry>(&mut self, line: &str, cvars: &mut R) -> Vec<Action> {
		line
			.split(';')
			.filter_map(|c| self.command(c, cvars))
			.collect()
	}

	fn command<R: Registry>(&mut self, text: &str, cvars: &mut R) -> Option<Action> {
		let mut words = text.split_whitespace();
		let name = words.next()?;
		let args: Vec<&str> = words.collect();
		match (name, args.as_slice()) {
			("help", _) => {
				for (c, h) in COMMANDS {
					self.print(format!("  {c:<12} {h}"));
				}
				self.print("  <cvar> shows a cvar, <cvar> <value> sets it; tab completes");
			}
			("cvarlist", filter) => {
				let filter = filter.first().copied().unwrap_or("");
				for n in R::NAMES.iter().filter(|n| n.contains(filter)) {
					self.print(format!(
						"  {n:<18} {:<8} {}",
						cvars.get(n).unwrap(),
						R::help(n).unwrap()
					));
				}
			}
			("reset", ["all"]) => {
				*cvars = R::default();
				self.print("all cvars reset");
			}
			("reset", [n]) => self.report(
				cvars
					.reset(n)
					.map(|()| format!("{n} = {}", cvars.get(n).unwrap())),
			),
			("toggle", [n]) => {
				let value = cvars.get(n).map(|v| if v == 0.0 { 1.0 } else { 0.0 });
				let result = match value {
					Some(v) => cvars.set(n, v).map(|()| format!("{n} = {v}")),
					None => Err(format!("unknown cvar \"{n}\"")),
				};
				self.report(result);
			}
			("exec", path) => {
				let path = path.first().map_or(self.config.clone(), PathBuf::from);
				self.exec(&path, cvars);
			}
			("writeconfig", path) => {
				let path = path.first().map_or(self.config.clone(), PathBuf::from);
				let result = std::fs::write(&path, cvars.to_toml())
					.map(|()| format!("wrote {}", path.display()))
					.map_err(|e| format!("{}: {e}", path.display()));
				self.report(result);
			}
			("respawn", []) => return Some(Action::Respawn),
			("noclip", []) => return Some(Action::Noclip),
			("quit" | "exit", []) => return Some(Action::Quit),
			("clear", []) => self.lines.clear(),
			("echo", words) => self.print(words.join(" ")),
			(n, []) if cvars.get(n).is_some() => {
				self.print(format!(
					"{n} = {}  ({})",
					cvars.get(n).unwrap(),
					R::help(n).unwrap()
				));
			}
			(n, [value]) if cvars.get(n).is_some() => {
				let result = value
					.parse::<f32>()
					.map_err(|_| format!("{n}: \"{value}\" is not a number"))
					.and_then(|v| cvars.set(n, v).map(|()| format!("{n} = {v}")));
				self.report(result);
			}
			_ => self.print(format!("unknown command \"{}\"; try help", text.trim())),
		}
		None
	}

	/// Applies a TOML config file, reporting what it could not apply.
	pub fn exec<R: Registry>(&mut self, path: &Path, cvars: &mut R) {
		let result = std::fs::read_to_string(path)
			.map_err(|e| e.to_string())
			.and_then(|text| cvars.apply_toml(&text));
		match result {
			Ok(problems) => {
				for p in problems {
					self.print(format!("{}: {p}", path.display()));
				}
				self.print(format!("executed {}", path.display()));
			}
			Err(e) => self.print(format!("{}: {e}", path.display())),
		}
	}

	fn report(&mut self, result: Result<String, String>) {
		match result {
			Ok(s) | Err(s) => self.print(s),
		}
	}

	/// Completes the first word to the longest prefix shared by matching commands
	/// and cvars of `R`, listing them when there is more than one.
	pub fn complete<R: Registry>(&mut self) {
		if self.input.contains(char::is_whitespace) {
			return;
		}
		let prefix = self.input.as_str();
		let matches: Vec<&str> = COMMANDS
			.iter()
			.map(|(c, _)| *c)
			.chain(R::NAMES.iter().copied())
			.filter(|n| n.starts_with(prefix))
			.collect();
		let Some(first) = matches.first() else {
			return;
		};
		let common = matches.iter().fold(first.len(), |len, m| {
			first
				.bytes()
				.zip(m.bytes())
				.take(len)
				.take_while(|(a, b)| a == b)
				.count()
		});
		if matches.len() > 1 {
			let listing = matches.join("  ");
			self.print(listing);
		}
		self.input = first[..common].to_owned();
		if matches.len() == 1 {
			self.input.push(' ');
		}
	}

	/// Steps back (`older`) or forward through submitted lines.
	pub fn recall(&mut self, older: bool) {
		if self.history.is_empty() {
			return;
		}
		let last = self.history.len() - 1;
		self.recall = match (self.recall, older) {
			(None, true) => Some(last),
			(None, false) => None,
			(Some(i), true) => Some(i.saturating_sub(1)),
			(Some(i), false) if i < last => Some(i + 1),
			(Some(_), false) => None,
		};
		self.input = self
			.recall
			.map_or_else(String::new, |i| self.history[i].clone());
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	cvar::cvars! {
		struct Test {
			sv_gravity = 800.0, "Gravity.";
			sv_airaccelerate = 10.0, "Air acceleration.";
			sv_maxspeed = 320.0, "Max speed.";
			sv_maxvelocity = 2000.0, "Max velocity.";
			fov = 90.0, "Field of view.";
			cl_showfps = 1.0, "Show fps.";
		}
	}

	fn run(console: &mut Console, cvars: &mut Test, line: &str) -> Vec<Action> {
		console.input = line.to_owned();
		console.submit(cvars)
	}

	#[test]
	fn sets_shows_and_resets_cvars() {
		let (mut con, mut cv) = (Console::new("unused.toml"), Test::default());
		run(&mut con, &mut cv, "sv_gravity 600; sv_airaccelerate 100");
		assert_eq!((cv.sv_gravity, cv.sv_airaccelerate), (600.0, 100.0));
		run(&mut con, &mut cv, "sv_gravity");
		assert!(con.lines().last().unwrap().starts_with("sv_gravity = 600"));
		run(&mut con, &mut cv, "sv_gravity fast");
		assert_eq!(cv.sv_gravity, 600.0);
		run(&mut con, &mut cv, "toggle cl_showfps");
		assert_eq!(cv.cl_showfps, 0.0);
		run(&mut con, &mut cv, "reset sv_gravity");
		assert_eq!(cv.sv_gravity, 800.0);
		run(&mut con, &mut cv, "reset all");
		assert_eq!(cv, Test::default());
	}

	#[test]
	fn returns_game_actions() {
		let (mut con, mut cv) = (Console::new("unused.toml"), Test::default());
		assert_eq!(
			run(&mut con, &mut cv, "noclip;respawn; quit"),
			[Action::Noclip, Action::Respawn, Action::Quit]
		);
		assert!(run(&mut con, &mut cv, "frobnicate").is_empty());
		assert!(con.lines().last().unwrap().contains("unknown command"));
	}

	#[test]
	fn writeconfig_then_exec_restores_values() {
		let path = std::env::temp_dir().join(format!("console-{}.toml", std::process::id()));
		let (mut con, mut cv) = (Console::new(&path), Test::default());
		run(&mut con, &mut cv, "fov 110; writeconfig; fov 90; exec");
		std::fs::remove_file(&path).unwrap();
		assert_eq!(cv.fov, 110.0);
	}

	#[test]
	fn completes_and_recalls() {
		let (mut con, mut cv) = (Console::new("unused.toml"), Test::default());
		con.input = "sv_air".into();
		con.complete::<Test>();
		assert_eq!(con.input, "sv_airaccelerate ");
		con.input = "sv_ma".into();
		con.complete::<Test>();
		assert_eq!(con.input, "sv_max");
		run(&mut con, &mut cv, "fov 100");
		run(&mut con, &mut cv, "fov 95");
		con.recall(true);
		assert_eq!(con.input, "fov 95");
		con.recall(true);
		assert_eq!(con.input, "fov 100");
		con.recall(false);
		con.recall(false);
		assert_eq!(con.input, "");
	}
}
