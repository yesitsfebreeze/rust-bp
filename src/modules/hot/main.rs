//! The game in a window that stays open while its code changes. Gameplay
//! builds as the `hot_game` library; when a gameplay source changes this
//! host rebuilds it and swaps the new build in between frames. The window,
//! the GPU device and the audio device live on; the new generation reloads
//! the map and assets and picks up the player's state from the old one.
//!
//! `just hot` starts it. Engine changes still need a restart: the shared
//! `engine_dylib` stays loaded, so cargo cannot rewrite it underneath.

use engine_dylib::engine::gpu::{Color, Window};
use libloading::Library;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, SystemTime};

/// `hot_game::hot_play`, which is `gameplay::play`.
type Play = fn(&mut Window, Option<&[u8]>, u32, &mut dyn FnMut() -> bool) -> Option<Vec<u8>>;

/// How often the gameplay sources are checked for changes.
const POLL: Duration = Duration::from_millis(300);

fn root() -> &'static Path {
	Path::new(env!("CARGO_MANIFEST_DIR"))
		.parent()
		.and_then(Path::parent)
		.and_then(Path::parent)
		.expect("hot crate must be under src/modules")
}

/// One loaded build of the game, from its own copy of the library: Windows
/// locks a loaded file, and cargo must be free to write the next build.
struct Game {
	library: Option<Library>,
	play: Play,
	file: PathBuf,
}

impl Game {
	fn open(source: &Path, generation: u32) -> Result<Game, String> {
		let dir = source.with_file_name("hot-reload");
		std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
		let file = dir.join(format!(
			"hot_game-{}-{generation}{}",
			std::process::id(),
			std::env::consts::DLL_SUFFIX
		));
		std::fs::copy(source, &file).map_err(|e| format!("copy {}: {e}", source.display()))?;
		// SAFETY: the library is this workspace's `hot_game`, built by the same
		// compiler against the same `engine_dylib` this host has loaded; a build
		// against another engine fails to load on its missing symbols.
		let opened = unsafe {
			Library::new(&file).and_then(|library| {
				let play = *library.get::<Play>(b"hot_play")?;
				Ok((library, play))
			})
		};
		match opened {
			Ok((library, play)) => Ok(Game {
				library: Some(library),
				play,
				file,
			}),
			Err(e) => {
				let _ = std::fs::remove_file(&file);
				Err(format!("load {}: {e}", file.display()))
			}
		}
	}
}

impl Drop for Game {
	fn drop(&mut self) {
		drop(self.library.take());
		let _ = std::fs::remove_file(&self.file);
	}
}

/// Newest modification time of the gameplay crate's sources.
fn newest(dir: &Path) -> Option<SystemTime> {
	let mut newest = None;
	for entry in std::fs::read_dir(dir).ok()?.flatten() {
		let path = entry.path();
		let time = if path.is_dir() {
			self::newest(&path)
		} else if path
			.extension()
			.is_some_and(|ext| ext == "rs" || ext == "toml")
		{
			entry.metadata().and_then(|m| m.modified()).ok()
		} else {
			None
		};
		newest = newest.max(time);
	}
	newest
}

/// Rebuilds the game whenever a gameplay source changes and reports each
/// build that succeeds. The build names the same packages `just hot` does, so
/// cargo resolves the same features and leaves the loaded engine alone.
fn watch() -> Receiver<()> {
	let (built, receiver) = mpsc::channel();
	std::thread::spawn(move || {
		let sources = root().join("src").join("modules").join("gameplay");
		let mut seen = newest(&sources);
		loop {
			std::thread::sleep(POLL);
			let now = newest(&sources);
			if now == seen {
				continue;
			}
			seen = now;
			eprintln!("hot: gameplay changed, rebuilding");
			let status = Command::new(std::env::var_os("CARGO").unwrap_or("cargo".into()))
				.args(["build", "--locked", "-p", "hot", "-p", "hot_game"])
				.current_dir(root())
				.status();
			match status {
				Ok(status) if status.success() => {
					if built.send(()).is_err() {
						return;
					}
				}
				Ok(status) => eprintln!("hot: build failed ({status}); the running game stays"),
				Err(e) => eprintln!("hot: cargo: {e}"),
			}
		}
	});
	receiver
}

/// The next generation, once a fresh build has loaded; a build that does not
/// load leaves the current one running.
fn next(built: &Receiver<()>, source: &Path, generation: u32) -> Option<Game> {
	built.try_recv().ok()?;
	Game::open(source, generation + 1)
		.inspect_err(|e| eprintln!("hot: keeping #{generation}: {e}"))
		.ok()
}

fn main() {
	let mut window = match Window::open("player", 1600, 900) {
		Ok(window) => window,
		Err(e) => {
			eprintln!("{e}");
			std::process::exit(1);
		}
	};
	let exe = std::env::current_exe().expect("own path");
	let source = exe.with_file_name(format!(
		"{}hot_game{}",
		std::env::consts::DLL_PREFIX,
		std::env::consts::DLL_SUFFIX
	));
	let mut generation = 1;
	let mut game = match Game::open(&source, generation) {
		Ok(game) => game,
		Err(e) => {
			eprintln!("hot: {e}; build it with `just hot`");
			std::process::exit(1);
		}
	};
	let built = watch();
	let mut carried: Option<Vec<u8>> = None;
	loop {
		let mut fresh = None;
		let mut reload = || {
			fresh = next(&built, &source, generation);
			fresh.is_some()
		};
		let play = game.play;
		let ran = panic::catch_unwind(AssertUnwindSafe(|| {
			play(&mut window, carried.as_deref(), generation, &mut reload)
		}));
		match ran {
			Ok(None) => return,
			Ok(Some(state)) => carried = Some(state),
			// The panic is printed already; the state this generation started
			// from goes to the build that fixes it.
			Err(_) => {
				window.grab_cursor(false);
				while fresh.is_none() {
					let gfx = window.gfx();
					let (_, height) = gfx.size();
					let message = "the game panicked; save a fix to reload it";
					gfx.text(
						message,
						16.0,
						height / 2.0,
						24.0,
						Color::new(1.0, 0.3, 0.3, 1.0),
					);
					if !window.next_frame() {
						return;
					}
					fresh = next(&built, &source, generation);
				}
			}
		}
		// The old generation is gone; its images go with it before it unloads.
		window.gfx().clear_textures();
		game = fresh.expect("a generation ends only for its successor");
		generation += 1;
		eprintln!("hot: generation #{generation}");
	}
}
