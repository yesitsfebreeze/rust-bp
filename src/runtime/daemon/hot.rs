//! The hot-reload daemon: watch the sources, rebuild, swap the
//! executable — the old build keeps serving until the new one is ready.
//!
//! `daemon serve` runs this loop in the foreground of a terminal:
//!
//! 1. build once, spawn the executable (default `./target/release/app`);
//! 2. poll the watched paths every `poll`; on change (or a `reload`
//!    request through the control socket) run `cargo build --release`;
//! 3. a good build that changed the executable (or a `reload` request) asks
//!    the old child to persist its state (`{"op":"persist"}` on the app
//!    socket, see `reload::Host`), kills it and spawns the new one, which
//!    restores that state; a build that changed only module libraries keeps
//!    the child running and its `reload::Host` swaps them in-process; a failed
//!    build keeps the old child serving and records the error;
//! 4. a child that exits on its own is respawned (counted as a restart).
//!
//! The child runs from a per-generation copy of the executable in the daemon
//! dir (`run/`), so the build can replace the original even where a running
//! executable is locked (Windows). Children get `DAEMON_SUPERVISED=1`.
//!
//! The control socket answers `status`, `reload` and `stop` — the `daemon`
//! CLI and the MCP `status`/`reload`/`stop` tools speak it.
//!
//! `daemon serve --detach` runs the same loop as a per-project daemon: its
//! own session, output in `daemon.log` in the daemon dir, and an idle
//! exit once nobody has asked the socket or touched the sources for
//! `--idle` (default 30 minutes). `ensure_running` is what the CLI and the
//! MCP tools call first: it starts that daemon when no daemon answers.

use crate::util::ProcessTree;
use crate::{events, lsp, util, watch, wire};
use serde_json::Value;
use serde_json::{json, Map};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct Daemon {
	/// Rebuild command; run with the project root as cwd.
	pub build_command: Vec<String>,
	/// The executable to supervise, argv from the project root.
	pub child_command: Vec<String>,
	/// What triggers a rebuild: a change anywhere under these paths.
	pub watch: Vec<PathBuf>,
	pub poll: Duration,
	pub debounce: Duration,
	pub build_timeout: Duration,
	pub control_socket: String,
	/// Exit after this long without a socket request or a source change;
	/// `None` runs until `stop`. Detached daemons default to
	/// [`DEFAULT_IDLE`], foreground ones to `None` (the terminal owns them).
	pub idle: Option<Duration>,
}

/// The default child, relative to the project root (`setup` renames it).
const APP: &str = "./target/release/app";

/// How long a persist request may take before the restart is rejected.
const PERSIST_TIMEOUT: Duration = Duration::from_secs(6);

/// Idle limit of a detached daemon.
pub const DEFAULT_IDLE: Duration = Duration::from_secs(30 * 60);

impl Default for Daemon {
	fn default() -> Self {
		Self {
			build_command: vec!["cargo".into(), "build".into(), "--release".into()],
			child_command: vec![std::env::var_os("CARGO_TARGET_DIR")
				.map(PathBuf::from)
				.map(|dir| {
					dir
						.join("release")
						.join(std::path::Path::new(APP).file_name().unwrap())
				})
				.unwrap_or_else(|| PathBuf::from(APP))
				.with_extension(if cfg!(windows) { "exe" } else { "" })
				.display()
				.to_string()],
			watch: vec![
				PathBuf::from("src"),
				PathBuf::from("Cargo.toml"),
				PathBuf::from("Cargo.lock"),
				PathBuf::from(".cargo"),
			],
			poll: Duration::from_millis(250),
			debounce: Duration::from_millis(300),
			build_timeout: Duration::from_secs(600),
			control_socket: crate::daemon_socket(),
			idle: None,
		}
	}
}

struct Shared {
	reload_requested: u64,
	reload_completed: u64,
	building: bool,
	child_pid: Option<u32>,
	child_started_ms: u64,
	swaps: u32,
	restarts: u32,
	last_build_ok: bool,
	last_build: String,
	last_build_ms: u64,
	last_swap_ms: u64,
	executable_stamp: Option<(SystemTime, u64)>,
	restore: Option<PathBuf>,
}

struct State {
	shared: Mutex<Shared>,
	reload: AtomicBool,
	stop: AtomicBool,
	/// Last socket request or source change; the idle clock.
	last_contact: Mutex<Instant>,
}

/// Where a detached daemon writes what the terminal would show.
pub fn log_path() -> PathBuf {
	crate::dir().join("daemon.log")
}

/// `30m`, `90s`, `2h` or `0` (never) for `--idle`.
pub fn parse_idle(text: &str) -> Result<Option<Duration>, String> {
	let text = text.trim();
	if text == "0" {
		return Ok(None);
	}
	let (number, unit) = text.split_at(
		text
			.trim_end_matches(|c: char| c.is_ascii_alphabetic())
			.len(),
	);
	let n: u64 = number
		.parse()
		.map_err(|_| format!("`{text}` is no duration: use 30m, 90s, 2h or 0"))?;
	let secs = match unit {
		"s" | "" => n,
		"m" => n * 60,
		"h" => n * 3600,
		_ => return Err(format!("`{text}` is no duration: use 30m, 90s, 2h or 0")),
	};
	Ok(Some(Duration::from_secs(secs)))
}

/// Start `command` as a daemon: its own session, cwd the project root, stdin
/// closed, stdout and stderr appended to `daemon.log`. Returns the pid.
pub fn spawn_detached(command: &mut Command) -> Result<u32, String> {
	crate::ensure_dir().map_err(|e| e.to_string())?;
	let log = std::fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(log_path())
		.map_err(|e| format!("{}: {e}", log_path().display()))?;
	let err = log.try_clone().map_err(|e| e.to_string())?;
	let child = command
		.current_dir(crate::project_root())
		.stdin(Stdio::null())
		.stdout(Stdio::from(log))
		.stderr(Stdio::from(err));
	crate::platform::detach(child);
	child.spawn().map(|c| c.id()).map_err(|e| e.to_string())
}

/// The daemon's status, starting a detached one through `launcher` when
/// none answers. `launcher` builds the command that runs `serve`
/// (`daemon serve --idle 30m`, with or without Cargo in front).
pub fn ensure_running(launcher: impl FnOnce() -> Command) -> Result<Value, String> {
	let socket = crate::daemon_socket();
	let status = || wire::ask(&socket, serde_json::json!({"op": "status"}));
	if let Ok(reply) = status() {
		return Ok(reply);
	}
	// Parallel tool items must not launch competing daemons from this process.
	static START: Mutex<()> = Mutex::new(());
	let _start = START.lock().unwrap();
	if let Ok(reply) = status() {
		return Ok(reply);
	}
	let pid = spawn_detached(&mut launcher())?;
	// Cargo may compile the daemon first; the socket binds before the
	// first app build, so a plain start answers within a second.
	let deadline = Instant::now() + Duration::from_secs(120);
	while Instant::now() < deadline {
		if let Ok(mut reply) = status() {
			reply["started_pid"] = serde_json::json!(pid);
			return Ok(reply);
		}
		std::thread::sleep(Duration::from_millis(200));
	}
	let log = std::fs::read_to_string(log_path()).unwrap_or_default();
	let tail: String = log
		.lines()
		.rev()
		.take(10)
		.collect::<Vec<_>>()
		.into_iter()
		.rev()
		.collect::<Vec<_>>()
		.join("\n");
	Err(format!(
		"started daemon pid {pid} but {socket} never answered; last lines of {}:\n{tail}",
		log_path().display()
	))
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|d| d.as_millis() as u64)
		.unwrap_or(0)
}

/// Modification time and length: enough to tell a rebuilt executable.
fn fingerprint(path: &std::path::Path) -> Option<(SystemTime, u64)> {
	let meta = std::fs::metadata(path).ok()?;
	Some((meta.modified().ok()?, meta.len()))
}

impl Daemon {
	/// Run the loop until a `stop` request; returns the exit code.
	pub fn run(self) -> i32 {
		// The default socket lives in the daemon dir; a custom one (tests)
		// only needs its parent.
		if self.control_socket == crate::daemon_socket() {
			let _ = crate::ensure_dir();
		} else if let Some(parent) = std::path::Path::new(&self.control_socket).parent() {
			let _ = std::fs::create_dir_all(parent);
		}
		let state = Arc::new(State {
			shared: Mutex::new(Shared {
				reload_requested: 0,
				reload_completed: 0,
				building: false,
				child_pid: None,
				child_started_ms: 0,
				swaps: 0,
				restarts: 0,
				last_build_ok: false,
				last_build: "never built".into(),
				last_build_ms: 0,
				last_swap_ms: 0,
				executable_stamp: None,
				restore: None,
			}),
			reload: AtomicBool::new(false),
			stop: AtomicBool::new(false),
			last_contact: Mutex::new(Instant::now()),
		});
		let listener = match wire::bind(&self.control_socket) {
			Ok(l) => l,
			Err(e) => {
				eprintln!("daemon: {e}");
				return 2;
			}
		};
		let control = state.clone();
		let daemon = self.clone_shallow();
		let idle_limit = self.idle;
		// One rust-analyzer per daemon, started on the first `lsp` request.
		// Hold the slot only for initialization, never while waiting for replies.
		let analyzer: Mutex<Option<Arc<lsp::Client>>> = Mutex::new(None);
		wire::serve_concurrent(listener, move |req| {
			let op = req["op"].as_str().unwrap_or("");
			*control.last_contact.lock().unwrap() = Instant::now();
			match op {
				"status" => {
					let sh = control.shared.lock().unwrap();
					json!({
							"op": "status", "ok": true,
							"pid": std::process::id(),
							"idle_limit_s": idle_limit.map(|d| d.as_secs()),
							"log": log_path().display().to_string(),
							"running": sh.child_pid.is_some(),
							"child_pid": sh.child_pid,
							"child_started_ms": sh.child_started_ms,
							"building": sh.building, "reload_completed": sh.reload_completed,
							"swaps": sh.swaps, "restarts": sh.restarts,
							"last_build": {"ok": sh.last_build_ok, "detail": sh.last_build, "took_ms": sh.last_build_ms},
							"last_swap_ms": sh.last_swap_ms,
							"build_command": daemon.build_command,
							"child_command": daemon.child_command,
							"watch": daemon.watch.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
							"app_socket": crate::app_socket(),
							"daemon_dir": crate::dir().display().to_string(),
							"lsp": analyzer.try_lock().ok().and_then(|slot| slot.as_ref().map(|client| client.status())),
					})
				}
				"lsp" => {
					let mut slot = analyzer.lock().unwrap();
					if slot.is_none() {
						match lsp::Client::start(&crate::project_root()) {
							Ok(client) => {
								println!(
									"daemon: started rust-analyzer, pid {}",
									client.status()["pid"]
								);
								*slot = Some(Arc::new(client));
							}
							Err(error) => return json!({"op": "lsp", "ok": false, "error": error}),
						}
					}
					let client = slot.as_ref().unwrap().clone();
					drop(slot);
					match client.ask(&req) {
						Ok(mut reply) => {
							reply["op"] = json!("lsp");
							reply["ok"] = json!(true);
							reply
						}
						Err(error) => json!({"op": "lsp", "ok": false, "error": error}),
					}
				}
				"reload" => {
					let mut sh = control.shared.lock().unwrap();
					sh.reload_requested += 1;
					control.reload.store(true, Ordering::Relaxed);
					json!({"op": "reload", "ok": true, "queued": true, "ticket": sh.reload_requested})
				}
				"stop" => {
					control.stop.store(true, Ordering::Relaxed);
					json!({"op": "stop", "ok": true})
				}
				other => {
					json!({"ok": false, "error": format!("unknown op `{other}`: status, reload, stop, lsp")})
				}
			}
		});

		println!(
			"daemon: first build, then watching {:?} (edit sources to swap)",
			self.watch
		);
		let mut baseline = watch::snapshot(&self.watch);
		let mut child = self.build_and_swap(None, &state, true);
		println!(
			"daemon: serving {} — reload through the control socket",
			self.child_command.join(" ")
		);

		loop {
			if state.stop.load(Ordering::Relaxed) || util::cancelled() {
				break;
			}
			if let Some(live) = child.as_mut() {
				if let Ok(Some(status)) = live.0.try_wait() {
					println!("daemon: child exited on its own ({status}); respawning");
					{
						let mut sh = state.shared.lock().unwrap();
						sh.restarts += 1;
					}
					drop(child.take());
					child = self.spawn_child(&state);
					self.note_child(&state, child.as_ref());
				}
			}
			if child.is_none() {
				child = self.spawn_child(&state);
				self.note_child(&state, child.as_ref());
				std::thread::sleep(Duration::from_secs(2));
			}
			let changed = watch::snapshot(&self.watch) != baseline;
			let requested = state.reload.swap(false, Ordering::Relaxed);
			if changed || requested {
				*state.last_contact.lock().unwrap() = Instant::now();
				if changed {
					std::thread::sleep(self.debounce); // editors write in bursts
				}
				// Capture before building: edits during this build trigger the next one.
				baseline = watch::snapshot(&self.watch);
				child = self.build_and_swap(child.take(), &state, requested);
			}
			if let Some(limit) = self.idle {
				let idle = state.last_contact.lock().unwrap().elapsed();
				if idle > limit {
					println!(
						"daemon: idle for {} s, exiting (serve --idle 0 to stay)",
						idle.as_secs()
					);
					break;
				}
			}
			std::thread::sleep(self.poll);
		}
		drop(child);
		let _ = std::fs::remove_file(&self.control_socket);
		println!("daemon: stopped");
		0
	}

	/// Build; then swap the child when the build changed the executable or
	/// `forced` (first start, `reload` request). An unchanged executable keeps
	/// the running child: its modules reload in-process.
	fn build_and_swap(
		&self,
		mut old: Option<ProcessTree>,
		state: &State,
		forced: bool,
	) -> Option<ProcessTree> {
		let started = Instant::now();
		let executable = self.executable();
		let before = state.shared.lock().unwrap().executable_stamp;
		let ticket = {
			let mut sh = state.shared.lock().unwrap();
			sh.building = true;
			sh.reload_requested
		};
		let mut recorded = events::Tail::new();
		match util::run_with_timeout_until(&self.build_command, self.build_timeout, 4_000_000, &|| {
			state.stop.load(Ordering::Relaxed)
		}) {
			Ok(_) => {
				let mut sh = state.shared.lock().unwrap();
				sh.last_build_ok = true;
				sh.last_build = self.build_command.join(" ");
				sh.last_build_ms = started.elapsed().as_millis() as u64;
				drop(sh);
				self.record_build(&mut recorded, None, started);
			}
			Err(error) => {
				println!("daemon: build failed, the old build keeps serving: {error}");
				let mut sh = state.shared.lock().unwrap();
				sh.building = false;
				sh.reload_completed = ticket;
				sh.last_build_ok = false;
				sh.last_build = error.chars().take(1500).collect();
				sh.last_build_ms = started.elapsed().as_millis() as u64;
				drop(sh);
				self.record_build(&mut recorded, Some(error), started);
				return old;
			}
		}
		let alive = old
			.as_mut()
			.is_some_and(|c| matches!(c.0.try_wait(), Ok(None)));
		let changed = executable.is_none() || executable.as_deref().and_then(fingerprint) != before;
		if alive && !forced && !changed {
			println!("daemon: build ok, executable unchanged; modules reload in the running child");
			let mut sh = state.shared.lock().unwrap();
			sh.building = false;
			sh.reload_completed = ticket;
			return old;
		}
		// Stage before touching the old process: copy failures cannot lose state.
		let prepared = self.prepare_child().and_then(|prepared| {
			if alive {
				let saved = self.persist()?;
				state.shared.lock().unwrap().restore = saved;
			}
			Ok(prepared)
		});
		let (mut command, staged) = match prepared {
			Ok(prepared) => prepared,
			Err(error) => {
				let _ = wire::ask(&crate::app_socket(), json!({"op": "resume"}));
				eprintln!("daemon: restart rejected, keeping old process: {error}");
				let mut sh = state.shared.lock().unwrap();
				sh.building = false;
				sh.reload_completed = ticket;
				sh.last_build_ok = false;
				sh.last_build = error;
				return old;
			}
		};
		drop(old);
		if util::cancelled() {
			return None;
		}
		let child = self.start_prepared(&mut command, staged, state);
		{
			let mut sh = state.shared.lock().unwrap();
			sh.swaps += 1;
			sh.last_swap_ms = now_ms();
		}
		if let Some(pid) = child.as_ref().map(|c| c.0.id()) {
			println!("daemon: swapped to pid {pid}");
		} else {
			println!(
				"daemon: cannot spawn {} — fix the child command",
				self.child_command.join(" ")
			);
		}
		self.note_child(state, child.as_ref());
		{
			let mut sh = state.shared.lock().unwrap();
			sh.building = false;
			sh.reload_completed = ticket;
		}
		child
	}

	/// One `compile_finished` / `compile_failed` event for this build — unless
	/// a configured `rustc-events` compiler wrapper already recorded the
	/// build's crates while it ran (then this would duplicate them).
	fn record_build(&self, recorded: &mut events::Tail, error: Option<String>, started: Instant) {
		let kind = if error.is_some() {
			"compile_failed"
		} else {
			"compile_finished"
		};
		if recorded
			.read_new()
			.unwrap_or_default()
			.iter()
			.any(|event| event["kind"] == kind && event["source"] == "rustc-wrapper")
		{
			return;
		}
		let diagnostics = error.unwrap_or_default();
		let crate_name = diagnostics
			.split("could not compile `")
			.nth(1)
			.and_then(|rest| rest.split('`').next())
			.unwrap_or("workspace");
		let mut data = Map::new();
		data.insert("crate".into(), json!(crate_name));
		data.insert("build".into(), json!(self.build_command.join(" ")));
		data.insert(
			"exit_code".into(),
			json!(if diagnostics.is_empty() { 0 } else { 1 }),
		);
		data.insert(
			"duration_ms".into(),
			json!(started.elapsed().as_millis() as u64),
		);
		data.insert("diagnostics".into(), json!(diagnostics));
		data.insert(
			"cwd".into(),
			json!(std::env::current_dir()
				.map(|p| p.to_string_lossy().into_owned())
				.unwrap_or_default()),
		);
		let kind = if data["exit_code"] == 0 {
			"compile_finished"
		} else {
			"compile_failed"
		};
		match events::publish(kind, data) {
			Ok(event) if kind == "compile_failed" => events::notify(&event),
			Ok(_) => (),
			Err(error) => println!("daemon: cannot record the build event: {error}"),
		}
	}

	/// The child's executable when it is a file in the project (a build
	/// output); `None` for a command such as `sh`.
	fn executable(&self) -> Option<PathBuf> {
		let head = PathBuf::from(self.child_command.first()?);
		(head.components().count() > 1 && head.is_file()).then_some(head)
	}

	/// Ask the running child to save its state for the next process; a child
	/// without `reload::Host` (or none listening) just answers no.
	fn persist(&self) -> Result<Option<PathBuf>, String> {
		let socket = crate::app_socket();
		if !std::path::Path::new(&socket).exists() {
			return Ok(None);
		}
		let reply = wire::ask_within(&socket, json!({"op": "persist"}), PERSIST_TIMEOUT)?;
		if reply["ok"] != true {
			return Err(format!("state save rejected: {reply}"));
		}
		if reply["supported"] == false {
			return Ok(None);
		}
		let path = reply["path"]
			.as_str()
			.map(PathBuf::from)
			.filter(|p| p.is_file())
			.ok_or("saved state file is missing")?;
		println!("daemon: child persisted its state");
		Ok(Some(path))
	}

	/// Spawn the child from a per-generation copy of its executable (the
	/// original stays writable for the next build), removing older copies.
	fn prepare_child(&self) -> Result<(Command, Option<crate::staging::Executable>), String> {
		let (head, tail) = self
			.child_command
			.split_first()
			.ok_or("empty child command")?;
		let mut command = Command::new(head);
		let mut staged = None;
		if let Some(executable) = self.executable() {
			let original = std::path::absolute(&executable).map_err(|e| e.to_string())?;
			let copy = crate::staging::Executable::copy(&original, &crate::dir().join("run"))
				.map_err(|e| e.to_string())?;
			command = Command::new(copy.path());
			command.env("DAEMON_ARTIFACT_DIR", original.parent().unwrap());
			staged = Some(copy);
		}
		command
			.args(tail)
			.env("DAEMON_SUPERVISED", "1")
			.env_remove("DAEMON_RESTORE_SNAPSHOT");
		Ok((command, staged))
	}

	fn start_prepared(
		&self,
		command: &mut Command,
		staged: Option<crate::staging::Executable>,
		state: &State,
	) -> Option<ProcessTree> {
		if let Some(path) = &state.shared.lock().unwrap().restore {
			command.env("DAEMON_RESTORE_SNAPSHOT", path);
		}
		match ProcessTree::spawn(command) {
			Ok(mut child) => {
				child.keep_executable(staged);
				state.shared.lock().unwrap().executable_stamp =
					self.executable().as_deref().and_then(fingerprint);
				Some(child)
			}
			Err(error) => {
				eprintln!("daemon: spawn failed: {error}");
				None
			}
		}
	}

	fn spawn_child(&self, state: &State) -> Option<ProcessTree> {
		let (mut command, staged) = self
			.prepare_child()
			.map_err(|e| eprintln!("daemon: {e}"))
			.ok()?;
		self.start_prepared(&mut command, staged, state)
	}

	fn note_child(&self, state: &State, child: Option<&ProcessTree>) {
		let mut sh = state.shared.lock().unwrap();
		sh.child_pid = child.map(|c| c.0.id());
		sh.child_started_ms = now_ms();
	}

	fn clone_shallow(&self) -> Daemon {
		Daemon {
			build_command: self.build_command.clone(),
			child_command: self.child_command.clone(),
			watch: self.watch.clone(),
			poll: self.poll,
			debounce: self.debounce,
			build_timeout: self.build_timeout,
			control_socket: self.control_socket.clone(),
			idle: self.idle,
		}
	}
}
