//! App-owned modules and the frame-boundary persistence handshake.
use crate::{library::Loaded, snapshot};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime};

const PERSIST_WAIT: Duration = Duration::from_secs(5);
const MODULE_PREFIX: &str = "module:";
const APP_PREFIX: &str = "app:";

struct Module {
	source: PathBuf,
	config: Vec<u8>,
	stamp: Option<SystemTime>,
	pending: Option<SystemTime>,
	generation: u64,
	loaded: Loaded,
}

#[derive(Default)]
struct Request {
	pending: bool,
	result: Option<Result<PathBuf, String>>,
}

#[derive(Default)]
struct Gate {
	state: Mutex<Request>,
	done: Condvar,
}

/// One host per application, owned by its simulation thread. Only trusted
/// native libraries built for this ABI may be loaded.
pub struct Host {
	modules: BTreeMap<String, Module>,
	live: PathBuf,
	restored: BTreeMap<String, Vec<u8>>,
	gate: Arc<Gate>,
}

impl Host {
	/// Read the snapshot explicitly passed by the supervisor, if any, and
	/// install the persist hook before opening the app's channel socket.
	pub fn new() -> Result<Host, String> {
		let restored = snapshot::read()?;
		let gate = Arc::new(Gate::default());
		let answer = gate.clone();
		daemon::channels::on_persist(move |op| {
			let mut state = answer.state.lock().unwrap();
			if op == "resume" {
				*state = Request::default();
				return serde_json::json!({"op": "resume", "ok": true});
			}
			*state = Request {
				pending: true,
				result: None,
			};
			let (mut state, _) = answer
				.done
				.wait_timeout_while(state, PERSIST_WAIT, |s| s.result.is_none())
				.unwrap();
			match state.result.as_ref() {
				Some(Ok(path)) => serde_json::json!({"op": "persist", "ok": true, "path": path}),
				Some(Err(error)) => serde_json::json!({"op": "persist", "ok": false, "error": error}),
				None => {
					state.pending = false;
					serde_json::json!({"op": "persist", "ok": false, "error": "app did not reach a persistence boundary"})
				}
			}
		});
		Ok(Host {
			modules: BTreeMap::new(),
			live: daemon::dir().join("modules"),
			restored,
			gate,
		})
	}

	/// Load a named module, restoring its saved state before exposing it.
	/// Duplicate names and incompatible state are errors, never silent resets.
	pub fn load(
		&mut self,
		name: &str,
		library: impl AsRef<Path>,
		config: &[u8],
	) -> Result<(), String> {
		if name.is_empty()
			|| !name
				.bytes()
				.all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
		{
			return Err("module name must contain only letters, numbers, underscores or hyphens".into());
		}
		if self.modules.contains_key(name) {
			return Err(format!("module {name} is already loaded"));
		}
		let source = library.as_ref().to_path_buf();
		std::fs::create_dir_all(&self.live).map_err(|e| format!("{}: {e}", self.live.display()))?;
		let mut loaded = Loaded::open(&source, self.copy_path(name, 1), config)?;
		let key = format!("{MODULE_PREFIX}{name}");
		if let Some(state) = self.restored.get(&key) {
			loaded.restore(state).map_err(|e| format!("{name}: {e}"))?;
		}
		self.restored.remove(&key);
		let stamp = modified(&source);
		self.modules.insert(
			name.into(),
			Module {
				source,
				config: config.to_vec(),
				stamp,
				pending: None,
				generation: 1,
				loaded,
			},
		);
		Ok(())
	}

	/// Poll once per frame. A stable changed timestamp triggers a load/save/
	/// restore transaction; any failure keeps the previous code and state.
	pub fn poll(&mut self) -> usize {
		if self.quiesced() {
			return 0;
		}
		let mut swapped = 0;
		let names: Vec<String> = self.modules.keys().cloned().collect();
		for name in names {
			let module = &self.modules[&name];
			let now = modified(&module.source);
			if now.is_none() || now == module.stamp {
				self.modules.get_mut(&name).unwrap().pending = None;
				continue;
			}
			if module.pending != now {
				self.modules.get_mut(&name).unwrap().pending = now;
				continue;
			}
			let next = self.copy_path(&name, module.generation + 1);
			let module = self.modules.get_mut(&name).unwrap();
			module.pending = None;
			module.stamp = now;
			let replacement = (|| {
				let mut fresh = Loaded::open(&module.source, next, &module.config)?;
				fresh.restore(&module.loaded.save()?)?;
				Ok::<_, String>(fresh)
			})();
			match replacement {
				Ok(fresh) => {
					module.loaded = fresh;
					module.generation += 1;
					swapped += 1;
				}
				Err(error) => eprintln!(
					"reload: {name} keeps generation {}: {error}",
					module.generation
				),
			}
		}
		swapped
	}

	/// Call at a simulation boundary. After persistence, calls are disabled
	/// until the supervisor replaces this process.
	pub fn call(&mut self, name: &str, method: &str, args: &[u8]) -> Vec<u8> {
		if self.quiesced() {
			return Vec::new();
		}
		self
			.modules
			.get_mut(name)
			.map(|m| m.loaded.call(method, args))
			.unwrap_or_default()
	}

	pub fn generation(&self, name: &str) -> u64 {
		self.modules.get(name).map(|m| m.generation).unwrap_or(0)
	}

	/// Take one application state part from the previous process.
	pub fn restored(&mut self, name: &str) -> Option<Vec<u8>> {
		self.restored.remove(&format!("{APP_PREFIX}{name}"))
	}

	pub fn persist_requested(&self) -> bool {
		let state = self.gate.state.lock().unwrap();
		state.pending && state.result.is_none()
	}

	/// The app must stop advancing its own simulation while this is true.
	pub fn quiesced(&self) -> bool {
		matches!(self.gate.state.lock().unwrap().result, Some(Ok(_)))
	}

	/// Save application parts and all module states together, then acknowledge
	/// the supervisor. Failure leaves the old process running and editable.
	pub fn persist(&mut self, parts: &[(&str, Vec<u8>)]) -> Result<PathBuf, String> {
		let mut state = self.gate.state.lock().unwrap();
		if !state.pending || state.result.is_some() {
			return Err("no pending persistence request".into());
		}
		let written = self.save_parts(parts);
		state.pending = false;
		state.result = Some(written.clone());
		self.gate.done.notify_all();
		written
	}

	fn save_parts(&self, parts: &[(&str, Vec<u8>)]) -> Result<PathBuf, String> {
		// Keep not-yet-consumed parts, e.g. a module that loads lazily.
		let mut entries = self.restored.clone();
		let mut app = BTreeMap::new();
		for (name, data) in parts {
			if app
				.insert(format!("{APP_PREFIX}{name}"), data.clone())
				.is_some()
			{
				return Err(format!("duplicate app state part {name}"));
			}
		}
		entries.extend(app);
		for (name, module) in &self.modules {
			entries.insert(format!("{MODULE_PREFIX}{name}"), module.loaded.save()?);
		}
		snapshot::write(&entries)
	}

	fn copy_path(&self, name: &str, generation: u64) -> PathBuf {
		self.live.join(format!(
			"{name}-{}-{generation}{}",
			std::process::id(),
			std::env::consts::DLL_SUFFIX
		))
	}
}

/// Cargo's output directory, supplied by the daemon when the executable is
/// staged elsewhere; otherwise next to the current executable.
pub fn library_path(name: &str) -> PathBuf {
	let dir = std::env::var_os("DAEMON_ARTIFACT_DIR")
		.map(PathBuf::from)
		.unwrap_or_else(|| {
			let exe = std::env::current_exe().unwrap_or_default();
			let mut dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();
			if dir
				.file_name()
				.is_some_and(|n| n == "deps" || n == "examples")
			{
				dir.pop();
			}
			dir
		});
	dir.join(format!(
		"{}{}{}",
		std::env::consts::DLL_PREFIX,
		name.replace('-', "_"),
		std::env::consts::DLL_SUFFIX
	))
}

fn modified(path: &Path) -> Option<SystemTime> {
	std::fs::metadata(path).and_then(|m| m.modified()).ok()
}
