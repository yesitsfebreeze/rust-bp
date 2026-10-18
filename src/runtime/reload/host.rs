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

#[cfg(test)]
mod tests {
	use crate as reload;
	use reload::{snapshot, Host};
	use std::path::{Path, PathBuf};
	use std::process::Command;
	use std::time::{Duration, Instant};

	struct Temp(PathBuf);
	impl Temp {
		fn new(label: &str) -> Self {
			let path = std::env::temp_dir().join(format!("reload-{}-{label}", std::process::id()));
			std::fs::create_dir_all(&path).unwrap();
			Self(path)
		}
	}
	impl Drop for Temp {
		fn drop(&mut self) {
			let _ = std::fs::remove_dir_all(&self.0);
		}
	}

	const MODULE: &str = r###"//! Compiled as a real shared library by reload's transaction tests.
#[path = "$ABI"]
pub mod abi;

struct Counter(u64);
impl abi::Module for Counter {
	fn create(_: &[u8]) -> Result<Self, String> {
		if cfg!(create_fail) { return Err("intentional create rejection".into()); }
		Ok(Self(0))
	}
	fn call(&mut self, method: &str, _: &[u8]) -> Vec<u8> {
		if method == "tick" { self.0 += if cfg!(v2) { 10 } else { 1 }; }
		self.0.to_le_bytes().to_vec()
	}
	fn save(&self) -> Result<Vec<u8>, String> {
		assert!(!cfg!(save_fail), "intentional save panic");
		Ok(self.0.to_le_bytes().to_vec())
	}
	fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
		if cfg!(restore_fail) { return Err("intentional restore rejection".into()); }
		self.0 = u64::from_le_bytes(bytes.try_into().map_err(|_| "bad state")?);
		Ok(())
	}
}

#[cfg(not(bad_abi))]
export_module!(Counter);

#[cfg(bad_abi)]
#[no_mangle]
pub extern "C" fn module_api() -> *const abi::ModuleApi {
	// The loader must read only the version before rejecting this pointer.
	static VERSION: u32 = 999;
	&VERSION as *const u32 as *const abi::ModuleApi
}"###;

	fn compile(output: &Path, cfg: &str) {
		let abi = Path::new(env!("CARGO_MANIFEST_DIR")).join("abi.rs");
		let source = output.with_extension("rs");
		std::fs::write(
			&source,
			MODULE.replace("$ABI", &abi.to_string_lossy().replace('\\', "/")),
		)
		.unwrap();
		let result = Command::new("rustc")
			.arg(&source)
			.args([
				"--crate-type",
				"cdylib",
				"--edition",
				"2021",
				"--cfg",
				cfg,
				"-o",
			])
			.arg(output)
			.output()
			.unwrap();
		assert!(
			result.status.success(),
			"{}",
			String::from_utf8_lossy(&result.stderr)
		);
	}

	fn count(host: &mut Host, method: &str) -> u64 {
		u64::from_le_bytes(host.call("counter", method, &[]).try_into().unwrap())
	}

	fn swap(host: &mut Host) -> usize {
		assert_eq!(host.poll(), 0, "first poll waits for stable build");
		host.poll()
	}

	#[test]
	fn module_reload_and_process_handoff_preserve_state_and_reject_bad_generations() {
		let temp = Temp::new("transactions");
		std::env::set_var("DAEMON_DIR", &temp.0);
		std::env::remove_var("DAEMON_RESTORE_SNAPSHOT");
		let library = temp
			.0
			.join(format!("counter{}", std::env::consts::DLL_SUFFIX));
		compile(&library, "v1");
		let mut host = Host::new().unwrap();
		assert!(host.load("../escape", &library, &[]).is_err());
		host.load("counter", &library, &[]).unwrap();
		assert!(host.load("counter", &library, &[]).is_err());
		assert_eq!(count(&mut host, "tick"), 1);
		assert_eq!(count(&mut host, "tick"), 2);
		compile(&library, "v2");
		assert_eq!(swap(&mut host), 1);
		assert_eq!(host.generation("counter"), 2);
		assert_eq!(count(&mut host, "tick"), 12);
		for variant in ["restore_fail", "create_fail", "bad_abi"] {
			compile(&library, variant);
			assert_eq!(swap(&mut host), 0, "{variant} must keep old module");
			assert_eq!(host.generation("counter"), 2);
			assert_eq!(count(&mut host, "get"), 12);
			assert_eq!(
				std::fs::read_dir(temp.0.join("modules")).unwrap().count(),
				1,
				"failed libraries must be unloaded before deletion"
			);
		}
		std::fs::write(&library, b"incomplete build").unwrap();
		assert_eq!(swap(&mut host), 0);
		assert_eq!(count(&mut host, "get"), 12);
		compile(&library, "save_fail");
		assert_eq!(swap(&mut host), 1);
		compile(&library, "v1");
		assert_eq!(swap(&mut host), 0, "save panic must reject swap");
		assert_eq!(count(&mut host, "get"), 12);
		drop(host);
		assert_eq!(
			std::fs::read_dir(temp.0.join("modules")).unwrap().count(),
			0
		);

		// Real channel handshake, persisted module and app state, then a new process.
		let mut host = Host::new().unwrap();
		host.load("counter", &library, &[]).unwrap();
		assert_eq!(count(&mut host, "tick"), 1);
		let _channels = daemon::channels::serve(&daemon::app_socket()).unwrap();
		let socket = daemon::app_socket();
		let request = std::thread::spawn(move || {
			daemon::wire::ask_within(
				&socket,
				serde_json::json!({"op":"persist"}),
				Duration::from_secs(7),
			)
			.unwrap()
		});
		let deadline = Instant::now() + Duration::from_secs(5);
		while !host.persist_requested() {
			assert!(Instant::now() < deadline);
			std::thread::sleep(Duration::from_millis(5));
		}
		let path = host
			.persist(&[("score", 42u64.to_le_bytes().to_vec())])
			.unwrap();
		assert!(host.quiesced());
		assert!(host.call("counter", "tick", &[]).is_empty());
		assert_eq!(request.join().unwrap()["ok"], true);
		let status = Command::new(std::env::current_exe().unwrap())
			.args(["--ignored", "--exact", "host::tests::restore_fixture"])
			.env("DAEMON_RESTORE_SNAPSHOT", path)
			.env("TEST_LIBRARY", &library)
			.status()
			.unwrap();
		assert!(status.success());
		assert!(snapshot::path().exists(), "retain recovery snapshot");
		assert!(
			Host::new().unwrap().restored("score").is_none(),
			"fresh sessions ignore old snapshot"
		);
	}

	#[test]
	#[ignore = "subprocess fixture"]
	fn restore_fixture() {
		let Ok(library) = std::env::var("TEST_LIBRARY") else {
			return;
		};
		let mut host = Host::new().unwrap();
		host.load("counter", library, &[]).unwrap();
		assert_eq!(host.restored("score"), Some(42u64.to_le_bytes().to_vec()));
		assert_eq!(count(&mut host, "tick"), 2);
	}
}
