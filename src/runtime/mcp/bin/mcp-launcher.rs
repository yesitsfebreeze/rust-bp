//! `mcp-launcher` — the stable stdio transport for the stateless
//! `mcp` worker. Each request starts the current worker binary, so a
//! successful cargo rebuild is live on the next request without disconnecting
//! the MCP client or interrupting an in-flight tool. stdout is JSON-RPC only.
//!
//! `.mcp.json` starts `mcp.sh`, which builds both binaries once and execs
//! the launcher with `--no-build`, so client signals reach it directly.
//! When launched directly, the first request builds the worker unless `--no-build`;
//! `--worker <path>` points at another worker binary. It also tails the
//! daemon dir's `events.jsonl` and streams compiler failures to the client
//! as MCP log notifications.

use daemon::events;
use daemon::util::{self, ProcessTree};
use serde_json::{json, Value};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

const WORKER_BUILD: [&str; 8] = [
	"cargo",
	"build",
	"--release",
	"--locked",
	"-p",
	"mcp",
	"--bin",
	"mcp",
];
const BUILD_TIMEOUT: Duration = Duration::from_secs(110);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(1200);
const TERMINATE_GRACE: Duration = Duration::from_secs(15);

type Version = (Option<std::time::SystemTime>, u64);

struct Launcher {
	worker: PathBuf,
	build_pending: bool,
	last_version: Option<Version>,
	initialized: Arc<AtomicBool>,
	/// The worker serving the current request; the signal watchdog terminates it.
	active: Arc<Mutex<Option<ProcessTree>>>,
}

fn emit(value: &Value) {
	let mut stdout = std::io::stdout().lock();
	let _ = writeln!(stdout, "{value}");
	let _ = stdout.flush();
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
	json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn parse_args() -> Launcher {
	let mut worker = None;
	let mut build_pending = true;
	let mut args = std::env::args().skip(1);
	while let Some(arg) = args.next() {
		match arg.as_str() {
			"--worker" => worker = args.next().map(PathBuf::from),
			"--no-build" => build_pending = false,
			other => {
				eprintln!("mcp-launcher: unknown argument {other}");
				std::process::exit(2);
			}
		}
	}
	let worker = worker.unwrap_or_else(|| {
		PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap_or_else(|| "target".into()))
			.join("release")
			.join(format!("mcp{}", std::env::consts::EXE_SUFFIX))
	});
	Launcher {
		worker: std::path::absolute(&worker).unwrap_or(worker),
		build_pending,
		last_version: None,
		initialized: Arc::new(AtomicBool::new(false)),
		active: Arc::new(Mutex::new(None)),
	}
}

/// SIGTERM/SIGINT: give the in-flight worker time to clean up, then exit.
fn watch_signals(active: Arc<Mutex<Option<ProcessTree>>>) {
	let _ = util::cancelled(); // install the handlers before any child exists
	std::thread::spawn(move || loop {
		if util::cancelled() {
			if let Some(mut tree) = active.lock().unwrap().take() {
				tree.terminate(TERMINATE_GRACE);
				drop(tree);
			}
			std::process::exit(128 + util::cancel_signal());
		}
		std::thread::sleep(Duration::from_millis(50));
	});
}

/// Stream compiler failures appended to the event log as MCP log messages.
fn feed_events(initialized: Arc<AtomicBool>) {
	std::thread::spawn(move || {
		let mut tail = events::Tail::new();
		loop {
			std::thread::sleep(Duration::from_millis(250));
			if !initialized.load(Ordering::Relaxed) {
				continue;
			}
			match tail.read_new() {
				Ok(fresh) => {
					for event in fresh.iter().filter(|e| e["kind"] == "compile_failed") {
						emit(&json!({"jsonrpc": "2.0", "method": "notifications/message",
                            "params": {"level": "error", "logger": "daemon.compiler", "data": event}}));
						events::notify(event);
					}
				}
				Err(error) => eprintln!("event feed: {error}"),
			}
		}
	});
}

fn build_worker() -> Result<(), String> {
	let command: Vec<String> = WORKER_BUILD.iter().map(|s| s.to_string()).collect();
	match util::run_with_timeout(&command, BUILD_TIMEOUT, 6000) {
		Ok(output) => {
			if !output.is_empty() {
				eprintln!("{output}");
			}
			Ok(())
		}
		Err(error) if error.starts_with("timed out") => Err(format!(
			"mcp worker build timed out; run {} in this workspace, then retry",
			WORKER_BUILD.join(" ")
		)),
		Err(error) => Err(format!("mcp worker build failed: {error}")),
	}
}

impl Launcher {
	fn serve(&mut self, request: &Value, line: &str) -> Result<Value, String> {
		if self.build_pending {
			// Read the request first so a build failure can answer its id;
			// closing stdout here would make required-server errors opaque.
			build_worker()?;
			self.build_pending = false;
		}
		let meta =
			std::fs::metadata(&self.worker).map_err(|e| format!("{}: {e}", self.worker.display()))?;
		// Portable rebuild check: a new worker build changes its mtime or length.
		let version = (meta.modified().ok(), meta.len());
		if self.initialized.load(Ordering::Relaxed) && self.last_version != Some(version) {
			emit(&json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}));
		}
		let staged = daemon::staging::Executable::copy(&self.worker, &daemon::dir().join("workers"))
			.map_err(|e| format!("stage worker: {e}"))?;
		let mut tree = ProcessTree::spawn(
			Command::new(staged.path())
				.stdin(Stdio::piped())
				.stdout(Stdio::piped()),
		)
		.map_err(|e| format!("cannot start {}: {e}", self.worker.display()))?;
		tree.keep_executable(Some(staged));
		let mut stdin = tree.0.stdin.take().ok_or("worker stdin")?;
		let mut stdout = tree.0.stdout.take().ok_or("worker stdout")?;
		let (sender, receiver) = mpsc::channel();
		std::thread::spawn(move || {
			let mut output = String::new();
			let _ = stdout.read_to_string(&mut output);
			let _ = sender.send(output);
		});
		stdin
			.write_all(format!("{line}\n").as_bytes())
			.map_err(|e| format!("worker stdin: {e}"))?;
		drop(stdin);
		*self.active.lock().unwrap() = Some(tree);
		let output = match receiver.recv_timeout(REQUEST_TIMEOUT) {
			Ok(output) => output,
			Err(_) => {
				drop(self.active.lock().unwrap().take()); // SIGKILL the group
				return Err(format!(
					"MCP worker timed out after {} seconds",
					REQUEST_TIMEOUT.as_secs()
				));
			}
		};
		let status = {
			let mut active = self.active.lock().unwrap();
			let tree = active.as_mut().ok_or("cancelled")?;
			tree.0.wait().map_err(|e| format!("worker wait: {e}"))?
		};
		drop(self.active.lock().unwrap().take());
		if !status.success() {
			return Err(format!("MCP worker exited with status {status}"));
		}
		let mut response: Value = serde_json::from_str(output.trim())
			.map_err(|e| format!("MCP worker returned no JSON: {e}"))?;
		if response.get("id") != request.get("id") {
			return Err("MCP worker returned a mismatched response id".into());
		}
		if request["method"] == "initialize" && response.get("result").is_some() {
			response["result"]["capabilities"]["tools"]["listChanged"] = json!(true);
			response["result"]["capabilities"]["logging"] = json!({});
			self.initialized.store(true, Ordering::Relaxed);
		}
		self.last_version = Some(version);
		Ok(response)
	}
}

fn main() {
	let mut launcher = parse_args();
	watch_signals(launcher.active.clone());
	feed_events(launcher.initialized.clone());
	let stdin = std::io::stdin();
	for line in stdin.lock().lines() {
		let Ok(line) = line else { break };
		let request: Value = match serde_json::from_str(&line) {
			Ok(value) => value,
			Err(_) => {
				emit(&rpc_error(Value::Null, -32700, "Invalid JSON"));
				continue;
			}
		};
		if !request.is_object() {
			emit(&rpc_error(Value::Null, -32600, "Expected a request object"));
			continue;
		}
		// A notification has no id; the worker keeps no session state to feed it.
		let Some(id) = request.get("id").cloned() else {
			continue;
		};
		match launcher.serve(&request, &line) {
			Ok(response) => emit(&response),
			Err(message) => emit(&rpc_error(id, -32603, &message)),
		}
	}
}
