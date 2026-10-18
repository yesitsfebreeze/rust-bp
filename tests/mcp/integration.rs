//! The MCP integration suite, the audit regressions and the Codex receipt
//! check. All three are `#[ignore]`: they build release binaries, start the
//! daemon and need valgrind or Docker. Every mutation happens in an
//! isolated copy of the project. Run one from the project root:
//!
//! ```sh
//! cargo test -p mcp --test integration -- --ignored --nocapture check
//! cargo test -p mcp --test integration -- --ignored --nocapture regressions
//! cargo test -p mcp --test integration -- --ignored --nocapture verify_receipt
//! ```
//!
//! `check` reads: `CHECK_JOURNEY=lifecycle` (daemon only, no
//! profilers), `CHECK_LIVE_JEV=1` (the external Jev service, needs
//! `TYPESAFE_API_KEY`; default is a local deterministic fixture) and
//! `CHECK_NOTIFY_THREAD=<codex session uuid>` (queue one clearly
//! marked intentional compile failure; `verify_receipt` checks receipt).

use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod stateful;

#[test]
#[ignore = "builds isolated release binaries and exercises real stateful reloads"]
fn stateful_reload() {
	stateful::run();
}

fn root() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("../../..")
		.canonicalize()
		.expect("project root")
}

fn pass(condition: bool, message: &str) {
	assert!(condition, "{message}");
	println!("PASS: {message}");
}

fn wait_for<T>(timeout: Duration, mut probe: impl FnMut() -> Option<T>) -> T {
	let deadline = Instant::now() + timeout;
	while Instant::now() < deadline {
		if let Some(value) = probe() {
			return value;
		}
		std::thread::sleep(Duration::from_millis(250));
	}
	panic!("timed out waiting for expected state");
}

fn process_running(pid: u64) -> bool {
	#[cfg(windows)]
	{
		daemon::platform::alive(pid as u32)
	}
	#[cfg(unix)]
	{
		let output = Command::new("ps")
			.args(["-o", "stat=", "-p", &pid.to_string()])
			.output()
			.unwrap();
		let state = String::from_utf8_lossy(&output.stdout).trim().to_string();
		!state.is_empty() && !state.starts_with('Z')
	}
}

fn on_path(program: &str) -> bool {
	std::env::var_os("PATH")
		.map(|path| {
			std::env::split_paths(&path).any(|dir| {
				dir
					.join(format!("{program}{}", std::env::consts::EXE_SUFFIX))
					.is_file()
			})
		})
		.unwrap_or(false)
		&& Command::new(program)
			.arg("--version")
			.output()
			.is_ok_and(|out| out.status.success())
}

fn read_json(path: &Path) -> Value {
	serde_json::from_str(
		&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
	)
	.unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn copy_dir(from: &Path, to: &Path) {
	std::fs::create_dir_all(to).unwrap();
	for entry in std::fs::read_dir(from).unwrap() {
		let entry = entry.unwrap();
		let target = to.join(entry.file_name());
		if entry.file_type().unwrap().is_dir() {
			copy_dir(&entry.path(), &target);
		} else {
			std::fs::copy(entry.path(), target).unwrap();
		}
	}
}

/// Environment for every process of the isolated copy: no inherited wrapper
/// or target dir, Jev pointed at the fixture unless the live service is used.
#[derive(Clone)]
struct Env {
	set: Vec<(String, String)>,
	remove: Vec<&'static str>,
}

impl Env {
	fn apply(&self, command: &mut Command) {
		for key in &self.remove {
			command.env_remove(key);
		}
		command.envs(self.set.iter().map(|(k, v)| (k.as_str(), v.as_str())));
	}
	fn run(&self, cwd: &Path, args: &[&str]) {
		let mut command = Command::new(args[0]);
		command.args(&args[1..]).current_dir(cwd);
		self.apply(&mut command);
		let status = command.status().unwrap();
		assert!(status.success(), "{} failed: {status}", args.join(" "));
	}
}

type Owned = daemon::util::ProcessTree;

#[cfg(unix)]
extern "C" {
	#[link_name = "kill"]
	fn libc_kill(pid: i32, signal: i32) -> i32;
}

#[cfg(unix)]
fn signal(pid: u32, signal: i32) {
	unsafe {
		libc_kill(pid as i32, signal);
	}
}
#[cfg(unix)]
const SIGTERM: i32 = 15;

/// The MCP client side: one launcher process on a pipe, requests by id.
struct Mcp {
	process: Owned,
	stdin: Option<ChildStdin>,
	incoming: Receiver<Value>,
	notifications: Vec<Value>,
	calls: BTreeSet<String>,
	seq: u64,
	app_pids: BTreeSet<u64>,
}

impl Mcp {
	fn start(launcher: &Path, args: &[&str], cwd: &Path, env: &Env) -> Mcp {
		let mut command = Command::new(launcher);
		if launcher.extension().is_some_and(|e| e == "ps1") {
			command = Command::new("powershell.exe");
			command
				.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
				.arg(launcher);
		}
		command
			.args(args)
			.current_dir(cwd)
			.stdin(Stdio::piped())
			.stdout(Stdio::piped());
		env.apply(&mut command);
		let mut child = Owned::spawn(&mut command).expect("start launcher");
		let stdin = child.0.stdin.take();
		let stdout = child.0.stdout.take().unwrap();
		let (sender, incoming) = mpsc::channel();
		std::thread::spawn(move || {
			for line in BufReader::new(stdout).lines() {
				let Ok(line) = line else { break };
				let value = serde_json::from_str(&line)
					.unwrap_or_else(|_| json!({"error": format!("non-JSON stdout: {line}")}));
				if sender.send(value).is_err() {
					break;
				}
			}
		});
		let mut mcp = Mcp {
			process: child,
			stdin,
			incoming,
			notifications: vec![],
			calls: BTreeSet::new(),
			seq: 0,
			app_pids: BTreeSet::new(),
		};
		let init = mcp.rpc(
			"initialize",
			json!({"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "mcp-check", "version": "1"}}),
		);
		pass(
			init["capabilities"]["tools"]["listChanged"] == true,
			"launcher advertises tool-change notifications",
		);
		assert!(init["instructions"]
			.as_str()
			.is_some_and(|text| text.contains("already loaded") && text.contains("audit_required")));
		mcp.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
		mcp
	}

	fn send(&mut self, line: &str) {
		let stdin = self.stdin.as_mut().expect("launcher stdin open");
		stdin.write_all(format!("{line}\n").as_bytes()).unwrap();
		stdin.flush().unwrap();
	}

	/// The `result` of one request; a JSON-RPC error panics.
	fn rpc(&mut self, method: &str, params: Value) -> Value {
		self.seq += 1;
		let request = json!({"jsonrpc": "2.0", "id": self.seq, "method": method, "params": params});
		self.send(&request.to_string());
		let deadline = Instant::now() + Duration::from_secs(900);
		loop {
			let remaining = deadline.saturating_duration_since(Instant::now());
			let response = match self
				.incoming
				.recv_timeout(remaining.max(Duration::from_millis(100)))
			{
				Ok(response) => response,
				Err(RecvTimeoutError::Timeout) => panic!("{method}: no response within 900 s"),
				Err(RecvTimeoutError::Disconnected) => panic!("{method}: launcher closed stdout"),
			};
			if response.get("id").is_none() && response.get("method").is_some() {
				self.notifications.push(response);
				continue;
			}
			if let Some(error) = response.get("error") {
				panic!("{method}: {error}");
			}
			assert_eq!(response["id"], json!(self.seq), "{response}");
			return response["result"].clone();
		}
	}

	/// One tool's text; a tool-level error (`isError`) panics.
	fn call(&mut self, name: &str, args: Value) -> String {
		self.calls.insert(name.to_string());
		let reply = self.rpc("tools/call", json!({"name": name, "arguments": args}));
		assert!(reply["isError"] != true, "{name}: {reply}");
		reply["content"][0]["text"].as_str().unwrap().to_string()
	}

	fn call_json(&mut self, name: &str, args: Value) -> Value {
		let text = self.call(name, args);
		serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name} answered no JSON ({e}): {text}"))
	}

	fn daemon_status(&mut self) -> Value {
		let text = self.call("status", json!({}));
		// The reply is pretty JSON followed by a channel-socket line.
		let mut stream = serde_json::Deserializer::from_str(&text).into_iter::<Value>();
		let state = stream.next().and_then(Result::ok).unwrap_or(json!({}));
		if let Some(pid) = state["child_pid"].as_u64() {
			self.app_pids.insert(pid);
		}
		state
	}

	fn rejects(&mut self, before: &str, after: &str) {
		let response = self.rpc(
			"tools/call",
			json!({"name": "compare", "arguments": {"before": before, "after": after}}),
		);
		pass(
			response["isError"] == true && response.to_string().contains("command differs"),
			&format!("{before}: mismatched workload rejected"),
		);
	}

	/// Close stdin and wait for the launcher to exit on EOF.
	fn close(&mut self) -> Option<std::process::ExitStatus> {
		drop(self.stdin.take());
		let deadline = Instant::now() + Duration::from_secs(10);
		while Instant::now() < deadline {
			if let Ok(Some(status)) = self.process.0.try_wait() {
				return Some(status);
			}
			std::thread::sleep(Duration::from_millis(50));
		}
		None
	}
}

/// A deterministic Jev protocol fixture: every `score` 2.0, every `noul`
/// 0.75, every choice the first criterion. Records each request's state.
fn jev_fixture() -> (u16, Arc<Mutex<Vec<Value>>>) {
	let listener = TcpListener::bind("127.0.0.1:0").unwrap();
	let port = listener.local_addr().unwrap().port();
	let states: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(vec![]));
	let seen = states.clone();
	std::thread::spawn(move || {
		for stream in listener.incoming() {
			let Ok(stream) = stream else { continue };
			let seen = seen.clone();
			std::thread::spawn(move || {
				let mut reader = BufReader::new(stream);
				let mut length = 0usize;
				loop {
					let mut line = String::new();
					if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
						break;
					}
					if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
						length = value.trim().parse().unwrap_or(0);
					}
				}
				let mut body = vec![0u8; length];
				reader.read_exact(&mut body).unwrap();
				let request: Value = serde_json::from_slice(&body).unwrap();
				seen.lock().unwrap().push(request["state"].clone());
				let mut answers = serde_json::Map::new();
				for (key, question) in request["questions"].as_object().unwrap() {
					let answer = match question["type"].as_str() {
						Some("score") => json!({"score": 2.0}),
						Some("noul") => json!({"noul": 0.75}),
						_ => {
							json!({"choice": question["criteria"].as_object().unwrap().keys().next().unwrap()})
						}
					};
					answers.insert(key.clone(), answer);
				}
				let body = json!({"answers": answers}).to_string();
				let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
				let mut stream = reader.into_inner();
				let _ = stream.write_all(response.as_bytes());
				let _ = stream.flush();
			});
		}
	});
	(port, states)
}

/// Removes the isolated checkout, build artifacts and daemon dir on drop.
struct Enclosure(PathBuf);
impl Drop for Enclosure {
	fn drop(&mut self) {
		// Resolve before removing: the hash needs the canonical path.
		let daemon_dir = daemon::dir_for(&self.0.join("boilerplate"));
		let _ = std::fs::remove_dir_all(&self.0);
		let _ = std::fs::remove_dir_all(daemon_dir);
	}
}

fn copy_fixture_sources(root: &Path, directory: &Path) {
	// The complete build closure must sit inside the single Docker /repo mount.
	copy_dir(&root.join("src"), &directory.join("src"));
	copy_dir(&root.join("tests"), &directory.join("tests"));
	std::fs::copy(
		root.join("tests/fixtures/workload.rs"),
		directory.join("src/main.rs"),
	)
	.unwrap();
	for name in [
		"Cargo.toml",
		"Cargo.lock",
		"valgrind.Dockerfile",
		"mcp.sh",
		"mcp.ps1",
	] {
		std::fs::copy(root.join(name), directory.join(name)).unwrap();
	}
}

fn env_flag(name: &str) -> bool {
	std::env::var(name)
		.map(|v| !v.is_empty() && v != "0")
		.unwrap_or(false)
}

#[test]
#[ignore = "builds release binaries, starts the daemon, needs valgrind or Docker"]
fn check() {
	let root = root();
	let lifecycle_only = std::env::var("CHECK_JOURNEY").as_deref() == Ok("lifecycle");
	let live_jev = env_flag("CHECK_LIVE_JEV");
	let notify_thread = std::env::var("CHECK_NOTIFY_THREAD")
		.ok()
		.filter(|t| !t.is_empty());
	if live_jev
		&& std::env::var("TYPESAFE_API_KEY")
			.map(|k| k.is_empty())
			.unwrap_or(true)
	{
		panic!("CHECK_LIVE_JEV requires TYPESAFE_API_KEY");
	}
	let mut env = Env {
		set: vec![],
		remove: vec!["RUSTC_WRAPPER", "CARGO_TARGET_DIR", "DAEMON_DIR"],
	};
	let fixture = (!live_jev).then(jev_fixture);
	if let Some((port, _)) = &fixture {
		env
			.set
			.push(("JEV_URL".into(), format!("http://127.0.0.1:{port}")));
		env
			.set
			.push(("TYPESAFE_API_KEY".into(), "local-test-fixture".into()));
	}
	let enclosure = Enclosure(std::env::temp_dir().join(format!("mcp-check-{}", std::process::id())));
	let directory = enclosure.0.join("boilerplate");
	std::fs::create_dir_all(&directory).unwrap();
	// Keep generated manifests and artifacts inside this fixture. A shared
	// target cache can retain source paths/binaries from a previous fixture.
	copy_fixture_sources(&root, &directory);
	let target = directory.join("target");

	env.run(
		&directory,
		&["cargo", "build", "--release", "--locked", "--workspace"],
	);
	let daemon_dir = daemon::dir_for(&directory);
	std::fs::create_dir_all(&daemon_dir).unwrap();
	let log = std::fs::File::create(daemon_dir.join("daemon.log")).unwrap();
	let mut daemon_command =
		Command::new(target.join(format!("release/daemon{}", std::env::consts::EXE_SUFFIX)));
	daemon_command
		.arg("serve")
		.current_dir(&directory)
		.stdout(log.try_clone().unwrap())
		.stderr(log);
	env.apply(&mut daemon_command);
	let mut daemon = Owned::spawn(&mut daemon_command).unwrap();
	let mut mcp = Mcp::start(
		&directory.join(if cfg!(windows) { "mcp.ps1" } else { "mcp.sh" }),
		&[],
		&directory,
		&env,
	);

	let tools: BTreeSet<String> = mcp.rpc("tools/list", json!({}))["tools"]
		.as_array()
		.unwrap()
		.iter()
		.map(|t| t["name"].as_str().unwrap().to_string())
		.collect();
	let expected: BTreeSet<String> = [
		"docs",
		"events",
		"lsp",
		"ast_grep",
		"audit",
		"usage",
		"compare",
		"benchmark",
		"status",
		"reload",
		"stop",
		"channels",
		"memory_report",
		"valgrind_run",
		"runs",
		"show",
		"rank",
	]
	.into_iter()
	.map(str::to_string)
	.collect();
	assert_eq!(tools, expected, "MCP tool contract");
	pass(true, "discovered the complete MCP tool contract");
	let docs = mcp.call_json("docs", json!({"topic":"startup"}));
	pass(
		docs["sections"][0]["text"]
			.as_str()
			.is_some_and(|text| text.contains("subagent")),
		"startup docs link to audit repair delegation",
	);
	let state = wait_for(Duration::from_secs(90), || {
		let s = mcp.daemon_status();
		s["child_pid"].as_u64().map(|_| s)
	});
	let first_pid = state["child_pid"].as_u64().unwrap();
	pass(
		mcp.call_json("status", json!({"start":false}))["daemon_running"] == true,
		"read-only status",
	);
	pass(
		mcp.call_json("reload", json!({"wait":true}))["queued"] == false,
		"reload waits for completed swap",
	);
	let usage = mcp.call_json("usage", json!({}));
	pass(
		usage["most_used_tools"].is_array()
			&& usage["tools_with_failures"].is_array()
			&& usage["nodes"].as_u64().is_some_and(|n| n > 0)
			&& usage["picture"]
				.as_str()
				.is_some_and(|s| s.contains("just viz")),
		"usage returns the activity summary and terminal viewer",
	);
	let audit = mcp.call_json("audit", json!({"mode":"local"}));
	pass(
		audit["findings"].is_array(),
		"local audit returns recorded evidence",
	);
	if on_path("ast-grep") {
		pass(
			mcp
				.call(
					"ast_grep",
					json!({"pattern":"daemon::channels::inc($X)","paths":["src/main.rs"]}),
				)
				.contains("match(es)"),
			"structural search",
		);
	}
	if on_path("rust-analyzer") {
		wait_for(Duration::from_secs(180), || {
			let state = mcp.call_json("lsp", json!({"method":"status"}));
			assert!(state["dead"].is_null(), "rust-analyzer exited: {state}");
			(state["ready"] == true).then_some(())
		});
		pass(
			mcp.call_json("lsp", json!({"method":"symbols","query":"parse_work"}))["symbols"]
				.as_array()
				.is_some_and(|s| !s.is_empty()),
			"LSP finds workspace symbol",
		);
		let batch = mcp.call_json(
			"lsp",
			json!({"items":[
			{"method":"symbols","query":"parse_work"},
			{"method":"symbols","query":"serve"},
			{"method":"status"}
		],"concurrency":3}),
		);
		pass(
			batch["failed"] == 0
				&& batch["results"]
					.as_array()
					.is_some_and(|items| items.len() == 3),
			"LSP batch returns all three items",
		);
		for item in batch["results"].as_array().unwrap().iter().take(2) {
			let reply: Value = serde_json::from_str(item["result"]["content"].as_str().unwrap()).unwrap();
			pass(
				reply["symbols"]
					.as_array()
					.is_some_and(|symbols| !symbols.is_empty()),
				"LSP batch preserves symbol results",
			);
		}
	}

	wait_for(Duration::from_secs(90), || {
		mcp
			.call("channels", json!({}))
			.contains("app.work.parse")
			.then_some(())
	});
	pass(
		true,
		"live channels contain bounded-workload instrumentation",
	);
	if !lifecycle_only {
		if cfg!(target_os = "macos") {
			let memory = mcp.call("memory_report", json!({"heap": true, "backtraces": 2}));
			pass(
				memory.contains("leaks for") && memory.contains("\"heap\""),
				"native leaks and heap inspection",
			);
		} else {
			let reply = mcp.rpc(
				"tools/call",
				json!({"name": "memory_report", "arguments": {"heap": true}}),
			);
			pass(
				reply["isError"] == true && reply.to_string().contains("macOS-only"),
				"memory_report explicitly reports unsupported platform",
			);
		}
	}
	let reload_request = mcp.call_json("reload", json!({}));
	pass(reload_request["queued"] == true, "reload accepted");
	let state = wait_for(Duration::from_secs(90), || {
		let s = mcp.daemon_status();
		match s["child_pid"].as_u64() {
			Some(pid)
				if pid != first_pid
					&& s["building"] == false
					&& s["reload_completed"].as_u64() >= reload_request["ticket"].as_u64() =>
			{
				Some(s)
			}
			_ => None,
		}
	});
	pass(
		state["swaps"].as_u64().unwrap_or(0) >= 2,
		"reload actually swaps app PID",
	);

	if let Some(thread) = &notify_thread {
		mcp.call("events", json!({"subscribe_thread": thread}));
	}
	let source = directory.join("src/main.rs");
	let original = std::fs::read_to_string(&source).unwrap();
	std::fs::write(
        &source,
        format!("{original}\ncompile_error!(\"INTEGRATION_EXPECTED_FAILURE — controlled test; the harness restores this file\");\n"),
    )
    .unwrap();
	let failures = wait_for(Duration::from_secs(90), || {
		let result = mcp.call_json("events", json!({"errors_only": true}));
		result["events"]
			.as_array()
			.map(|events| events.iter().any(|e| e["kind"] == "compile_failed"))
			.unwrap_or(false)
			.then_some(result)
	});
	pass(
		failures
			.to_string()
			.contains("INTEGRATION_EXPECTED_FAILURE"),
		"compiler diagnostics captured in event feed",
	);
	wait_for(Duration::from_secs(90), || {
		(mcp.daemon_status()["last_build"]["ok"] == false).then_some(())
	});
	pass(
		mcp.daemon_status()["child_pid"] == state["child_pid"],
		"failed build preserves running app",
	);
	let failed_reload = mcp.rpc(
		"tools/call",
		json!({"name":"reload","arguments":{"wait":true,"timeout_s":30}}),
	);
	pass(
		failed_reload["isError"] == true && failed_reload.to_string().contains("reload failed"),
		"waiting reload reports compiler failure as MCP error",
	);

	wait_for(Duration::from_secs(90), || {
		mcp.rpc("ping", json!({}));
		mcp
			.notifications
			.iter()
			.any(|n| n["method"] == "notifications/message")
			.then_some(())
	});
	pass(true, "compile failure streamed as live MCP notification");
	if let Some(thread) = &notify_thread {
		wait_for(Duration::from_secs(90), || {
			mcp.call_json("events", json!({"limit": 100,"kind":"all"}))["events"]
				.as_array()
				.unwrap()
				.iter()
				.any(|e| e["kind"] == "notification_queued")
				.then_some(())
		});
		pass(
			true,
			"Codex queue accepted the event; actual session receipt is a separate check",
		);
		let event = failures["events"]
			.as_array()
			.unwrap()
			.iter()
			.find(|e| {
				e["kind"] == "compile_failed"
					&& e["diagnostics"]
						.as_str()
						.unwrap_or_default()
						.contains("INTEGRATION_EXPECTED_FAILURE")
			})
			.unwrap();
		let receipt =
			json!({"event_id": event["id"], "thread": thread, "workspace": directory.to_string_lossy()});
		std::fs::write(
			daemon::dir_for(&root).join("check-receipt.json"),
			format!("{receipt:#}\n"),
		)
		.unwrap();
		println!("RECEIPT CHECK: cargo test -p mcp --test integration -- --ignored verify_receipt");
		mcp.call("events", json!({"unsubscribe": true}));
	}
	std::fs::write(&source, &original).unwrap();
	mcp.call("reload", json!({}));
	wait_for(Duration::from_secs(90), || {
		let s = mcp.daemon_status();
		(s["last_build"]["ok"] == true && s["child_pid"] != state["child_pid"]).then_some(())
	});
	pass(
		mcp.call_json("stop", json!({}))["ok"] == true,
		"stop accepted",
	);
	let exit = wait_for(Duration::from_secs(15), || {
		daemon.0.try_wait().ok().flatten()
	});
	pass(exit.success(), "daemon exited successfully");
	let pids = mcp.app_pids.clone();
	wait_for(Duration::from_secs(10), || {
		pids.iter().all(|pid| !process_running(*pid)).then_some(())
	});
	pass(true, "all observed app PIDs are gone after stop");
	verify_worker_rebuild(&directory, &env, &mut mcp);
	if lifecycle_only {
		let status = mcp.close().expect("launcher exits on EOF");
		pass(status.success(), "lifecycle MCP closes successfully on EOF");
		println!("LIFECYCLE JOURNEY PASSED");
		return;
	}

	for name in ["native-before", "native-after"] {
		let report = mcp.call_json(
			"benchmark",
			json!({"name": name, "command": ["./target/release/app", "3"], "repeats": 3}),
		);
		pass(
			report["samples_ms"].as_array().map(Vec::len) == Some(3)
				&& report["channel_metrics"]["app.work.alloc.rounds/count"].as_f64() == Some(3.0),
			&format!("{name}: three timing samples and final workload channels"),
		);
	}
	let compare = mcp.call_json(
		"compare",
		json!({"before": "native-before", "after": "native-after"}),
	);
	pass(
		compare["metrics"].get("wall_p50_ms").is_some()
			&& compare["metrics"].get("app.work.parse/p99_us").is_some(),
		"native latency comparisons include channel distributions",
	);
	mcp.call(
		"benchmark",
		json!({"name": "native-mismatch", "command": ["./target/release/app", "4"], "repeats": 3}),
	);
	mcp.rejects("native-before", "native-mismatch");

	let docker = !on_path("valgrind");
	let binary = if docker {
		format!("{}/target-linux/release/", daemon_dir.display())
	} else {
		"./target/release/".to_string()
	};
	for (tool, executable) in [("callgrind", "app"), ("memcheck", "app")] {
		let mut record = json!({});
		for suffix in ["before", "after"] {
			let name = format!("{tool}-{suffix}");
			mcp.call(
				"valgrind_run",
				json!({"name": name, "tool": tool, "command": [format!("{binary}{executable}"), "3"],
                       "build": "cargo build --release --locked -p app", "timeout_s": 600}),
			);
			record = read_json(&daemon_dir.join(format!("valgrind/{name}.json")));
			pass(
				record["run_error"].is_null(),
				&format!("{name}: profiler completed"),
			);
			if tool == "callgrind" {
				let raw = std::fs::read_to_string(daemon_dir.join(format!("valgrind/{name}.out"))).unwrap();
				let raw_total: u64 = raw
					.lines()
					.find_map(|l| l.strip_prefix("summary:"))
					.and_then(|v| v.split_whitespace().next())
					.and_then(|v| v.parse().ok())
					.expect("callgrind summary line");
				let self_sum: u64 = record["functions"]
					.as_array()
					.unwrap()
					.iter()
					.map(|f| f["ir"].as_u64().unwrap())
					.sum();
				pass(
					record["total_ir"] == raw_total && raw_total == self_sum,
					"raw Ir equals sum of self costs",
				);
				pass(
					!record["call_edges"].as_array().unwrap().is_empty(),
					"caller/callee evidence retained",
				);
			} else {
				pass(
					record["allocations"]["allocs"].as_u64().unwrap_or(0) > 0
						&& record["errors_total"].as_u64() == Some(0),
					"allocation counts recorded with no memory errors",
				);
				let raw = std::fs::read_to_string(daemon_dir.join(format!("valgrind/{name}.log"))).unwrap();
				let total: u64 = raw
					.lines()
					.find_map(|l| l.split("ERROR SUMMARY:").nth(1))
					.and_then(|v| v.split_whitespace().next())
					.and_then(|v| v.replace(',', "").parse().ok())
					.expect("memcheck error summary");
				pass(
					record["errors_total"] == total,
					"Memcheck error total equals raw ERROR SUMMARY",
				);
			}
		}
		let diff = mcp.call_json(
			"compare",
			json!({"before": format!("{tool}-before"), "after": format!("{tool}-after")}),
		);
		pass(
			diff["metrics"]
				.as_object()
				.map(|m| !m.is_empty())
				.unwrap_or(false),
			&format!("{tool} before/after comparison"),
		);
		let shown = mcp.call("show", json!({"name": format!("{tool}-after"), "top": 100}));
		if tool == "callgrind" {
			pass(
				shown.contains("current definition"),
				"hotspots link to local source definitions",
			);
			let flagged = record["call_edges"]
				.as_array()
				.unwrap()
				.iter()
				.take(100)
				.any(|e| {
					e["attribution_flags"]
						.as_array()
						.map(|f| !f.is_empty())
						.unwrap_or(false)
				});
			if flagged {
				pass(
					shown.contains("attribution warning"),
					"suspicious call costs flagged",
				);
			}
		}
		if tool == "callgrind" {
			let ranked = mcp.call("rank", json!({"name": format!("{tool}-after"), "top": 2}));
			pass(
				ranked.contains("priority"),
				&format!(
					"{tool} ranking via {}",
					if live_jev {
						"live Jev"
					} else {
						"local Jev fixture"
					}
				),
			);
		}
		let mismatch = format!("{tool}-mismatch");
		mcp.call(
            "valgrind_run",
            json!({"name": mismatch, "tool": tool, "command": [format!("{binary}{executable}"), "4"], "timeout_s": 600}),
        );
		mcp.rejects(&format!("{tool}-before"), &mismatch);
	}
	for name in ["clean-before", "clean-after"] {
		mcp.call(
			"valgrind_run",
			json!({"name": name, "tool": "memcheck", "command": ["/bin/true"], "timeout_s": 60}),
		);
	}
	let clean = mcp.call_json(
		"compare",
		json!({"before": "clean-before", "after": "clean-after"}),
	);
	pass(
		clean["metrics"]["leak_summary/definitely_lost_bytes"]["delta"].as_f64() == Some(0.0),
		"clean Memcheck runs compare zero leaks",
	);
	pass(
		mcp.call("runs", json!({})).contains("native-before"),
		"runs lists all stored profile types",
	);
	if let Some((_, states)) = &fixture {
		let evidence = states
			.lock()
			.unwrap()
			.iter()
			.any(|s| s.get("source").is_some() && s.get("call_edges").is_some());
		pass(evidence, "ranking receives source and call-graph evidence");
	}

	pass(mcp.calls == tools, "every advertised MCP tool exercised");
	println!("ALL INTEGRATION CHECKS PASSED");
	let _ = mcp.close();
	let pids = mcp.app_pids.clone();
	drop(mcp);
	drop(daemon);
	drop(enclosure);
	pass(
		pids.iter().all(|pid| !process_running(*pid)),
		"cleanup left no observed app processes",
	);
}

fn verify_worker_rebuild(directory: &Path, env: &Env, mcp: &mut Mcp) {
	// Real worker rebuild under the SAME launcher process and client pipe.
	let launcher_pid = mcp.process.0.id();
	let worker_source = directory.join("src/runtime/mcp/main.rs");
	let text = std::fs::read_to_string(&worker_source).unwrap();
	assert!(
		text.contains("Daemon status (pid, running app pid"),
		"worker marker string moved"
	);
	std::fs::write(
		&worker_source,
		text.replace(
			"Daemon status (pid, running app pid",
			"REBUILT WORKER: Daemon status (pid, running app pid",
		),
	)
	.unwrap();
	env.run(
		directory,
		&["cargo", "build", "--release", "--locked", "-p", "mcp"],
	);
	let listed = mcp.rpc("tools/list", json!({}));
	pass(
		listed.to_string().contains("REBUILT WORKER") && mcp.process.0.id() == launcher_pid,
		"rebuilt worker picked up without reconnecting MCP client",
	);
	pass(
		mcp
			.notifications
			.iter()
			.any(|n| n["method"] == "notifications/tools/list_changed"),
		"worker replacement sends tool-list notification",
	);
}

#[cfg(unix)]
fn owned_containers() -> BTreeSet<String> {
	let output = Command::new("docker")
		.args(["ps", "-aq", "--filter", "label=valgrind.owned=true"])
		.output()
		.expect("docker on PATH");
	assert!(
		output.status.success(),
		"docker ps failed: the regressions need a running Docker daemon"
	);
	String::from_utf8_lossy(&output.stdout)
		.split_whitespace()
		.map(str::to_string)
		.collect()
}

/// The independently reproduced audit defects that need the real
/// transport and the real Docker daemon. Rust unit tests cover process groups,
/// signals, build races, parser and comparison failures.
#[test]
#[ignore = "needs a running Docker daemon and the profiler image"]
#[cfg(unix)]
fn regressions() {
	let root = root();
	let plain = Env {
		set: vec![],
		remove: vec![],
	};
	plain.run(
		&root,
		&["cargo", "build", "--workspace", "--release", "--locked"],
	);
	let worker = root.join("target/release/mcp");
	let launcher = root.join("target/release/mcp-launcher");
	let baseline = owned_containers();
	let tmp = Enclosure(PathBuf::from(format!(
		"/tmp/mcp-regressions-{}",
		std::process::id()
	)));
	let result = std::panic::catch_unwind(|| {
		let tmp = &tmp.0;
		// Deliberately failing Cargo fixture: the transport returns diagnostics
		// for initialize, stays open, and can retry successfully.
		std::fs::create_dir_all(tmp.join("bin")).unwrap();
		std::fs::write(tmp.join("fail-build"), "").unwrap();
		let fake_cargo = tmp.join("bin/cargo");
		std::fs::write(
            &fake_cargo,
            "#!/bin/sh\nif [ -e fail-build ]; then echo \"LOCKFILE_MISMATCH_FIXTURE\" >&2; exit 101; fi\n",
        )
        .unwrap();
		std::fs::set_permissions(
			&fake_cargo,
			std::os::unix::fs::PermissionsExt::from_mode(0o755),
		)
		.unwrap();
		let path = format!(
			"{}:{}",
			tmp.join("bin").display(),
			std::env::var("PATH").unwrap_or_default()
		);
		let env = Env {
			set: vec![("PATH".into(), path)],
			remove: vec![],
		};
		let mut probe = Command::new(&launcher);
		probe
			.args(["--worker", worker.to_str().unwrap()])
			.current_dir(tmp)
			.stdin(Stdio::piped())
			.stdout(Stdio::piped())
			.stderr(Stdio::null())
			.process_group(0);
		env.apply(&mut probe);
		let mut probe = Owned::spawn(&mut probe).unwrap();
		let mut stdin = probe.0.stdin.take().unwrap();
		let mut stdout = BufReader::new(probe.0.stdout.take().unwrap());
		let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "bootstrap-regression", "version": "1"}}});
		let mut initialize = || -> Value {
			stdin.write_all(format!("{init}\n").as_bytes()).unwrap();
			stdin.flush().unwrap();
			let (sender, receiver) = mpsc::channel();
			let mut line = String::new();
			// Answer within 10 s or the transport hung on the failed build.
			let read = std::thread::scope(|scope| {
				scope.spawn(|| {
					let _ = stdout.read_line(&mut line);
					let _ = sender.send(());
				});
				receiver.recv_timeout(Duration::from_secs(10))
			});
			assert!(read.is_ok(), "initialize did not answer");
			serde_json::from_str(&line).unwrap()
		};
		let response = initialize();
		assert!(
			response["id"] == 1 && response["error"]["code"] == -32603,
			"{response}"
		);
		assert!(
			response["error"]["message"]
				.as_str()
				.unwrap()
				.contains("LOCKFILE_MISMATCH_FIXTURE"),
			"{response}"
		);
		std::fs::remove_file(tmp.join("fail-build")).unwrap();
		let response = initialize();
		assert!(
			response["result"]["capabilities"]["tools"]["listChanged"] == true,
			"{response}"
		);
		drop(stdin);
		wait_for(Duration::from_secs(10), || {
			probe.0.try_wait().ok().flatten()
		});
		println!("PASS: bootstrap build failure returns diagnostics and same transport recovers (Cargo fixture, real worker)");

		let request = |name: &str, timeout: u64| {
			json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "valgrind_run", "arguments": {"name": name, "tool": "memcheck",
                "command": ["/bin/sh", "-c", "sleep 120"], "timeout_s": timeout}}})
			.to_string()
				+ "\n"
		};
		let started = Instant::now();
		let mut direct = Command::new(&worker)
			.current_dir(tmp)
			.stdin(Stdio::piped())
			.stdout(Stdio::piped())
			.stderr(Stdio::piped())
			.spawn()
			.unwrap();
		direct
			.stdin
			.take()
			.unwrap()
			.write_all(request("timeout", 1).as_bytes())
			.unwrap();
		let output = direct.wait_with_output().unwrap();
		let response: Value = serde_json::from_slice(&output.stdout).unwrap();
		assert!(response["result"]["isError"] == true, "{response}");
		assert!(response.to_string().contains("timed out"), "{response}");
		assert!(started.elapsed() < Duration::from_secs(25));
		assert!(
			owned_containers() == baseline,
			"timeout left an owned container"
		);
		println!("PASS: actual Docker timeout removes container and descendants");

		// Terminate the real transport with a Docker worker in flight.
		let mut launcher_command = Command::new(&launcher);
		launcher_command
			.args(["--no-build", "--worker", worker.to_str().unwrap()])
			.current_dir(tmp)
			.stdin(Stdio::piped())
			.stdout(Stdio::piped())
			.stderr(Stdio::piped())
			.process_group(0);
		let mut running = Owned::spawn(&mut launcher_command).unwrap();
		let mut stdin = running.0.stdin.take().unwrap();
		stdin.write_all(request("cancel", 120).as_bytes()).unwrap();
		stdin.flush().unwrap();
		wait_for(Duration::from_secs(30), || {
			(!owned_containers().difference(&baseline).next().is_none()).then_some(())
		});
		signal(running.0.id(), SIGTERM);
		wait_for(Duration::from_secs(20), || {
			running.0.try_wait().ok().flatten()
		});
		assert!(
			owned_containers() == baseline,
			"launcher cancellation left a container"
		);
		println!("PASS: actual launcher cancellation waits for worker container cleanup");
	});
	let survivors: Vec<String> = owned_containers().difference(&baseline).cloned().collect();
	if !survivors.is_empty() {
		let status = Command::new("docker")
			.args(["rm", "-f"])
			.args(&survivors)
			.status()
			.unwrap();
		assert!(
			status.success(),
			"docker rm -f of leftover containers failed"
		);
	}
	if let Err(panic) = result {
		std::panic::resume_unwind(panic);
	}
	println!("ALL AUDIT REGRESSIONS PASSED");
}

fn jsonl_files(dir: &Path, needle: &str, found: &mut Vec<PathBuf>) {
	let Ok(entries) = std::fs::read_dir(dir) else {
		return;
	};
	for entry in entries.flatten() {
		let path = entry.path();
		if path.is_dir() {
			jsonl_files(&path, needle, found);
		} else if path.extension().is_some_and(|e| e == "jsonl")
			&& path
				.file_name()
				.is_some_and(|n| n.to_string_lossy().contains(needle))
		{
			found.push(path);
		}
	}
}

/// Actual Codex receipt, independently of `codex queue` acceptance: the event
/// id must appear as an incoming user message in the named session's
/// persisted transcript under `$CODEX_HOME/sessions` (default `~/.codex`).
#[test]
#[ignore = "reads the local Codex transcripts after a notify-thread check run"]
fn verify_receipt() {
	let root = root();
	let record = read_json(&daemon::dir_for(&root).join("check-receipt.json"));
	let (event_id, thread) = (
		record["event_id"].as_str().unwrap(),
		record["thread"].as_str().unwrap(),
	);
	let codex_home = std::env::var_os("CODEX_HOME")
		.map(PathBuf::from)
		.unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap()).join(".codex"));
	let marker = format!("[compiler failure {event_id}]");
	let mut transcripts = vec![];
	jsonl_files(&codex_home.join("sessions"), thread, &mut transcripts);
	let receipt = transcripts.iter().find_map(|path| {
		let file = std::fs::File::open(path).ok()?;
		BufReader::new(file)
			.lines()
			.map_while(Result::ok)
			.find_map(|line| {
				let row: Value = serde_json::from_str(&line).ok()?;
				let payload = &row["payload"];
				if row["type"] != "response_item" || payload["role"] != "user" {
					return None;
				}
				let received = payload["content"].as_array()?.iter().any(|item| {
					item["text"]
						.as_str()
						.unwrap_or_default()
						.starts_with(&marker)
				});
				received.then(|| {
					json!({"event_id": event_id, "thread": thread, "received_at": row["timestamp"],
                       "transcript": path.to_string_lossy()})
				})
			})
	});
	let receipt = receipt.unwrap_or_else(|| {
        panic!("NOT VERIFIED: event {event_id} has not appeared as an incoming user message in session {thread}")
    });
	println!("PASS: actual session receipt {receipt}");
	std::fs::write(
		daemon::dir_for(&root).join("verified-receipt.json"),
		format!("{receipt:#}\n"),
	)
	.unwrap();
}
