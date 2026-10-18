//! A long-lived `rust-analyzer` the daemon owns, so the index is
//! built once per project and survives every MCP worker swap. The daemon
//! starts it on the first `lsp` request and answers navigation, diagnostics,
//! call hierarchy and edit-preview questions plus `status`, and lets it die with the daemon: its stdin is a pipe, and
//! rust-analyzer exits on EOF.
//!
//! Files are never opened through `didOpen`; rust-analyzer reads the tree
//! from disk and watches it itself, so a source edit is live on the next
//! question without any sync from this side. Positions on this wire are
//! 1-based (what editors and compilers print); LSP's 0-based UTF-16 columns
//! are converted at the edge. Ceiling: a non-ASCII character before the
//! column shifts it by its extra UTF-16 units; convert through the line text
//! when that matters.

use crate::util::ProcessTree;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long one question may wait for rust-analyzer; the socket client's
/// read timeout has to be longer.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
/// How long `diagnostics` waits for `cargo check` to finish.
pub const CHECK_TIMEOUT: Duration = Duration::from_secs(120);

struct Inbox {
	/// Replies by request id.
	replies: HashMap<u64, Value>,
	/// `$/progress` tokens with a `begin` and no `end` yet.
	active: HashSet<String>,
	/// rust-analyzer's own word (`experimental/serverStatus`): the
	/// workspace is loaded and nothing is being indexed. Progress tokens
	/// alone have gaps between phases, so they cannot tell this.
	quiescent: bool,
	/// Count of `begin` reports, so a caller can wait for the one it triggered.
	begins: u64,
	/// Last `publishDiagnostics` per file (uri), replaced wholesale.
	diagnostics: BTreeMap<String, Vec<Value>>,
	/// rust-analyzer exited or its pipe broke.
	dead: Option<String>,
}

struct Link {
	inbox: Mutex<Inbox>,
	changed: Condvar,
}

pub struct Client {
	process: ProcessTree,
	stdin: Arc<Mutex<std::process::ChildStdin>>,
	link: Arc<Link>,
	next_id: Mutex<u64>,
	// Flycheck notifications share global progress state; never overlap them.
	check: Mutex<()>,
	root: PathBuf,
}

impl Client {
	/// Spawn `rust-analyzer` for the project at `root` and run the
	/// initialize handshake; the index builds in the background.
	pub fn start(root: &Path) -> Result<Client, String> {
		let root = root
			.canonicalize()
			.map_err(|e| format!("{}: {e}", root.display()))?;
		let mut command = Command::new("rust-analyzer");
		command
			.current_dir(&root)
			.stdin(Stdio::piped())
			.stdout(Stdio::piped())
			.stderr(Stdio::inherit());
		let mut process = ProcessTree::spawn(&mut command).map_err(|e| {
			format!(
				"cannot start rust-analyzer: {e}; install it with `rustup component add rust-analyzer`"
			)
		})?;
		let stdin = process.0.stdin.take().ok_or("no stdin pipe")?;
		let stdout = process.0.stdout.take().ok_or("no stdout pipe")?;
		let link = Arc::new(Link {
			inbox: Mutex::new(Inbox {
				replies: HashMap::new(),
				active: HashSet::new(),
				quiescent: false,
				begins: 0,
				diagnostics: BTreeMap::new(),
				dead: None,
			}),
			changed: Condvar::new(),
		});
		let client = Client {
			process,
			stdin: Arc::new(Mutex::new(stdin)),
			link: link.clone(),
			next_id: Mutex::new(1),
			check: Mutex::new(()),
			root,
		};
		let reader_link = link.clone();
		let reader_stdin = client.stdin.clone();
		std::thread::spawn(move || read_loop(stdout, reader_link, reader_stdin));

		let uri = uri_for(&client.root);
		let initialize = json!({
				"processId": std::process::id(),
				"rootUri": uri,
				"workspaceFolders": [{"uri": uri, "name": client.root.file_name().and_then(|n| n.to_str()).unwrap_or("project")}],
				"capabilities": {
						"textDocument": {
								"hover": {"contentFormat": ["plaintext", "markdown"]},
								"publishDiagnostics": {"relatedInformation": true}
						},
						"window": {"workDoneProgress": true},
						"workspace": {"configuration": true},
						"experimental": {"serverStatusNotification": true}
				},
				"initializationOptions": {"checkOnSave": true}
		});
		client.request("initialize", initialize)?;
		client.notify("initialized", json!({}))?;
		Ok(client)
	}

	/// `{"ready": bool, "pid": u32, "indexing": [tokens], "diagnostics": n}`.
	pub fn status(&self) -> Value {
		let inbox = self.link.inbox.lock().unwrap();
		json!({
				"pid": self.process.0.id(),
				"ready": inbox.quiescent && inbox.active.is_empty() && inbox.dead.is_none(),
				"indexing": inbox.active.iter().cloned().collect::<Vec<_>>(),
				"dead": inbox.dead,
				"diagnostics": inbox.diagnostics.values().map(Vec::len).sum::<usize>(),
		})
	}

	/// Answer one question from the daemon socket (`method`, `file`,
	/// `line`, `col`, `query`); positions 1-based in and out.
	pub fn ask(&self, request: &Value) -> Result<Value, String> {
		let method = request["method"].as_str().unwrap_or("status");
		if method == "status" {
			return Ok(self.status());
		}
		if let Some(wait) = self.wait_ready(REQUEST_TIMEOUT) {
			return Err(wait);
		}
		match method {
			"rename" => {
                let (file, position) = self.position(request)?;
                let new_name = request["new_name"].as_str().filter(|s| !s.is_empty()).ok_or("new_name is required")?;
                let edit = self.request("textDocument/rename", json!({"textDocument":{"uri":uri_for(&file)},"position":position,"newName":new_name}))?;
                Ok(json!({"workspace_edit":edit,"applied":false,"coordinates":"LSP edits use zero-based UTF-16 positions; preview only"}))
            }
            "call_hierarchy" => {
                let (file, position) = self.position(request)?;
                let items = self.request("textDocument/prepareCallHierarchy", json!({"textDocument":{"uri":uri_for(&file)},"position":position}))?;
                let direction = request["direction"].as_str().unwrap_or("incoming");
                let method = match direction {"incoming"=>"callHierarchy/incomingCalls", "outgoing"=>"callHierarchy/outgoingCalls", _=>return Err("direction must be incoming or outgoing".into())};
                let mut results = Vec::new();
                for item in items.as_array().into_iter().flatten().take(20) {
                    results.push(json!({"item":item,"calls":self.request(method,json!({"item":item}))?}));
                }
                Ok(json!({"direction":direction,"hierarchy":results,"coordinates":"LSP zero-based UTF-16 positions"}))
            }
            "code_actions" => {
                let (file, position) = self.position(request)?;
                let uri = uri_for(&file);
                let diagnostics = self.link.inbox.lock().unwrap().diagnostics.get(&uri).cloned().unwrap_or_default();
                let actions = self.request("textDocument/codeAction",json!({"textDocument":{"uri":uri},"range":{"start":position,"end":position},"context":{"diagnostics":diagnostics}}))?;
                Ok(json!({"actions":actions,"applied":false,"coordinates":"LSP zero-based UTF-16 positions; preview only"}))
            }
            "definition" | "references" | "hover" => {
				let (file, position) = self.position(request)?;
				let mut params = json!({"textDocument": {"uri": uri_for(&file)}, "position": position});
				if method == "references" {
					params["context"] = json!({"includeDeclaration": true});
				}
				let reply = self.request(&format!("textDocument/{method}"), params)?;
				Ok(match method {
					"hover" => json!({"hover": hover_text(&reply)}),
					_ => json!({"locations": locations(&reply, &self.root)}),
				})
			}
			"symbols" => {
				let query = request["query"].as_str().ok_or("`query` is required")?;
				let reply = self.request("workspace/symbol", json!({"query": query}))?;
				let symbols = reply
					.as_array()
					.map(|list| {
						list
							.iter()
							.take(100)
							.map(|s| {
								let mut entry = location(&s["location"], &self.root);
								entry["name"] = s["name"].clone();
								entry["kind"] = json!(symbol_kind(s["kind"].as_u64().unwrap_or(0)));
								if let Some(container) = s.get("containerName") {
									entry["container"] = container.clone();
								}
								entry
							})
							.collect::<Vec<_>>()
					})
					.unwrap_or_default();
				Ok(json!({"symbols": prefer_name_matches(symbols, query)}))
			}
			"diagnostics" => {
				let _check = self.check.lock().unwrap();
				let before = self.link.inbox.lock().unwrap().begins;
				self.notify("rust-analyzer/runFlycheck", json!({"textDocument": null}))?;
				self.wait_begin(before, Duration::from_secs(5));
				if let Some(wait) = self.wait_ready(CHECK_TIMEOUT) {
					return Err(wait);
				}
				let only = request["file"].as_str().map(|f| uri_for(&self.resolve(f)));
				let inbox = self.link.inbox.lock().unwrap();
				let list = inbox
					.diagnostics
					.iter()
					.filter(|(uri, _)| only.as_ref().is_none_or(|o| o == *uri))
					.flat_map(|(uri, items)| {
						items.iter().map(move |d| {
							let range = &d["range"]["start"];
							json!({
									"file": relative(uri, &self.root),
									"line": range["line"].as_u64().unwrap_or(0) + 1,
									"col": range["character"].as_u64().unwrap_or(0) + 1,
									"severity": severity(d["severity"].as_u64().unwrap_or(0)),
									"code": d["code"].clone(),
									"message": d["message"],
							})
						})
					})
					.take(200)
					.collect::<Vec<_>>();
				Ok(json!({"diagnostics": list}))
			}
			other => Err(format!(
				"unknown lsp method `{other}`: definition, references, hover, symbols, diagnostics, status, rename, call_hierarchy, code_actions"
			)),
		}
	}

	/// Wait until a progress report began after `before`, at most `limit`:
	/// flycheck announces itself a moment after `runFlycheck`.
	fn wait_begin(&self, before: u64, limit: Duration) {
		let start = Instant::now();
		let mut inbox = self.link.inbox.lock().unwrap();
		while inbox.begins == before && inbox.dead.is_none() {
			let Some(left) = limit.checked_sub(start.elapsed()) else {
				break;
			};
			inbox = self.link.changed.wait_timeout(inbox, left).unwrap().0;
		}
	}

	/// `None` once the index is quiet; else why the caller should retry.
	fn wait_ready(&self, limit: Duration) -> Option<String> {
		let start = Instant::now();
		let mut inbox = self.link.inbox.lock().unwrap();
		loop {
			if let Some(dead) = &inbox.dead {
				return Some(format!("rust-analyzer is gone: {dead}"));
			}
			if inbox.quiescent && inbox.active.is_empty() {
				return None;
			}
			let Some(left) = limit.checked_sub(start.elapsed()) else {
				return Some("rust-analyzer is still indexing; retry in a few seconds".into());
			};
			let (next, timeout) = self.link.changed.wait_timeout(inbox, left).unwrap();
			inbox = next;
			if timeout.timed_out() {
				let active = inbox.active.iter().cloned().collect::<Vec<_>>();
				return Some(format!(
					"rust-analyzer still indexing after {} s ({}); retry in a few seconds",
					limit.as_secs(),
					if active.is_empty() {
						"loading the workspace".into()
					} else {
						active.join(", ")
					}
				));
			}
		}
	}

	fn position(&self, request: &Value) -> Result<(PathBuf, Value), String> {
		let file = request["file"].as_str().ok_or("`file` is required")?;
		let line = request["line"]
			.as_u64()
			.ok_or("`line` is required (1-based)")?;
		let col = request["col"].as_u64().unwrap_or(1);
		Ok((
			self.resolve(file),
			json!({"line": line.saturating_sub(1), "character": col.saturating_sub(1)}),
		))
	}

	fn resolve(&self, file: &str) -> PathBuf {
		let path = Path::new(file);
		if path.is_absolute() {
			path.to_path_buf()
		} else {
			self.root.join(path)
		}
	}

	fn request(&self, method: &str, params: Value) -> Result<Value, String> {
		let id = {
			let mut next = self.next_id.lock().unwrap();
			let id = *next;
			*next += 1;
			id
		};
		send(
			&mut *self.stdin.lock().unwrap(),
			&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
		)?;
		let start = Instant::now();
		let mut inbox = self.link.inbox.lock().unwrap();
		loop {
			if let Some(reply) = inbox.replies.remove(&id) {
				return match reply.get("error") {
					Some(error) => Err(format!("{method}: {}", error["message"])),
					None => Ok(reply.get("result").cloned().unwrap_or(Value::Null)),
				};
			}
			if let Some(dead) = &inbox.dead {
				return Err(format!("rust-analyzer is gone: {dead}"));
			}
			let left = REQUEST_TIMEOUT
				.checked_sub(start.elapsed())
				.ok_or_else(|| format!("{method}: no answer within {} s", REQUEST_TIMEOUT.as_secs()))?;
			inbox = self.link.changed.wait_timeout(inbox, left).unwrap().0;
		}
	}

	fn notify(&self, method: &str, params: Value) -> Result<(), String> {
		send(
			&mut *self.stdin.lock().unwrap(),
			&json!({"jsonrpc": "2.0", "method": method, "params": params}),
		)
	}
}

/// Read framed messages until EOF: route replies by id, track progress,
/// keep diagnostics, and answer the server's own requests (configuration,
/// progress tokens, capability registration) so it never stalls on us.
fn read_loop(
	stdout: std::process::ChildStdout,
	link: Arc<Link>,
	stdin: Arc<Mutex<std::process::ChildStdin>>,
) {
	let mut reader = BufReader::new(stdout);
	let outcome = loop {
		let message = match read_frame(&mut reader) {
			Ok(Some(message)) => message,
			Ok(None) => break "exited (EOF)".to_string(),
			Err(error) => break error,
		};
		let mut inbox = link.inbox.lock().unwrap();
		if let Some(id) = message
			.get("id")
			.and_then(Value::as_u64)
			.filter(|_| message.get("method").is_none())
		{
			inbox.replies.insert(id, message);
		} else if let Some(method) = message["method"].as_str() {
			match method {
				"experimental/serverStatus" => {
					inbox.quiescent = message["params"]["quiescent"].as_bool().unwrap_or(false);
				}
				"$/progress" => {
					let token = message["params"]["token"].to_string();
					match message["params"]["value"]["kind"].as_str() {
						Some("begin") => {
							inbox.active.insert(token);
							inbox.begins += 1;
						}
						Some("end") => {
							inbox.active.remove(&token);
						}
						_ => {}
					}
				}
				"textDocument/publishDiagnostics" => {
					let uri = message["params"]["uri"].as_str().unwrap_or("").to_string();
					let items = message["params"]["diagnostics"]
						.as_array()
						.cloned()
						.unwrap_or_default();
					if items.is_empty() {
						inbox.diagnostics.remove(&uri);
					} else {
						inbox.diagnostics.insert(uri, items);
					}
				}
				_ => {}
			}
			if let Some(id) = message.get("id") {
				let result = match method {
					"workspace/configuration" => {
						let n = message["params"]["items"].as_array().map_or(0, Vec::len);
						Value::Array(vec![Value::Null; n])
					}
					_ => Value::Null,
				};
				let _ = send(
					&mut *stdin.lock().unwrap(),
					&json!({"jsonrpc": "2.0", "id": id, "result": result}),
				);
			}
		}
		drop(inbox);
		link.changed.notify_all();
	};
	link.inbox.lock().unwrap().dead = Some(outcome);
	link.changed.notify_all();
}

/// Write one `Content-Length` framed message.
pub fn send(out: &mut impl Write, message: &Value) -> Result<(), String> {
	let body = message.to_string();
	out
		.write_all(format!("Content-Length: {}\r\n\r\n{body}", body.len()).as_bytes())
		.and_then(|_| out.flush())
		.map_err(|e| format!("rust-analyzer stdin: {e}"))
}

/// Read one `Content-Length` framed message; `None` at EOF.
pub fn read_frame(reader: &mut impl BufRead) -> Result<Option<Value>, String> {
	let mut length = None;
	loop {
		let mut line = String::new();
		if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
			return Ok(None);
		}
		let line = line.trim_end();
		if line.is_empty() {
			break;
		}
		if let Some(value) = line.strip_prefix("Content-Length:") {
			length = value.trim().parse::<usize>().ok();
		}
	}
	let length = length.ok_or("frame without Content-Length")?;
	let mut body = vec![0; length];
	reader.read_exact(&mut body).map_err(|e| e.to_string())?;
	serde_json::from_slice(&body)
		.map(Some)
		.map_err(|e| format!("bad frame: {e}"))
}

pub fn uri_for(path: &Path) -> String {
	url::Url::from_file_path(path)
		.expect("LSP paths are absolute")
		.into()
}

/// `src/x.rs` for a uri under `root`, the plain path otherwise.
pub fn relative(uri: &str, root: &Path) -> String {
	let Some(path) = url::Url::parse(uri)
		.ok()
		.and_then(|url| url.to_file_path().ok())
	else {
		return uri.into();
	};
	// URL conversion normalizes Windows verbatim paths (\\?\C:\...) too.
	let root = url::Url::from_file_path(root)
		.ok()
		.and_then(|url| url.to_file_path().ok())
		.unwrap_or_else(|| root.to_path_buf());
	path
		.strip_prefix(root)
		.map(|p| p.to_string_lossy().replace('\\', "/"))
		.unwrap_or_else(|_| path.display().to_string())
}

/// One `Location` or `LocationLink` as `{file, line, col}` (1-based).
fn location(value: &Value, root: &Path) -> Value {
	let (uri, range) = match value.get("targetUri") {
		Some(uri) => (uri, &value["targetSelectionRange"]),
		None => (&value["uri"], &value["range"]),
	};
	json!({
			"file": relative(uri.as_str().unwrap_or(""), root),
			"line": range["start"]["line"].as_u64().unwrap_or(0) + 1,
			"col": range["start"]["character"].as_u64().unwrap_or(0) + 1,
	})
}

fn locations(reply: &Value, root: &Path) -> Vec<Value> {
	match reply {
		Value::Array(list) => list.iter().take(200).map(|l| location(l, root)).collect(),
		Value::Null => vec![],
		one => vec![location(one, root)],
	}
}

/// The hover body as text: `MarkupContent`, a string, or a list of either.
fn hover_text(reply: &Value) -> Value {
	fn text(contents: &Value) -> String {
		match contents {
			Value::String(s) => s.clone(),
			Value::Array(list) => list.iter().map(text).collect::<Vec<_>>().join("\n"),
			other => other["value"].as_str().unwrap_or("").to_string(),
		}
	}
	match reply.get("contents") {
		Some(contents) => json!(text(contents)),
		None => Value::Null,
	}
}

/// rust-analyzer's `workspace/symbol` is fuzzy: `parse_work` also matches
/// `compares_..._workloads`. Exact names first, then names containing the
/// query; the fuzzy rest only when nothing else matched.
pub fn prefer_name_matches(symbols: Vec<Value>, query: &str) -> Vec<Value> {
	let query = query.to_lowercase();
	let name = |s: &Value| s["name"].as_str().unwrap_or("").to_lowercase();
	let (exact, rest): (Vec<Value>, Vec<Value>) = symbols.into_iter().partition(|s| name(s) == query);
	let (contains, fuzzy): (Vec<Value>, Vec<Value>) =
		rest.into_iter().partition(|s| name(s).contains(&query));
	let mut out = exact;
	out.extend(contains);
	if out.is_empty() {
		return fuzzy;
	}
	out
}

fn severity(code: u64) -> &'static str {
	match code {
		1 => "error",
		2 => "warning",
		3 => "info",
		4 => "hint",
		_ => "unknown",
	}
}

fn symbol_kind(code: u64) -> &'static str {
	match code {
		2 => "module",
		5 => "class",
		6 => "method",
		7 => "property",
		8 => "field",
		9 => "constructor",
		10 => "enum",
		11 => "interface",
		12 => "function",
		13 => "variable",
		14 => "constant",
		22 => "enum member",
		23 => "struct",
		25 => "type parameter",
		26 => "trait",
		_ => "symbol",
	}
}

#[cfg(test)]
mod tests {
	use crate as daemon;
	use daemon::lsp::{prefer_name_matches, read_frame, relative, send, uri_for};
	use serde_json::json;
	use std::io::Cursor;
	use std::path::Path;

	#[test]
	fn frames_round_trip_and_end_at_eof() {
		let mut wire = Vec::new();
		send(&mut wire, &json!({"id": 1, "method": "a"})).unwrap();
		send(&mut wire, &json!({"id": 2, "params": {"ü": "ß"}})).unwrap();
		assert!(wire.starts_with(b"Content-Length: "));
		let mut reader = Cursor::new(wire);
		assert_eq!(read_frame(&mut reader).unwrap().unwrap()["id"], 1);
		assert_eq!(
			read_frame(&mut reader).unwrap().unwrap()["params"]["ü"],
			"ß"
		);
		assert!(read_frame(&mut reader).unwrap().is_none());
		let mut bad = Cursor::new(b"Content-Type: x\r\n\r\n{}".to_vec());
		assert!(read_frame(&mut bad).is_err());
	}

	#[test]
	fn paths_relativize_under_the_root_only() {
		let root = if cfg!(windows) {
			Path::new(r"C:\w\p")
		} else {
			Path::new("/w/p")
		};
		let uri = uri_for(root);
		assert_eq!(
			uri,
			if cfg!(windows) {
				"file:///C:/w/p"
			} else {
				"file:///w/p"
			}
		);
		assert_eq!(relative(&format!("{uri}/src/a.rs"), root), "src/a.rs");
		let file = root.join("src/a b#ü.rs");
		assert!(uri_for(&file).ends_with("/src/a%20b%23%C3%BC.rs"));
		assert_eq!(relative(&uri_for(&file), root), "src/a b#ü.rs");
		let outside = root.parent().unwrap().join("outside.rs");
		assert_eq!(
			relative(&uri_for(&outside), root),
			outside.display().to_string()
		);
	}

	/// Needs `rust-analyzer` on PATH and indexes this workspace (10–60 s):
	/// `cargo test -p daemon --test lsp -- --ignored`.
	#[test]
	#[ignore]
	fn live_rust_analyzer_answers_definition() {
		let root = daemon::project_root(); // Cargo runs tests from target/gen/<crate>
		let client = daemon::lsp::Client::start(&root).unwrap();
		let line = std::fs::read_to_string(root.join("src/runtime/daemon/lib.rs"))
			.unwrap()
			.lines()
			.position(|l| l.starts_with("pub fn dir_for"))
			.unwrap()
			+ 1;
		let reply = client
			.ask(&json!({"method": "hover", "file": "src/runtime/daemon/lib.rs", "line": line, "col": 8}))
			.unwrap();
		assert!(reply["hover"].as_str().unwrap().contains("pub fn dir_for"));
		let reply = client
			.ask(&json!({"method": "symbols", "query": "ensure_running"}))
			.unwrap();
		assert_eq!(reply["symbols"][0]["file"], "src/runtime/daemon/hot.rs");
		let rename = client.ask(&json!({"method":"rename","file":"src/runtime/daemon/lib.rs","line":line,"col":8,"new_name":"dir_for_preview"})).unwrap();
		assert_eq!(rename["applied"], false);
		assert!(!rename["workspace_edit"].is_null());
		let hierarchy = client
			.ask(
				&json!({"method":"call_hierarchy","file":"src/runtime/daemon/lib.rs","line":line,"col":8}),
			)
			.unwrap();
		assert!(hierarchy["hierarchy"]
			.as_array()
			.is_some_and(|a| !a.is_empty()));
		let actions = client
			.ask(&json!({"method":"code_actions","file":"src/runtime/daemon/lib.rs","line":line,"col":8}))
			.unwrap();
		assert_eq!(actions["applied"], false);
	}

	#[test]
	fn symbols_rank_exact_then_substring_and_drop_fuzzy_noise() {
		let list = vec![
			json!({"name": "compares_costs_and_rejects_different_workloads"}),
			json!({"name": "parse_workload"}),
			json!({"name": "parse_work"}),
		];
		let ranked = prefer_name_matches(list.clone(), "parse_work");
		assert_eq!(ranked[0]["name"], "parse_work");
		assert_eq!(ranked[1]["name"], "parse_workload");
		assert_eq!(ranked.len(), 2);
		assert_eq!(prefer_name_matches(list.clone(), "PARSE_WORK").len(), 2);
		assert_eq!(prefer_name_matches(list, "prs_wrk").len(), 3);
	}
}
