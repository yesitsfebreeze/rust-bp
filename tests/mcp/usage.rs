//! The tool-call recorder: what a call touched, and the event it appends.

use mcp::usage::{pattern_symbol, record, symbol_at, touched};
use serde_json::{json, Value};
use std::path::PathBuf;

fn temp(label: &str) -> PathBuf {
	let dir = std::env::temp_dir().join(format!("rt-usage-{label}-{}", std::process::id()));
	std::fs::create_dir_all(&dir).unwrap();
	dir
}

#[test]
fn symbol_under_the_cursor_and_pattern_heads() {
	let dir = temp("sym");
	let file = dir.join("a.rs");
	std::fs::write(
		&file,
		"fn main() {\n\tlet n = read_frame(&mut reader);\n}\n",
	)
	.unwrap();
	let path = file.to_str().unwrap();
	assert_eq!(symbol_at(path, 2, 10).as_deref(), Some("read_frame"));
	assert_eq!(symbol_at(path, 2, 19).as_deref(), Some("read_frame"));
	assert_eq!(symbol_at(path, 2, 20), None); // the parenthesis
	assert_eq!(symbol_at(path, 1, 4).as_deref(), Some("main"));
	assert_eq!(symbol_at(path, 9, 1), None);
	assert_eq!(
		pattern_symbol("read_frame($$$A)").as_deref(),
		Some("read_frame")
	);
	assert_eq!(pattern_symbol("$A.unwrap()"), None);
	assert_eq!(pattern_symbol("  Foo::bar($X)").as_deref(), Some("Foo"));
	let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn touched_reads_arguments_and_replies() {
	let dir = temp("touch");
	let file = dir.join("b.rs");
	std::fs::write(&file, "pub fn send() {}\nsend();\n").unwrap();
	let path = file.to_str().unwrap().to_string();

	let reply = json!({"locations": [{"file": "src/x.rs", "line": 1, "col": 1}, {"file": &path, "line": 2, "col": 1}]}).to_string();
	let (files, symbols, runs) = touched(
		"lsp",
		&json!({"method": "references", "file": &path, "line": 1, "col": 9}),
		&Ok(reply),
	);
	assert_eq!(files, vec![path.clone(), "src/x.rs".to_string()]);
	assert_eq!(
		symbols,
		vec![
			("send".to_string(), Some(path.clone())),
			("send".to_string(), Some("src/x.rs".to_string()))
		]
	);
	assert!(runs.is_empty());

	let reply =
		json!({"symbols": [{"name": "read_frame", "file": "src/lsp.rs", "kind": "function"}]})
			.to_string();
	let (files, symbols, _) = touched(
		"lsp",
		&json!({"method": "symbols", "query": "read_frame"}),
		&Ok(reply),
	);
	assert!(files.is_empty());
	assert_eq!(
		symbols,
		vec![("read_frame".to_string(), Some("src/lsp.rs".to_string()))]
	);

	let text = "src/lsp.rs:329:23  read_frame(&mut reader)\ntests/lsp.rs:13:14  read_frame(&mut r)\n2 match(es)".to_string();
	let (files, symbols, _) = touched(
		"ast_grep",
		&json!({"pattern": "read_frame($$$A)"}),
		&Ok(text),
	);
	assert_eq!(files, ["src/lsp.rs", "tests/lsp.rs"]);
	assert_eq!(symbols.len(), 2);
	assert_eq!(
		symbols[0],
		("read_frame".to_string(), Some("src/lsp.rs".to_string()))
	);

	let (_, _, runs) = touched(
		"compare",
		&json!({"before": "a", "after": "b"}),
		&Ok(String::new()),
	);
	assert_eq!(runs, ["a", "b"]);
	let (_, _, runs) = touched("show", &json!({"name": "a"}), &Err("boom".into()));
	assert_eq!(runs, ["a"]);
	let (files, symbols, runs) = touched("status", &json!({}), &Ok("{}".into()));
	assert!(files.is_empty() && symbols.is_empty() && runs.is_empty());
	let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn record_appends_a_capped_tool_call_event_with_touches() {
	let dir = temp("record");
	std::env::set_var("DAEMON_DIR", &dir);
	let long = "x".repeat(500);
	let reply = json!({"symbols": [{"name": "relative", "file": "src/lsp.rs"}]}).to_string();
	record(
		"lsp",
		&json!({"method": "symbols", "query": long}),
		&Ok(reply.clone()),
		12,
	);
	record("lsp", &json!({"method": "hover"}), &Err("boom".into()), 3);
	let log = std::fs::read_to_string(dir.join("events.jsonl")).unwrap();
	let mut lines = log.lines().rev();
	let last: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
	assert_eq!(last["kind"], "tool_call");
	assert_eq!(last["ok"], false);
	assert_eq!(last["error"], "boom");
	assert_eq!(last["ms"], 3);
	assert_eq!(last["bytes"], 4);
	assert!(last.get("files").is_none() && last.get("symbols").is_none());
	let first: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
	assert_eq!(first["ok"], true);
	assert_eq!(first["bytes"], reply.len());
	assert_eq!(
		first["args"]["query"].as_str().unwrap().chars().count(),
		121
	);
	assert_eq!(
		first["symbols"],
		json!([{"name": "relative", "file": "src/lsp.rs"}])
	);
	let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn audit_cli_uses_jev_and_preserves_evidence_on_service_failure() {
	use std::io::{Read, Write};
	use std::net::TcpListener;
	use std::process::Command;
	use std::time::{Duration, Instant};
	let dir = temp("audit");
	let seed = json!({"id":"broken-build","kind":"compile_failed","time_ms":1,
		"crate":"workspace","build":"cargo build -p app","duration_ms":200,
		"diagnostics":"error: missing symbol"});
	let run = |mode: &str, url: &str, key: bool| {
		// Each call gets identical evidence, independent of the recorder's output.
		std::fs::write(dir.join("events.jsonl"), format!("{seed}\n")).unwrap();
		let mut command = Command::new(env!("CARGO_BIN_EXE_mcp"));
		command
			.env("DAEMON_DIR", &dir)
			.env("JEV_URL", url)
			.env_remove("TYPESAFE_API_KEY");
		if key {
			command.env("TYPESAFE_API_KEY", "fixture-key");
		}
		command
			.args(["--call", "audit", &json!({"mode":mode}).to_string()])
			.output()
			.unwrap()
	};
	let listener = TcpListener::bind("127.0.0.1:0").unwrap();
	let url = format!("http://{}", listener.local_addr().unwrap());
	listener.set_nonblocking(true).unwrap();
	let server = std::thread::spawn(move || {
		let deadline = Instant::now() + Duration::from_secs(10);
		let (mut socket, _) = loop {
			match listener.accept() {
				Ok(pair) => break pair,
				Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => {
					std::thread::sleep(Duration::from_millis(10))
				}
				Err(e) => panic!("fixture accept: {e}"),
			}
		};
		socket
			.set_read_timeout(Some(Duration::from_secs(5)))
			.unwrap();
		let mut request = vec![];
		let mut chunk = [0u8; 4096];
		loop {
			let n = socket.read(&mut chunk).unwrap();
			assert!(n > 0);
			request.extend_from_slice(&chunk[..n]);
			if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
				let headers = String::from_utf8_lossy(&request[..end]);
				let len: usize = headers
					.lines()
					.find_map(|l| {
						let (k, v) = l.split_once(':')?;
						k.eq_ignore_ascii_case("content-length")
							.then(|| v.trim().parse().unwrap())
					})
					.unwrap();
				if request.len() >= end + 4 + len {
					let body: Value = serde_json::from_slice(&request[end + 4..end + 4 + len]).unwrap();
					assert_eq!(
						body["state"]["findings"][0]["evidence"]["event_id"],
						"broken-build"
					);
					assert_eq!(body["questions"]["finding_0"]["type"], "choice");
					break;
				}
			}
		}
		let body = r#"{"answers":{"finding_0":{"choice":"investigate"}}}"#;
		write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
	});
	let output = run("jev", &url, true);
	server.join().unwrap();
	assert!(
		output.status.success(),
		"{}",
		String::from_utf8_lossy(&output.stderr)
	);
	let report: Value = serde_json::from_slice(&output.stdout).unwrap();
	assert_eq!(report["mode"], "jev");
	assert_eq!(report["findings"][0]["jev_assessment"], "investigate");
	// The fixture listener is closed: auto must still return usable local evidence.
	let output = run("auto", &url, true);
	assert!(output.status.success());
	let report: Value = serde_json::from_slice(&output.stdout).unwrap();
	assert_eq!(report["mode"], "local");
	assert!(report["jev_error"].is_string());
	assert_eq!(
		report["findings"][0]["evidence"]["event_id"],
		"broken-build"
	);
	let output = run("local", &url, false);
	assert!(output.status.success());
	assert!(!run("jev", &url, false).status.success());
	std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn compiler_event_pages_do_not_lose_equal_timestamps_or_partial_lines() {
	let log = "{\"kind\":\"tool_call\",\"time_ms\":1}\n{\"kind\":\"compile_failed\",\"time_ms\":1,\"id\":\"a\"}\n{\"kind\":\"compile_failed\",\"time_ms\":1,\"id\":\"b\"}\n{\"kind\":";
	let first = mcp::events::select(log, &serde_json::json!({"cursor":0,"limit":1}));
	assert_eq!(first["events"][0]["id"], "a");
	let second = mcp::events::select(
		log,
		&serde_json::json!({"cursor":first["next_cursor"],"limit":1}),
	);
	assert_eq!(second["events"][0]["id"], "b");
	let last = mcp::events::select(log, &serde_json::json!({"cursor":second["next_cursor"]}));
	assert_eq!(last["next_cursor"], second["next_cursor"]);
	assert_eq!(last["events"], serde_json::json!([]));
	assert_eq!(
		mcp::events::select(log, &serde_json::json!({"kind":"all"}))["events"]
			.as_array()
			.unwrap()
			.len(),
		3
	);
	assert_eq!(
		mcp::events::select(log, &serde_json::json!({"cursor":99999}))["cursor_reset"],
		true
	);
}
