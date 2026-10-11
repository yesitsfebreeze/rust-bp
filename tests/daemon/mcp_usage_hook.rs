use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

struct Fixture(PathBuf);
impl Fixture {
	fn new(name: &str) -> Self {
		let path = std::env::temp_dir().join(format!("mcp-hook-{}-{name}", std::process::id()));
		std::fs::create_dir(&path).unwrap();
		std::fs::write(path.join("Cargo.toml"), "[workspace]\n").unwrap();
		std::fs::write(
			path.join(".mcp.json"),
			r#"{"mcpServers":{"app":{"command":"./mcp.sh"}}}"#,
		)
		.unwrap();
		Self(path)
	}
	fn call(&self, event: &Value) {
		let mut child = Command::new(env!("CARGO_BIN_EXE_mcp-usage-hook"))
			.current_dir(&self.0)
			.env("DAEMON_DIR", self.0.join("log"))
			.stdin(Stdio::piped())
			.stdout(Stdio::piped())
			.stderr(Stdio::piped())
			.spawn()
			.unwrap();
		child
			.stdin
			.take()
			.unwrap()
			.write_all(event.to_string().as_bytes())
			.unwrap();
		let out = child.wait_with_output().unwrap();
		assert!(out.status.success());
		assert!(out.stdout.is_empty(), "hooks must not add model context");
		assert!(
			out.stderr.is_empty(),
			"{}",
			String::from_utf8_lossy(&out.stderr)
		);
	}
	fn log(&self) -> String {
		std::fs::read_to_string(self.0.join("log/events.jsonl")).unwrap_or_default()
	}
}
impl Drop for Fixture {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.0);
	}
}

#[test]
fn host_calls_record_duration_outcome_and_no_payloads() {
	let f = Fixture::new("metadata");
	let mut event = json!({
		"session_id":"session-one", "tool_use_id":"call-one",
		"tool_name":"mcp__git_fs__read", "hook_event_name":"PreToolUse",
		"tool_input":{"path":"private-secret.rs", "token":"never-log-this"}
	});
	f.call(&event);
	event["hook_event_name"] = json!("PostToolUse");
	event["tool_response"] = json!({"isError":false,"content":[{"text":"private response"}]});
	f.call(&event);
	let events: Vec<Value> = f
		.log()
		.lines()
		.map(|l| serde_json::from_str(l).unwrap())
		.collect();
	assert_eq!(events.len(), 1);
	assert_eq!(events[0]["tool"], "mcp__git_fs__read");
	assert_eq!(events[0]["source"], "codex-hook");
	assert_eq!(events[0]["ok"], true);
	assert!(events[0]["ms"].is_u64());
	assert!(!f.log().contains("private"));
	assert!(!f.log().contains("never-log"));
	assert_eq!(
		std::fs::read_dir(f.0.join("log/mcp-pending"))
			.unwrap()
			.count(),
		0
	);

	event["tool_use_id"] = json!("without-pre");
	event["tool_response"] = json!({"isError":true});
	f.call(&event);
	let last: Value = serde_json::from_str(f.log().lines().last().unwrap()).unwrap();
	assert_eq!(last["ok"], false);
	assert!(
		last.get("ms").is_none(),
		"unknown duration must not become zero"
	);
}

#[test]
fn local_mcp_is_not_double_counted_and_session_ids_isolate_timing() {
	let f = Fixture::new("scope");
	for name in ["mcp__app__status", "Bash", "functions.exec"] {
		f.call(&json!({"session_id":"s","tool_use_id":"c","tool_name":name,"hook_event_name":"PostToolUse","tool_response":{}}));
	}
	assert!(f.log().is_empty());
	for session in ["one", "two"] {
		f.call(&json!({"session_id":session,"tool_use_id":"same","tool_name":"mcp__taskview__task","hook_event_name":"PreToolUse"}));
	}
	assert_eq!(
		std::fs::read_dir(f.0.join("log/mcp-pending"))
			.unwrap()
			.count(),
		2
	);
	for session in ["one", "two"] {
		f.call(&json!({"session_id":session,"tool_use_id":"same","tool_name":"mcp__taskview__task","hook_event_name":"PostToolUse","tool_response":{}}));
	}
	assert_eq!(f.log().lines().count(), 2);
	assert_eq!(
		std::fs::read_dir(f.0.join("log/mcp-pending"))
			.unwrap()
			.count(),
		0
	);
}
