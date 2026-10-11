//! Minimal Codex Pre/PostToolUse recorder. No arguments or responses are logged.
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis() as u64
}

fn record(input: &Value) -> Result<(), Box<dyn std::error::Error>> {
	let name = input["tool_name"].as_str().unwrap_or("");
	if !name.starts_with("mcp__") {
		return Ok(());
	}
	// Calls to our own MCP already carry richer instrumentation. Keep one record.
	let config: Value = std::fs::read_to_string(daemon::project_root().join(".mcp.json"))
		.ok()
		.and_then(|s| serde_json::from_str(&s).ok())
		.unwrap_or(Value::Null);
	if config["mcpServers"].as_object().is_some_and(|servers| {
		servers.iter().any(|(server, config)| {
			config["command"]
				.as_str()
				.is_some_and(|c| c.ends_with("/mcp.sh"))
				&& name.starts_with(&format!("mcp__{server}__"))
		})
	}) {
		return Ok(());
	}
	let session = input["session_id"].as_str().unwrap_or("");
	let id = input["tool_use_id"].as_str().unwrap_or("");
	if session.is_empty() || id.is_empty() {
		return Ok(());
	}
	let dir = daemon::ensure_dir()?.join("mcp-pending");
	std::fs::create_dir_all(&dir)?;
	let mut hash = DefaultHasher::new();
	(session, id).hash(&mut hash);
	let pending = dir.join(format!("{:016x}", hash.finish()));
	match input["hook_event_name"].as_str() {
		Some("PreToolUse") => {
			use std::io::Write;
			let mut file = std::fs::OpenOptions::new()
				.write(true)
				.create(true)
				.truncate(true)
				.mode(0o600)
				.open(pending)?;
			writeln!(file, "{}", now_ms())?;
		}
		Some("PostToolUse") => {
			let started = std::fs::read_to_string(&pending)
				.ok()
				.and_then(|s| s.trim().parse::<u64>().ok());
			let _ = std::fs::remove_file(pending);
			let response = &input["tool_response"];
			let response = response
				.as_str()
				.and_then(|s| serde_json::from_str::<Value>(s).ok())
				.unwrap_or_else(|| response.clone());
			let failed = response["isError"] == true
				|| response["result"]["isError"] == true
				|| response.get("error").is_some_and(|v| !v.is_null());
			let mut data = json!({
				"tool": name, "source": "codex-hook", "session_id": session,
				"tool_use_id": id, "detail": "Host MCP call; arguments and response omitted.",
			})
			.as_object()
			.unwrap()
			.clone();
			if !response.is_null() {
				data.insert("ok".into(), json!(!failed));
			}
			if let Some(started) = started {
				data.insert("ms".into(), json!(now_ms().saturating_sub(started)));
			}
			if failed {
				data.insert(
					"error".into(),
					json!("MCP call reported an error; inspect the original tool response."),
				);
			}
			daemon::events::publish("tool_call", data)?;
		}
		_ => {}
	}
	Ok(())
}

fn main() {
	// Hook input can include large tool responses. Only metadata is persisted.
	let mut text = String::new();
	if std::io::stdin().read_to_string(&mut text).is_err() {
		return;
	}
	if let Ok(input) = serde_json::from_str::<Value>(&text) {
		if let Err(error) = record(&input) {
			eprintln!("MCP usage recording: {error}");
		}
	}
}
