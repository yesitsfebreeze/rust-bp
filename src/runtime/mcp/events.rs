use daemon::events::{log_path, subscription_path};
use serde_json::{json, Value};

pub fn tool(args: &Value) -> Result<String, String> {
	daemon::ensure_dir().map_err(|e| e.to_string())?;
	let subscription = subscription_path();
	if let Some(thread) = args["subscribe_thread"].as_str() {
		if thread.len() != 36 || !thread.bytes().all(|c| c.is_ascii_hexdigit() || c == b'-') {
			return Err("subscribe_thread must be an explicit Codex session UUID".into());
		}
		std::fs::write(&subscription, thread).map_err(|e| e.to_string())?;
	}
	if args["unsubscribe"].as_bool() == Some(true) {
		match std::fs::remove_file(&subscription) {
			Ok(_) => (),
			Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
			Err(e) => return Err(e.to_string()),
		}
	}
	let log = log_path();
	let text = match std::fs::read_to_string(&log) {
		Ok(text) => text,
		Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
		Err(e) => return Err(e.to_string()),
	};
	let mut reply = select(&text, args);
	reply["subscribed_thread"] = json!(std::fs::read_to_string(&subscription).ok());
	reply["log"] = json!(log.display().to_string());
	Ok(reply.to_string())
}

/// Page complete log lines; the byte cursor distinguishes equal timestamps.
pub fn select(text: &str, args: &Value) -> Value {
	let limit = args["limit"].as_u64().unwrap_or(20).clamp(1, 100) as usize;
	let after = args["after_ms"].as_u64().unwrap_or(0);
	let original_cursor = args["cursor"].as_u64();
	let cursor = original_cursor.map(|c| if c > text.len() as u64 { 0 } else { c });
	let kind = args["kind"].as_str().unwrap_or("compiler");
	let lines: Vec<_> = text.split_inclusive('\n').collect();
	let mut offset = 0u64;
	let mut next_cursor = cursor.unwrap_or(0);
	let mut events = Vec::new();
	for line in lines {
		if !line.ends_with('\n') {
			break;
		}
		offset += line.len() as u64;
		if cursor.is_some_and(|c| offset <= c) {
			continue;
		}
		let Ok(event) = serde_json::from_str::<Value>(line) else {
			next_cursor = offset;
			continue;
		};
		let event_kind = event["kind"].as_str().unwrap_or("");
		let matches = event["time_ms"].as_u64().unwrap_or(0) > after
			&& (kind == "all"
				|| kind == event_kind
				|| (kind == "compiler" && event_kind.starts_with("compile_")))
			&& (!args["errors_only"].as_bool().unwrap_or(false)
				|| event_kind == "compile_failed"
				|| event_kind == "notification_failed");
		next_cursor = offset;
		if matches {
			events.push(event);
		}
		if cursor.is_some() && events.len() >= limit {
			break;
		}
	}
	if cursor.is_none() && events.len() > limit {
		events.drain(..events.len() - limit);
	}
	json!({"events":events,"next_cursor":next_cursor,"cursor_reset":original_cursor.is_some_and(|c| c>text.len() as u64)})
}
