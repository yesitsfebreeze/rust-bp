use crate as daemon;
use daemon::events::log_path;
use serde_json::{json, Value};

pub fn tool(args: &Value) -> Result<String, String> {
	daemon::ensure_dir().map_err(|e| e.to_string())?;
	let log = log_path();
	let text = match std::fs::read_to_string(&log) {
		Ok(text) => text,
		Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
		Err(e) => return Err(e.to_string()),
	};
	let mut reply = select(&text, args);
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

#[cfg(test)]
mod tests {
	use crate::mcp;
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
}
