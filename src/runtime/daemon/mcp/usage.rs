//! usage — what the agent asks this MCP for, recorded next to the compiler
//! events in the daemon log. Every tool call lands as a `tool_call` event
//! with the files, symbols and stored runs it touched (`touched` reads them
//! off the arguments and the reply), so the daemon's live graph
//! (`daemon::usage`) can draw tools, files and symbols as one picture.

use crate as daemon;
use daemon::events::publish;
use daemon::usage::{tool_call_data, Symbol};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Longest string kept from arguments and errors in the log.
const VALUE_CAP: usize = 120;

/// Append one `tool_call` event. A failure to log never fails the call.
pub fn record(tool: &str, args: &Value, result: &Result<String, String>, ms: u64) {
	let (files, symbols, runs) = touched(tool, args, result);
	let mut data = tool_call_data(tool, &files, &symbols, &runs);
	data.insert("args".into(), trim(args));
	data.insert("ok".into(), json!(result.is_ok()));
	data.insert("ms".into(), json!(ms));
	let bytes = result.as_ref().map(String::len).unwrap_or_else(String::len);
	data.insert("bytes".into(), json!(bytes));
	if let Err(error) = result {
		data.insert("error".into(), json!(cap(error)));
	}
	let _ = publish("tool_call", data);
}

/// Attach global unresolved failure state after all batch items are recorded.
/// Preserve the tool's success/error outcome and every original result.
pub fn feedback(result: Result<String, String>) -> Result<String, String> {
	let state = match daemon::health::load() {
		Ok(health) => health.report(daemon::health::now_ms()),
		Err(error) => json!({"audit_state_unavailable":error}),
	};
	if state["audit_required"] != true && state.get("audit_state_unavailable").is_none() {
		return result;
	}
	let failed = result.is_err();
	let text = result.unwrap_or_else(|error| error);
	let mut reply = serde_json::from_str::<Value>(&text)
		.ok()
		.filter(Value::is_object)
		.unwrap_or_else(|| json!({"content":text}));
	if state["audit_required"] == true {
		reply["audit_required"] = json!(true);
		reply["audit_state"] = state;
	} else {
		reply["audit_state_unavailable"] = state["audit_state_unavailable"].clone();
	}
	let text = reply.to_string();
	if failed {
		Err(text)
	} else {
		Ok(text)
	}
}

fn cap(text: &str) -> String {
	let mut chars = text.chars();
	let head: String = chars.by_ref().take(VALUE_CAP).collect();
	if chars.next().is_some() {
		format!("{head}…")
	} else {
		head
	}
}

fn trim(value: &Value) -> Value {
	match value {
		Value::String(s) => json!(cap(s)),
		Value::Array(items) => Value::Array(items.iter().map(trim).collect()),
		Value::Object(map) => Value::Object(map.iter().map(|(k, v)| (k.clone(), trim(v))).collect()),
		other => other.clone(),
	}
}

/// `src/x.rs` for a path under the project root, the path as given otherwise.
fn relative(path: &str) -> String {
	let root = daemon::project_root();
	Path::new(path)
		.strip_prefix(&root)
		.map(|p| p.display().to_string())
		.unwrap_or_else(|_| path.to_string())
}

/// The identifier under `line:col` (1-based) in `file`, read from disk; the
/// name behind a `definition` / `references` / `hover` question.
pub fn symbol_at(file: &str, line: usize, col: usize) -> Option<String> {
	let path = PathBuf::from(file);
	let path = if path.is_absolute() {
		path
	} else {
		daemon::project_root().join(path)
	};
	let text = std::fs::read_to_string(path).ok()?;
	let line = text.lines().nth(line.checked_sub(1)?)?;
	let chars: Vec<char> = line.chars().collect();
	let ident = |c: &char| c.is_alphanumeric() || *c == '_';
	let at = col.checked_sub(1)?.min(chars.len().checked_sub(1)?);
	if !ident(&chars[at]) {
		return None;
	}
	let start = (0..=at).rev().take_while(|&i| ident(&chars[i])).last()?;
	let end = (at..chars.len()).take_while(|&i| ident(&chars[i])).last()? + 1;
	Some(chars[start..end].iter().collect())
}

/// The leading identifier of an ast-grep pattern (`read_frame($$$A)` →
/// `read_frame`); none for patterns that start with a metavariable or syntax.
pub fn pattern_symbol(pattern: &str) -> Option<String> {
	let head: String = pattern
		.trim()
		.chars()
		.take_while(|c| c.is_alphanumeric() || *c == '_')
		.collect();
	(!head.is_empty() && !head.chars().next()?.is_ascii_digit()).then_some(head)
}

/// Files, symbols and stored runs one call touched, from its arguments and
/// its reply: `lsp` (the file asked about, every location answered, the
/// symbols listed, the identifier under the cursor), `ast_grep` (every
/// `file:line:col` hit, the pattern's leading identifier), and the run names
/// the valgrind tools take.
pub fn touched(
	tool: &str,
	args: &Value,
	result: &Result<String, String>,
) -> (Vec<String>, Vec<Symbol>, Vec<String>) {
	let mut files: Vec<String> = vec![];
	let mut symbols: Vec<Symbol> = vec![];
	let mut runs: Vec<String> = vec![];
	let reply: Option<Value> = result
		.as_ref()
		.ok()
		.and_then(|text| serde_json::from_str(text).ok());
	match tool {
		"lsp" => {
			let asked = args["file"].as_str().map(relative);
			if let Some(file) = &asked {
				files.push(file.clone());
			}
			if let Some(reply) = &reply {
				for location in reply["locations"].as_array().into_iter().flatten() {
					if let Some(file) = location["file"].as_str() {
						files.push(relative(file));
					}
				}
				for symbol in reply["symbols"].as_array().into_iter().flatten() {
					if let Some(name) = symbol["name"].as_str() {
						symbols.push((name.to_string(), symbol["file"].as_str().map(relative)));
					}
				}
			}
			if let (Some(file), Some(line), Some(col)) = (
				args["file"].as_str(),
				args["line"].as_u64(),
				args["col"].as_u64(),
			) {
				if let Some(name) = symbol_at(file, line as usize, col as usize) {
					// The name sits in the file asked about and in every file answered.
					let mut homes = files.clone();
					homes.sort();
					homes.dedup();
					for home in homes {
						symbols.push((name.clone(), Some(home)));
					}
				}
			}
		}
		"ast_grep" => {
			if let Ok(text) = result {
				for line in text.lines() {
					let mut parts = line.splitn(3, ':');
					if let (Some(file), Some(line_no), Some(_)) = (parts.next(), parts.next(), parts.next()) {
						if file.ends_with(".rs") && line_no.parse::<u64>().is_ok() {
							files.push(relative(file));
						}
					}
				}
			}
			if let Some(name) = args["pattern"].as_str().and_then(pattern_symbol) {
				let mut homes = files.clone();
				homes.sort();
				homes.dedup();
				if homes.is_empty() {
					symbols.push((name.clone(), None));
				}
				for home in homes {
					symbols.push((name.clone(), Some(home)));
				}
			}
		}
		"show" | "rank" | "valgrind_run" | "benchmark" => {
			if let Some(name) = args["name"].as_str() {
				runs.push(name.to_string());
			}
		}
		"compare" => {
			for key in ["before", "after"] {
				if let Some(name) = args[key].as_str() {
					runs.push(name.to_string());
				}
			}
		}
		_ => {}
	}
	files.sort();
	files.dedup();
	symbols.sort();
	symbols.dedup();
	(files, symbols, runs)
}

#[cfg(test)]
mod tests {
	use crate::mcp;
	// The tool-call recorder: what a call touched, and the event it appends.

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
		if std::env::var_os("USAGE_TEST_CHILD").is_none() {
			let status = std::process::Command::new(std::env::current_exe().unwrap())
				.args([
					"--exact",
					"mcp::usage::tests::record_appends_a_capped_tool_call_event_with_touches",
					"--nocapture",
				])
				.env("USAGE_TEST_CHILD", "1")
				.status()
				.unwrap();
			assert!(status.success());
			return;
		}
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
}
