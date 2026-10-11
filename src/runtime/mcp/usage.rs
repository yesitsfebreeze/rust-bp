//! usage — what the agent asks this MCP for, recorded next to the compiler
//! events in the daemon log. Every tool call lands as a `tool_call` event
//! with the files, symbols and stored runs it touched (`touched` reads them
//! off the arguments and the reply), so the daemon's live graph
//! (`daemon::usage`) can draw tools, files and symbols as one picture.

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
