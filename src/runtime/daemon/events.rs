//! Compiler events: an append-only log (`events.jsonl` in the daemon dir)
//! read by the runtime's MCP transport for standard failure notifications.
//!
//! Writers: the daemon records one event per build it runs; the optional
//! `rustc-events` compiler wrapper (`src/runtime/daemon/bin`) records one per
//! crate compile for every build once a developer configures it. Readers: the
//! MCP `events` tool and the runtime, which streams failures to the client.
//!
//! All paths come from [`crate::dir`]; the compiler wrapper derives the
//! project root from `--out-dir`, since Cargo runs it from the compiled
//! crate's own directory.

use serde_json::{json, Map, Value};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn log_path() -> PathBuf {
	crate::dir().join("events.jsonl")
}
fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|d| d.as_millis() as u64)
		.unwrap_or(0)
}

/// 32 hex characters from the OS random source; time and pid when it is unreadable.
#[doc(hidden)]
pub fn fresh_id() -> String {
	let mut bytes = [0u8; 16];
	let random = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut bytes));
	if random.is_err() {
		let seed = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.map(|d| d.as_nanos())
			.unwrap_or(0)
			^ ((std::process::id() as u128) << 64);
		bytes = seed.to_le_bytes();
	}
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Append one event; `data` are the event's own fields. Returns the full event.
pub fn publish(kind: &str, data: Map<String, Value>) -> std::io::Result<Value> {
	crate::ensure_dir()?;
	let mut event = Map::new();
	event.insert("id".into(), json!(fresh_id()));
	event.insert("time_ms".into(), json!(now_ms()));
	event.insert("kind".into(), json!(kind));
	event.extend(data);
	let event = Value::Object(event);
	let mut file =
		crate::platform::private_file(OpenOptions::new().create(true).append(true)).open(log_path())?;
	// One write per line: appends below PIPE_BUF stay whole across writers.
	file.write_all(format!("{event}\n").as_bytes())?;
	Ok(event)
}

/// Reads complete new lines of the log; starts at the current end.
pub struct Tail {
	path: PathBuf,
	offset: u64,
}

impl Tail {
	/// A tail over `path` that starts at the beginning of the file.
	pub fn at(path: PathBuf) -> Self {
		Self { path, offset: 0 }
	}

	pub fn new() -> Self {
		let path = log_path();
		let offset = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
		Self { path, offset }
	}

	/// Events appended since the last call; a truncated log restarts at zero.
	pub fn read_new(&mut self) -> std::io::Result<Vec<Value>> {
		let Ok(mut file) = std::fs::File::open(&self.path) else {
			return Ok(vec![]);
		};
		if file.metadata()?.len() < self.offset {
			self.offset = 0;
		}
		file.seek(SeekFrom::Start(self.offset))?;
		let mut text = String::new();
		file.read_to_string(&mut text)?;
		let mut events = vec![];
		for line in text.split_inclusive('\n') {
			if !line.ends_with('\n') {
				break; // a writer is mid-line; read it next time
			}
			self.offset += line.len() as u64;
			if let Ok(event) = serde_json::from_str::<Value>(line) {
				events.push(event);
			}
		}
		Ok(events)
	}
}

impl Default for Tail {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use crate as daemon;
	use daemon::events::{fresh_id, Tail};
	use std::fs::OpenOptions;
	use std::io::Write;

	#[test]
	fn ids_are_32_hex_and_distinct() {
		let (a, b) = (fresh_id(), fresh_id());
		assert_eq!(a.len(), 32);
		assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
		assert_ne!(a, b);
	}

	#[test]
	fn tail_only_yields_complete_lines_and_survives_truncation() {
		let dir = std::env::temp_dir().join(format!("rt-events-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("events.jsonl");
		std::fs::write(&path, "").unwrap();
		let mut tail = Tail::at(path.clone());
		std::fs::write(&path, "{\"kind\":\"a\"}\n{\"kind\":\"b\"").unwrap();
		let first = tail.read_new().unwrap();
		assert_eq!(first.len(), 1);
		assert_eq!(first[0]["kind"], "a");
		let mut file = OpenOptions::new().append(true).open(&path).unwrap();
		file.write_all(b"}\n").unwrap();
		let second = tail.read_new().unwrap();
		assert_eq!(second.len(), 1);
		assert_eq!(second[0]["kind"], "b");
		std::fs::write(&path, "{\"kind\":\"c\"}\n").unwrap();
		assert_eq!(tail.read_new().unwrap()[0]["kind"], "c");
		let _ = std::fs::remove_dir_all(dir);
	}
}
