//! The tiny one-line JSON wire both sockets speak: one request line in, one
//! reply line out. Used by the channel socket, the daemon control socket
//! and their clients.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::time::Duration;

/// Bind `path`, refusing when another process still answers there, and
/// removing the stale socket file a dead process left behind.
pub fn bind(path: &str) -> std::io::Result<UnixListener> {
	if UnixStream::connect(path).is_ok() {
		return Err(std::io::Error::new(
			std::io::ErrorKind::AddrInUse,
			format!("another process still answers at {path}"),
		));
	}
	let _ = std::fs::remove_file(path);
	UnixListener::bind(path)
}

/// Serve one connection per thread-less loop: read a request line, answer
/// with `handler`'s reply. An unreadable or unparseable line answers the
/// handler with `{}`.
pub fn serve_loop(
	listener: UnixListener,
	handler: impl Fn(Value) -> Value + Send + Sync + 'static,
) {
	serve_workers(listener, handler, 1);
}

/// Bounded control-socket workers let independent LSP requests overlap without
/// spawning an unbounded thread per connection. Channel sockets remain serial.
pub fn serve_concurrent(
	listener: UnixListener,
	handler: impl Fn(Value) -> Value + Send + Sync + 'static,
) {
	serve_workers(listener, handler, 8);
}

fn serve_workers(
	listener: UnixListener,
	handler: impl Fn(Value) -> Value + Send + Sync + 'static,
	workers: usize,
) {
	let handler = Arc::new(handler);
	for _ in 0..workers {
		let listener = listener.try_clone().expect("clone listener");
		let handler = handler.clone();
		std::thread::spawn(move || {
			for conn in listener.incoming() {
				let Ok(mut conn) = conn else { continue };
				let _ = conn.set_read_timeout(Some(Duration::from_secs(5)));
				let _ = conn.set_write_timeout(Some(Duration::from_secs(5)));
				let mut reader = BufReader::new(conn.try_clone().expect("clone socket"));
				let mut line = String::new();
				if reader.read_line(&mut line).unwrap_or(0) == 0 {
					continue;
				}
				let request = serde_json::from_str::<Value>(line.trim()).unwrap_or_else(|_| json!({}));
				let reply = handler(request);
				let _ = conn.write_all(format!("{reply}\n").as_bytes());
			}
		});
	}
}

/// One request over the wire; the reply is the parsed reply line.
pub fn ask(path: &str, request: Value) -> Result<Value, String> {
	ask_within(path, request, Duration::from_secs(5))
}

/// `ask` with its own read timeout, for ops that wait on a build or an index.
pub fn ask_within(path: &str, request: Value, timeout: Duration) -> Result<Value, String> {
	let mut stream = UnixStream::connect(path).map_err(|_| format!("no socket answers at {path}"))?;
	stream.set_read_timeout(Some(timeout)).ok();
	stream
		.write_all(format!("{request}\n").as_bytes())
		.map_err(|e| format!("{path}: {e}"))?;
	let mut reply = String::new();
	let mut stream = BufReader::new(stream);
	stream
		.read_to_string(&mut reply)
		.map_err(|e| format!("{path}: {e}"))?;
	serde_json::from_str(reply.trim())
		.map_err(|_| format!("{path} answered no JSON: {}", reply.trim()))
}

/// A request against a socket that may be down, phrased for an agent: the
/// reply when it answers, a line that names how to start it when it does not.
pub fn ask_or_guide(path: &str, request: Value, start_hint: &str) -> String {
	match ask(path, request) {
		Ok(reply) => serde_json::to_string_pretty(&reply).unwrap_or_default(),
		Err(_) => format!("no socket answers at {path} — {start_hint}"),
	}
}
