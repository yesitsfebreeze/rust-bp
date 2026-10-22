//! The tiny one-line JSON wire both sockets speak: one request line in, one
//! reply line out. Used by the channel socket, the daemon control socket
//! and their clients.

use crate::platform::{LocalListener as UnixListener, LocalStream as UnixStream};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use std::time::Duration;

/// Bind `path`, refusing when another process still answers there, and
/// removing the stale socket file a dead process left behind. The listener
/// never blocks: Winsock's AF_UNIX accept has no reliable blocking mode, so
/// [`serve_workers`] polls it instead.
pub fn bind(path: &str) -> std::io::Result<UnixListener> {
	if UnixStream::connect(path).is_ok() {
		return Err(std::io::Error::new(
			std::io::ErrorKind::AddrInUse,
			format!("another process still answers at {path}"),
		));
	}
	let _ = std::fs::remove_file(path);
	let listener = UnixListener::bind(path)?;
	listener.set_nonblocking(true)?;
	crate::platform::not_inherited(&listener);
	Ok(listener)
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
	// One acceptor works on Winsock too; bounded queues limit idle clients.
	let handler = Arc::new(handler);
	let (send, receive) = std::sync::mpsc::sync_channel::<UnixStream>(workers * 4);
	let receive = Arc::new(std::sync::Mutex::new(receive));
	for _ in 0..workers {
		let receive = receive.clone();
		let handler = handler.clone();
		std::thread::spawn(move || loop {
			let next = receive.lock().unwrap().recv();
			match next {
				Ok(conn) => answer(conn, &*handler),
				Err(_) => break,
			}
		});
	}
	std::thread::spawn(move || loop {
		match listener.accept() {
			Ok((conn, _)) => {
				crate::platform::not_inherited(&conn);
				// Winsock inherits the listener mode; request reads must block.
				if conn.set_nonblocking(false).is_err() {
					continue;
				}
				// A saturated pool rejects the connection instead of blocking
				// the acceptor.
				if send.try_send(conn).is_err() {
					break;
				}
			}
			Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
				std::thread::sleep(Duration::from_millis(20));
			}
			Err(_) => break,
		}
	});
}

fn answer(mut conn: UnixStream, handler: &(impl Fn(Value) -> Value + ?Sized)) {
	let _ = conn.set_read_timeout(Some(Duration::from_secs(5)));
	let _ = conn.set_write_timeout(Some(Duration::from_secs(5)));
	let Ok(clone) = conn.try_clone() else { return };
	// A duplicated Windows socket is a new handle, inheritable again.
	crate::platform::not_inherited(&clone);
	let mut reader = BufReader::new(clone);
	let mut line = String::new();
	if reader.read_line(&mut line).unwrap_or(0) == 0 {
		return;
	}
	let request = serde_json::from_str::<Value>(line.trim()).unwrap_or_else(|_| json!({}));
	let reply = handler(request);
	let _ = writeln!(conn, "{reply}");
}

/// One request over the wire; the reply is the parsed reply line.
pub fn ask(path: &str, request: Value) -> Result<Value, String> {
	ask_within(path, request, Duration::from_secs(5))
}

/// `ask` with its own read timeout, for ops that wait on a build or an index.
pub fn ask_within(path: &str, request: Value, timeout: Duration) -> Result<Value, String> {
	let mut stream =
		UnixStream::connect(path).map_err(|e| format!("no socket answers at {path}: {e}"))?;
	crate::platform::not_inherited(&stream);
	stream.set_read_timeout(Some(timeout)).ok();
	stream
		.write_all(format!("{request}\n").as_bytes())
		.map_err(|e| format!("{path}: {e}"))?;
	let mut reply = String::new();
	let mut stream = BufReader::new(stream);
	stream
		.read_line(&mut reply)
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
