//! The tiny one-line JSON wire both sockets speak: one request line in, one
//! reply line out. Used by the channel socket, the daemon control socket
//! and their clients.

use crate::platform::{LocalListener as UnixListener, LocalStream as UnixStream};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::sync::{
	atomic::{AtomicBool, Ordering},
	Arc,
};
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
	let listener = UnixListener::bind(path)?;
	crate::platform::not_inherited(&listener);
	listener.set_nonblocking(true)?;
	Ok(listener)
}

/// Serve connections serially: read a request line, answer
/// with `handler`'s reply. An unreadable or unparseable line answers the
/// handler with `{}`.
pub fn serve_loop(
	listener: UnixListener,
	handler: impl Fn(Value) -> Value + Send + Sync + 'static,
) -> Server {
	serve_workers(listener, handler, 1)
}

/// Bounded control-socket workers let independent LSP requests overlap without
/// spawning an unbounded thread per connection. Channel sockets remain serial.
pub fn serve_concurrent(
	listener: UnixListener,
	handler: impl Fn(Value) -> Value + Send + Sync + 'static,
) -> Server {
	serve_workers(listener, handler, 8)
}

/// Owns the listener and workers. Dropping it closes the endpoint and releases
/// the handler's resources, including any language-server process.
pub struct Server {
	stop: Arc<AtomicBool>,
	threads: Vec<std::thread::JoinHandle<()>>,
}

impl Drop for Server {
	fn drop(&mut self) {
		self.stop.store(true, Ordering::Relaxed);
		for thread in self.threads.drain(..) {
			let _ = thread.join();
		}
	}
}

fn serve_workers(
	listener: UnixListener,
	handler: impl Fn(Value) -> Value + Send + Sync + 'static,
	workers: usize,
) -> Server {
	// One acceptor works on Winsock too; bounded queues limit idle clients.
	let handler = Arc::new(handler);
	let stop = Arc::new(AtomicBool::new(false));
	let mut threads = Vec::new();
	let (send, receive) = std::sync::mpsc::sync_channel::<UnixStream>(workers * 4);
	let receive = Arc::new(std::sync::Mutex::new(receive));
	for _ in 0..workers {
		let receive = receive.clone();
		let handler = handler.clone();
		let stop = stop.clone();
		threads.push(std::thread::spawn(move || {
			while !stop.load(Ordering::Relaxed) {
				let next = receive.lock().unwrap().recv();
				match next {
					Ok(conn) if !stop.load(Ordering::Relaxed) => answer(conn, &*handler),
					_ => break,
				}
			}
		}));
	}
	let stopping = stop.clone();
	threads.push(std::thread::spawn(move || {
		while !stopping.load(Ordering::Relaxed) {
			match listener.accept() {
				Ok((conn, _)) => {
					crate::platform::not_inherited(&conn);
					// Winsock inherits the listener mode; request reads must block.
					if conn.set_nonblocking(false).is_err() {
						continue;
					}
					// A saturated pool rejects the connection instead of blocking shutdown.
					let _ = send.try_send(conn);
				}
				Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
					std::thread::sleep(Duration::from_millis(20));
				}
				Err(_) => break,
			}
		}
	}));
	Server { stop, threads }
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
	let mut stream = UnixStream::connect(path).map_err(|_| format!("no socket answers at {path}"))?;
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

#[cfg(test)]
mod tests {
	use crate as daemon;
	use serde_json::json;
	use std::sync::{Condvar, Mutex};
	use std::time::Duration;
	struct Fixture(std::path::PathBuf);
	impl Fixture {
		fn new() -> Self {
			let dir = std::env::temp_dir().join(format!("wire-{}", crate::events::fresh_id()));
			std::fs::create_dir(&dir).unwrap();
			Self(dir)
		}
	}
	impl Drop for Fixture {
		fn drop(&mut self) {
			let _ = std::fs::remove_dir_all(&self.0);
		}
	}
	#[test]
	fn daemon_socket_services_concurrent_requests() {
		let fixture = Fixture::new();
		// Keep socket paths below macOS's 104-byte limit.
		let path = fixture.0.join("d.sock").to_string_lossy().into_owned();
		let listener = daemon::wire::bind(&path).unwrap();
		let gate = (Mutex::new(0), Condvar::new());
		let server = daemon::wire::serve_concurrent(listener, move |request| {
			let mut arrived = gate.0.lock().unwrap();
			*arrived += 1;
			gate.1.notify_all();
			let (arrived, _) = gate
				.1
				.wait_timeout_while(arrived, Duration::from_secs(3), |n| *n < 3)
				.unwrap();
			json!({"overlapped":*arrived >= 3,"n":request["n"]})
		});
		std::thread::scope(|scope| {
			let calls: Vec<_> = (0..3)
				.map(|n| {
					let path = &path;
					scope.spawn(move || daemon::wire::ask(path, json!({"n":n})).unwrap())
				})
				.collect();
			for (n, call) in calls.into_iter().enumerate() {
				let reply = call.join().unwrap();
				assert_eq!(reply["n"], n);
				assert_eq!(reply["overlapped"], true);
			}
		});
		drop(server);
		assert!(daemon::wire::ask(&path, json!({})).is_err());
		let listener = daemon::wire::bind(&path).unwrap();
		let _restarted = daemon::wire::serve_concurrent(listener, |_| json!({"restarted":true}));
		assert_eq!(
			daemon::wire::ask(&path, json!({})).unwrap()["restarted"],
			true
		);
	}
}
