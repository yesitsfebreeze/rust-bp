//! Small process helpers shared by the daemon and the valgrind runner.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A process tree owned by this operation (process group on Unix, Job Object
/// on Windows: [`crate::platform::Group`]). Drop cleans descendants even when
/// the leader has already exited (including error and cancellation paths).
pub struct ProcessTree(
	pub std::process::Child,
	crate::platform::Group,
	Option<crate::staging::Executable>,
);
impl ProcessTree {
	pub fn spawn(command: &mut Command) -> std::io::Result<Self> {
		let _ = cancelled(); // install handlers before creating owned children
		crate::platform::spawn_group(command).map(|(child, group)| Self(child, group, None))
	}

	/// Remove a staged executable only after this process has exited.
	pub fn keep_executable(&mut self, executable: Option<crate::staging::Executable>) {
		self.2 = executable;
	}
	/// Ask the tree to stop and wait up to `grace` for the leader to exit, so
	/// it can clean up (containers, sockets); Drop kills whatever remains.
	pub fn terminate(&mut self, grace: Duration) {
		self.1.request_stop(self.0.id());
		let start = Instant::now();
		while start.elapsed() < grace {
			if let Ok(Some(_)) = self.0.try_wait() {
				return;
			}
			std::thread::sleep(Duration::from_millis(20));
		}
	}
}
impl Drop for ProcessTree {
	fn drop(&mut self) {
		self.1.kill(self.0.id());
		let _ = self.0.wait();
	}
}

static CANCEL_SIGNAL: std::sync::OnceLock<std::sync::Arc<std::sync::atomic::AtomicUsize>> =
	std::sync::OnceLock::new();

/// The signal (SIGTERM, SIGINT or SIGHUP) that requested cancellation; 0 before any.
pub fn cancel_signal() -> i32 {
	let _ = cancelled();
	CANCEL_SIGNAL
		.get()
		.map(|s| s.load(std::sync::atomic::Ordering::Relaxed))
		.unwrap_or(0) as i32
}

/// SIGINT/SIGTERM (and SIGHUP on Unix) request cancellation so RAII cleanup
/// still runs; SIGHUP is the terminal closing, which must not orphan the app child.
pub fn cancelled() -> bool {
	use std::sync::{
		atomic::{AtomicBool, Ordering},
		Arc, OnceLock,
	};
	static STOP: OnceLock<Arc<AtomicBool>> = OnceLock::new();
	STOP
		.get_or_init(|| {
			let flag = Arc::new(AtomicBool::new(false));
			let which = CANCEL_SIGNAL
				.get_or_init(|| Arc::new(std::sync::atomic::AtomicUsize::new(0)))
				.clone();
			for &signal in crate::platform::TERMINATION_SIGNALS {
				signal_hook::flag::register(signal, flag.clone()).expect("register termination handler");
				signal_hook::flag::register_usize(signal, which.clone(), signal as usize)
					.expect("register termination handler");
			}
			flag
		})
		.load(Ordering::Relaxed)
}

/// Run a command with a deadline covering both process exit and pipe draining.
pub fn run_with_timeout(
	command: &[String],
	timeout: Duration,
	keep: usize,
) -> Result<String, String> {
	run_bounded(command, timeout, keep, true, &|| false)
}

/// Cleanup must still execute after cancellation was requested.
pub fn run_cleanup(command: &[String]) -> Result<String, String> {
	run_bounded(command, Duration::from_secs(10), 2000, false, &|| false)
}

pub fn run_with_timeout_until(
	command: &[String],
	timeout: Duration,
	keep: usize,
	stop: &dyn Fn() -> bool,
) -> Result<String, String> {
	run_bounded(command, timeout, keep, true, stop)
}

fn run_bounded(
	command: &[String],
	timeout: Duration,
	keep: usize,
	interruptible: bool,
	stop: &dyn Fn() -> bool,
) -> Result<String, String> {
	let (head, tail) = command.split_first().ok_or("empty command")?;
	if interruptible && (cancelled() || stop()) {
		return Err("cancelled".into());
	}
	let mut child = ProcessTree::spawn(
		Command::new(head)
			.args(tail)
			.stdout(Stdio::piped())
			.stderr(Stdio::piped()),
	)
	.map_err(|e| format!("cannot run {head}: {e}"))?;
	// Nonblocking readers also terminate when a detached descendant retains a
	// pipe. A deadline must not turn into an unbounded JoinHandle::join.
	fn drain(
		mut stream: impl Read + Send + 'static,
		keep: usize,
		done: std::sync::Arc<std::sync::atomic::AtomicBool>,
	) -> std::thread::JoinHandle<Vec<u8>> {
		use std::sync::atomic::Ordering;
		std::thread::spawn(move || {
			let mut saved = Vec::new();
			let mut buffer = [0; 8192];
			while !done.load(Ordering::Relaxed) {
				match stream.read(&mut buffer) {
					Ok(0) => break,
					Ok(count) => {
						let retain = count.min(keep.saturating_sub(saved.len()));
						saved.extend_from_slice(&buffer[..retain]);
					}
					Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
						std::thread::sleep(Duration::from_millis(5))
					}
					Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
					Err(_) => break,
				}
			}
			saved
		})
	}
	let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
	let (out, err) = (
		child.0.stdout.take().unwrap(),
		child.0.stderr.take().unwrap(),
	);
	crate::platform::nonblocking(&out);
	crate::platform::nonblocking(&err);
	let stdout = drain(out, keep, done.clone());
	let stderr = drain(err, keep, done.clone());
	let started = Instant::now();
	let result = loop {
		if interruptible && (cancelled() || stop()) {
			break Err("cancelled".into());
		}
		if started.elapsed() >= timeout {
			break Err(format!("timed out after {:.3} s", timeout.as_secs_f64()));
		}
		match child.0.try_wait() {
			Ok(Some(status)) if stdout.is_finished() && stderr.is_finished() => break Ok(status),
			Ok(_) => std::thread::sleep(Duration::from_millis(5)),
			Err(e) => break Err(e.to_string()),
		}
	};
	drop(child);
	if result.is_err() {
		done.store(true, std::sync::atomic::Ordering::Relaxed);
	}
	let mut text = String::from_utf8_lossy(&stdout.join().unwrap_or_default()).into_owned();
	text.push_str(&String::from_utf8_lossy(&stderr.join().unwrap_or_default()));
	let diagnostic = || text.trim().chars().take(keep.min(1500)).collect::<String>();
	let status = result.map_err(|error| {
		let diagnostic = diagnostic();
		if diagnostic.is_empty() {
			error
		} else {
			format!("{error}: {diagnostic}")
		}
	})?;
	if status.success() {
		Ok(text.trim().chars().take(keep).collect())
	} else {
		Err(format!("{status}: {}", diagnostic()))
	}
}

/// Run `command`, return only whether it succeeded (used to probe docker).
pub fn succeeds(command: &[String]) -> bool {
	run_with_timeout(command, Duration::from_secs(10), 0).is_ok()
}
