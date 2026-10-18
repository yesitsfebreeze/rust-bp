//! Every operating-system difference the framework has, in one place, so the
//! rest of the crate stays portable: local sockets, owned process trees,
//! detached daemons, nonblocking pipes, owner-only files, termination
//! signals, process liveness, the local clock and the terminal.
//!
//! Unix (Linux, macOS) uses the POSIX calls the framework was written
//! against. Windows uses AF_UNIX sockets (Windows 10 1803 and later, so
//! socket paths keep their meaning), Job Objects for process trees and
//! process creation flags for detaching.

use std::io;
use std::process::{Child, Command};
#[cfg(windows)]
use std::time::Duration;

// ---- local sockets ----

#[cfg(unix)]
pub use std::os::unix::net::{UnixListener as LocalListener, UnixStream as LocalStream};
#[cfg(windows)]
pub use uds_windows::{UnixListener as LocalListener, UnixStream as LocalStream};

/// Keep a socket out of child processes. Unix sockets from std are already
/// close-on-exec; Windows AF_UNIX sockets are inheritable by default, and a
/// child holding a connection keeps it open after we close our end, so the
/// peer never sees the reply end.
#[cfg(unix)]
pub fn not_inherited<T>(_socket: &T) {}

#[cfg(windows)]
pub fn not_inherited(socket: &impl std::os::windows::io::AsRawSocket) {
	use windows_sys::Win32::Foundation::{SetHandleInformation, HANDLE_FLAG_INHERIT};
	// SAFETY: the socket handle is live for the call.
	unsafe { SetHandleInformation(socket.as_raw_socket() as _, HANDLE_FLAG_INHERIT, 0) };
}

// ---- owned process trees ----

/// The OS handle on a spawned child's process tree: its process group on
/// Unix, a Job Object on Windows. [`crate::util::ProcessTree`] pairs it with
/// the child.
pub struct Group {
	#[cfg(windows)]
	job: windows::Job,
}

/// Spawn `command` as the leader of a new process tree.
pub fn spawn_group(command: &mut Command) -> io::Result<(Child, Group)> {
	#[cfg(unix)]
	{
		use std::os::unix::process::CommandExt;
		command
			.process_group(0)
			.spawn()
			.map(|child| (child, Group {}))
	}
	#[cfg(windows)]
	{
		use std::os::windows::process::CommandExt;
		let job = windows::Job::new()?;
		let mut child = command
			.creation_flags(windows::CREATE_NEW_PROCESS_GROUP | windows::CREATE_NO_WINDOW)
			.spawn()?;
		if let Err(error) = job.assign(&child) {
			let _ = child.kill();
			let _ = child.wait();
			return Err(error);
		}
		Ok((child, Group { job }))
	}
}

impl Group {
	/// Ask the tree led by `pid` to stop: SIGTERM to the group, Ctrl-Break on
	/// Windows (console programs receive it, GUI programs ignore it).
	pub fn request_stop(&self, pid: u32) {
		#[cfg(unix)]
		unsafe {
			libc::kill(-(pid as i32), libc::SIGTERM);
		}
		#[cfg(windows)]
		windows::ctrl_break(pid);
	}

	/// Kill every process of the tree — only the group or job this Group
	/// created, never the caller's own.
	pub fn kill(&self, pid: u32) {
		#[cfg(unix)]
		unsafe {
			libc::kill(-(pid as i32), libc::SIGKILL);
		}
		#[cfg(windows)]
		{
			let _ = pid;
			self.job.kill();
		}
	}
}

/// Resident memory and platform-specific counters, in bytes. These are
/// process accounting values, not a leak detector.
pub fn memory_usage(pid: u32) -> Result<serde_json::Value, String> {
	#[cfg(windows)]
	{
		use windows_sys::Win32::{
			Foundation::CloseHandle,
			System::{
				ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX},
				Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ},
			},
		};
		// SAFETY: the handle is checked and closed on every path; the counters
		// buffer has the size and alignment required by GetProcessMemoryInfo.
		unsafe {
			let handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, pid);
			if handle.is_null() {
				return Err(io::Error::last_os_error().to_string());
			}
			let mut counters: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
			counters.cb = std::mem::size_of_val(&counters) as u32;
			let ok = GetProcessMemoryInfo(handle, &mut counters as *mut _ as _, counters.cb);
			let error = io::Error::last_os_error();
			CloseHandle(handle);
			if ok == 0 {
				return Err(error.to_string());
			}
			Ok(serde_json::json!({"source":"GetProcessMemoryInfo",
				"resident_bytes":counters.WorkingSetSize,
				"peak_resident_bytes":counters.PeakWorkingSetSize,
				"private_bytes":counters.PrivateUsage}))
		}
	}
	#[cfg(target_os = "linux")]
	{
		let text = std::fs::read_to_string(format!("/proc/{pid}/status")).map_err(|e| e.to_string())?;
		let bytes = |key: &str| {
			text.lines().find_map(|line| {
				line
					.strip_prefix(key)?
					.split_whitespace()
					.next()?
					.parse::<u64>()
					.ok()?
					.checked_mul(1024)
			})
		};
		let resident = bytes("VmRSS:").ok_or("process has no resident memory (it may have exited)")?;
		Ok(
			serde_json::json!({"source":"/proc/pid/status", "resident_bytes":resident,
			"peak_resident_bytes":bytes("VmHWM:"), "virtual_bytes":bytes("VmSize:")}),
		)
	}
	#[cfg(target_os = "macos")]
	{
		let text = crate::util::run_with_timeout(
			&[
				"ps".into(),
				"-o".into(),
				"rss=,vsz=".into(),
				"-p".into(),
				pid.to_string(),
			],
			std::time::Duration::from_secs(5),
			4096,
		)?;
		let values: Vec<u64> = text
			.split_whitespace()
			.filter_map(|v| v.parse().ok())
			.collect();
		if values.len() != 2 {
			return Err("process has no memory counters".into());
		}
		Ok(
			serde_json::json!({"source":"ps", "resident_bytes":values[0] * 1024,
			"virtual_bytes":values[1] * 1024}),
		)
	}
}

// ---- detached daemons ----

/// Start `command` outside our session / console, so it outlives the
/// terminal that launched it.
pub fn detach(command: &mut Command) {
	#[cfg(unix)]
	{
		use std::os::unix::process::CommandExt;
		// SAFETY: setsid only detaches the child from our session and terminal.
		unsafe {
			command.pre_exec(|| {
				libc::setsid();
				Ok(())
			});
		}
	}
	#[cfg(windows)]
	{
		use std::os::windows::process::CommandExt;
		command.creation_flags(windows::DETACHED_PROCESS | windows::CREATE_NEW_PROCESS_GROUP);
	}
}

// ---- pipes ----

/// Make a child's output pipe nonblocking where the OS allows it, so a
/// detached descendant holding the pipe cannot hang a reader past its
/// deadline. Windows reads stay blocking: killing the Job closes every
/// writer, which ends the read.
#[cfg(unix)]
pub fn nonblocking(stream: &impl std::os::fd::AsRawFd) {
	unsafe {
		let flags = libc::fcntl(stream.as_raw_fd(), libc::F_GETFL);
		assert!(
			flags >= 0 && libc::fcntl(stream.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) >= 0
		);
	}
}

#[cfg(windows)]
pub fn nonblocking<T>(_stream: &T) {}

// ---- owner-only files ----

/// The daemon dir is private to the user (0700 on Unix; Windows already
/// keeps the per-user temp dir private).
pub fn private_dir(builder: &mut std::fs::DirBuilder) -> &mut std::fs::DirBuilder {
	#[cfg(unix)]
	{
		use std::os::unix::fs::DirBuilderExt;
		builder.mode(0o700)
	}
	#[cfg(windows)]
	builder
}

/// Files only the user may read (0600 on Unix).
pub fn private_file(options: &mut std::fs::OpenOptions) -> &mut std::fs::OpenOptions {
	#[cfg(unix)]
	{
		use std::os::unix::fs::OpenOptionsExt;
		options.mode(0o600)
	}
	#[cfg(windows)]
	options
}

// ---- signals ----

/// Signals that request a clean stop: SIGHUP (terminal closed) exists on Unix only.
#[cfg(unix)]
pub const TERMINATION_SIGNALS: &[i32] = &[libc::SIGTERM, libc::SIGINT, libc::SIGHUP];
#[cfg(windows)]
pub const TERMINATION_SIGNALS: &[i32] =
	&[signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT];

/// The exit code a shell reports for a process: its code, or 128 + the
/// killing signal on Unix.
pub fn exit_code(status: std::process::ExitStatus) -> i32 {
	#[cfg(unix)]
	{
		use std::os::unix::process::ExitStatusExt;
		status
			.code()
			.unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
	}
	#[cfg(windows)]
	status.code().unwrap_or(1)
}

// ---- liveness ----

/// Whether a process with this pid is still running.
pub fn alive(pid: u32) -> bool {
	#[cfg(unix)]
	unsafe {
		libc::kill(pid as i32, 0) == 0
	}
	#[cfg(windows)]
	windows::alive(pid)
}

// ---- clock ----

/// Local wall-clock `HH:MM:SS` for a Unix time in milliseconds, with the
/// host timezone's daylight-saving rules.
pub fn local_hms(time_ms: u64) -> Option<(u32, u32, u32)> {
	#[cfg(unix)]
	{
		let seconds = (time_ms / 1000) as libc::time_t;
		// SAFETY: an all-zero tm is valid storage for localtime_r to fill.
		let mut local: libc::tm = unsafe { std::mem::zeroed() };
		// SAFETY: localtime_r reads a valid time_t and writes the supplied tm.
		if unsafe { libc::localtime_r(&seconds, &mut local) }.is_null() {
			return None;
		}
		Some((
			local.tm_hour as u32,
			local.tm_min as u32,
			local.tm_sec as u32,
		))
	}
	#[cfg(windows)]
	windows::local_hms(time_ms)
}

// ---- terminal ----

/// Columns and rows of the terminal on stdout.
pub fn terminal_size() -> Option<(usize, usize)> {
	#[cfg(unix)]
	{
		let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
		// SAFETY: TIOCGWINSZ fills a winsize; a failure leaves it zeroed.
		if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) } != 0 || ws.ws_col == 0
		{
			return None;
		}
		Some((ws.ws_col as usize, ws.ws_row as usize))
	}
	#[cfg(windows)]
	windows::terminal_size()
}

/// Raw, unechoed keyboard input with VT escape sequences, restored on drop.
/// Reads through [`RawTerminal::read`] return within about 100 ms when no
/// key is pressed.
pub struct RawTerminal {
	#[cfg(unix)]
	saved: libc::termios,
	#[cfg(windows)]
	saved: windows::ConsoleModes,
	#[cfg(windows)]
	keys: std::sync::mpsc::Receiver<Vec<u8>>,
}

impl RawTerminal {
	pub fn enter() -> Option<RawTerminal> {
		#[cfg(unix)]
		{
			let mut t: libc::termios = unsafe { std::mem::zeroed() };
			// SAFETY: tcgetattr writes a termios for a tty fd and fails otherwise.
			if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut t) } != 0 {
				return None;
			}
			let saved = t;
			t.c_lflag &= !(libc::ECHO | libc::ICANON | libc::ISIG | libc::IEXTEN);
			t.c_iflag &= !(libc::IXON | libc::ICRNL);
			t.c_cc[libc::VMIN] = 0;
			t.c_cc[libc::VTIME] = 1;
			// SAFETY: a termios we read and adjusted, applied to the same fd.
			unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t) };
			Some(RawTerminal { saved })
		}
		#[cfg(windows)]
		{
			let saved = windows::ConsoleModes::raw()?;
			// Console reads block; a reader thread makes them pollable.
			let (send, keys) = std::sync::mpsc::channel();
			std::thread::spawn(move || {
				use std::io::Read;
				let mut stdin = io::stdin();
				let mut buf = [0u8; 64];
				while let Ok(n) = stdin.read(&mut buf) {
					if n == 0 || send.send(buf[..n].to_vec()).is_err() {
						break;
					}
				}
			});
			Some(RawTerminal { saved, keys })
		}
	}

	/// Pressed bytes, or an empty read after about 100 ms.
	pub fn read(&mut self, buf: &mut [u8]) -> usize {
		#[cfg(unix)]
		{
			use std::io::Read;
			io::stdin().lock().read(buf).unwrap_or(0)
		}
		#[cfg(windows)]
		match self.keys.recv_timeout(Duration::from_millis(100)) {
			Ok(bytes) => {
				let n = bytes.len().min(buf.len());
				buf[..n].copy_from_slice(&bytes[..n]);
				n
			}
			Err(_) => 0,
		}
	}
}

impl Drop for RawTerminal {
	fn drop(&mut self) {
		#[cfg(unix)]
		// SAFETY: the termios tcgetattr gave us, restored on the same fd.
		unsafe {
			libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.saved)
		};
		#[cfg(windows)]
		self.saved.restore();
	}
}

#[cfg(windows)]
mod windows {
	use std::io;
	use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, SYSTEMTIME};
	use windows_sys::Win32::System::Console::{
		GenerateConsoleCtrlEvent, GetConsoleMode, GetConsoleScreenBufferInfo, GetStdHandle,
		SetConsoleMode, CONSOLE_MODE, CONSOLE_SCREEN_BUFFER_INFO, CTRL_BREAK_EVENT, ENABLE_ECHO_INPUT,
		ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT, ENABLE_VIRTUAL_TERMINAL_INPUT,
		ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
	};
	use windows_sys::Win32::System::JobObjects::{
		AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
		SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
		JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
	};
	use windows_sys::Win32::System::Threading::{
		GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
	};
	use windows_sys::Win32::System::Time::{FileTimeToSystemTime, SystemTimeToTzSpecificLocalTime};

	pub const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
	pub const DETACHED_PROCESS: u32 = 0x0000_0008;
	pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
	const STILL_ACTIVE: u32 = 259;

	/// A Job Object that kills its processes when closed or killed.
	pub struct Job(HANDLE);

	// SAFETY: a job handle may be used and closed from any thread.
	unsafe impl Send for Job {}
	unsafe impl Sync for Job {}

	impl Job {
		pub fn new() -> io::Result<Job> {
			// SAFETY: an anonymous job with default security.
			let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
			if handle.is_null() {
				return Err(io::Error::last_os_error());
			}
			let job = Job(handle);
			// SAFETY: zeroed limit information is valid; only the flag is set.
			let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
			limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
			let ok = unsafe {
				SetInformationJobObject(
					job.0,
					JobObjectExtendedLimitInformation,
					&limits as *const _ as *const _,
					std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
				)
			};
			if ok == 0 {
				return Err(io::Error::last_os_error());
			}
			Ok(job)
		}

		pub fn assign(&self, child: &std::process::Child) -> io::Result<()> {
			use std::os::windows::io::AsRawHandle;
			// SAFETY: both handles are live for the call.
			if unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle() as HANDLE) } == 0 {
				return Err(io::Error::last_os_error());
			}
			Ok(())
		}

		pub fn kill(&self) {
			// SAFETY: the job handle is live until Drop.
			unsafe { TerminateJobObject(self.0, 1) };
		}
	}

	impl Drop for Job {
		fn drop(&mut self) {
			// SAFETY: closing the last handle also kills the job (KILL_ON_JOB_CLOSE).
			unsafe { CloseHandle(self.0) };
		}
	}

	pub fn ctrl_break(pid: u32) {
		// SAFETY: fails harmlessly when the process shares no console with us.
		unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid) };
	}

	pub fn alive(pid: u32) -> bool {
		// SAFETY: query-only access; the handle is closed before returning.
		unsafe {
			let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
			if handle.is_null() {
				return false;
			}
			let mut code = 0u32;
			let ok = GetExitCodeProcess(handle, &mut code) != 0;
			CloseHandle(handle);
			ok && code == STILL_ACTIVE
		}
	}

	pub fn local_hms(time_ms: u64) -> Option<(u32, u32, u32)> {
		// FILETIME counts 100 ns intervals since 1601-01-01.
		let ticks = (time_ms + 11_644_473_600_000) * 10_000;
		let file = FILETIME {
			dwLowDateTime: ticks as u32,
			dwHighDateTime: (ticks >> 32) as u32,
		};
		// SAFETY: plain conversions between caller-owned structs.
		unsafe {
			let mut utc: SYSTEMTIME = std::mem::zeroed();
			let mut local: SYSTEMTIME = std::mem::zeroed();
			if FileTimeToSystemTime(&file, &mut utc) == 0
				|| SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut local) == 0
			{
				return None;
			}
			Some((
				local.wHour as u32,
				local.wMinute as u32,
				local.wSecond as u32,
			))
		}
	}

	pub fn terminal_size() -> Option<(usize, usize)> {
		// SAFETY: the info struct is caller-owned; failure leaves it unused.
		unsafe {
			let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
			if GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &mut info) == 0 {
				return None;
			}
			let w = info.srWindow.Right - info.srWindow.Left + 1;
			let h = info.srWindow.Bottom - info.srWindow.Top + 1;
			(w > 0 && h > 0).then_some((w as usize, h as usize))
		}
	}

	/// Saved console modes of stdin and stdout.
	pub struct ConsoleModes {
		input: CONSOLE_MODE,
		output: CONSOLE_MODE,
	}

	impl ConsoleModes {
		/// Raw VT input (no line editing, echo or Ctrl-C processing) and VT
		/// output; None when stdin is no console.
		pub fn raw() -> Option<ConsoleModes> {
			// SAFETY: modes are read from and written to the process's std handles.
			unsafe {
				let (stdin, stdout) = (
					GetStdHandle(STD_INPUT_HANDLE),
					GetStdHandle(STD_OUTPUT_HANDLE),
				);
				let (mut input, mut output) = (0, 0);
				if GetConsoleMode(stdin, &mut input) == 0 || GetConsoleMode(stdout, &mut output) == 0 {
					return None;
				}
				let raw_in = (input & !(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT))
					| ENABLE_VIRTUAL_TERMINAL_INPUT;
				SetConsoleMode(stdin, raw_in);
				SetConsoleMode(stdout, output | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
				Some(ConsoleModes { input, output })
			}
		}

		pub fn restore(&self) {
			// SAFETY: restores the modes read in raw().
			unsafe {
				SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), self.input);
				SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), self.output);
			}
		}
	}
}
