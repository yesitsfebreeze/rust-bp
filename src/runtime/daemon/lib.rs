//! The hot-reload + profiler framework an agent drives through the
//! `mcp` control server. (module docs: see README.md)
//!
//! Conventions: nothing the framework writes lands inside the project. Every
//! process resolves one daemon dir ([`dir`]) under the OS temp dir, keyed by
//! the project root; the daemon answers `daemon.sock` there, the
//! executable serves its channels at `app.sock`, valgrind runs go to
//! `valgrind/`, compile events to `events.jsonl`, container builds to
//! `target-linux/`. `daemon dir` prints the path; `DAEMON_DIR` overrides it.
//! The daemon also owns one `rust-analyzer` ([`lsp`]), started on the first
//! `lsp` request and kept for the daemon's lifetime.

pub mod channels;
pub mod events;
pub mod health;
pub mod hot;
pub mod lsp;
pub mod native;
pub mod platform;
pub mod profiles;
pub mod staging;
pub mod usage;
pub mod util;
pub mod valgrind;
pub mod watch;
pub mod wire;

use std::path::{Path, PathBuf};

/// The directory every framework process writes into: `DAEMON_DIR` when
/// set, else `<OS temp dir>/sv-<hash of the canonical project root>`. The
/// name is short on purpose: a Unix socket path may not exceed 104 bytes
/// on macOS, and `$TMPDIR` there is already ~50. Deterministic, so the
/// daemon, the executable, the MCP worker and the compiler wrapper
/// meet in the same place without passing it around.
pub fn dir() -> PathBuf {
	match std::env::var_os("DAEMON_DIR") {
		Some(dir) if !dir.is_empty() => PathBuf::from(dir),
		_ => dir_for(&project_root()),
	}
}

/// The daemon dir of the project at `root` (the env override aside).
pub fn dir_for(root: &Path) -> PathBuf {
	let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
	let hash = fnv1a(root.as_os_str().as_encoded_bytes()) & 0xffff_ffff_ffff;
	std::env::temp_dir().join(format!("sv-{hash:012x}"))
}

/// The topmost ancestor of the working directory that holds a `Cargo.toml`:
/// the workspace root even when run from a member crate (Cargo runs tests
/// there). The working directory itself when there is none.
pub fn project_root() -> PathBuf {
	let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
	cwd
		.ancestors()
		.filter(|p| p.join("Cargo.toml").is_file())
		.last()
		.map(Path::to_path_buf)
		.unwrap_or(cwd)
}

/// Create the daemon dir (owner-only) and leave a `project` note in it
/// naming the root it serves, so a human browsing the temp dir can tell.
pub fn ensure_dir() -> std::io::Result<PathBuf> {
	let dir = dir();
	platform::private_dir(std::fs::DirBuilder::new().recursive(true)).create(&dir)?;
	let note = dir.join("project");
	if !note.exists() {
		let _ = std::fs::write(note, format!("{}\n", project_root().display()));
	}
	Ok(dir)
}

/// Where the daemon answers `status` / `reload` / `stop`.
pub fn daemon_socket() -> String {
	dir().join("daemon.sock").to_string_lossy().into_owned()
}
/// Where the executable serves its channel snapshots (`channels::serve`).
pub fn app_socket() -> String {
	dir().join("app.sock").to_string_lossy().into_owned()
}
/// Where valgrind runs, benchmark logs and their parsed results are stored.
pub fn valgrind_dir() -> PathBuf {
	dir().join("valgrind")
}

fn fnv1a(bytes: &[u8]) -> u64 {
	bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
		(h ^ *b as u64).wrapping_mul(0x100_0000_01b3)
	})
}
