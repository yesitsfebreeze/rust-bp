//! The hot-reload + profiler framework an agent drives through the
//! `boiler-mcp` control server. (module docs: see README.md)
//!
//! Conventions: everything the framework writes lands under `.boiler/` of
//! the project root (the current directory of whoever runs it); the
//! supervisor answers `.boiler/supervisor.sock`, the executable serves its
//! channels at `.boiler/app.sock`, valgrind runs under `.boiler/valgrind/`.

pub mod channels;
pub mod hot;
#[cfg(feature = "jev")]
pub mod jev;
pub mod native;
pub mod util;
pub mod valgrind;
pub mod watch;
pub mod wire;

/// Where the supervisor answers `status` / `reload` / `stop`.
pub const SUPERVISOR_SOCKET: &str = ".boiler/supervisor.sock";
/// Where the executable serves its channel snapshots (`channels::serve`).
pub const APP_SOCKET: &str = ".boiler/app.sock";
/// Where valgrind runs and their parsed results are stored.
pub const VALGRIND_DIR: &str = ".boiler/valgrind";