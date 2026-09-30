//! boiler — the hot-reload + profiler framework an agent drives through the
//! `boiler-mcp` control server.
//!
//! One project, three loops:
//!
//! 1. **Hot reload** — [`hot::Supervisor`] watches the sources, rebuilds on
//!    change and swaps the executable, keeping the old build on a failed one.
//! 2. **Live channels** — [`channels`] is the preset profiler channel:
//!    counters, latency histograms and last-values recorded in the
//!    executable and read live over a Unix socket.
//! 3. **Valgrind + Jev** — [`valgrind`] runs any command under callgrind or
//!    memcheck inside a Linux container (the host Mac has no valgrind) and
//!    stores parsed runs; [`jev`] ranks them over the Jev wire.
//!
//! Conventions: everything the framework writes lands under `.boiler/` of
//! the project root (the current directory of whoever runs it); the
//! supervisor answers `.boiler/supervisor.sock`, the executable serves its
//! channels at `.boiler/app.sock`, valgrind runs under `.boiler/valgrind/`.

pub mod channels;
pub mod hot;
pub mod jev;
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