# rust-boilerplate

A Rust framework an agent drives through **one MCP server**: hot-reload the
executable, read its live profiler channels, debug its memory where it runs,
and rank valgrind analyses with [Jev](https://api.typesafe.ai) (TypeSafe
System One). Copy this repo as the start of a project; `app/` is yours, the
rest is the harness.

```text
crates/boiler        the framework: channels, supervisor, valgrind runner, Jev wire
crates/boiler-mcp    the control MCP server (stdio, JSON-RPC 2.0)
app/                 your executable (the demo shows the contract)
valgrind.Dockerfile  the Linux container valgrind falls back to
.mcp.json            wires `boiler` as an MCP server for the agent
```

Everything the framework writes lands under `.boiler/` (gitignored):
the supervisor control socket `supervisor.sock`, the app channel socket
`app.sock`, valgrind runs under `valgrind/`, container builds under
`target-linux/`.

## Quickstart

```sh
cargo build --release
cargo run --release -p boiler -- serve     # terminal 1: the supervisor
cargo run --release -p boiler-mcp -- --call status
```

Edit `app/src/main.rs` — the supervisor rebuilds and swaps the executable.
`app/src/bin/leaky.rs` is an intentionally broken target (use-after-free,
uninitialized read, leaks) that proves the memcheck loop — never fix it.

## The loops

1. **Hot reload** — `boiler serve` watches `crates/`, `app/`, `Cargo.toml`,
   rebuilds on change and SIGKILLs the old child only when the new build is
   good; a failed build keeps the old one serving. `boiler status / reload /
   stop` (and the MCP tools of the same names) drive it. Replace the child
   with `boiler serve -- <command>`.

2. **Live channels** — the preset profiler channel, in-process:
   `channels::inc("items")`, `channels::timer("op.x").done()`,
   `channels::set("depth", 1.5)`. The executable serves them once at startup:

   ```rust
   boiler::channels::serve(boiler::APP_SOCKET)?;
   ```

   `boiler channels` (or the MCP `channels` tool) reads the snapshot live:
   counters, latency histograms (mean/p50/p90/p99/max µs) and last values.
   Channels are per-process — every hot swap starts them from zero.

3. **Memory, native where it runs** — `memory_report` reads the *live*
   process on this machine (macOS `leaks`, opt-in `heap`): leaked bytes with
   allocation callstacks, the honest view for graphics programs (Metal and
   the window server live in the same process). The pid defaults to the
   supervised app. Nothing containerized.

4. **Valgrind + Jev** — `valgrind_run {name, command, build?, tool?}` runs a
   bounded command under **callgrind** (per-function instruction counts) or
   **memcheck** (memory errors, leak records), native when a valgrind is on
   PATH — same machine, same libc, the exact binary you build — and inside
   the Linux container otherwise (Apple Silicon has no valgrind; Docker/
   OrbStack, the project mounted at `/repo`, container builds in
   `.boiler/target-linux`). `runs` lists the stored runs, `show` prints one,
   and `rank` feeds it to Jev:

   ```sh
   cargo run --release -p boiler-mcp -- --call valgrind_run '{"name":"demo",
     "command":[".boiler/target-linux/release/app","200"],
     "build":"cargo build --release -p app"}'
   cargo run --release -p boiler-mcp -- --call rank '{"name":"demo"}'
   ```

   Callgrind rows answer *how pressing* (0-4), *what work dominates*
   (hash/copy/parse/math/syscall/lock/by_design) and *whether a
   behavior-preserving reduction is plausible*; priority = pressing/4 ×
   P(safe), ×0.2 for startup-by-design. Memcheck rows answer *severity* and
   *real-defect-vs-allocator-noise*; priority = severity/4 × P(real). Work
   the top row, change the code, swap, re-run, compare. Needs
   `TYPESAFE_API_KEY` (`JEV_URL` overrides the endpoint).

## Wiring the agent

The MCP server must run with this directory as cwd (that is how MCP servers
launch); the framework's paths are project-root relative.

- **pi** — already wired: `.mcp.json` at the root, `boiler` server.
- **Claude Code** — same shape: `.mcp.json` with `mcpServers.boiler`.
- **Codex** — `~/.codex/config.toml`:

  ```toml
  [mcp_servers.boiler]
  command = "cargo"
  args = ["run", "--release", "--quiet", "-p", "boiler-mcp"]
  ```

Every tool also runs from the shell:
`cargo run --release -p boiler-mcp -- --call <tool> '<json args>'`.

## Tools

| Tool | What it does |
|------|--------------|
| `status` | supervisor state (pid, swaps, last build) + whether the channel socket answers |
| `reload` / `stop` | rebuild-and-swap now / stop the supervisor |
| `channels` | the live channel snapshot as a table |
| `memory_report` | native `leaks` (+`heap`) on the live executable, pid defaults to the supervised app |
| `valgrind_run` | one callgrind or memcheck run, parsed and stored under `.boiler/valgrind/` |
| `runs` / `show` | list stored runs / print one as a table |
| `rank` | rank a stored run with Jev, most pressing first |

## Notes

- The container image builds once (`docker build -f valgrind.Dockerfile -t
  boiler-valgrind .`), with a rustup stable toolchain — the distro rustc
  cannot read a v4 Cargo.lock. A `boiler-cargo` volume caches the registry;
  the first in-container build is slow, later ones are incremental.
- `cargo test` covers the channels snapshot, the watcher, and the annotate/
  memcheck/leak-record parsers.
- To make it yours: delete `app/src/bin/leaky.rs`, rewrite `app/src/main.rs`
  (keep the one `channels::serve` call), change the supervisor child with
  `boiler serve -- <command>` if your executable is elsewhere.