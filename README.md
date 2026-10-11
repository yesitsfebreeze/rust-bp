# rust-boilerplate

A Rust framework an agent drives through **one MCP server**: hot-reload the
executable, read its live profiler channels, debug its memory where it runs,
and rank valgrind analyses with [Jev](https://api.typesafe.ai) (TypeSafe
System One). Copy this repo as the start of a project; `src/main.rs` is yours,
`src/runtime/` is the harness. The workspace is self-contained: no path
dependencies outside this repository.

## Layout

```text
Cargo.toml           app package, workspace members, shared dependency versions and build profiles
src/main.rs          bin `app`: your executable (the demo shows the contract); src/bin/leaky.rs beside it
src/runtime/         the harness; daemon/ and mcp/ each have a normal Cargo.toml
  daemon/            lib `daemon` + bin/daemon.rs: channels, daemon, valgrind runner, wire
                     + bin/rustc-events.rs: optional RUSTC_WRAPPER recording per-crate compile events
  mcp/               lib `mcp` (Jev wire, events and usage tools) + main.rs: the control MCP server (stdio, JSON-RPC 2.0)
                     + bin/mcp-launcher.rs: stable MCP transport; starts the latest worker per request
tests/<crate>/       integration tests, explicitly listed in each crate's Cargo.toml
  daemon/            <module>.rs per module (parsers, channels, watcher, ...), lifecycle.rs: daemon lifecycle
  mcp/               integration.rs: isolated end-to-end checks, Docker regressions, receipt check; jev.rs, usage.rs
justfile             the recipes: just build / check / ci / serve / call <tool>, grouped build / check / run; `just` lists them
setup.sh             optional project rename and build: ./setup.sh <name>
mcp.sh               build the MCP binaries together, then start the persistent launcher
valgrind.Dockerfile  the Linux container valgrind falls back to
.mcp.json            wires `app` as an MCP server for Claude Code, through mcp.sh
```

## Development

Requires Rust/Cargo and a Unix host (macOS or Linux). `just` is optional.
From a fresh checkout, without a setup step:

```sh
cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo run --release --bin app -- 3     # bounded demo workload
just ci                               # format, clippy, tests
just smoke                            # real MCP + daemon journey; no Docker or API key
```

The workspace uses ordinary manifests at the root, `src/runtime/daemon/Cargo.toml`
and `src/runtime/mcp/Cargo.toml`. Dependencies and targets are explicit; Cargo
and rust-analyzer can load them before a build and after `cargo clean`.
Add app dependencies to the root manifest, framework dependencies to the
relevant crate, and shared versions to `[workspace.dependencies]`.

Optional tools: `rustup component add rust-analyzer` for navigation and
`ast-grep` for structural search. Docker/Valgrind and a Jev API key are only
needed for the corresponding profiling paths; local ranking needs no key.

## Name the project

```sh
./setup.sh <name>
```

Renames the app package in `Cargo.toml` to `<name>` (binary
`target/release/<name>`), points the daemon, the checks and this README
at it, and names the MCP server `<name>` in `.mcp.json`, so the agent sees
`mcp__<name>__status`. The framework crates stay `daemon` and
`mcp`. Run it once after copying the repo; run it again to rename
again. Needs perl and cargo; it ends with `cargo build --release`, so
`Cargo.lock` follows.

## How it works

Three processes, two Unix sockets, one directory:

```text
agent ──stdio/JSON-RPC──▶ mcp-launcher ──per request──▶ target/release/mcp
                                                              │ <daemon dir>/daemon.sock
                                                              ▼
                              cargo build --release ◀── daemon serve (daemon)
                                      │ swaps on success        │ spawn / SIGKILL
                                      ▼                         ▼
                              target/release/app ───────▶ <daemon dir>/app.sock (channels)
```

1. **Daemon** (`daemon serve`, `src/runtime/daemon/hot.rs`): builds once,
   spawns `./target/release/app`, polls `src/` and `Cargo.toml` for changes,
   rebuilds, and swaps the child only when the build succeeded. It answers
   `status` / `reload` / `stop` on `daemon.sock` in the daemon dir.
   Nobody has to start it: the first `status`, `reload` or `channels` (CLI
   or MCP) that finds no socket starts `daemon serve --detach --idle 30m`,
   a per-project daemon in its own session that logs to `daemon.log`
   in the daemon dir and exits after 30 minutes without a request or a
   source edit. `daemon serve` in a terminal is the same loop without the
   idle exit; the socket is the lock, so only one runs per project.
2. **Channels** (`src/runtime/daemon/channels.rs`): the executable calls
   `daemon::channels::serve(&daemon::app_socket())` once, then records
   counters, latency histograms and values in-process. Any reader asks the
   socket for a snapshot. Channels are per-process; each swap restarts at zero.
3. **MCP worker** (`src/runtime/mcp`): stateless. `mcp-launcher` (same
   crate, started by `.mcp.json` through `mcp.sh`) keeps the stdio
   connection to the agent open and starts the current worker binary for every
   request, so a rebuilt worker is live on the next call without reconnecting.
   The worker talks to the two sockets, runs `leaks` / valgrind, and sends
   results to Jev. Every tool also runs from the shell with `--call`.
4. **Compile events** (`daemon::events`): the daemon appends one event
   per build it runs to `events.jsonl` in the daemon dir; the launcher tails it
   and forwards failures to the agent as MCP log notifications. The optional
   `rustc-events` wrapper records every crate of every build instead.

5. **Code intelligence** (`src/runtime/daemon/lsp.rs`, `src/runtime/mcp/astgrep.rs`): the
   daemon starts one `rust-analyzer` on the first `lsp` request and keeps it
   for its lifetime, so the index is built once per project and survives
   every worker swap; rust-analyzer watches the sources itself, so an edit
   is live on the next question. Positions are 1-based on the wire. While
   it still indexes, the reply says so and the agent retries. `ast_grep` is
   stateless and runs the `ast-grep` binary per call from the project root.
   Both need the tool on PATH (`rustup component add rust-analyzer`,
   `brew install ast-grep`).

## Build profiles

Defined in the root `Cargo.toml`; the choice follows what each build is for.

| profile        | used by                         | debug info                                   | why                                                                   |
|----------------|---------------------------------|----------------------------------------------|-----------------------------------------------------------------------|
| `dev`          | `cargo test`, `cargo check`     | line tables for workspace crates, none for deps | fast inner loop, readable backtraces                                 |
| `release`      | daemon swaps, valgrind, `leaks` | line tables for workspace crates, none for deps | callgrind annotation and leak callstacks need file:line, nothing more |
| `distribution` | `cargo build --profile distribution` | none; `strip`, `lto`, `codegen-units = 1`  | the small binary to ship; nothing in the framework uses it            |

`release` deliberately has no `strip`, `lto` or `panic = "abort"`: they remove
the symbols the profilers report against and change the unwinding and heap
paths being measured. Put size work into `distribution`.

## Quickstart

```sh
cargo build --release
cargo run --release -p mcp -- --call status   # starts the daemon, reports it
cargo run --release -p daemon -- logs -f     # its output; `daemon serve` runs it in a terminal instead
```

Edit `src/main.rs` — the daemon rebuilds and swaps the
executable. `src/bin/leaky.rs` is an intentionally broken target
(use-after-free, uninitialized read, leaks) that proves the memcheck loop —
never fix it.

## The loops

1. **Hot reload** — the daemon watches `src/` and `Cargo.toml`,
   rebuilds on change and SIGKILLs the old child only when the new build is
   good; a failed build keeps the old one serving. `daemon status / reload /
   stop` (and the MCP tools of the same names) drive it; `status`, `reload`
   and `channels` start it as a daemon when none runs, `stop` ends it, and
   it ends itself after 30 idle minutes (`serve --detach --idle 0` to stay).
   Replace the child with `daemon serve [--detach] -- <command>`.

2. **Live channels** — the preset profiler channel, in-process:
   `channels::inc("items")`, `channels::timer("op.x").done()`,
   `channels::set("depth", 1.5)`. The executable serves them once at startup:

   ```rust
   daemon::channels::serve(&daemon::app_socket())?;
   ```

   `daemon channels` (or the MCP `channels` tool) reads the snapshot live:
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
   OrbStack, the project mounted at `/repo`, the daemon dir mounted at its
   own path, container builds in `<daemon dir>/target-linux`). `runs` lists
   the stored runs, `show` prints one,
   and `rank` feeds it to Jev:

   ```sh
   dir=$(cargo run -q --release -p daemon -- dir)   # same path inside the container
   cargo run --release -p mcp -- --call valgrind_run "{\"name\":\"demo\",
     \"command\":[\"$dir/target-linux/release/app\",\"200\"],
     \"build\":\"cargo build --release -p app\"}"
   cargo run --release -p mcp -- --call rank '{"name":"demo"}'
   ```

   Callgrind totals count **executed instructions**, including repetitions,
   from the raw `Ir` summary. The self-cost source/function rows must sum to
   that total; program totals, call edges and source annotations are excluded.
   `show` also lists the most frequent caller → callee edges, call counts,
   inclusive instructions and instructions per call. These costs overlap:
   never add them to self costs. Cycles, inlining and ARM call detection can
   distort edge attribution; a high call count is evidence to investigate,
   not proof of redundant work. `rank` receives the relevant edges too.
   Legacy reports are corrected from their artifacts when read; rerun them
   to collect the call tree if their annotation predates this change.

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

- **pi** — register `./mcp.sh` as a stdio MCP server from the project root.
- **Claude Code** — `.mcp.json` supplies `mcpServers.app`. First run
  `cargo build --release --locked -p mcp` to avoid a cold-build connection
  timeout, then `claude mcp get app` from this project root. If Pending
  approval, approve the project server in Claude or explicitly register it
  locally: `claude mcp add --scope local app -- ./mcp.sh`
  (run from the project root; the launcher resolves paths against its working directory).
  A local registration applies only to that checkout. Reconnect existing sessions.
- **Codex** — `~/.codex/config.toml`:

  ```toml
  [mcp_servers.app]
  command = "/absolute/path/to/your/project/mcp.sh"
  args = []
  cwd = "/absolute/path/to/your/project"
  env_vars = ["TYPESAFE_API_KEY", "JEV_URL"]
  startup_timeout_sec = 120
  tool_timeout_sec = 1200
  ```

This checkout also supplies project-local `.codex/config.toml`. Start Codex
from the project root and trust the project to load it. See the
[official OpenAI MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
`mcp.sh` builds the launcher and worker in one Cargo invocation, then executes
the launcher without a redundant worker build. Build diagnostics go to stderr;
stdout carries JSON-RPC only. The launcher
keeps the MCP connection open. Each
request uses a fresh stateless worker, so a successful `cargo build --release
-p mcp` is picked up on the next request, with a tool-list change
notification. In-flight requests finish on their original binary. A failed
build leaves the last successful worker usable. This is not source watching:
use the daemon or Cargo to rebuild. Changing the launcher/configuration
itself still requires reconnecting once.

Every tool also runs from the shell:
`cargo run --release -p mcp -- --call <tool> '<json args>'`.

## Tools

| Tool | What it does |
|------|--------------|
| `status` | structured daemon state and readiness; `start:false` inspects without starting |
| `reload` / `stop` | rebuild-and-swap (`wait:true` waits for success/failure) / idempotent stop |
| `channels` | live snapshot; `top` limits rows, `filter` selects name substrings; quantiles are approximate bucket lower bounds |
| `memory_report` | native `leaks` (+`heap`) on the live executable, pid defaults to the supervised app |
| `valgrind_run` | one callgrind or memcheck run, parsed and stored under `valgrind/` in the daemon dir |
| `runs` / `show` | filter/sort runs with timing and age / compact report (`raw:true` includes benchmark snapshots; `group_by_function:true` groups Callgrind self costs) |
| `rank` | `mode:auto` uses Jev when configured, otherwise local measured-cost/error-class ordering; `local` and `jev` select explicitly |
| `benchmark` | native timings and channel summaries; `warmups` defaults to 1; records SHA-256 of executable and sample spread |
| `compare` | guarded before/after instruction, allocation, or native timing comparisons |
| `events` | compiler events by default; `kind:all` or exact kind; forward paging via `cursor`/`next_cursor`; opt-in delivery |
| `lsp` | rust-analyzer, owned by the daemon: `definition` / `references` / `hover` (file, line, col), `symbols` (query), `diagnostics` (cargo check), `status`, `rename` (new_name), `call_hierarchy` (direction), `code_actions`; edits are previews, raw LSP ranges use zero-based UTF-16 |
| `ast_grep` | structural search or rewrite with ast-grep: a `pattern` (+ `rewrite`) or a YAML `rule`; dry run unless `apply` |
| `audit` | read-only failure and latency evidence; `mode:local` uses no external service |
| `usage` | frequent tools, failures and activity counts; `just viz` shows live local-time rows with selected-event details, and `--heatmap` selects the colour timeline |

Status/reload/channels failures return MCP `isError`. Benchmark min/max and median absolute deviation describe sample variability, not confidence intervals. Warmups count toward the total timeout. Executable hashes identify the command binary; scripts, input files, and environment variables still need to be held constant. The usage summary lists frequent tools and tools with failures; idle graph nodes do not prove dead code.

## Live activity

`just viz` shows local time, outcome, duration and the operation. Its clock
updates once a second even when no events arrive; `HISTORY (g follows)`
means the cursor is browsing older events. Press `g` to follow new calls.
Use `cargo run --release -p daemon --bin viz -- --once` for a snapshot or
`--heatmap` for the colour timeline.

The app MCP and daemon builds record directly. To include **all MCP servers
called by Codex in this repository**, build once with `cargo build --release`,
then review and trust the project's PreToolUse/PostToolUse entries in Codex
`/hooks` and start/resume a session that loads them. This is the
[Codex hook trust requirement](https://learn.chatgpt.com/docs/hooks#review-and-trust-hooks);
installing a hook does not activate it automatically.

`.codex/hooks.json` runs the small Rust `mcp-usage-hook` binary. It records
server/tool name, session/call IDs, elapsed time and outcome in the same
project log. Arguments and response bodies are omitted. The app MCP's own
calls are not duplicated. Calls are recorded when completed; a missing
pre-hook leaves duration unknown. Native shell/file calls and other projects
are outside this MCP feed. Tests use isolated logs, so their calls do not
appear in your project's viewer. `cargo run -q -p daemon -- dir` prints the
log directory (`events.jsonl`); `DAEMON_DIR` overrides it.

## Repeatable checks

```sh
cargo test --workspace --locked
cargo test -p mcp --test integration -- --ignored --nocapture check
# Daemon lifecycle only, without the profilers:
CHECK_JOURNEY=lifecycle cargo test -p mcp --test integration -- --ignored --nocapture check
# Also exercise the real Jev service (requires TYPESAFE_API_KEY):
CHECK_LIVE_JEV=1 cargo test -p mcp --test integration -- --ignored --nocapture check
```

The `check` test copies the workspace into an owned temporary directory and
checks the advertised tool contract and exercises the tools over the persistent stdio MCP launcher.
It asserts PID swaps, preservation of the app after a deliberate compile
error, compiler notifications, native memory inspection (macOS), both Valgrind
modes, comparison arithmetic, source context, ranking, and worker replacement
without reconnecting. Linux checks the explicit unsupported-platform result
for native memory tools. Default ranking uses a local Jev protocol fixture;
`CHECK_LIVE_JEV=1` uses the external service. Docker must be running when native
Valgrind is unavailable. The temporary checkout and its isolated build artifacts
are removed after the test.

## Before/after profiles

Run the same `valgrind_run` command twice, using different names before and
after an edit, then call `compare {"before":"baseline","after":"candidate"}`.
Callgrind compares self instruction totals and per-location changes; Memcheck
compares allocations, frees, allocated bytes and definitely lost bytes. A clean
Memcheck run is valid too. Inclusive call costs are never summed as work.

For native timing, call `benchmark` twice around the edit:

```json
{"name":"baseline","command":["./target/release/app","100"],"repeats":5,"workload":"100 demo iterations"}
```

The tool stores individual wall-time samples and p50/p90. Apps opt into final
channel capture by calling `daemon::channels::save_requested_snapshot()` at
normal exit (the demo does this). Without that call, wall times are still
available but channel metrics are absent. Channel comparisons use medians of
per-run summaries, not pooled quantiles. The comparison refuses different
commands, runner/environment metadata, workload labels, repeat counts,
channel sets, workload counts, and failed runs. Keep input files and relevant
environment variables fixed yourself; these are not captured. Repeated timing
runs are needed to distinguish a change from noise. Legacy reports without
environment or allocation metadata must be rerun before comparison.

`show` flags edges whose inclusive cost exceeds the entire run (the number
is dropped, the call count stays) and direct recursive edges. The optimizer
folds small hot functions into their caller; mark the ones you want their own
row with `#[inline(never)]`, as `parse_work` in the demo app does. Where a file is inside this checkout and a Rust definition
can be resolved, it includes its path, line and snippet. This is a current
source definition, **not** an inferred measured call site. External library
sources and optimized/inlined names may not resolve. `rank` receives those
same flags and source evidence.

## Live compiler events in Codex

The daemon records every build it runs in the daemon dir's `events.jsonl`: one
`compile_finished` or `compile_failed` event with the build's diagnostics
(`daemon::events`). Terminal builds are not captured by default. To record
every crate of every build, including terminal ones, install the compiler
wrapper once and configure it for all builds — Cargo fingerprints include the
wrapper, so switching it on and off recompiles everything each time:

```sh
cargo install --path src/runtime/daemon --bin rustc-events
# then, in a .cargo/config.toml that is not checked in (or RUSTC_WRAPPER=rustc-events):
# [build]
# rustc-wrapper = "rustc-events"
```

The wrapper preserves compiler output and exit status. When it records a
build's crates, the daemon adds no event of its own.

Subscribe the intended session once with the MCP `events` tool:

```json
{"subscribe_thread":"YOUR-CODEX-SESSION-UUID"}
```

Failures are sent to that session with `codex queue --thread ... --message ...`.
The launcher also streams live MCP error notifications; these alone do not
guarantee the host wakes an agent. Queue acceptance/failure is recorded in the
event log, and full diagnostics remain there even when the queued message is
truncated. Each event is claimed to avoid duplicate delivery by a compiler
wrapper and launcher. Native builds can queue messages without the launcher;
container builds need the host launcher running to deliver their events.
Use `events {"errors_only":true}` to inspect failures, or
`events {"unsubscribe":true}` to stop queued messages. A subscription persists
in this workspace; update it when switching sessions.

To queue one intentionally broken build to the current session (the isolated
harness restores the file):

```sh
CHECK_LIVE_JEV=1 CHECK_NOTIFY_THREAD="$CODEX_THREAD_ID" \
  cargo test -p mcp --test integration -- --ignored --nocapture check
# After the failure message arrives in that session:
cargo test -p mcp --test integration -- --ignored --nocapture verify_receipt
```

Queue success is recorded as `notification_queued`, not a delivery receipt.
The receipt verifier requires the exact event ID in an incoming user message
in the named session's persisted transcript. An accepted queue request alone
cannot pass it; the message may arrive after the active turn finishes.

Run `cargo test -p mcp --test integration -- --ignored --nocapture regressions`
for the audit regressions, including real Docker timeout/cancellation cleanup.
It requires a running Docker daemon and the profiler image. The normal
integration suite uses a local Jev fixture; `CHECK_LIVE_JEV=1` uses the
configured external service and starts no fixture server.

Profile names accept only ASCII letters, digits, hyphens and underscores.
Native comparisons check counts in every recorded repetition. Input files and
unrecorded environment variables must still be held constant by the caller.
Source links identify unique matching definitions in the current checkout;
ambiguous matches are omitted. Memcheck errors use the raw `ERROR SUMMARY`,
while finding counts describe displayed records or leaked blocks.

Commands own process groups and clean them on exit, timeout, SIGINT and SIGTERM.
Docker jobs have unique container names and remove their container after failure
or cancellation. SIGKILL and deliberately detached daemons cannot be cleaned by
an in-process handler. Cleanup errors are reported rather than counted as success.

If a fresh Codex session reports an initialization failure, run
`cargo build --release --locked -p mcp` in that workspace. The entry script
prints startup build diagnostics to stderr; retry after correcting the build. If Cargo reports an outdated lockfile after copying or changing
manifests, run `cargo build --release -p mcp` once, review `Cargo.lock`,
then retry. Copy manifests and the matching lockfile together.

## Notes

- The container image builds once (`docker build -f valgrind.Dockerfile -t
  valgrind-v2 .`), with a rustup stable toolchain — the distro rustc
  cannot read a v4 Cargo.lock. A `valgrind-cargo` volume caches the registry;
  the first in-container build is slow, later ones are incremental.
- `cargo test` covers the channels snapshot, the watcher, and the annotate/
  memcheck/leak-record parsers.
- To make it yours: delete `src/bin/leaky.rs`, rewrite
  `src/main.rs`
  (keep the one `channels::serve` call), change the daemon child with
  `daemon serve -- <command>` if your executable is elsewhere.
