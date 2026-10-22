# cs — Counter-Strike movement port (macroquad arena)

## Layout
- App crate (root): game code in `src/*.rs` (`main`, `bsp`, `bullets`, `cvars`, `map`, `materials`, `render`, `showcase`, `textures`, `weapons`); extra binaries in `src/bin/`.
- Game libs (workspace members, path deps of the app): `src/modules/engine/{console,cvar,linalg,lightbake,physics,parallel}` and `src/modules/{audio,gameplay}`.
- `parallel` (std-only, no deps): fixed worker pool — `Pool::new(workers)`, `execute(closure) -> JobHandle`, `handle.join()`, `run_all(jobs) -> Vec` in submission order; job panics propagate to the joiner, workers survive. The fan-out point for batch CPU work. `lightbake`'s per-chart ray passes are the known candidate, but only if the bake ever runs interactively (in-game rebake, live geometry edits) — the bake-once + disk-cache flow stays serial.
- Hot reload in an open window (`just hot`): `src/modules/hot` hosts the window and swaps `src/modules/hot/game` (gameplay as a Rust dylib) on gameplay source changes; both link engine+audio from the never-unloaded `src/modules/engine/dylib`. State crosses as `gameplay::state::Carried` bytes. Engine edits need a restart.
- Harness: `src/runtime/{daemon,mcp,reload}`. Tests: `tests/daemon`, `tests/mcp`; workloads: `tests/fixtures`.
- Crates move mid-session: list directories before building from stale context. `tests/daemon/lifecycle.rs` can flake under parallel load.

## Verify
- Rust changes: `cargo check --workspace --all-targets --locked`, focused tests while iterating, finish with `just ci`.
- MCP startup/reload behavior: also `just smoke` (`CHECK_JOURNEY=lifecycle` + `stateful_reload`).
- Prose-only changes need review, not a build. Configuration changes need an effective-configuration check.
- Before builds/tests, inspect MCP `conflicts` and `errors`; reconcile divergent disk changes before overwriting.

## MCP harness
- `./mcp.sh` starts MCP; `just call <tool> '<json>'` is the shell fallback.
- Inspect with `status {start:false}`; build and run with `reload {wait:true}`.
- Navigate with `lsp` (symbols, definitions, diagnostics) and `ast_grep` (structural search). Batch reads; tools accept one object or an `items` batch.
- On `audit_required:true`: run `audit {mode:local}`, read `docs {topic:audit}`, fix the identified problem with evidence, rerun the exact failed operation, audit again.
- Profiling and visualization instructions: ask `docs` on demand.

## Never
- Do not add `strip`, `lto`, or `panic = "abort"` to `[profile.release]`: profilers and leak callstacks need those symbols and unwind paths.

## Docs
- `llms.txt` indexes task-specific guides (getting-started, architecture, reload, testing, agent-setup, tools, profiling, events). Read a guide only when a task needs it.