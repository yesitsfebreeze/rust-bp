# cs — Counter-Strike movement port (macroquad arena)

## Layout
- App crate (root): game code in `src/*.rs` (`main`, `bsp`, `bullets`, `cvars`, `map`, `materials`, `render`, `showcase`, `textures`, `weapons`); extra binaries in `src/bin/`.
- Game libs (workspace members, path deps of the app): `src/modules/{console,cvar,linalg,lightbake,physics,player}`.
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