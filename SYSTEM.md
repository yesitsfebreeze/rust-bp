# Rust development

- The app is `src/main.rs` and `src/bin/`. The reusable harness is
  `src/runtime/daemon/` and `src/runtime/mcp/`, with standard Cargo manifests.
- Use GitFS for repository reads, searches and edits when available. Batch
  independent reads. Scope searches to source/test paths; broad globs can
  include `target/` and GitFS metadata. Materialize before builds or tests;
  inspect `conflicts` and `errors`, since a tool-level success can contain a
  materialization conflict. Do not test stale disk files. After formatting,
  capture only the files you changed. Reconcile any later disk guard conflict
  against the captured revision before overwriting a path.
  If GitFS fails, report the specific failure and use the narrowest permitted
  fallback. Never overwrite divergent disk changes without reconciling them.
- Use the shell for Cargo, Git and process execution. Start with
  `cargo check --workspace --all-targets --locked`; use focused tests while
  iterating and `just ci` before finishing. No manifest generation is needed.
- `just smoke` tests real MCP startup, navigation, reload, deliberate compiler
  failure, recovery and cleanup in a temporary checkout. `just e2e` also
  exercises native memory and Valgrind; it needs the profiler backend.
- `./mcp.sh` is the MCP entry point. `just call <tool> '<json>'` invokes a tool
  from the shell. Prefer `lsp` for definitions/references/diagnostics and
  `ast_grep` for structural searches. LSP edits and structural rewrites are
  previews; apply repository changes through GitFS when it is active.
- Preserve the intentionally faulty `src/bin/leaky.rs` profiling fixture.
  Do not infer dead code from tools or symbols with no recorded usage.
- `just viz` shows local-time activity. Project Codex hooks add other MCP
  servers after the user trusts them via `/hooks`; they store metadata only.
  Native shell/file calls and isolated test projects are not in this feed.
- Keep dependencies and workflows small. Add configuration only for a
  demonstrated need; retain compiler symbols in the release profiling build.
