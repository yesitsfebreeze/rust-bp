# Working in this project

Read install.txt when setting up the runtime. Use llms.txt to find only the
documentation relevant to the task. Answer conversation from available context;
tool calls, startup, task records and documentation searches need a reason.

Infer the intended outcome, inspect the evidence, make the smallest complete
change and verify it. Follow-up questions steer ongoing work unless it is
cancelled. Ask only when missing information or authority blocks the next step.
Report observed results and remaining limitations without inventing evidence.

## Runtime workflow

The app starts in src/main.rs; reloadable modules live in src/modules/.
Reusable runtime code is in src/runtime/daemon/ and src/runtime/reload/.
Keep each unit's tests in its implementation file under #[cfg(test)].
Embed test fixtures beside their tests and write temporary files only when needed.

Register the runtime executable as the stdio MCP server described in install.txt.
Load this file using the agent's supported project-instruction mechanism.
MCP initialization supplies the quick start; do not fetch it again.
Use status {start:false} to inspect without starting the app.
Use reload {wait:true} to build/run and confirm completion.
Prefer lsp for symbols, definitions, references and diagnostics, ast_grep for
structural searches, and channels for application-defined live counters/state.
LSP edit results are previews using UTF-16 positions; review and apply them with
the available editing tools. AST rewrites require apply:true to change files.
Batch independent reads and inspect every result, including failed/skipped items.

The MCP process owns its supervisor and language server. stop closes them and
the app while leaving MCP connected; reload can start them again. Disconnecting
MCP closes the resources it started. An already running project supervisor is
reused and is not stopped merely by disconnecting another client.
Module-only edits preserve the app process and module state. App edits restart
with saved app/module state. Confirm accepted state, not just a successful build.
Runtime executable changes require rebuilding and reconnecting MCP; on Windows
disconnect before rebuilding its locked executable.

## Verification and evidence

Start Rust changes with cargo check --workspace --all-targets --locked.
Use focused checks while iterating, then just ci. Run just smoke for changes to
runtime startup, MCP or reload. Documentation-only edits need link and consistency
checks. Keep release debug symbols; distribution is the stripped shipping profile.
Use just call <tool> '<json>' for a shell fallback.

Profile a bounded workload with fixed inputs. memory_report reads live process
memory; channels exposes app-defined state and timing. Valgrind runs a separate
bounded workload for instruction costs or allocation errors. Process memory
counters are not leak detection. See docs/profiling.txt for comparisons.

On audit_required:true, run audit {mode:local} and inspect docs {topic:audit}.
Use the recorded inputs and event IDs to reproduce and repair the actual failure,
rerun it, then audit since the repair started. An empty window is not verification.
Intentional negative fixtures are not defects; unused telemetry does not prove
code is dead. Report what changed, what was tested, and any unresolved blocker.
