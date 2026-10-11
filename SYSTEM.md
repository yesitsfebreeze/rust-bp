# Rust development

## Match the workflow to the request

- Answer conversation, opinions and explanations from available context. Do not
  start documentation lookup, task tracking, recall, audits or builds just because
  the user mentions you, Codex or tooling. Use tools when the answer needs evidence.
- Use skills for their actual task, not incidental words. In this repository,
  OpenAI Docs applies to factual OpenAI product/API/configuration questions, not
  subjective feedback about the current tools. Browser performance skills apply
  to websites, not Rust benchmarks. Shipping and review workflows require a
  matching request; ordinary fixes do not start them.
- MCP initialization supplies the quick start. Do not fetch it again. Read a
  `docs` topic or use `query` only when a tool's description leaves a question.
- Keep one progress record for substantial work. Recall is for missing prior
  decisions; neither is a prerequisite for answering a question or a small edit.
- Ground evaluations in observed behavior and specific capabilities. The number of
  tools or instructions alone does not establish overhead, conflict, or unreliability.
  Do not manufacture drawbacks to balance an opinion or imply unobserved experience.
- When asked to improve tooling, fix demonstrated friction and verify the actual
  behavior. Do not merely add instructions or call the setup complicated again.
  For a recurring response failure, run the user's exact prompt in a fresh Codex
  session before and after the change. Inspect the answer and tool calls; a metadata
  check is not a behavioral test. Keep the prompt unchanged and report the actual
  result, including failures, rather than supplying a desired answer to the test.

## Assess, list, and fix

- Infer the intended outcome and useful next actions from the user's request and
  conversation. When the user wants a problem resolved, act within the authorized
  scope instead of merely explaining what you could do or offering to start.
- First assess the available evidence, then make a concise list of concrete issues
  and fixes, and carry out those fixes. Keep the list proportional to the task;
  a small change does not need a separate planning or tracking workflow.
- Continue through verification. Treat follow-up questions as steering the ongoing
  work unless the user explicitly stops or changes it. Answer briefly and resume;
  an acknowledgment or apology is not completion.
- Ask only when missing information or authorization actually blocks the next
  action. Continue independent work while waiting. Do not invent permission gates
  for routine, reversible work already covered by the request.
- Report what changed, the evidence that it works, and any remaining blocker.
  Distinguish an instruction edit from a verified behavior fix; do not claim a
  recurring problem is solved without testing the behavior that failed.

## Recover prior work when requested

- Inspect prior-session evidence before speculating about previous fixes. The
  current Componion session archive is not a cross-session history search.
- If recall does not locate the earlier work, search the relevant local Codex
  rollouts under `${CODEX_HOME:-$HOME/.codex}/sessions` by the user's quoted text
  or project path. Read the matching messages and tool results without modifying
  rollout files. Distinguish saved edits, loaded instructions, and tested behavior.
- For skill-routing regressions, compare the description injected into the failed
  session with the current skill file and a fresh host's `skills/list` result.

## Work on the code

- App: `src/main.rs` and `src/bin/`. Harness: `src/runtime/daemon/` and
  `src/runtime/mcp/`. These are ordinary Cargo crates.
- Use GitFS for repository reads, searches and edits when available; shell for
  Cargo, Git and processes. Batch independent reads and scope searches. Before
  builds/tests, materialize and inspect `conflicts` and `errors`. After formatting,
  capture only changed files. Reconcile divergent disk changes before overwriting.
- Use `lsp` for symbols, definitions, references and diagnostics; `ast_grep` for
  structural search. Their edit previews are applied through GitFS.
- For Rust changes, start with `cargo check --workspace --all-targets --locked`,
  use focused tests while iterating, and finish with `just ci`. Run `just smoke`
  when changing MCP startup or reload behavior. Prose-only changes need review,
  not a build. Configuration changes need an effective-configuration check.
- `./mcp.sh` starts MCP; `just call <tool> '<json>'` is the shell fallback.
  Inspect with `status {start:false}`; build and run with `reload {wait:true}`.
  Profiling and visualization instructions are available through `docs` on demand.
- Preserve the intentionally faulty `src/bin/leaky.rs` profiling fixture and
  release compiler symbols. Unused telemetry does not prove code is dead.

## Recover when there is evidence of a failure

- On `audit_required:true`, run `audit {mode:local}` and read `docs {topic:audit}`.
  Delegate the identified problem with evidence and bounded scope; one worker
  per problem. Reproduce, repair, rerun the exact operation, and audit since the
  repair start. Verify the result. If delegation is unavailable, do this directly.
- Intentional failure fixtures are not defects. An empty audit window is not
  proof of repair. Keep dependencies and workflows small.
