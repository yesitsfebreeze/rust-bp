//! Built-in, offline operating instructions. No external service or daemon needed.
use serde_json::{json, Value};

pub const STARTUP: &str = "Rust workspace tools. Use only when the task needs them; conversation needs no startup calls. This quick start is already loaded; do not fetch it again. Inspect: status {start:false}. Build and run: reload {wait:true}. Navigate: lsp or ast_grep. Tools accept one object or an items batch; inspect each result. For unfamiliar workflows, docs {query:...} searches offline guides. On audit_required:true, run audit {mode:local}, then docs {topic:audit} for evidence-based subagent repair.";

const TOOLS: &str = "docs: offline instructions; topic selects a guide, query searches guides. status: daemon/app readiness; start:false is read-only. reload: rebuild and swap; wait:true reports completion. stop: stop daemon and app. channels: live counters/timings, filter and top. lsp: method status, symbols(query), definition/references/hover(file,line,col), diagnostics(optional file), rename(new_name), call_hierarchy(direction), code_actions; coordinates are 1-based, returned edits are unapplied LSP UTF-16 ranges. ast_grep: structural pattern or YAML rule; paths defaults to src/tests, rewrite is a preview unless apply:true. events: compiler/tool activity; kind:all for all events, next_cursor for paging, subscribe_thread/unsubscribe change failure delivery. usage: activity graph summary. audit: read-only failure/slow-call diagnosis; mode:local is offline, since_ms/limit/top bound evidence. benchmark: named repeated native command with repeats/warmups/timeout_s/workload. valgrind_run: named command under callgrind or memcheck. memory_report: native process memory (pid optional), heap/backtraces. runs: list saved profiles with filter/tool/sort. show: read a saved name. rank: prioritize findings for name, mode:local avoids Jev. compare: before/after run names, requires matching workloads/environments. tools/list supplies exact schemas.";

const BATCHING: &str = "Call a tool with its normal object or {items:[argument objects],concurrency:4}. 1..32 items, concurrency 1..8, default 4. Results keep input order as {results:[{index,result:{content,error}}],failed,skipped}. Read batches collect all outcomes concurrently. A batch containing a mutation or profiling operation runs sequentially and stops at its first failure; completed work is not rolled back. Serialized tools: reload, stop, benchmark, valgrind_run, memory_report; also lsp diagnostics, ast_grep apply:true and events subscription changes. Each executed item records one visualizer event. Inspect failed/skipped and audit_required; MCP isError or CLI exit 1 preserves the successful items too. Mixed envelopes and nested batches are rejected before execution.";

const PROFILING: &str = "Use channels for live counters and timings. Run benchmark twice with the same bounded command, repeats, workload and inputs, with names baseline and candidate; compare {before:baseline,after:candidate}. For CPU instruction counts use valgrind_run tool:callgrind; for leaks and memory errors use tool:memcheck. Native Valgrind is preferred; otherwise the Docker profiler image is needed. memory_report uses native leaks/heap on macOS. runs lists stored runs; show explains one; rank mode:local prioritizes measured evidence.";

const VISUALIZER: &str = "Run just viz. The selected activity row uses reverse video; OK is green, FAIL red, and unknown EVENT yellow. The header retains total errors and failed MCP calls, plus the number of operations currently requiring audit. Selected-operation counts distinguish a noisy operation from isolated failures. j/k or arrows move, g/G go newest/oldest, q quits. The live clock still advances when no events arrive. --heatmap selects the duration heatmap. Counts come from the same event log used by audit, including trusted host hooks. Missing telemetry is not proof that an operation succeeded.";

pub fn tool(args: &Value) -> Result<String, String> {
	let topics = [
		("startup", STARTUP),
		("tools", TOOLS),
		("batching", BATCHING),
		("audit", daemon::health::REPAIR_INSTRUCTION),
		("profiling", PROFILING),
		("visualizer", VISUALIZER),
	];
	let query = match args.get("query") {
		None => None,
		Some(value) => Some(
			value
				.as_str()
				.ok_or("query must be a string")?
				.to_lowercase(),
		),
	};
	let topic = args
		.get("topic")
		.map(|v| v.as_str().ok_or("topic must be a string"))
		.transpose()?
		.unwrap_or(if query.is_some() { "all" } else { "startup" });
	if topic != "all" && !topics.iter().any(|(name, _)| *name == topic) {
		return Err(
			"unknown docs topic; use startup, tools, batching, audit, profiling, visualizer or all"
				.into(),
		);
	}
	let sections: Vec<_> = topics
		.iter()
		.filter(|(name, text)| {
			(topic == "all" || topic == *name)
				&& query
					.as_ref()
					.is_none_or(|q| format!("{name} {text}").to_lowercase().contains(q))
		})
		.map(|(name, text)| json!({"topic":name,"text":text}))
		.collect();
	let includes_audit = sections.iter().any(|section| section["topic"] == "audit");
	let mut result = json!({"topic":topic,"sections":sections,
		"topics":topics.iter().map(|(name,_)| *name).collect::<Vec<_>>(),
		"reference":"llms.txt"});
	if includes_audit {
		let recovery = json!({
		"audit_threshold":daemon::health::FAILURE_THRESHOLD,"audit_window_ms":daemon::health::WINDOW_MS,
		"audit_recovery":"Three failures since the last matching success within ten minutes require audit. Match recorded operation, source, inputs and cwd. Audit is read-only and does not clear alerts; rerun the failing operation successfully. A window expiring is not proof of repair.",
		"repair_verification":"Give the subagent event IDs/log lines, inputs and expected behavior. Reproduce before editing, preserve intentional negative fixtures, fix the root cause within scope, run focused tests, rerun the exact operation, then audit with since_ms set to repair start. An empty audit window is not verification. Use one subagent per distinct problem and verify its result before marking resolved.",
		});
		result
			.as_object_mut()
			.unwrap()
			.extend(recovery.as_object().unwrap().clone());
	}
	Ok(result.to_string())
}
