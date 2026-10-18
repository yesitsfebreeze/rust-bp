//! Persistent MCP access to the runtime, its application and development tools.

use crate as daemon;
use daemon::{app_socket, channels, daemon_socket, hot, native, profiles, valgrind, wire};
pub mod astgrep;
pub mod audit;
pub mod batch;
pub mod docs;
pub mod events;
pub mod jev;
pub mod usage;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

const SERVE_HINT: &str = "the executable serves its channels by calling daemon::channels::serve(&daemon::app_socket()) once at startup and retaining the returned guard for its lifetime";

/// Persistent MCP runs supervision in this process. One-shot CLI calls
/// reuse or start a detached supervisor from this same executable.
static EMBEDDED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn ensure_runtime() -> Result<Value, String> {
	if EMBEDDED.load(std::sync::atomic::Ordering::Relaxed) {
		hot::ensure_background()
	} else {
		hot::ensure_running(|| {
			let mut command = std::process::Command::new(std::env::current_exe().expect("runtime path"));
			command.args(["serve", "--idle", "30m"]);
			command
		})
	}
}

/// Ask the daemon, starting it first when needed.
fn daemon(op: &str) -> Result<Value, String> {
	ensure_runtime()?;
	wire::ask(&daemon_socket(), json!({"op": op}))
}

/// Forward one `lsp` question to the daemon's rust-analyzer; the wait
/// covers a full `cargo check` for `diagnostics`.
fn tool_lsp(args: &Value) -> Result<String, String> {
	ensure_runtime()?;
	let mut request = args.clone();
	request["op"] = json!("lsp");
	let reply = wire::ask_within(
		&daemon_socket(),
		request,
		daemon::lsp::CHECK_TIMEOUT + Duration::from_secs(10),
	)?;
	if reply["ok"] == json!(false) {
		return Err(reply["error"].as_str().unwrap_or("lsp failed").to_string());
	}
	serde_json::to_string_pretty(&reply).map_err(|e| e.to_string())
}

fn tool_usage(args: &Value) -> Result<String, String> {
	let graph = daemon::usage::graph(
		&daemon::usage::load(),
		args["gap_ms"]
			.as_u64()
			.unwrap_or(daemon::usage::CHAIN_GAP_MS),
		args["since_ms"].as_u64().unwrap_or(0),
		&daemon::usage::Skeleton::scan(&daemon::project_root()),
	);
	let count = |key: &str, kind: &str| {
		graph[key]
			.as_array()
			.map(|items| items.iter().filter(|n| n["kind"] == kind).count())
			.unwrap_or(0)
	};
	let mut observed: Vec<_> = graph["nodes"]
		.as_array()
		.into_iter()
		.flatten()
		.filter(|n| n["kind"] == "tool" && n["calls"].as_u64().unwrap_or(0) > 0)
		.cloned()
		.collect();
	observed.sort_by_key(|n| std::cmp::Reverse(n["errors"].as_u64().unwrap_or(0)));
	let failures: Vec<_> = observed
		.iter()
		.filter(|n| n["errors"].as_u64().unwrap_or(0) > 0)
		.take(10)
		.cloned()
		.collect();
	observed.sort_by_key(|n| std::cmp::Reverse(n["calls"].as_u64().unwrap_or(0)));
	observed.truncate(10);
	serde_json::to_string_pretty(&json!({
        "most_used_tools":observed,"tools_with_failures":failures,
        "note":"Idle source nodes are unobserved by these tools, not proven unused code. Investigate tools_with_failures first.",
        "picture": "run `just viz` for the live terminal stream of these calls",
		"nodes": graph["nodes"].as_array().map(Vec::len).unwrap_or(0),
		"edges": graph["edges"].as_array().map(Vec::len).unwrap_or(0),
		"tools": count("nodes", "tool"), "files": count("nodes", "file"), "symbols": count("nodes", "symbol"),
		"runs": count("nodes", "run"), "events": count("nodes", "event"), "jev": count("nodes", "jev"),
		"idle": graph["nodes"].as_array().map(|items| items.iter().filter(|n| n["calls"] == 0).count()).unwrap_or(0),
	}))
	.map_err(|e| e.to_string())
}

fn pretty(value: &Value) -> Result<String, String> {
	serde_json::to_string_pretty(value).map_err(|e| e.to_string())
}

fn tool_status(args: &Value) -> Result<String, String> {
	let mut status = if args["start"].as_bool() == Some(false) {
		if daemon::platform::LocalStream::connect(daemon_socket()).is_err() {
			return pretty(&json!({"ok":true,"running":false,"daemon_running":false}));
		}
		wire::ask(&daemon_socket(), json!({"op":"status"}))?
	} else {
		ensure_runtime()?
	};
	status["daemon_running"] = json!(true);
	status["ready"] = json!(
		status["running"] == true && status["last_build"]["ok"] == true && status["building"] != true
	);
	status["channels"] = match channels::read(&app_socket()) {
		Ok(snapshot) => {
			json!({"available":true,"process":snapshot["process"],"total":snapshot["channels_total"]})
		}
		Err(error) => json!({"available":false,"detail":error,"hint":SERVE_HINT}),
	};
	pretty(&status)
}

fn tool_reload(args: &Value) -> Result<String, String> {
	let reply = daemon("reload")?;
	if !args["wait"].as_bool().unwrap_or(false) {
		return pretty(&reply);
	}
	let ticket = reply["ticket"]
		.as_u64()
		.ok_or("restart the daemon to enable reload completion tracking")?;
	let started = Instant::now();
	let timeout = Duration::from_secs(args["timeout_s"].as_u64().unwrap_or(120).clamp(1, 600));
	loop {
		let status = wire::ask(&daemon_socket(), json!({"op":"status"}))?;
		if status["reload_completed"].as_u64().unwrap_or(0) >= ticket {
			if status["last_build"]["ok"] != true || status["running"] != true {
				return Err(format!("reload failed: {}", status["last_build"]));
			}
			return pretty(&json!({"ok":true,"queued":false,"ticket":ticket,"status":status}));
		}
		if started.elapsed() >= timeout {
			return Err("reload wait timed out; the daemon may still be building, inspect status".into());
		}
		std::thread::sleep(Duration::from_millis(100));
	}
}

fn tool_stop(_: &Value) -> Result<String, String> {
	if daemon::platform::LocalStream::connect(daemon_socket()).is_err() {
		return pretty(&json!({"op":"stop","ok":true,"already_stopped":true}));
	}
	let mut reply = wire::ask(&daemon_socket(), json!({"op":"stop"}))?;
	if EMBEDDED.load(std::sync::atomic::Ordering::Relaxed) {
		hot::stop_background();
	}
	reply["already_stopped"] = json!(false);
	pretty(&reply)
}

fn tool_channels(args: &Value) -> Result<String, String> {
	ensure_runtime()?;
	let snapshot = channels::read(&app_socket())
		.map_err(|e| format!("{e}; the app may still be building. {SERVE_HINT}"))?;
	Ok(channels::render_filtered(
		&snapshot,
		args["top"].as_u64().unwrap_or(100).clamp(1, 1000) as usize,
		args["filter"].as_str().unwrap_or(""),
	))
}

fn supervised_pid() -> Option<u32> {
	let status = wire::ask(&daemon_socket(), json!({"op": "status"})).ok()?;
	status["child_pid"].as_u64().map(|pid| pid as u32)
}

fn tool_memory_report(args: &Value) -> Result<String, String> {
	let requested = args
		.get("pid")
		.map(|value| {
			value
				.as_u64()
				.and_then(|pid| u32::try_from(pid).ok())
				.filter(|pid| *pid > 0)
				.ok_or("pid must be a positive 32-bit process ID")
		})
		.transpose()?;
	let pid = match requested {
        Some(pid) => pid,
        None => match supervised_pid().or_else(|| {
            channels::read(&app_socket())
                .ok()
                .and_then(|snapshot| snapshot["process"]["pid"].as_u64().map(|pid| pid as u32))
        }) {
            Some(pid) => pid,
            None => {
                return Err(
                    "no pid to inspect: pass `pid`, or start the daemon (or serve the channel socket) so the framework knows the executable".into(),
                )
            }
        },
    };
	let memory = daemon::platform::memory_usage(pid)?;
	if !cfg!(target_os = "macos") {
		return pretty(&json!({"pid":pid,"runner":"native","memory":memory,
			"leak_detection":false,
			"note":"Process memory accounting. For allocation errors/leaks, run a bounded workload with valgrind_run tool:memcheck; channels provides application-defined live state."}));
	}
	let backtraces = args["backtraces"].as_u64().unwrap_or(5) as usize;
	let leaks = native::leaks_report(pid, backtraces.clamp(1, 30))?;
	let heap = if args["heap"].as_bool().unwrap_or(false) {
		let text = native::heap_summary(pid)?;
		let lines: Vec<_> = text.lines().collect();
		Some(
			json!({"text":lines.iter().take(40).copied().collect::<Vec<_>>().join("\n"),"truncated":lines.len()>40}),
		)
	} else {
		None
	};
	pretty(
		&json!({"pid":pid,"runner":"native","memory":memory,"leaks":leaks,"heap":heap,
        "allocation_stacks":"If leak stacks are absent, launch the app with MallocStackLogging=1 and reproduce; availability is determined by the inspected process."}),
	)
}

fn tool_valgrind_run(args: &Value) -> Result<String, String> {
	let name = args["name"]
		.as_str()
		.ok_or("`name` is required: one word naming the run (<daemon dir>/valgrind/<name>.json)")?;
	let command: Vec<String> = args["command"]
		.as_array()
		.and_then(|a| {
			let words: Vec<String> = a
				.iter()
				.filter_map(|v| v.as_str().map(String::from))
				.collect();
			(words.len() == a.len()).then_some(words)
		})
		.filter(|w| !w.is_empty())
		.ok_or(
			"`command` is required: the argv array the container runs under valgrind (from /repo)",
		)?;
	let tool = valgrind::Tool::from_name(args["tool"].as_str().unwrap_or("callgrind"))?;
	let request = valgrind::Request {
		name: name.to_string(),
		tool,
		command,
		build: args["build"].as_str().map(String::from),
		timeout_s: args["timeout_s"].as_u64().unwrap_or(600),
	};
	let record = valgrind::run(request)?;
	Ok(format!(
		"{}\nrunner: {} · build: {}\nenvironment: {}",
		render_summary(&record),
		record["runner"],
		record["build"],
		record["environment"]
	))
}

fn tool_runs(args: &Value) -> Result<String, String> {
	let list = valgrind::list()?;
	let mut rows = list["runs"].as_array().cloned().unwrap_or_default();
	rows.retain(|r| {
		args["tool"].as_str().is_none_or(|t| r["tool"] == t)
			&& args["filter"]
				.as_str()
				.is_none_or(|f| r["name"].as_str().unwrap_or("").contains(f))
	});
	match args["sort"].as_str().unwrap_or("newest") {
		"name" => rows.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str())),
		"newest" => rows.sort_by_key(|r| std::cmp::Reverse(r["created_ms"].as_u64().unwrap_or(0))),
		_ => return Err("sort must be newest or name".into()),
	}
	let total = rows.len();
	rows.truncate(args["limit"].as_u64().unwrap_or(50).clamp(1, 1000) as usize);
	let now = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis() as u64;
	for row in &mut rows {
		row["age_seconds"] = json!(now.saturating_sub(row["created_ms"].as_u64().unwrap_or(0)) / 1000);
	}
	pretty(&json!({"runs":rows,"total":total}))
}

fn tool_show(args: &Value) -> Result<String, String> {
	let name = args["name"]
		.as_str()
		.ok_or("`name` is required: a stored run")?;
	let top = args["top"].as_u64().unwrap_or(25).clamp(1, 1000) as usize;
	let record = valgrind::load(name)?;
	if record["tool"] == "benchmark" {
		let mut compact = record.clone();
		if args["raw"].as_bool() != Some(true) {
			compact.as_object_mut().unwrap().remove("channel_snapshots");
		}
		return pretty(&compact);
	}
	if record["tool"] == "callgrind" {
		let functions = record["functions"].as_array().cloned().unwrap_or_default();
		let functions = if args["group_by_function"].as_bool() == Some(true) {
			profiles::group_functions(&functions)
		} else {
			functions
		};
		let mut lines = vec![format!(
            "run `{}` (callgrind): {} executed instructions across {} source/function entries (self costs)\n{:>12}  {:>6}  function",
            name,
            record["total_ir"].as_u64().unwrap_or(0),
            record["functions_total"].as_u64().unwrap_or(0),
            "self ir", "share"
        )];
		for function in functions.iter().take(top) {
			lines.push(format!(
				"{:>12}  {:>5.1}%  {}",
				function["ir"].as_u64().unwrap_or(0),
				function["share"].as_f64().unwrap_or(0.0) * 100.0,
				profiles::display_location(function["location"].as_str().unwrap_or("?"))
			));
			let root = std::env::current_dir().map_err(|e| e.to_string())?;
			if let Some(source) =
				profiles::source_context(function["location"].as_str().unwrap_or(""), &root)
			{
				lines.push(format!(
					"  source {}:{} (current definition, not a measured call site)\n{}",
					source["path"].as_str().unwrap(),
					source["line"],
					source["snippet"].as_str().unwrap()
				));
			}
		}
		if record["call_graph_available"] == true {
			lines.push("\nMost frequent caller -> callee edges (overlapping inclusive costs):".into());
			lines.push(format!(
				"{:>10} {:>14} {:>12}  caller -> callee",
				"calls", "inclusive ir", "ir/call"
			));
			if let Some(edges) = record["call_edges"].as_array() {
				for edge in edges.iter().take(top) {
					let inclusive = edge["inclusive_ir"]
						.as_u64()
						.map_or("n/a".to_string(), |v| v.to_string());
					let per_call = edge["ir_per_call"]
						.as_f64()
						.map_or("n/a".to_string(), |v| format!("{v:.1}"));
					lines.push(format!(
						"{:>10} {:>14} {:>12}  {} -> {}",
						edge["calls"].as_u64().unwrap_or(0),
						inclusive,
						per_call,
						profiles::display_location(edge["caller"].as_str().unwrap_or("?")),
						profiles::display_location(edge["callee"].as_str().unwrap_or("?"))
					));
					if edge["attribution_flags"]
						.as_array()
						.is_some_and(|a| !a.is_empty())
					{
						lines.push(format!(
							"  attribution warning: {}",
							edge["attribution_flags"]
						));
					}
				}
			}
			lines.push(record["call_graph_note"].as_str().unwrap_or("").into());
		} else {
			lines.push(
				"Call edges unavailable in this legacy annotation; rerun to collect caller/callee counts."
					.into(),
			);
		}
		Ok(lines.join("\n"))
	} else {
		let findings = record["findings"].as_array().cloned().unwrap_or_default();
		let leak = &record["leak_summary"];
		let mut lines = vec![format!(
			"run `{}` (memcheck): {} findings, {} errors — {}",
			name,
			record["findings_total"].as_u64().unwrap_or(0),
			record["errors_total"].as_u64().unwrap_or(0),
			record["error_summary"].as_str().unwrap_or("?")
		)];
		let definitely = leak["definitely_lost_bytes"].as_u64().unwrap_or(0);
		if definitely > 0 {
			lines.push(format!(
				"leak summary: definitely lost {} bytes in {} blocks",
				definitely,
				leak["definitely_lost_blocks"].as_u64().unwrap_or(0)
			));
		}
		for finding in findings.iter().take(top) {
			let frame = finding["stack"][0]
				.as_str()
				.or_else(|| finding["frame"].as_str())
				.filter(|f| !f.is_empty());
			lines.push(format!(
				"[{:>3}] {} — {} {}",
				finding["count"].as_u64().unwrap_or(0),
				finding["kind"].as_str().unwrap_or("?"),
				finding["message"].as_str().unwrap_or(""),
				frame.map(|f| format!("({f})")).unwrap_or_default()
			));
		}
		Ok(lines.join("\n"))
	}
}

fn tool_rank(args: &Value) -> Result<String, String> {
	let name = args["name"]
		.as_str()
		.ok_or("`name` is required: a stored run")?;
	let top = args["top"].as_u64().unwrap_or(12).clamp(1, 30) as usize;
	let record = valgrind::load(name)?;
	let mode = args["mode"].as_str().unwrap_or("auto");
	if !["auto", "local", "jev"].contains(&mode) {
		return Err("mode must be auto, local or jev".into());
	}
	if mode == "local"
		|| (mode == "auto"
			&& std::env::var("TYPESAFE_API_KEY")
				.unwrap_or_default()
				.is_empty())
	{
		return jev::rank_local(&record, top);
	}
	if record["tool"] == "callgrind" {
		jev::rank_callgrind(&record, top)
	} else if record["tool"] == "memcheck" {
		jev::rank_memcheck(&record, top)
	} else {
		Err("rank supports Callgrind or Memcheck; use compare for native benchmarks".into())
	}
}

fn tool_compare(args: &Value) -> Result<String, String> {
	let before = valgrind::load(args["before"].as_str().ok_or("before run name required")?)?;
	let after = valgrind::load(args["after"].as_str().ok_or("after run name required")?)?;
	let mut report = profiles::compare(&before, &after)?;
	let top = args["top"].as_u64().unwrap_or(20).clamp(1, 100) as usize;
	if let Some(functions) = report["functions"].as_array_mut() {
		functions.truncate(top);
	}
	serde_json::to_string_pretty(&report).map_err(|e| e.to_string())
}

fn tool_benchmark(args: &Value) -> Result<String, String> {
	let mut report = profiles::benchmark(args)?;
	// Full samples remain on disk; keep the tool response compact.
	report.as_object_mut().unwrap().remove("channel_snapshots");
	serde_json::to_string_pretty(&report).map_err(|e| e.to_string())
}

fn render_summary(record: &Value) -> String {
	if record["tool"] == "callgrind" {
		let mut summary = format!(
            "run `{}` (callgrind): {} executed instructions across {} source/function entries · stored {}/valgrind/{}.json",
            record["name"].as_str().unwrap_or("?"),
            record["total_ir"].as_u64().unwrap_or(0),
            record["functions_total"].as_u64().unwrap_or(0),
            daemon::dir().display(),
            record["name"].as_str().unwrap_or("?"),
        );
		if let Some(error) = record["run_error"].as_str() {
			summary.push_str(&format!(
				"\nthe container run reported: {error} — the numbers above are what it recorded anyway"
			));
		}
		let functions = record["functions"].as_array().cloned().unwrap_or_default();
		let mut lines = vec![format!("\n{:>12}  {:>6}  function", "self ir", "share")];
		for function in functions.iter().take(5) {
			lines.push(format!(
				"{:>12}  {:>5.1}%  {}",
				function["ir"].as_u64().unwrap_or(0),
				function["share"].as_f64().unwrap_or(0.0) * 100.0,
				profiles::display_location(function["location"].as_str().unwrap_or("?"))
			));
		}
		format!("{summary}\n{}", lines.join("\n"))
	} else {
		let leak = &record["leak_summary"];
		format!(
            "run `{}` (memcheck): {} findings, {} errors — {} · stored {}/valgrind/{}.json\nleak summary: definitely lost {} bytes in {} blocks",
            record["name"].as_str().unwrap_or("?"),
            record["findings_total"].as_u64().unwrap_or(0),
            record["errors_total"].as_u64().unwrap_or(0),
            record["error_summary"].as_str().unwrap_or("?"),
            daemon::dir().display(),
            record["name"].as_str().unwrap_or("?"),
            leak["definitely_lost_bytes"].as_u64().unwrap_or(0),
            leak["definitely_lost_blocks"].as_u64().unwrap_or(0),
        )
	}
}

type Tool = (
	&'static str,
	&'static str,
	Value,
	fn(&Value) -> Result<String, String>,
);

fn tools() -> Vec<Tool> {
	let object =
		|properties: Value| json!({"type": "object", "properties": properties, "required": []});
	vec![
        (
            "docs",
            "Offline help for unfamiliar workflows; topic selects a guide and query searches guides. Quick start is already supplied at initialization. Read audit guidance when audit_required is true.",
            object(json!({"topic":{"type":"string","enum":["startup","tools","batching","audit","profiling","visualizer","all"]},
                "query":{"type":"string"}})),
            docs::tool,
        ),
        (
            "events",
            "Recent compiler events (kind defaults to compiler; use all or an exact event kind). Use next_cursor as cursor for lossless forward paging; without cursor returns latest events. The runtime streams standard MCP error notifications.",
            object(json!({
                "after_ms":{"type":"integer"}, "limit":{"type":"integer","minimum":1,"maximum":100},
                "errors_only":{"type":"boolean"},"kind":{"type":"string","default":"compiler"},"cursor":{"type":"integer","minimum":0}})),
            events::tool,
        ),
        (
            "lsp",
            "Ask the daemon's rust-analyzer (started on first use, kept across rebuilds): method definition | references | hover (file, line, col; 1-based) | symbols (query) | diagnostics (optional file; runs cargo check, up to 2 min) | status. rename (new_name) and code_actions return unapplied LSP edits; call_hierarchy accepts direction incoming/outgoing. Raw LSP edits/ranges use zero-based UTF-16. While it still indexes the reply says so: retry in a few seconds.",
            json!({"type":"object", "properties":{
                "method":{"type":"string","enum":["definition","references","hover","symbols","diagnostics","status","rename","call_hierarchy","code_actions"]},
                "file":{"type":"string"}, "line":{"type":"integer","minimum":1}, "col":{"type":"integer","minimum":1},
                "query":{"type":"string"},"new_name":{"type":"string"},"direction":{"type":"string","enum":["incoming","outgoing"]}}, "required":["method"]}),
            tool_lsp,
        ),
        (
            "ast_grep",
            "Structural search or rewrite with ast-grep. `pattern` (ast-grep syntax: $NAME, $$$ARGS) with optional `rewrite`; or `rule` (YAML, several separated by ---). Dry run unless `apply` is true. Defaults: lang rust, paths src, limit 100.",
            json!({"type":"object", "properties":{
                "pattern":{"type":"string"}, "rule":{"type":"string"}, "rewrite":{"type":"string"},
                "lang":{"type":"string"}, "paths":{"type":"array","items":{"type":"string"}},
                "apply":{"type":"boolean"}, "limit":{"type":"integer","minimum":1,"maximum":1000}}, "required":[]}),
            astgrep::tool,
        ),
        (
            "audit",
            "Read-only audit of the activity graph: failures with recorded evidence and slow calls against matching prior successful calls. Returns event IDs, log lines, touched files, baselines and repair/verification steps. mode:auto asks Jev when configured, otherwise local; local keeps evidence offline. Jev assessments are hypotheses. since_ms scopes a repair verification window; limit bounds analyzed events; top bounds findings.",
            object(json!({"mode":{"type":"string","enum":["auto","local","jev"],"default":"auto"},
                "since_ms":{"type":"integer","minimum":0}, "limit":{"type":"integer","minimum":1,"maximum":10000,"default":2000},
                "top":{"type":"integer","minimum":1,"maximum":50,"default":10}})),
            audit::tool,
        ),
        (
            "usage",
            "The live graph of what this MCP gets asked: tools, the files and symbols their calls touched, stored runs, compiler events, and what followed what. Returns the most used tools, the tools with failures and node and edge counts; `just viz` paints the live stream in the terminal. gap_ms: how far apart two calls may be and still chain; since_ms: leave out older events.",
            json!({"type":"object", "properties":{"gap_ms":{"type":"integer","minimum":0}, "since_ms":{"type":"integer","minimum":0}}, "required":[]}),
            tool_usage,
        ),
        (
            "compare",
            "Compare two completed profiling runs with identical commands and recorded environments. Instruction, allocation or native latency deltas; rejects mismatched workloads. Positive deltas mean more cost.",
            json!({"type":"object", "properties":{"before":{"type":"string"}, "after":{"type":"string"},
                "top":{"type":"integer","minimum":1,"maximum":100}}, "required":["before","after"]}),
            tool_compare,
        ),
        (
            "benchmark",
            "Run a bounded command repeatedly on this machine, storing native wall-time distributions and optional final channels. Apps call channels::save_requested_snapshot at exit to include channel latencies. No Valgrind timing distortion. Keep inputs fixed for before/after comparisons.",
            json!({"type":"object", "properties":{"name":{"type":"string"},
                "command":{"type":"array","items":{"type":"string"},"minItems":1},
                "repeats":{"type":"integer","minimum":2,"maximum":30}, "warmups":{"type":"integer","minimum":0,"maximum":10,"default":1},
                "timeout_s":{"type":"integer","minimum":1,"maximum":600}, "workload":{"type":"string"}},
                "required":["name","command"]}),
            tool_benchmark,
        ),
        (
            "status",
            "Daemon status (pid, running app pid, swaps, last build, idle limit, log path) and whether the app's channel socket answers. start:false inspects without starting. Otherwise starts the supervisor inside this runtime when none answers.",
            object(json!({"start":{"type":"boolean","default":true}})),
            tool_status,
        ),
        (
            "reload",
            "Ask the daemon to rebuild and swap the executable now (same as editing a watched file). wait:true waits for completion or failure up to timeout_s (default 120). Starts the supervisor when none answers.",
            object(json!({"wait":{"type":"boolean","default":false},"timeout_s":{"type":"integer","minimum":1,"maximum":600}})),
            tool_reload,
        ),
        (
            "stop",
            "Stop the supervisor, app and language server. The MCP connection remains available for restart. Never starts a supervisor.",
            object(json!({})),
            tool_stop,
        ),
        (
            "channels",
            "The app's live preset profiler channel snapshot: counters, latency histograms (mean/p50/p90/p99/max µs) and last-values, served over the app socket. Starts the supervisor when none answers.",
            object(json!({"top": {"type": "integer", "minimum": 1,"maximum":1000},"filter":{"type":"string"}})),
            tool_channels,
        ),
        (
            "memory_report",
            "Read live process memory: Windows resident/private bytes, Linux resident/virtual bytes, macOS resident/virtual bytes plus leaks and optional heap. Process counters are not leak detection; use Valgrind memcheck for allocation errors on Linux or Docker. Defaults to the supervised app PID.",
            json!({
                "type": "object",
                "properties": {
                    "pid": {"type": "integer"},
                    "backtraces": {"type": "integer", "minimum": 1, "maximum": 30},
                    "heap": {"type": "boolean"}
                }
            }),
            tool_memory_report,
        ),
        (
            "valgrind_run",
            "Run a command under valgrind (callgrind or memcheck) — natively when a valgrind is on PATH (same machine, same libc, the exact binary this machine builds), otherwise inside the Linux container with the project mounted at /repo (Apple Silicon has no valgrind). Parses and stores the run under <daemon dir>/valgrind/<name>.json (`daemon dir` prints it; the container mounts it at the same path, so container builds sit in <daemon dir>/target-linux/release/). `build` (a shell line, e.g. `cargo build --release -p app`) runs first — on the host natively, in the container otherwise. In the container a `./target/release/X` command word is mapped to that Linux build, so the same `command` works in both runners. The command must terminate — give a bounded workload.",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "command": {"type": "array", "items": {"type": "string"}},
                    "build": {"type": "string"},
                    "tool": {"type": "string", "enum": ["callgrind", "memcheck"]},
                    "timeout_s": {"type": "integer", "minimum": 1}
                },
                "required": ["name", "command"]
            }),
            tool_valgrind_run,
        ),
        (
            "runs",
            "Stored native benchmark and Valgrind runs, newest first: timing, totals, timestamps and age. Filter by tool/name; sort by newest or name.",
            object(json!({"tool":{"type":"string","enum":["benchmark","callgrind","memcheck"]},"filter":{"type":"string"},"sort":{"type":"string","enum":["newest","name"]},"limit":{"type":"integer","minimum":1,"maximum":1000}})),
            tool_runs,
        ),
        (
            "show",
            "One stored run; benchmark snapshots omitted unless raw:true. group_by_function sums Callgrind self costs for identical names across source entries. Valgrind table: callgrind — self instruction counts and shares plus caller/callee counts and overlapping inclusive edge costs; memcheck — findings with the leak and error summaries.",
            json!({
                "type": "object",
                "properties": {"name": {"type": "string"}, "top": {"type": "integer", "minimum": 1}, "raw":{"type":"boolean"},"group_by_function":{"type":"boolean"}},
                "required": ["name"]
            }),
            tool_show,
        ),
        (
            "rank",
            "Rank a stored run. mode auto uses Jev with TYPESAFE_API_KEY, otherwise deterministic local ranking; local always avoids network; jev requires credentials. Local ranks measured cost or error classes, not optimization feasibility. Jev: callgrind — how pressing each function is, what kind of work dominates it, whether a behavior-preserving reduction is plausible; memcheck — how severe each finding is and whether it is a real defect. Ordered by priority, most pressing first.",
            json!({
                "type": "object",
                "properties": {"name": {"type": "string"}, "top": {"type": "integer", "minimum": 1, "maximum": 30},"mode":{"type":"string","enum":["auto","local","jev"]}},
                "required": ["name"]
            }),
            tool_rank,
        ),
    ]
}

/// Run one tool and log the call (`usage::record`). The outer error is an
/// unknown tool; the inner one is the tool's own failure.
fn call(name: &str, arguments: &Value) -> Result<Result<String, String>, String> {
	let (_, _, _, run) = tools()
		.into_iter()
		.find(|(tool, _, _, _)| *tool == name)
		.ok_or_else(|| format!("no tool `{name}`"))?;
	let executed = std::sync::atomic::AtomicUsize::new(0);
	let result = batch::execute(name, arguments, |item| {
		executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
		let started = Instant::now();
		let result = run(item);
		usage::record(name, item, &result, started.elapsed().as_millis() as u64);
		result
	});
	if executed.load(std::sync::atomic::Ordering::Relaxed) == 0 {
		usage::record(name, arguments, &result, 0);
	}
	Ok(usage::feedback(result))
}

fn handle(method: &str, params: &Value) -> Result<Value, String> {
	match method {
		"initialize" => Ok(json!({
			"protocolVersion": params.get("protocolVersion").cloned().unwrap_or(json!("2025-06-18")),
			"capabilities": {"tools": {}, "logging": {}},
			"serverInfo": {"name": "runtime", "version": "0.1.0"},
			"instructions": docs::STARTUP
		})),
		"tools/list" => Ok(json!({
				"tools": tools().iter().map(|(name, description, schema, _)| json!({
						"name": name, "description": format!("{description}{}", batch::DESCRIPTION), "inputSchema": batch::schema(schema.clone())
				})).collect::<Vec<_>>()
		})),
		"tools/call" => {
			let name = params["name"].as_str().unwrap_or_default();
			let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
			Ok(match call(name, &arguments)? {
				Ok(text) => json!({"content": [{"type": "text", "text": text}]}),
				Err(error) => {
					json!({"isError": true, "content": [{"type": "text", "text": error}]})
				}
			})
		}
		"ping" | "logging/setLevel" => Ok(json!({})),
		other => Err(format!("no method `{other}`")),
	}
}

/// Run a tool once from the CLI, or keep a persistent stdio connection.
pub fn run(args: &[String]) -> i32 {
	if args.first().is_some_and(|arg| arg == "--call") {
		let Some(name) = args.get(1) else {
			eprintln!("--call requires a tool name");
			return 2;
		};
		let arguments = match args.get(2).map(|s| serde_json::from_str(s)).transpose() {
			Ok(value) => value.unwrap_or(json!({})),
			Err(error) => {
				eprintln!("invalid arguments: {error}");
				return 2;
			}
		};
		return match call(name, &arguments).and_then(|result| result) {
			Ok(text) => {
				println!("{text}");
				0
			}
			Err(error) => {
				println!("{error}");
				1
			}
		};
	}
	if !args.is_empty() {
		eprintln!("usage: daemon mcp | daemon --call <tool> '<json>'");
		return 2;
	}
	serve();
	0
}

fn emit(value: &Value) {
	let mut stdout = std::io::stdout().lock();
	let _ = writeln!(stdout, "{value}");
	let _ = stdout.flush();
}

/// Drop also handles a panicking request: owned children must not outlive stdin.
struct Session {
	done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
impl Drop for Session {
	fn drop(&mut self) {
		self.done.store(true, std::sync::atomic::Ordering::Relaxed);
		hot::stop_background();
	}
}

fn serve() {
	use std::sync::{
		atomic::{AtomicBool, Ordering},
		mpsc, Arc,
	};
	EMBEDDED.store(true, Ordering::Relaxed);
	let _ = daemon::util::cancelled();
	let session = Session {
		done: Arc::new(AtomicBool::new(false)),
	};
	let initialized = Arc::new(AtomicBool::new(false));
	let ready = initialized.clone();
	let done = session.done.clone();
	let mut tail = daemon::events::Tail::new();
	std::thread::spawn(move || {
		while !done.load(Ordering::Relaxed) {
			if ready.load(Ordering::Relaxed) {
				match tail.read_new() {
					Ok(events) => {
						for event in events.iter().filter(|e| e["kind"] == "compile_failed") {
							emit(&json!({"jsonrpc":"2.0","method":"notifications/message",
                            "params":{"level":"error","logger":"runtime.compiler","data":event}}));
						}
					}
					Err(error) => eprintln!("event feed: {error}"),
				}
			}
			std::thread::sleep(Duration::from_millis(100));
		}
	});
	// A blocking stdin read must not prevent signal-driven cleanup.
	let (send, receive) = mpsc::channel();
	std::thread::spawn(move || {
		for line in std::io::stdin().lock().lines() {
			let Ok(line) = line else { break };
			if send.send(line).is_err() {
				break;
			}
		}
	});
	while !daemon::util::cancelled() {
		let line = match receive.recv_timeout(Duration::from_millis(50)) {
			Ok(line) => line,
			Err(mpsc::RecvTimeoutError::Timeout) => continue,
			Err(mpsc::RecvTimeoutError::Disconnected) => break,
		};
		if line.trim().is_empty() {
			continue;
		}
		let message: Value = match serde_json::from_str(&line) {
			Ok(value) => value,
			Err(_) => {
				emit(&json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Invalid JSON"}}));
				continue;
			}
		};
		if !message.is_object() {
			emit(
				&json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"Expected a request object"}}),
			);
			continue;
		}
		let Some(id) = message.get("id").cloned() else {
			continue;
		};
		let method = message["method"].as_str().unwrap_or_default();
		let params = message.get("params").cloned().unwrap_or(json!({}));
		match handle(method, &params) {
			Ok(result) => emit(&json!({"jsonrpc":"2.0","id":id,"result":result})),
			Err(error) => emit(&json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":error}})),
		}
		if method == "initialize" {
			initialized.store(true, Ordering::Relaxed);
		}
	}
}

#[cfg(test)]
mod audit_cli_tests {
	use serde_json::{json, Value};
	fn temp(label: &str) -> std::path::PathBuf {
		let dir = std::env::temp_dir().join(format!("rt-{label}-{}", crate::events::fresh_id()));
		std::fs::create_dir_all(&dir).unwrap();
		dir
	}
	#[test]
	fn audit_cli_uses_jev_and_preserves_evidence_on_service_failure() {
		use std::io::{Read, Write};
		use std::net::TcpListener;
		use std::time::{Duration, Instant};
		// Compilation must complete before the HTTP fixture starts its deadline.
		let _ = super::test_command();
		let dir = temp("audit");
		let seed = json!({"id":"broken-build","kind":"compile_failed","time_ms":1,
		"crate":"workspace","build":"cargo build -p app","duration_ms":200,
		"diagnostics":"error: missing symbol"});
		let run = |mode: &str, url: &str, key: bool| {
			// Each call gets identical evidence, independent of the recorder's output.
			std::fs::write(dir.join("events.jsonl"), format!("{seed}\n")).unwrap();
			let mut command = super::test_command();
			command
				.env("DAEMON_DIR", &dir)
				.env("JEV_URL", url)
				.env_remove("TYPESAFE_API_KEY");
			if key {
				command.env("TYPESAFE_API_KEY", "fixture-key");
			}
			command
				.args(["--call", "audit", &json!({"mode":mode}).to_string()])
				.output()
				.unwrap()
		};
		let listener = TcpListener::bind("127.0.0.1:0").unwrap();
		let url = format!("http://{}", listener.local_addr().unwrap());
		listener.set_nonblocking(true).unwrap();
		let server = std::thread::spawn(move || {
			let deadline = Instant::now() + Duration::from_secs(10);
			let (mut socket, _) = loop {
				match listener.accept() {
					Ok(pair) => break pair,
					Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => {
						std::thread::sleep(Duration::from_millis(10))
					}
					Err(e) => panic!("fixture accept: {e}"),
				}
			};
			socket
				.set_read_timeout(Some(Duration::from_secs(5)))
				.unwrap();
			let mut request = vec![];
			let mut chunk = [0u8; 4096];
			loop {
				let n = socket.read(&mut chunk).unwrap();
				assert!(n > 0);
				request.extend_from_slice(&chunk[..n]);
				if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
					let headers = String::from_utf8_lossy(&request[..end]);
					let len: usize = headers
						.lines()
						.find_map(|l| {
							let (k, v) = l.split_once(':')?;
							k.eq_ignore_ascii_case("content-length")
								.then(|| v.trim().parse().unwrap())
						})
						.unwrap();
					if request.len() >= end + 4 + len {
						let body: Value = serde_json::from_slice(&request[end + 4..end + 4 + len]).unwrap();
						assert_eq!(
							body["state"]["findings"][0]["evidence"]["event_id"],
							"broken-build"
						);
						assert_eq!(body["questions"]["finding_0"]["type"], "choice");
						break;
					}
				}
			}
			let body = r#"{"answers":{"finding_0":{"choice":"investigate"}}}"#;
			write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
		});
		let output = run("jev", &url, true);
		server.join().unwrap();
		assert!(
			output.status.success(),
			"{}",
			String::from_utf8_lossy(&output.stderr)
		);
		let report: Value = serde_json::from_slice(&output.stdout).unwrap();
		assert_eq!(report["mode"], "jev");
		assert_eq!(report["findings"][0]["jev_assessment"], "investigate");
		// The fixture listener is closed: auto must still return usable local evidence.
		let output = run("auto", &url, true);
		assert!(output.status.success());
		let report: Value = serde_json::from_slice(&output.stdout).unwrap();
		assert_eq!(report["mode"], "local");
		assert!(report["jev_error"].is_string());
		assert_eq!(
			report["findings"][0]["evidence"]["event_id"],
			"broken-build"
		);
		let output = run("local", &url, false);
		assert!(output.status.success());
		assert!(!run("jev", &url, false).status.success());
		std::fs::remove_dir_all(dir).unwrap();
	}
}

#[cfg(test)]
mod tests {
	use crate as daemon;
	use serde_json::{json, Value};
	use std::io::Write;
	use std::path::PathBuf;
	use std::process::{Command, Stdio};
	struct Fixture(PathBuf);
	impl Fixture {
		fn new() -> Self {
			let path = std::env::temp_dir().join(format!("batch-{}", daemon::events::fresh_id()));
			std::fs::create_dir(&path).unwrap();
			Self(path)
		}
		fn command(&self) -> Command {
			let mut command = super::test_command();
			command.current_dir(&self.0).env("DAEMON_DIR", &self.0);
			command
		}
		fn rpc(&self, method: &str, params: Value) -> Value {
			let mut child = self
				.command()
				.stdin(Stdio::piped())
				.stdout(Stdio::piped())
				.spawn()
				.unwrap();
			writeln!(
				child.stdin.take().unwrap(),
				"{}",
				json!({"jsonrpc":"2.0","id":1,"method":method,"params":params})
			)
			.unwrap();
			let output = child.wait_with_output().unwrap();
			assert!(output.status.success());
			serde_json::from_slice::<Value>(&output.stdout).unwrap()["result"].clone()
		}
	}
	impl Drop for Fixture {
		fn drop(&mut self) {
			let _ = std::fs::remove_dir_all(&self.0);
		}
	}

	#[test]
	fn every_advertised_tool_supports_batch_and_rejects_bad_envelopes() {
		let fixture = Fixture::new();
		let listed = fixture.rpc("tools/list", json!({}));
		let tools = listed["tools"].as_array().unwrap();
		assert_eq!(tools.len(), 17);
		for tool in tools {
			let schema = &tool["inputSchema"];
			assert_eq!(schema["type"], "object");
			assert_eq!(schema["properties"]["items"]["maxItems"], 32);
			assert_eq!(
				schema["properties"]["items"]["items"]["properties"]["items"],
				false
			);
			assert_eq!(
				schema["properties"]["items"]["items"]["required"],
				schema["oneOf"][0]["required"]
			);
			assert!(tool["description"].as_str().unwrap().contains("items"));
			let response = fixture.rpc(
				"tools/call",
				json!({"name":tool["name"],"arguments":{"items":[{},null]}}),
			);
			assert_eq!(response["isError"], true, "{tool}");
			assert!(response["content"][0]["text"]
				.as_str()
				.unwrap()
				.contains("items[1]"));
		}
		let log = std::fs::read_to_string(fixture.0.join("events.jsonl")).unwrap();
		assert_eq!(log.lines().count(), tools.len());
		assert!(log
			.lines()
			.all(|line| serde_json::from_str::<Value>(line).unwrap()["ok"] == false));
	}

	#[test]
	fn mcp_and_cli_return_partial_results_and_record_each_item_once() {
		let fixture = Fixture::new();
		let args = json!({"items":[{"start":false},{"start":false}]});
		let reply = fixture.rpc("tools/call", json!({"name":"status","arguments":args}));
		assert_ne!(reply["isError"], true);
		let batch: Value = serde_json::from_str(reply["content"][0]["text"].as_str().unwrap()).unwrap();
		for item in batch["results"].as_array().unwrap() {
			let content: Value =
				serde_json::from_str(item["result"]["content"].as_str().unwrap()).unwrap();
			assert_eq!(content["daemon_running"], false);
		}
		let output = fixture
		.command()
		.args([
			"--call",
			"benchmark",
			r#"{"items":[{"name":"ok","command":["rustc","--version"],"repeats":2,"warmups":0},{"name":"bad","command":[]},{"name":"skip","command":[]}]} "#,
		])
		.output()
		.unwrap();
		assert!(!output.status.success());
		let batch: Value = serde_json::from_slice(&output.stdout).unwrap();
		assert_eq!(batch["failed"], 1);
		assert_eq!(batch["skipped"], 1);
		assert_eq!(batch["results"].as_array().unwrap().len(), 2);
		let log = std::fs::read_to_string(fixture.0.join("events.jsonl")).unwrap();
		let events: Vec<Value> = log
			.lines()
			.map(|line| serde_json::from_str(line).unwrap())
			.collect();
		assert_eq!(events.len(), 4);
		assert!(events
			.iter()
			.all(|event| event["kind"] == "tool_call" && event["args"].get("items").is_none()));
		assert_eq!(
			events.iter().filter(|event| event["ok"] == false).count(),
			1
		);
	}

	#[test]
	fn startup_is_read_only_and_docs_remain_available_on_demand() {
		let fixture = Fixture::new();
		let init = fixture.rpc("initialize", json!({"protocolVersion":"2025-06-18"}));
		let instructions = init["instructions"].as_str().unwrap();
		assert!(instructions.contains("already loaded"));
		assert!(instructions.contains("audit_required"));
		assert!(instructions.len() < 700, "quick start must stay compact");
		assert!(
			!fixture.0.join("events.jsonl").exists(),
			"initialization must not record synthetic tool activity"
		);
		let reply = fixture.rpc(
			"tools/call",
			json!({"name":"docs","arguments":{"items":[
					{"topic":"audit"},{"query":"Valgrind"}
			]}}),
		);
		assert_ne!(reply["isError"], true);
		let body: Value = serde_json::from_str(reply["content"][0]["text"].as_str().unwrap()).unwrap();
		assert_eq!(body["results"].as_array().unwrap().len(), 2);
		let guide: Value =
			serde_json::from_str(body["results"][0]["result"]["content"].as_str().unwrap()).unwrap();
		assert!(guide["repair_verification"].as_str().is_some());
		assert!(guide["sections"][0]["text"]
			.as_str()
			.unwrap()
			.contains("audit"));
		let search: Value =
			serde_json::from_str(body["results"][1]["result"]["content"].as_str().unwrap()).unwrap();
		assert!(search.get("repair_verification").is_none());
		assert!(search.get("audit_recovery").is_none());
		assert!(search["sections"]
			.as_array()
			.unwrap()
			.iter()
			.any(|s| s["topic"] == "profiling"));
	}

	#[test]
	fn repeated_failures_escalate_across_calls_and_recovery_clears_only_matching_alert() {
		let fixture = Fixture::new();
		let call =
			|name: &str, args: Value| fixture.rpc("tools/call", json!({"name":name,"arguments":args}));
		for n in 1..=3 {
			let reply = call("show", json!({"name":"repair"}));
			assert_eq!(reply["isError"], true);
			let text = reply["content"][0]["text"].as_str().unwrap();
			if n < 3 {
				assert!(!text.contains("audit_required"));
			} else {
				let body: Value = serde_json::from_str(text).unwrap();
				assert_eq!(body["audit_required"], true);
				assert_eq!(body["audit_state"]["failed_calls"], 3);
				assert_eq!(body["audit_state"]["problems"][0]["recent_failures"], 3);
				assert!(body["content"].as_str().is_some());
			}
		}
		for (name, args) in [
			("status", json!({"start":false})),
			("audit", json!({"mode":"local"})),
		] {
			let reply = call(name, args);
			assert_ne!(reply["isError"], true);
			let body: Value =
				serde_json::from_str(reply["content"][0]["text"].as_str().unwrap()).unwrap();
			assert_eq!(body["audit_required"], true);
			assert_eq!(body["audit_state"]["failed_calls"], 3);
			if name == "audit" {
				assert_eq!(body["findings"][0]["failure_count"], 3);
			}
		}
		std::fs::create_dir_all(fixture.0.join("valgrind")).unwrap();
		std::fs::write(
			fixture.0.join("valgrind/repair.json"),
			r#"{"tool":"benchmark","name":"repair","channel_snapshots":[]}"#,
		)
		.unwrap();
		let reply = call("show", json!({"name":"repair"}));
		assert_ne!(reply["isError"], true);
		let body: Value = serde_json::from_str(reply["content"][0]["text"].as_str().unwrap()).unwrap();
		assert_ne!(body["audit_required"], true);
		let reply = call("audit", json!({"mode":"local"}));
		let body: Value = serde_json::from_str(reply["content"][0]["text"].as_str().unwrap()).unwrap();
		assert_eq!(body["audit_required"], false);
		assert_eq!(body["audit_state"]["failed_calls"], 3);

		let reply = call(
			"show",
			json!({"items":[{"name":"other"},{"name":"other"},{"name":"other"}]}),
		);
		assert_eq!(reply["isError"], true);
		let body: Value = serde_json::from_str(reply["content"][0]["text"].as_str().unwrap()).unwrap();
		assert_eq!(body["failed"], 3);
		assert_eq!(body["results"].as_array().unwrap().len(), 3);
		assert_eq!(body["audit_required"], true);
		assert_eq!(body["audit_state"]["problems_total"], 1);
		assert_eq!(body["audit_state"]["problems"][0]["args"]["name"], "other");
	}
}

#[cfg(test)]
mod integration_tests {
	use crate as daemon;
	const WORKLOAD: &str = r###"//! Integration-test workload, copied into the temporary app workspace.
// Shows the framework contract: serve your channels once, then work; the
// daemon swaps you on every source change, the `channels` tool reads
// you live, and `valgrind_run` builds and runs you under callgrind or
// memcheck inside the Linux container.
//
// One optional argument: the number of work iterations before exiting
// (absent = until killed — the daemon's normal mode; bounded = what a
// valgrind run wants, since the run needs a program that terminates).

use std::hint::black_box;
use std::time::Duration;

fn main() {
	let iterations: Option<u64> = std::env::args().nth(1).and_then(|arg| arg.parse().ok());
	let _channels = daemon::channels::serve(&daemon::app_socket()).expect("serve workload channels");
	let mut done = 0;
	loop {
		if let Some(limit) = iterations {
			if done >= limit {
				break;
			}
		}
		let parsed = daemon::channels::timer("app.work.parse");
		let items = parse_work(done);
		parsed.done();
		daemon::channels::add("app.work.items", items as u64);
		daemon::channels::set("app.work.last", items as f64);

		let copied = daemon::channels::timer("app.work.alloc");
		let held: Vec<u8> = (0..4096u16).map(|i| i as u8).collect();
		black_box(&held);
		copied.done();
		daemon::channels::inc("app.work.alloc.rounds");

		std::thread::sleep(Duration::from_millis(50));
		done += 1;
	}
	if let Err(error) = daemon::channels::save_requested_snapshot() {
		eprintln!("app: cannot save final channels: {error}");
		std::process::exit(1);
	}
}

// Without this the optimizer folds the body into `main` and callgrind
// shows no `parse_work` row; keep it on functions you want to profile.
#[inline(never)]
fn parse_work(round: u64) -> usize {
	let text = format!("round {round} alpha beta gamma delta epsilon zeta").repeat(64);
	black_box(
		text
			.split_whitespace()
			.filter(|word| word.len() > 3)
			.count(),
	)
}
"###;
	use serde_json::{json, Value};
	use std::collections::BTreeSet;
	use std::io::{BufRead, BufReader, Read, Write};
	use std::net::TcpListener;
	#[cfg(unix)]
	use std::os::unix::process::CommandExt;
	use std::path::{Path, PathBuf};
	use std::process::{ChildStdin, Command, Stdio};
	use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
	use std::sync::{Arc, Mutex};
	use std::time::{Duration, Instant};

	mod stateful {
		// The actual starter under the daemon, using the integration harness's
		// isolated checkout and process ownership helpers.
		use super::*;

		fn status(dir: &Path) -> Value {
			daemon::wire::ask(
				dir.join("daemon.sock").to_str().unwrap(),
				json!({"op":"status"}),
			)
			.unwrap()
		}

		fn sample(dir: &Path) -> Option<Value> {
			daemon::channels::read(dir.join("app.sock").to_str()?).ok()
		}

		fn gauge(sample: &Value, name: &str) -> f64 {
			sample["channels"]
				.as_array()
				.unwrap()
				.iter()
				.find(|c| c["name"] == name)
				.unwrap()["value"]
				.as_f64()
				.unwrap()
		}

		fn loaded(dir: &Path) -> Value {
			wait_for(Duration::from_secs(90), || {
				let s = sample(dir)?;
				s["channels"]
					.as_array()?
					.iter()
					.any(|c| c["name"] == "module_count")
					.then_some(s)
			})
		}

		fn reload(dir: &Path, ok: bool) -> Value {
			let request = daemon::wire::ask(
				dir.join("daemon.sock").to_str().unwrap(),
				json!({"op":"reload"}),
			)
			.unwrap();
			wait_for(Duration::from_secs(90), || {
				let s = status(dir);
				(s["building"] == false
					&& s["reload_completed"].as_u64() >= request["ticket"].as_u64()
					&& s["last_build"]["ok"] == ok)
					.then_some(s)
			})
		}

		pub fn run() {
			let root = root();
			let enclosure =
				Enclosure(std::env::temp_dir().join(format!("state-check-{}", std::process::id())));
			let directory = enclosure.0.join("boilerplate");
			std::fs::create_dir_all(&directory).unwrap();
			copy_fixture_sources(&root, &directory);
			std::fs::copy(root.join("src/main.rs"), directory.join("src/main.rs")).unwrap();
			let env = Env {
				set: vec![],
				remove: vec![
					"RUSTC_WRAPPER",
					"CARGO_TARGET_DIR",
					"DAEMON_DIR",
					"DAEMON_RESTORE_SNAPSHOT",
				],
			};
			env.run(
				&directory,
				&["cargo", "build", "--release", "--locked", "--workspace"],
			);
			let dir = daemon::dir_for(&directory);
			std::fs::create_dir_all(&dir).unwrap();
			let log = std::fs::File::create(dir.join("daemon.log")).unwrap();
			let mut command = Command::new(directory.join(format!(
				"target/release/daemon{}",
				std::env::consts::EXE_SUFFIX
			)));
			command
				.arg("serve")
				.current_dir(&directory)
				.stdout(log.try_clone().unwrap())
				.stderr(log);
			env.apply(&mut command);
			let mut process = Owned::spawn(&mut command).unwrap();
			let first = loaded(&dir);
			let first_pid = first["process"]["pid"].as_u64().unwrap();
			assert_eq!(gauge(&first, "module_count"), gauge(&first, "app_ticks"));

			let module = directory.join("src/modules/counter/lib.rs");
			let source = std::fs::read_to_string(&module).unwrap();
			std::fs::write(
				&module,
				source.replace("const STEP: u64 = 1;", "const STEP: u64 = 10;"),
			)
			.unwrap();
			let swapped = wait_for(Duration::from_secs(90), || {
				let s = sample(&dir)?;
				(gauge(&s, "module_generation") == 2.0
					&& gauge(&s, "module_count") > gauge(&s, "app_ticks"))
				.then_some(s)
			});
			assert_eq!(
				swapped["process"]["pid"].as_u64(),
				Some(first_pid),
				"module edit must keep the process"
			);
			let offset = gauge(&swapped, "module_count") - 10.0 * gauge(&swapped, "app_ticks");
			pass(
				true,
				"module rebuild changes behavior, retains state and keeps app PID",
			);

			let app = directory.join("src/main.rs");
			let source = std::fs::read_to_string(&app).unwrap();
			std::fs::write(&app, source.replace("from_millis(100)", "from_millis(120)")).unwrap();
			let restarted = wait_for(Duration::from_secs(90), || {
				let s = sample(&dir)?;
				(s["process"]["pid"].as_u64()? != first_pid && s["channels"].as_array()?.len() >= 3)
					.then_some(s)
			});
			assert!(gauge(&restarted, "app_ticks") >= gauge(&swapped, "app_ticks"));
			assert_eq!(
				gauge(&restarted, "module_count") - 10.0 * gauge(&restarted, "app_ticks"),
				offset,
				"both states must survive together"
			);
			pass(
				true,
				"app rebuild changes PID and restores module and app state",
			);

			// Force a write error without relying on OS permission semantics.
			let snapshot = dir.join("state/snapshot.bin");
			std::fs::rename(&snapshot, dir.join("state/backup.bin")).unwrap();
			std::fs::create_dir(&snapshot).unwrap();
			let rejected = reload(&dir, false);
			assert_eq!(rejected["child_pid"], restarted["process"]["pid"]);
			let live = loaded(&dir);
			std::thread::sleep(Duration::from_millis(300));
			assert!(
				gauge(&loaded(&dir), "app_ticks") > gauge(&live, "app_ticks"),
				"rejected save must resume simulation"
			);
			std::fs::remove_dir(&snapshot).unwrap();
			let accepted = reload(&dir, true);
			assert_ne!(accepted["child_pid"], rejected["child_pid"]);
			let after = loaded(&dir);
			assert_eq!(
				gauge(&after, "module_count") - 10.0 * gauge(&after, "app_ticks"),
				offset
			);
			pass(
				true,
				"save failure keeps old process running; repaired destination permits reload",
			);
			assert_eq!(
				std::fs::read_dir(dir.join("run")).unwrap().count(),
				1,
				"old executable copies are removed"
			);
			daemon::wire::ask(
				dir.join("daemon.sock").to_str().unwrap(),
				json!({"op":"stop"}),
			)
			.unwrap();
			assert!(wait_for(Duration::from_secs(10), || process
				.0
				.try_wait()
				.ok()
				.flatten())
			.success());
		}
	}

	#[test]
	#[ignore = "builds isolated release binaries and exercises real stateful reloads"]
	fn stateful_reload() {
		stateful::run();
	}

	fn root() -> PathBuf {
		Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("../../..")
			.canonicalize()
			.expect("project root")
	}

	fn pass(condition: bool, message: &str) {
		assert!(condition, "{message}");
		println!("PASS: {message}");
	}

	fn wait_for<T>(timeout: Duration, mut probe: impl FnMut() -> Option<T>) -> T {
		let deadline = Instant::now() + timeout;
		while Instant::now() < deadline {
			if let Some(value) = probe() {
				return value;
			}
			std::thread::sleep(Duration::from_millis(250));
		}
		panic!("timed out waiting for expected state");
	}

	fn process_running(pid: u64) -> bool {
		#[cfg(windows)]
		{
			daemon::platform::alive(pid as u32)
		}
		#[cfg(unix)]
		{
			let output = Command::new("ps")
				.args(["-o", "stat=", "-p", &pid.to_string()])
				.output()
				.unwrap();
			let state = String::from_utf8_lossy(&output.stdout).trim().to_string();
			!state.is_empty() && !state.starts_with('Z')
		}
	}

	fn program_path(program: &str) -> PathBuf {
		// Resolve PATH explicitly: Windows otherwise finds System32's WSL bash
		// before the Git Bash configured on PATH.
		std::env::var_os("PATH")
			.and_then(|path| {
				std::env::split_paths(&path)
					.map(|dir| dir.join(format!("{program}{}", std::env::consts::EXE_SUFFIX)))
					.find(|file| file.is_file())
			})
			.unwrap_or_else(|| PathBuf::from(program))
	}

	fn on_path(program: &str) -> bool {
		let path = program_path(program);
		path.is_file()
			&& Command::new(path)
				.arg("--version")
				.output()
				.is_ok_and(|out| out.status.success())
	}

	fn read_json(path: &Path) -> Value {
		serde_json::from_str(
			&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
		)
		.unwrap_or_else(|e| panic!("{}: {e}", path.display()))
	}

	fn copy_dir(from: &Path, to: &Path) {
		std::fs::create_dir_all(to).unwrap();
		for entry in std::fs::read_dir(from).unwrap() {
			let entry = entry.unwrap();
			let target = to.join(entry.file_name());
			if entry.file_type().unwrap().is_dir() {
				copy_dir(&entry.path(), &target);
			} else {
				std::fs::copy(entry.path(), target).unwrap();
			}
		}
	}

	/// Environment for every process of the isolated copy: no inherited wrapper
	/// or target dir, Jev pointed at the fixture unless the live service is used.
	#[derive(Clone)]
	struct Env {
		set: Vec<(String, String)>,
		remove: Vec<&'static str>,
	}

	impl Env {
		fn apply(&self, command: &mut Command) {
			for key in &self.remove {
				command.env_remove(key);
			}
			command.envs(self.set.iter().map(|(k, v)| (k.as_str(), v.as_str())));
		}
		fn run(&self, cwd: &Path, args: &[&str]) {
			let mut command = Command::new(args[0]);
			command.args(&args[1..]).current_dir(cwd);
			self.apply(&mut command);
			let status = command.status().unwrap();
			assert!(status.success(), "{} failed: {status}", args.join(" "));
		}
	}

	type Owned = daemon::util::ProcessTree;

	#[cfg(unix)]
	extern "C" {
		#[link_name = "kill"]
		fn libc_kill(pid: i32, signal: i32) -> i32;
	}

	#[cfg(unix)]
	fn signal(pid: u32, signal: i32) {
		unsafe {
			libc_kill(pid as i32, signal);
		}
	}
	#[cfg(unix)]
	const SIGTERM: i32 = 15;

	/// The MCP client side: one runtime process on a pipe, requests by id.
	struct Mcp {
		process: Owned,
		stdin: Option<ChildStdin>,
		incoming: Receiver<Value>,
		notifications: Vec<Value>,
		calls: BTreeSet<String>,
		seq: u64,
		app_pids: BTreeSet<u64>,
	}

	impl Mcp {
		fn start(runtime: &Path, args: &[&str], cwd: &Path, env: &Env) -> Mcp {
			let mut command = Command::new(runtime);
			command
				.args(args)
				.current_dir(cwd)
				.stdin(Stdio::piped())
				.stdout(Stdio::piped());
			env.apply(&mut command);
			let mut child = Owned::spawn(&mut command).expect("start runtime");
			let stdin = child.0.stdin.take();
			let stdout = child.0.stdout.take().unwrap();
			let (sender, incoming) = mpsc::channel();
			std::thread::spawn(move || {
				for line in BufReader::new(stdout).lines() {
					let Ok(line) = line else { break };
					let value = serde_json::from_str(&line)
						.unwrap_or_else(|_| json!({"error": format!("non-JSON stdout: {line}")}));
					if sender.send(value).is_err() {
						break;
					}
				}
			});
			let mut mcp = Mcp {
				process: child,
				stdin,
				incoming,
				notifications: vec![],
				calls: BTreeSet::new(),
				seq: 0,
				app_pids: BTreeSet::new(),
			};
			let init = mcp.rpc(
				"initialize",
				json!({"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "mcp-check", "version": "1"}}),
			);
			pass(
				init["capabilities"]["logging"].is_object(),
				"runtime advertises standard MCP logging",
			);
			assert!(init["instructions"]
				.as_str()
				.is_some_and(|text| text.contains("already loaded") && text.contains("audit_required")));
			mcp.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
			mcp
		}

		fn send(&mut self, line: &str) {
			let stdin = self.stdin.as_mut().expect("runtime stdin open");
			stdin.write_all(format!("{line}\n").as_bytes()).unwrap();
			stdin.flush().unwrap();
		}

		/// The `result` of one request; a JSON-RPC error panics.
		fn rpc(&mut self, method: &str, params: Value) -> Value {
			self.seq += 1;
			let request = json!({"jsonrpc": "2.0", "id": self.seq, "method": method, "params": params});
			self.send(&request.to_string());
			let deadline = Instant::now() + Duration::from_secs(900);
			loop {
				let remaining = deadline.saturating_duration_since(Instant::now());
				let response = match self
					.incoming
					.recv_timeout(remaining.max(Duration::from_millis(100)))
				{
					Ok(response) => response,
					Err(RecvTimeoutError::Timeout) => panic!("{method}: no response within 900 s"),
					Err(RecvTimeoutError::Disconnected) => panic!("{method}: runtime closed stdout"),
				};
				if response.get("id").is_none() && response.get("method").is_some() {
					self.notifications.push(response);
					continue;
				}
				if let Some(error) = response.get("error") {
					panic!("{method}: {error}");
				}
				assert_eq!(response["id"], json!(self.seq), "{response}");
				return response["result"].clone();
			}
		}

		/// One tool's text; a tool-level error (`isError`) panics.
		fn call(&mut self, name: &str, args: Value) -> String {
			self.calls.insert(name.to_string());
			let reply = self.rpc("tools/call", json!({"name": name, "arguments": args}));
			assert!(reply["isError"] != true, "{name}: {reply}");
			reply["content"][0]["text"].as_str().unwrap().to_string()
		}

		fn call_json(&mut self, name: &str, args: Value) -> Value {
			let text = self.call(name, args);
			serde_json::from_str(&text)
				.unwrap_or_else(|e| panic!("{name} answered no JSON ({e}): {text}"))
		}

		fn daemon_status(&mut self) -> Value {
			let text = self.call("status", json!({}));
			// The reply is pretty JSON followed by a channel-socket line.
			let mut stream = serde_json::Deserializer::from_str(&text).into_iter::<Value>();
			let state = stream.next().and_then(Result::ok).unwrap_or(json!({}));
			if let Some(pid) = state["child_pid"].as_u64() {
				self.app_pids.insert(pid);
			}
			state
		}

		fn rejects(&mut self, before: &str, after: &str) {
			let response = self.rpc(
				"tools/call",
				json!({"name": "compare", "arguments": {"before": before, "after": after}}),
			);
			pass(
				response["isError"] == true && response.to_string().contains("command differs"),
				&format!("{before}: mismatched workload rejected"),
			);
		}

		/// Close stdin and wait for the runtime to exit on EOF.
		fn close(&mut self) -> Option<std::process::ExitStatus> {
			drop(self.stdin.take());
			let deadline = Instant::now() + Duration::from_secs(10);
			while Instant::now() < deadline {
				if let Ok(Some(status)) = self.process.0.try_wait() {
					return Some(status);
				}
				std::thread::sleep(Duration::from_millis(50));
			}
			None
		}
	}

	/// A deterministic Jev protocol fixture: every `score` 2.0, every `noul`
	/// 0.75, every choice the first criterion. Records each request's state.
	fn jev_fixture() -> (u16, Arc<Mutex<Vec<Value>>>) {
		let listener = TcpListener::bind("127.0.0.1:0").unwrap();
		let port = listener.local_addr().unwrap().port();
		let states: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(vec![]));
		let seen = states.clone();
		std::thread::spawn(move || {
			for stream in listener.incoming() {
				let Ok(stream) = stream else { continue };
				let seen = seen.clone();
				std::thread::spawn(move || {
					let mut reader = BufReader::new(stream);
					let mut length = 0usize;
					loop {
						let mut line = String::new();
						if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
							break;
						}
						if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
							length = value.trim().parse().unwrap_or(0);
						}
					}
					let mut body = vec![0u8; length];
					reader.read_exact(&mut body).unwrap();
					let request: Value = serde_json::from_slice(&body).unwrap();
					seen.lock().unwrap().push(request["state"].clone());
					let mut answers = serde_json::Map::new();
					for (key, question) in request["questions"].as_object().unwrap() {
						let answer = match question["type"].as_str() {
							Some("score") => json!({"score": 2.0}),
							Some("noul") => json!({"noul": 0.75}),
							_ => {
								json!({"choice": question["criteria"].as_object().unwrap().keys().next().unwrap()})
							}
						};
						answers.insert(key.clone(), answer);
					}
					let body = json!({"answers": answers}).to_string();
					let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
					let mut stream = reader.into_inner();
					let _ = stream.write_all(response.as_bytes());
					let _ = stream.flush();
				});
			}
		});
		(port, states)
	}

	/// Removes the isolated checkout, build artifacts and daemon dir on drop.
	struct Enclosure(PathBuf);
	impl Drop for Enclosure {
		fn drop(&mut self) {
			// Resolve before removing: the hash needs the canonical path.
			let daemon_dir = daemon::dir_for(&self.0.join("boilerplate"));
			let _ = std::fs::remove_dir_all(&self.0);
			let _ = std::fs::remove_dir_all(daemon_dir);
		}
	}

	fn copy_fixture_sources(root: &Path, directory: &Path) {
		// The complete build closure must sit inside the single Docker /repo mount.
		copy_dir(&root.join("src"), &directory.join("src"));
		std::fs::write(directory.join("src/main.rs"), WORKLOAD).unwrap();
		std::fs::create_dir_all(directory.join("tools")).unwrap();
		for name in ["Cargo.toml", "Cargo.lock", "tools/valgrind.Dockerfile"] {
			std::fs::copy(root.join(name), directory.join(name)).unwrap();
		}
	}

	fn env_flag(name: &str) -> bool {
		std::env::var(name)
			.map(|v| !v.is_empty() && v != "0")
			.unwrap_or(false)
	}

	#[test]
	#[ignore = "builds release binaries, starts the daemon, needs valgrind or Docker"]
	fn check() {
		let root = root();
		let lifecycle_only = std::env::var("CHECK_JOURNEY").as_deref() == Ok("lifecycle");
		let live_jev = env_flag("CHECK_LIVE_JEV");
		if live_jev
			&& std::env::var("TYPESAFE_API_KEY")
				.map(|k| k.is_empty())
				.unwrap_or(true)
		{
			panic!("CHECK_LIVE_JEV requires TYPESAFE_API_KEY");
		}
		let mut env = Env {
			set: vec![],
			remove: vec!["RUSTC_WRAPPER", "CARGO_TARGET_DIR", "DAEMON_DIR"],
		};
		let fixture = (!live_jev).then(jev_fixture);
		if let Some((port, _)) = &fixture {
			env
				.set
				.push(("JEV_URL".into(), format!("http://127.0.0.1:{port}")));
			env
				.set
				.push(("TYPESAFE_API_KEY".into(), "local-test-fixture".into()));
		}
		let enclosure =
			Enclosure(std::env::temp_dir().join(format!("mcp-check-{}", std::process::id())));
		let directory = enclosure.0.join("boilerplate");
		std::fs::create_dir_all(&directory).unwrap();
		// Keep generated manifests and artifacts inside this fixture. A shared
		// target cache can retain source paths/binaries from a previous fixture.
		copy_fixture_sources(&root, &directory);
		let target = directory.join("target");

		env.run(
			&directory,
			&["cargo", "build", "--release", "--locked", "--workspace"],
		);
		let daemon_dir = daemon::dir_for(&directory);
		std::fs::create_dir_all(&daemon_dir).unwrap();
		let mut mcp = Mcp::start(
			&target.join(format!("release/daemon{}", std::env::consts::EXE_SUFFIX)),
			&["mcp"],
			&directory,
			&env,
		);
		assert_eq!(
			mcp.call_json("status", json!({"start":false}))["daemon_running"],
			false
		);

		let tools: BTreeSet<String> = mcp.rpc("tools/list", json!({}))["tools"]
			.as_array()
			.unwrap()
			.iter()
			.map(|t| t["name"].as_str().unwrap().to_string())
			.collect();
		let expected: BTreeSet<String> = [
			"docs",
			"events",
			"lsp",
			"ast_grep",
			"audit",
			"usage",
			"compare",
			"benchmark",
			"status",
			"reload",
			"stop",
			"channels",
			"memory_report",
			"valgrind_run",
			"runs",
			"show",
			"rank",
		]
		.into_iter()
		.map(str::to_string)
		.collect();
		assert_eq!(tools, expected, "MCP tool contract");
		pass(true, "discovered the complete MCP tool contract");
		let docs = mcp.call_json("docs", json!({"topic":"startup"}));
		pass(
			docs["sections"][0]["text"]
				.as_str()
				.is_some_and(|text| text.contains("audit")),
			"startup docs link to audit recovery",
		);
		let state = wait_for(Duration::from_secs(90), || {
			let s = mcp.daemon_status();
			s["child_pid"].as_u64().map(|_| s)
		});
		let first_pid = state["child_pid"].as_u64().unwrap();
		assert_eq!(
			state["pid"],
			mcp.process.0.id(),
			"supervisor must share the MCP runtime process"
		);
		pass(
			mcp.call_json("status", json!({"start":false}))["daemon_running"] == true,
			"read-only status",
		);
		pass(
			mcp.call_json("reload", json!({"wait":true}))["queued"] == false,
			"reload waits for completed swap",
		);
		let usage = mcp.call_json("usage", json!({}));
		pass(
			usage["most_used_tools"].is_array()
				&& usage["tools_with_failures"].is_array()
				&& usage["nodes"].as_u64().is_some_and(|n| n > 0)
				&& usage["picture"]
					.as_str()
					.is_some_and(|s| s.contains("just viz")),
			"usage returns the activity summary and terminal viewer",
		);
		let audit = mcp.call_json("audit", json!({"mode":"local"}));
		pass(
			audit["findings"].is_array(),
			"local audit returns recorded evidence",
		);
		if on_path("ast-grep") {
			pass(
				mcp
					.call(
						"ast_grep",
						json!({"pattern":"daemon::channels::inc($X)","paths":["src/main.rs"]}),
					)
					.contains("match(es)"),
				"structural search",
			);
		}
		if on_path("rust-analyzer") {
			wait_for(Duration::from_secs(180), || {
				let state = mcp.call_json("lsp", json!({"method":"status"}));
				assert!(state["dead"].is_null(), "rust-analyzer exited: {state}");
				(state["ready"] == true).then_some(())
			});
			pass(
				mcp.call_json("lsp", json!({"method":"symbols","query":"parse_work"}))["symbols"]
					.as_array()
					.is_some_and(|s| !s.is_empty()),
				"LSP finds workspace symbol",
			);
			let batch = mcp.call_json(
				"lsp",
				json!({"items":[
			{"method":"symbols","query":"parse_work"},
			{"method":"symbols","query":"serve"},
			{"method":"status"}
		],"concurrency":3}),
			);
			pass(
				batch["failed"] == 0
					&& batch["results"]
						.as_array()
						.is_some_and(|items| items.len() == 3),
				"LSP batch returns all three items",
			);
			for item in batch["results"].as_array().unwrap().iter().take(2) {
				let reply: Value =
					serde_json::from_str(item["result"]["content"].as_str().unwrap()).unwrap();
				pass(
					reply["symbols"]
						.as_array()
						.is_some_and(|symbols| !symbols.is_empty()),
					"LSP batch preserves symbol results",
				);
			}
		}

		wait_for(Duration::from_secs(90), || {
			mcp
				.call("channels", json!({}))
				.contains("app.work.parse")
				.then_some(())
		});
		pass(
			true,
			"live channels contain bounded-workload instrumentation",
		);
		let memory = mcp.call_json("memory_report", json!({}));
		pass(
			memory["memory"]["resident_bytes"].as_u64().unwrap_or(0) > 0,
			"native live process memory",
		);

		let reload_request = mcp.call_json("reload", json!({}));
		pass(reload_request["queued"] == true, "reload accepted");
		let state = wait_for(Duration::from_secs(90), || {
			let s = mcp.daemon_status();
			match s["child_pid"].as_u64() {
				Some(pid)
					if pid != first_pid
						&& s["building"] == false
						&& s["reload_completed"].as_u64() >= reload_request["ticket"].as_u64() =>
				{
					Some(s)
				}
				_ => None,
			}
		});
		pass(
			state["swaps"].as_u64().unwrap_or(0) >= 2,
			"reload actually swaps app PID",
		);

		let source = directory.join("src/main.rs");
		let original = std::fs::read_to_string(&source).unwrap();
		std::fs::write(
        &source,
        format!("{original}\ncompile_error!(\"INTEGRATION_EXPECTED_FAILURE — controlled test; the harness restores this file\");\n"),
    )
    .unwrap();
		let failures = wait_for(Duration::from_secs(90), || {
			let result = mcp.call_json("events", json!({"errors_only": true}));
			result["events"]
				.as_array()
				.map(|events| events.iter().any(|e| e["kind"] == "compile_failed"))
				.unwrap_or(false)
				.then_some(result)
		});
		pass(
			failures
				.to_string()
				.contains("INTEGRATION_EXPECTED_FAILURE"),
			"compiler diagnostics captured in event feed",
		);
		wait_for(Duration::from_secs(90), || {
			(mcp.daemon_status()["last_build"]["ok"] == false).then_some(())
		});
		pass(
			mcp.daemon_status()["child_pid"] == state["child_pid"],
			"failed build preserves running app",
		);
		let failed_reload = mcp.rpc(
			"tools/call",
			json!({"name":"reload","arguments":{"wait":true,"timeout_s":30}}),
		);
		pass(
			failed_reload["isError"] == true && failed_reload.to_string().contains("reload failed"),
			"waiting reload reports compiler failure as MCP error",
		);

		wait_for(Duration::from_secs(90), || {
			mcp.rpc("ping", json!({}));
			mcp
				.notifications
				.iter()
				.any(|n| n["method"] == "notifications/message")
				.then_some(())
		});
		pass(true, "compile failure streamed as live MCP notification");
		std::fs::write(&source, &original).unwrap();
		mcp.call("reload", json!({}));
		wait_for(Duration::from_secs(90), || {
			let s = mcp.daemon_status();
			(s["last_build"]["ok"] == true && s["child_pid"] != state["child_pid"]).then_some(())
		});
		let old_analyzer = mcp.daemon_status()["lsp"]["pid"].as_u64();
		pass(
			mcp.call_json("stop", json!({}))["ok"] == true,
			"stop accepted",
		);
		assert_eq!(
			mcp.call_json("status", json!({"start":false}))["daemon_running"],
			false
		);
		assert!(
			mcp.process.0.try_wait().unwrap().is_none(),
			"MCP must remain available after stopping the app"
		);
		mcp.rpc("ping", json!({}));
		if let Some(pid) = old_analyzer {
			assert!(!process_running(pid), "stop closes the old language server");
		}
		let pids = mcp.app_pids.clone();
		wait_for(Duration::from_secs(10), || {
			pids.iter().all(|pid| !process_running(*pid)).then_some(())
		});
		pass(true, "all observed app PIDs are gone after stop");
		if lifecycle_only {
			let restarted = mcp.call_json("reload", json!({"wait":true}));
			assert_eq!(
				restarted["status"]["pid"].as_u64(),
				Some(mcp.process.0.id() as u64)
			);
			let child = restarted["status"]["child_pid"].as_u64().unwrap();
			mcp.call("lsp", json!({"method":"status"}));
			let analyzer = mcp.daemon_status()["lsp"]["pid"].as_u64().unwrap();
			let status = mcp.close().expect("runtime exits on EOF");
			wait_for(Duration::from_secs(10), || {
				(!process_running(child) && !process_running(analyzer)).then_some(())
			});
			pass(
				true,
				"same-process restart and EOF clean app and language server",
			);
			pass(status.success(), "lifecycle MCP closes successfully on EOF");
			println!("LIFECYCLE JOURNEY PASSED");
			return;
		}

		for name in ["native-before", "native-after"] {
			let report = mcp.call_json(
				"benchmark",
				json!({"name": name, "command": ["./target/release/app", "3"], "repeats": 3}),
			);
			pass(
				report["samples_ms"].as_array().map(Vec::len) == Some(3)
					&& report["channel_metrics"]["app.work.alloc.rounds/count"].as_f64() == Some(3.0),
				&format!("{name}: three timing samples and final workload channels"),
			);
		}
		let compare = mcp.call_json(
			"compare",
			json!({"before": "native-before", "after": "native-after"}),
		);
		pass(
			compare["metrics"].get("wall_p50_ms").is_some()
				&& compare["metrics"].get("app.work.parse/p99_us").is_some(),
			"native latency comparisons include channel distributions",
		);
		mcp.call(
			"benchmark",
			json!({"name": "native-mismatch", "command": ["./target/release/app", "4"], "repeats": 3}),
		);
		mcp.rejects("native-before", "native-mismatch");

		let docker = !on_path("valgrind");
		let binary = if docker {
			format!("{}/target-linux/release/", daemon_dir.display())
		} else {
			"./target/release/".to_string()
		};
		for (tool, executable) in [("callgrind", "app"), ("memcheck", "app")] {
			let mut record = json!({});
			for suffix in ["before", "after"] {
				let name = format!("{tool}-{suffix}");
				mcp.call(
					"valgrind_run",
					json!({"name": name, "tool": tool, "command": [format!("{binary}{executable}"), "3"],
                       "build": "cargo build --release --locked -p app", "timeout_s": 600}),
				);
				record = read_json(&daemon_dir.join(format!("valgrind/{name}.json")));
				pass(
					record["run_error"].is_null(),
					&format!("{name}: profiler completed"),
				);
				if tool == "callgrind" {
					let raw =
						std::fs::read_to_string(daemon_dir.join(format!("valgrind/{name}.out"))).unwrap();
					let raw_total: u64 = raw
						.lines()
						.find_map(|l| l.strip_prefix("summary:"))
						.and_then(|v| v.split_whitespace().next())
						.and_then(|v| v.parse().ok())
						.expect("callgrind summary line");
					let self_sum: u64 = record["functions"]
						.as_array()
						.unwrap()
						.iter()
						.map(|f| f["ir"].as_u64().unwrap())
						.sum();
					pass(
						record["total_ir"] == raw_total && raw_total == self_sum,
						"raw Ir equals sum of self costs",
					);
					pass(
						!record["call_edges"].as_array().unwrap().is_empty(),
						"caller/callee evidence retained",
					);
				} else {
					let raw =
						std::fs::read_to_string(daemon_dir.join(format!("valgrind/{name}.log"))).unwrap();
					assert!(
						record["allocations"]["allocs"].as_u64().unwrap_or(0) > 0
							&& record["errors_total"].as_u64() == Some(0),
						"allocation counts {:?}, errors {:?}\n{raw}",
						record["allocations"],
						record["errors_total"]
					);
					pass(true, "allocation counts recorded with no memory errors");
					let total: u64 = raw
						.lines()
						.find_map(|l| l.split("ERROR SUMMARY:").nth(1))
						.and_then(|v| v.split_whitespace().next())
						.and_then(|v| v.replace(',', "").parse().ok())
						.expect("memcheck error summary");
					pass(
						record["errors_total"] == total,
						"Memcheck error total equals raw ERROR SUMMARY",
					);
				}
			}
			let diff = mcp.call_json(
				"compare",
				json!({"before": format!("{tool}-before"), "after": format!("{tool}-after")}),
			);
			pass(
				diff["metrics"]
					.as_object()
					.map(|m| !m.is_empty())
					.unwrap_or(false),
				&format!("{tool} before/after comparison"),
			);
			let shown = mcp.call("show", json!({"name": format!("{tool}-after"), "top": 100}));
			if tool == "callgrind" {
				pass(
					shown.contains("current definition"),
					"hotspots link to local source definitions",
				);
				let flagged = record["call_edges"]
					.as_array()
					.unwrap()
					.iter()
					.take(100)
					.any(|e| {
						e["attribution_flags"]
							.as_array()
							.map(|f| !f.is_empty())
							.unwrap_or(false)
					});
				if flagged {
					pass(
						shown.contains("attribution warning"),
						"suspicious call costs flagged",
					);
				}
			}
			if tool == "callgrind" {
				let ranked = mcp.call("rank", json!({"name": format!("{tool}-after"), "top": 2}));
				pass(
					ranked.contains("priority"),
					&format!(
						"{tool} ranking via {}",
						if live_jev {
							"live Jev"
						} else {
							"local Jev fixture"
						}
					),
				);
			}
			let mismatch = format!("{tool}-mismatch");
			mcp.call(
            "valgrind_run",
            json!({"name": mismatch, "tool": tool, "command": [format!("{binary}{executable}"), "4"], "timeout_s": 600}),
        );
			mcp.rejects(&format!("{tool}-before"), &mismatch);
		}
		for name in ["clean-before", "clean-after"] {
			mcp.call(
				"valgrind_run",
				json!({"name": name, "tool": "memcheck", "command": ["/bin/true"], "timeout_s": 60}),
			);
		}
		let clean = mcp.call_json(
			"compare",
			json!({"before": "clean-before", "after": "clean-after"}),
		);
		pass(
			clean["metrics"]["leak_summary/definitely_lost_bytes"]["delta"].as_f64() == Some(0.0),
			"clean Memcheck runs compare zero leaks",
		);
		pass(
			mcp.call("runs", json!({})).contains("native-before"),
			"runs lists all stored profile types",
		);
		if let Some((_, states)) = &fixture {
			let evidence = states
				.lock()
				.unwrap()
				.iter()
				.any(|s| s.get("source").is_some() && s.get("call_edges").is_some());
			pass(evidence, "ranking receives source and call-graph evidence");
		}

		pass(mcp.calls == tools, "every advertised MCP tool exercised");
		println!("ALL INTEGRATION CHECKS PASSED");
		let _ = mcp.close();
		let pids = mcp.app_pids.clone();
		drop(mcp);
		drop(enclosure);
		pass(
			pids.iter().all(|pid| !process_running(*pid)),
			"cleanup left no observed app processes",
		);
	}

	#[cfg(unix)]
	fn owned_containers() -> BTreeSet<String> {
		let output = Command::new("docker")
			.args(["ps", "-aq", "--filter", "label=valgrind.owned=true"])
			.output()
			.expect("docker on PATH");
		assert!(
			output.status.success(),
			"docker ps failed: the regressions need a running Docker daemon"
		);
		String::from_utf8_lossy(&output.stdout)
			.split_whitespace()
			.map(str::to_string)
			.collect()
	}

	/// The independently reproduced audit defects that need the real
	/// transport and the real Docker daemon. Rust unit tests cover process groups,
	/// signals, build races, parser and comparison failures.
	#[test]
	#[ignore = "needs a running Docker daemon and the profiler image"]
	#[cfg(unix)]
	fn regressions() {
		let root = root();
		let plain = Env {
			set: vec![],
			remove: vec![],
		};
		plain.run(
			&root,
			&["cargo", "build", "--workspace", "--release", "--locked"],
		);
		let runtime = root.join("target/release/daemon");
		let baseline = owned_containers();
		let tmp = Enclosure(PathBuf::from(format!(
			"/tmp/mcp-regressions-{}",
			std::process::id()
		)));
		std::fs::create_dir_all(&tmp.0).unwrap();
		let result = std::panic::catch_unwind(|| {
			let tmp = &tmp.0;
			let request = |name: &str, timeout: u64| {
				json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "valgrind_run", "arguments": {"name": name, "tool": "memcheck",
                "command": ["/bin/sh", "-c", "sleep 120"], "timeout_s": timeout}}})
				.to_string()
					+ "\n"
			};
			let started = Instant::now();
			let mut direct = Command::new(&runtime)
				.current_dir(tmp)
				.stdin(Stdio::piped())
				.stdout(Stdio::piped())
				.stderr(Stdio::piped())
				.spawn()
				.unwrap();
			direct
				.stdin
				.take()
				.unwrap()
				.write_all(request("timeout", 1).as_bytes())
				.unwrap();
			let output = direct.wait_with_output().unwrap();
			let response: Value = serde_json::from_slice(&output.stdout).unwrap();
			assert!(response["result"]["isError"] == true, "{response}");
			assert!(response.to_string().contains("timed out"), "{response}");
			assert!(started.elapsed() < Duration::from_secs(25));
			assert!(
				owned_containers() == baseline,
				"timeout left an owned container"
			);
			println!("PASS: actual Docker timeout removes container and descendants");

			// Terminate the real transport with a Docker command in flight.
			let mut runtime_command = Command::new(&runtime);
			runtime_command
				.arg("mcp")
				.current_dir(tmp)
				.stdin(Stdio::piped())
				.stdout(Stdio::piped())
				.stderr(Stdio::piped())
				.process_group(0);
			let mut running = Owned::spawn(&mut runtime_command).unwrap();
			let mut stdin = running.0.stdin.take().unwrap();
			stdin.write_all(request("cancel", 120).as_bytes()).unwrap();
			stdin.flush().unwrap();
			wait_for(Duration::from_secs(30), || {
				(!owned_containers().difference(&baseline).next().is_none()).then_some(())
			});
			signal(running.0.id(), SIGTERM);
			wait_for(Duration::from_secs(20), || {
				running.0.try_wait().ok().flatten()
			});
			assert!(
				owned_containers() == baseline,
				"runtime cancellation left a container"
			);
			println!("PASS: actual runtime cancellation waits for container cleanup");
		});
		let survivors: Vec<String> = owned_containers().difference(&baseline).cloned().collect();
		if !survivors.is_empty() {
			let status = Command::new("docker")
				.args(["rm", "-f"])
				.args(&survivors)
				.status()
				.unwrap();
			assert!(
				status.success(),
				"docker rm -f of leftover containers failed"
			);
		}
		if let Err(panic) = result {
			std::panic::resume_unwind(panic);
		}
		println!("ALL AUDIT REGRESSIONS PASSED");
	}
}

#[cfg(test)]
fn test_command() -> std::process::Command {
	crate::test_command("daemon")
}
