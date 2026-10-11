//! mcp — the control MCP (stdio, JSON-RPC 2.0) an agent drives the
//! project through: hot reload (status/reload/stop), the live preset
//! profiler channel (channels), and the valgrind loop (valgrind_run, runs,
//! show, rank — rank feeds callgrind functions or memcheck findings to Jev).
//!
//! Wired in `.mcp.json` as `app`; every tool also runs from the shell:
//! `cargo run --release -p mcp -- --call <tool> '<json args>'`.
//! It must run with the project root as its working directory (that is how
//! MCP servers are launched), because every path it touches — the
//! daemon socket, the app channel socket, the daemon dir's `valgrind/` runs
//! — is project-root relative.

use daemon::{app_socket, channels, daemon_socket, hot, native, profiles, valgrind, wire};
use mcp::{astgrep, batch, events, jev, usage};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

const SERVE_HINT: &str = "the executable serves its channels by calling daemon::channels::serve(&daemon::app_socket()) once at startup";

/// The daemon `status`/`reload`/`channels` start when no daemon answers:
/// through Cargo, so a stale or missing `target/release/daemon` is rebuilt.
fn launcher() -> std::process::Command {
	let mut command = std::process::Command::new("cargo");
	command.args([
		"run",
		"--quiet",
		"--release",
		"-p",
		"daemon",
		"--",
		"serve",
		"--idle",
		"30m",
	]);
	command
}

/// Ask the daemon, starting it first when needed.
fn daemon(op: &str) -> Result<Value, String> {
	hot::ensure_running(launcher)?;
	wire::ask(&daemon_socket(), json!({"op": op}))
}

/// Forward one `lsp` question to the daemon's rust-analyzer; the wait
/// covers a full `cargo check` for `diagnostics`.
fn tool_lsp(args: &Value) -> Result<String, String> {
	hot::ensure_running(launcher)?;
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
		if std::os::unix::net::UnixStream::connect(daemon_socket()).is_err() {
			return pretty(&json!({"ok":true,"running":false,"daemon_running":false}));
		}
		wire::ask(&daemon_socket(), json!({"op":"status"}))?
	} else {
		hot::ensure_running(launcher)?
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
	if std::os::unix::net::UnixStream::connect(daemon_socket()).is_err() {
		return pretty(&json!({"op":"stop","ok":true,"already_stopped":true}));
	}
	let mut reply = wire::ask(&daemon_socket(), json!({"op":"stop"}))?;
	reply["already_stopped"] = json!(false);
	pretty(&reply)
}

fn tool_channels(args: &Value) -> Result<String, String> {
	hot::ensure_running(launcher)?;
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
	let pid = match args["pid"].as_u64().map(|pid| pid as u32) {
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
		&json!({"pid":pid,"runner":"native","leaks":leaks,"heap":heap,
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
            mcp::docs::tool,
        ),
        (
            "events",
            "Recent compiler events (kind defaults to compiler; use all or an exact event kind). Use next_cursor as cursor for lossless forward paging; without cursor returns latest events. subscribe_thread opts this workspace into delivering failures to that Codex session via codex queue; unsubscribe stops delivery. The launcher also streams MCP error notifications.",
            object(json!({"subscribe_thread":{"type":"string"}, "unsubscribe":{"type":"boolean"},
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
            "Structural search or rewrite with ast-grep. `pattern` (ast-grep syntax: $NAME, $$$ARGS) with optional `rewrite`; or `rule` (YAML, several separated by ---). Dry run unless `apply` is true. Defaults: lang rust, paths src and tests, limit 100.",
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
            mcp::audit::tool,
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
            "Daemon status (pid, running app pid, swaps, last build, idle limit, log path) and whether the app's channel socket answers. start:false inspects without starting. Otherwise starts a detached daemon (`daemon serve --idle 30m`) when none answers; it exits after 30 minutes without requests or source edits.",
            object(json!({"start":{"type":"boolean","default":true}})),
            tool_status,
        ),
        (
            "reload",
            "Ask the daemon to rebuild and swap the executable now (same as editing a watched file). wait:true waits for completion or failure up to timeout_s (default 120). Starts the detached daemon when none answers.",
            object(json!({"wait":{"type":"boolean","default":false},"timeout_s":{"type":"integer","minimum":1,"maximum":600}})),
            tool_reload,
        ),
        (
            "stop",
            "Stop the daemon (the executable dies with it). Never starts one.",
            object(json!({})),
            tool_stop,
        ),
        (
            "channels",
            "The app's live preset profiler channel snapshot: counters, latency histograms (mean/p50/p90/p99/max µs) and last-values, served over the app socket. Starts the detached daemon when none answers.",
            object(json!({"top": {"type": "integer", "minimum": 1,"maximum":1000},"filter":{"type":"string"}})),
            tool_channels,
        ),
        (
            "memory_report",
            "Native memory debugging of the live executable, on this machine — nothing containerized (macOS: the system's `leaks`, opt-in `heap`). Leaked bytes with the allocation callstacks, read from the process where it actually runs — the honest view for graphics programs (Metal, the window server, driver allocations). The pid defaults to the supervised app.",
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
			"capabilities": {"tools": {}},
			"serverInfo": {"name": "mcp", "version": "0.1.0"},
			"instructions": mcp::docs::STARTUP
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

fn main() {
	let args: Vec<String> = std::env::args().collect();
	if args.len() >= 3 && args[1] == "--call" {
		// one tool from the shell: --call <tool> '<json args>'
		let arguments = args
			.get(3)
			.map(|a| serde_json::from_str(a).expect("json args"));
		let result =
			call(&args[2], &arguments.unwrap_or(json!({}))).unwrap_or_else(|error| panic!("{error}"));
		match result {
			Ok(text) => println!("{text}"),
			Err(error) => {
				println!("{error}");
				std::process::exit(1);
			}
		}
		return;
	}
	let stdin = std::io::stdin();
	for line in stdin.lock().lines() {
		let Ok(line) = line else { break };
		if line.trim().is_empty() {
			continue;
		}
		let Ok(message) = serde_json::from_str::<Value>(&line) else {
			continue;
		};
		let Some(id) = message.get("id").cloned() else {
			continue;
		};
		let (method, params) = (
			message["method"].as_str().unwrap_or_default(),
			message.get("params").cloned().unwrap_or(json!({})),
		);
		let reply = handle(method, &params);
		let reply = match reply {
			Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
			Err(error) => {
				json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": error}})
			}
		};
		let mut stdout = std::io::stdout();
		let _ = writeln!(stdout, "{reply}");
		let _ = stdout.flush();
	}
}
