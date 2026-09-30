//! boiler-mcp — the control MCP (stdio, JSON-RPC 2.0) an agent drives the
//! project through: hot reload (status/reload/stop), the live preset
//! profiler channel (channels), and the valgrind loop (valgrind_run, runs,
//! show, rank — rank feeds callgrind functions or memcheck findings to Jev).
//!
//! Wired in `.mcp.json` as `boiler`; every tool also runs from the shell:
//! `cargo run --release -p boiler-mcp -- --call <tool> '<json args>'`.
//! It must run with the project root as its working directory (that is how
//! MCP servers are launched), because every path it touches — the
//! supervisor socket, the app channel socket, the `.boiler/valgrind/` runs
//! — is project-root relative.

use boiler::{channels, jev, valgrind, wire, APP_SOCKET, SUPERVISOR_SOCKET};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

const START_HINT: &str = "start one with `cargo run --release -p boiler -- serve` (from this project root)";
const SERVE_HINT: &str = "the executable serves its channels by calling boiler::channels::serve(boiler::APP_SOCKET) once at startup";

fn tool_status() -> String {
    let mut reply = wire::ask_or_guide(SUPERVISOR_SOCKET, json!({"op": "status"}), START_HINT);
    match channels::read(APP_SOCKET) {
        Ok(snapshot) => {
            let process = &snapshot["process"];
            reply.push_str(&format!(
                "\n\nchannel socket answers: pid {} uptime {} s, {} channels",
                process["pid"], process["uptime_s"], snapshot["channels_total"]
            ));
        }
        Err(_) => reply.push_str(&format!("\n\nno channel socket answers at {APP_SOCKET} — {SERVE_HINT}")),
    }
    reply
}

fn tool_reload() -> String {
    wire::ask_or_guide(SUPERVISOR_SOCKET, json!({"op": "reload"}), START_HINT)
}

fn tool_stop() -> String {
    wire::ask_or_guide(SUPERVISOR_SOCKET, json!({"op": "stop"}), START_HINT)
}

fn tool_channels() -> String {
    match channels::read(APP_SOCKET) {
        Ok(snapshot) => channels::render_table(&snapshot),
        Err(error) => format!("{error}\n{SERVE_HINT}"),
    }
}

fn tool_valgrind_run(args: &Value) -> Result<String, String> {
    let name = args["name"]
        .as_str()
        .ok_or("`name` is required: one word naming the run (.boiler/valgrind/<name>.json)")?;
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
        .ok_or("`command` is required: the argv array the container runs under valgrind (from /repo)")?;
    let tool = valgrind::Tool::from_name(args["tool"].as_str().unwrap_or("callgrind"))?;
    let request = valgrind::Request {
        name: name.to_string(),
        tool,
        command,
        build: args["build"].as_str().map(String::from),
        timeout_s: args["timeout_s"].as_u64().unwrap_or(600),
    };
    let record = valgrind::run(request)?;
    Ok(render_summary(&record))
}

fn tool_runs() -> Result<String, String> {
    let list = valgrind::list()?;
    let rows = list["runs"].as_array().cloned().unwrap_or_default();
    if rows.is_empty() {
        return Ok("no valgrind runs stored yet; make one with valgrind_run".into());
    }
    let mut lines = vec![format!("{:<20}{:<11}{:>16}{:>12}  created", "name", "tool", "total_ir", "errors")];
    for row in rows {
        let ir = row["total_ir"]
            .as_u64()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into());
        let errors = row["errors_total"]
            .as_u64()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into());
        lines.push(format!(
            "{:<20}{:<11}{ir:>16}{errors:>12}  {}",
            row["name"].as_str().unwrap_or("?"),
            row["tool"].as_str().unwrap_or("?"),
            row["created_ms"].as_u64().unwrap_or(0)
        ));
    }
    Ok(lines.join("\n"))
}

fn tool_show(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().ok_or("`name` is required: a stored run")?;
    let top = args["top"].as_u64().unwrap_or(25) as usize;
    let record = valgrind::load(name)?;
    if record["tool"] == "callgrind" {
        let functions = record["functions"].as_array().cloned().unwrap_or_default();
        let mut lines = vec![format!(
            "run `{}` (callgrind): {} total instructions across {} functions\n{:>12}  {:>6}  function",
            name,
            record["total_ir"].as_u64().unwrap_or(0),
            record["functions_total"].as_u64().unwrap_or(0),
            "ir", "share"
        )];
        for function in functions.iter().take(top) {
            lines.push(format!(
                "{:>12}  {:>5.1}%  {}",
                function["ir"].as_u64().unwrap_or(0),
                function["share"].as_f64().unwrap_or(0.0) * 100.0,
                function["location"].as_str().unwrap_or("?")
            ));
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
            let frame = finding["frame"].as_str().filter(|f| !f.is_empty());
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
    let name = args["name"].as_str().ok_or("`name` is required: a stored run")?;
    let top = args["top"].as_u64().unwrap_or(12) as usize;
    let record = valgrind::load(name)?;
    if record["tool"] == "callgrind" {
        jev::rank_callgrind(&record, top)
    } else {
        jev::rank_memcheck(&record, top)
    }
}

fn render_summary(record: &Value) -> String {
    if record["tool"] == "callgrind" {
        let mut summary = format!(
            "run `{}` (callgrind): {} instructions across {} functions · stored .boiler/valgrind/{}.json",
            record["name"].as_str().unwrap_or("?"),
            record["total_ir"].as_u64().unwrap_or(0),
            record["functions_total"].as_u64().unwrap_or(0),
            record["name"].as_str().unwrap_or("?"),
        );
        if let Some(error) = record["run_error"].as_str() {
            summary.push_str(&format!("\nthe container run reported: {error} — the numbers above are what it recorded anyway"));
        }
        let functions = record["functions"].as_array().cloned().unwrap_or_default();
        let mut lines = vec![format!("\n{:>12}  {:>6}  function", "ir", "share")];
        for function in functions.iter().take(5) {
            lines.push(format!(
                "{:>12}  {:>5.1}%  {}",
                function["ir"].as_u64().unwrap_or(0),
                function["share"].as_f64().unwrap_or(0.0) * 100.0,
                function["location"].as_str().unwrap_or("?")
            ));
        }
        format!("{summary}\n{}", lines.join("\n"))
    } else {
        let leak = &record["leak_summary"];
        format!(
            "run `{}` (memcheck): {} findings, {} errors — {} · stored .boiler/valgrind/{}.json\nleak summary: definitely lost {} bytes in {} blocks",
            record["name"].as_str().unwrap_or("?"),
            record["findings_total"].as_u64().unwrap_or(0),
            record["errors_total"].as_u64().unwrap_or(0),
            record["error_summary"].as_str().unwrap_or("?"),
            record["name"].as_str().unwrap_or("?"),
            leak["definitely_lost_bytes"].as_u64().unwrap_or(0),
            leak["definitely_lost_blocks"].as_u64().unwrap_or(0),
        )
    }
}

type Tool = (&'static str, &'static str, Value, fn(&Value) -> Result<String, String>);

fn tools() -> Vec<Tool> {
    let object = |properties: Value| json!({"type": "object", "properties": properties, "required": []});
    vec![
        (
            "status",
            "Supervisor status (running pid, swaps, last build) and whether the app's channel socket answers.",
            object(json!({})),
            |_| Ok(tool_status()),
        ),
        (
            "reload",
            "Ask the supervisor to rebuild and swap the executable now (same as editing a watched file).",
            object(json!({})),
            |_| Ok(tool_reload()),
        ),
        (
            "stop",
            "Stop the supervisor (the executable dies with it).",
            object(json!({})),
            |_| Ok(tool_stop()),
        ),
        (
            "channels",
            "The app's live preset profiler channel snapshot: counters, latency histograms (mean/p50/p90/p99/max µs) and last-values, served over the app socket.",
            object(json!({"top": {"type": "integer", "minimum": 1}})),
            |_| Ok(tool_channels()),
        ),
        (
            "valgrind_run",
            "Run a command under valgrind (callgrind or memcheck) inside the Linux container, the project mounted at /repo; parse and store the run under .boiler/valgrind/<name>.json. `build` (a shell line, e.g. `cargo build --release -p app`) runs first in the container; its cargo target is .boiler/target-linux, so host builds are untouched. The command must terminate — give a bounded workload.",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "command": {"type": "array", "items": {"type": "string"}},
                    "build": {"type": "string"},
                    "tool": {"type": "string", "enum": ["callgrind", "memcheck"]},
                    "timeout_s": {"type": "integer", "minimum": 30}
                },
                "required": ["name", "command"]
            }),
            tool_valgrind_run,
        ),
        (
            "runs",
            "Every stored valgrind run: name, tool, totals, when it was made.",
            object(json!({})),
            |_| tool_runs(),
        ),
        (
            "show",
            "One stored valgrind run as a table: callgrind — per-function instruction counts and shares; memcheck — findings with the leak and error summaries.",
            json!({
                "type": "object",
                "properties": {"name": {"type": "string"}, "top": {"type": "integer", "minimum": 1}},
                "required": ["name"]
            }),
            tool_show,
        ),
        (
            "rank",
            "Rank a stored valgrind run with Jev (TypeSafe System One, needs TYPESAFE_API_KEY): callgrind — how pressing each function is, what kind of work dominates it, whether a behavior-preserving reduction is plausible; memcheck — how severe each finding is and whether it is a real defect. Ordered by priority, most pressing first.",
            json!({
                "type": "object",
                "properties": {"name": {"type": "string"}, "top": {"type": "integer", "minimum": 1, "maximum": 30}},
                "required": ["name"]
            }),
            tool_rank,
        ),
    ]
}

fn handle(method: &str, params: &Value) -> Result<Value, String> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": params.get("protocolVersion").cloned().unwrap_or(json!("2025-06-18")),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "boiler", "version": "0.1.0"}
        })),
        "tools/list" => Ok(json!({
            "tools": tools().iter().map(|(name, description, schema, _)| json!({
                "name": name, "description": description, "inputSchema": schema
            })).collect::<Vec<_>>()
        })),
        "tools/call" => {
            let name = params["name"].as_str().unwrap_or_default();
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            let tool = tools()
                .into_iter()
                .find(|(tool, _, _, _)| *tool == name)
                .ok_or_else(|| format!("no tool `{name}`"))?;
            tool.3(&arguments).map(|text| json!({"content": [{"type": "text", "text": text}]}))
        }
        "ping" => Ok(json!({})),
        other => Err(format!("no method `{other}`")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "--call" {
        // one tool from the shell: --call <tool> '<json args>'
        let arguments = args.get(3).map(|a| serde_json::from_str(a).expect("json args"));
        let (_, _, _, run) = tools()
            .into_iter()
            .find(|(name, _, _, _)| *name == args[2])
            .unwrap_or_else(|| panic!("no tool `{}`", args[2]));
        match run(&arguments.unwrap_or(json!({}))) {
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
        let reply = match (
            message["method"].as_str().unwrap_or_default(),
            message.get("params").cloned().unwrap_or(json!({})),
        ) {
            (method, params) => handle(method, &params),
        };
        let reply = match reply {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err(error) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": error}}),
        };
        let mut stdout = std::io::stdout();
        let _ = writeln!(stdout, "{reply}");
        let _ = stdout.flush();
    }
}