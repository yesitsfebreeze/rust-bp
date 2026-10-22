//! The valgrind runner: callgrind and memcheck runs of any command —
//! **native first**. When a valgrind is on the machine's PATH (Linux, Intel
//! Mac) it runs the exact binary this machine builds, no container, same
//! libc, same values. When there is none (Apple Silicon macOS builds no
//! valgrind), the run falls back to the `valgrind-v2` Linux container
//! (OrbStack or any Docker), project mounted at `/repo`, the daemon dir
//! mounted at its own path, container builds in `<daemon dir>/target-linux`
//! so they never clobber the host's `target/`.
//!
//! Every run is stored parsed under `<daemon dir>/valgrind/<name>.json`
//! (callgrind: per-function instruction counts; memcheck: grouped findings
//! with leak and error summaries), next to the raw files for reference.

use crate::util;
use serde_json::{json, Map, Value};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const IMAGE: &str = "valgrind-v2";
const CONTAINER_CARGO: &str = "valgrind-cargo";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
	Callgrind,
	Memcheck,
}

impl Tool {
	pub fn from_name(name: &str) -> Result<Tool, String> {
		match name {
			"callgrind" => Ok(Tool::Callgrind),
			"memcheck" => Ok(Tool::Memcheck),
			other => Err(format!(
				"unknown valgrind tool `{other}`: callgrind or memcheck"
			)),
		}
	}

	pub fn name(&self) -> &'static str {
		match self {
			Tool::Callgrind => "callgrind",
			Tool::Memcheck => "memcheck",
		}
	}
}

pub struct Request {
	/// Stored as `<daemon dir>/valgrind/<name>.json`; one path component.
	pub name: String,
	pub tool: Tool,
	/// The argv valgrind runs, from the project root.
	pub command: Vec<String>,
	/// A shell line run before valgrind (e.g. `cargo build --release -p
	/// app`) — natively on the host, or in the container when the docker
	/// fallback serves the run. A failed build refuses the run.
	pub build: Option<String>,
	/// Bound on the whole run. Default 600.
	pub timeout_s: u64,
}

#[doc(hidden)]
pub fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|d| d.as_millis() as u64)
		.unwrap_or(0)
}

/// A valgrind on the machine's PATH — the native runner serves the run.
pub fn native_available() -> bool {
	util::succeeds(&["valgrind".into(), "--version".into()])
}

/// Run one valgrind job: native when a valgrind exists on PATH, otherwise
/// the Linux container. The reply is the stored run record.
pub fn run(req: Request) -> Result<Value, String> {
	crate::profiles::valid_name(&req.name)?;
	if req.command.is_empty() {
		return Err("`command` is empty: give the argv valgrind runs".into());
	}
	let root = std::env::current_dir().map_err(|e| e.to_string())?;
	let dir = crate::valgrind_dir();
	crate::ensure_dir().map_err(|e| e.to_string())?;
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

	let runner = if native_available() {
		"native"
	} else {
		"docker"
	};
	// Never interpret leftover artifacts from an earlier run under this name.
	for path in [
		artifacts(&req.name).0,
		artifacts(&req.name).1,
		artifacts(&req.name).2,
		dir
			.join(format!("{}.json", req.name))
			.to_string_lossy()
			.into_owned(),
	] {
		match std::fs::remove_file(&path) {
			Ok(()) => {}
			Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
			Err(e) => return Err(e.to_string()),
		}
	}
	let started = Instant::now();
	let run_error = if runner == "native" {
		run_native(&req)?
	} else {
		run_docker(&req, &root)?
	};
	let mut record = json!({
			"name": req.name,
			"tool": req.tool.name(),
			"runner": runner,
			"command": req.command,
			"build": req.build,
			"created_ms": now_ms(),
			"run_seconds": started.elapsed().as_secs(),
			"run_error": run_error,
			"environment": crate::profiles::environment(runner),
			"provenance": crate::profiles::provenance(),
	});
	parse_and_store(&req, &root, &mut record)?;
	let path = dir.join(format!("{}.json", req.name));
	std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap())
		.map_err(|e| e.to_string())?;
	Ok(record)
}

/// Paths every run writes: absolute, under the daemon dir, which the
/// container mounts at the same path, so host and container agree.
/// Windows backslash forms break docker's `-v` host:container split and the
/// container's `sh`, so artifacts carry forward slashes on every platform.
fn artifacts(name: &str) -> (String, String, String) {
	let dir = crate::valgrind_dir();
	let path = |ext: &str| {
		dir
			.join(format!("{name}.{ext}"))
			.to_string_lossy()
			.replace('\\', "/")
	};
	(path("out"), path("log"), path("ann"))
}

/// The native runner: valgrind runs the host's own binary directly.
/// Ok(Some(text)) when valgrind exited nonzero (the numbers it recorded may
/// still be usable); Err when the build or valgrind itself failed.
fn run_native(req: &Request) -> Result<Option<String>, String> {
	let (out, log, _) = artifacts(&req.name);
	let timeout = Duration::from_secs(req.timeout_s.max(1));
	if let Some(build) = &req.build {
		util::run_with_timeout(&["sh".into(), "-c".into(), build.clone()], timeout, 2000)
			.map_err(|e| format!("the build line refused the run: {e}"))?;
	}
	let mut valgrind = vec!["valgrind".to_string()];
	match req.tool {
		Tool::Callgrind => valgrind.extend([
			"--tool=callgrind".to_string(),
			format!("--callgrind-out-file={out}"),
			format!("--log-file={log}"),
			"--".to_string(),
		]),
		Tool::Memcheck => valgrind.extend([
			"--tool=memcheck".to_string(),
			"--leak-check=full".to_string(),
			"--show-leak-kinds=definite,indirect,possible".to_string(),
			"--track-origins=yes".to_string(),
			format!("--log-file={log}"),
			"--".to_string(),
		]),
	}
	valgrind.extend(req.command.iter().cloned());
	let outcome = util::run_with_timeout(&valgrind, timeout, 2000);
	if req.tool == Tool::Callgrind {
		// Write directly to disk: large call trees would fill the helper's
		// stdout pipe before it waits, or be truncated by its output limit.
		let ann = artifacts(&req.name).2;
		util::run_with_timeout(
            &[
                "sh".into(), "-c".into(),
                format!("callgrind_annotate --threshold=100 --auto=no --tree=calling --inclusive=no --show=Ir {} > {}",
                    shell_word(&out), shell_word(&ann)),
            ],
            timeout, 2000,
        ).map_err(|e| format!("callgrind annotation failed: {e}"))?;
	}
	Ok(outcome.err())
}

/// `./target/release/X`, `target/release/X`, or an absolute path to the
/// host's or container's `target-linux` artifact all name the container
/// build, at `<container daemon dir>/target-linux/release/X`.
#[doc(hidden)]
pub fn container_word(word: &str, host_dir: &str, container_dir: &str) -> String {
	for prefix in [
		format!("{host_dir}/target-linux/"),
		format!("{container_dir}/target-linux/"),
	] {
		if let Some(rest) = word.strip_prefix(&prefix) {
			return format!("{container_dir}/target-linux/{rest}");
		}
	}
	let rest = word
		.strip_prefix("./target/")
		.or_else(|| word.strip_prefix("target/"));
	match rest {
		Some(rest) if !rest.starts_with("gen/") => format!("{container_dir}/target-linux/{rest}"),
		_ => word.to_string(),
	}
}

/// The container runner: the project mounted at /repo, the daemon dir at
/// [`CONTAINER_DAEMON_DIR`] (its own path where that is a POSIX absolute
/// path; a Windows drive path is not one), valgrind inside.
const CONTAINER_DAEMON_DIR: &str = "/host-daemon";

fn run_docker(req: &Request, root: &std::path::Path) -> Result<Option<String>, String> {
	let (out, log, ann) = artifacts(&req.name);
	// Forward slashes: docker's `-v` splits host:container at the drive
	// colon, and backslashes read as escapes in the container's `sh`.
	let daemon_dir = crate::dir().to_string_lossy().replace('\\', "/");
	// Everything the container reads or writes under the daemon dir travels
	// under its container-side path; the host still finds the artifacts at
	// the daemon dir after the same bind.
	let inside = |host: &str| shell_word(&host.replace(&daemon_dir, CONTAINER_DAEMON_DIR));
	let (out, log, ann) = (inside(&out), inside(&log), inside(&ann));
	docker_up()?;
	ensure_image()?;
	// The host's binaries cannot run in the container: a `target/release/X`
	// word means the container build of X, which lands in target-linux.
	let command = req
		.command
		.iter()
		.map(|w| container_word(w, &daemon_dir, CONTAINER_DAEMON_DIR))
		.map(|w| shell_word(&w))
		.collect::<Vec<_>>()
		.join(" ");
	let mut script = String::from("set -e\ncd /repo\n");
	if let Some(build) = &req.build {
		script.push_str(build);
		script.push('\n');
	}
	match req.tool {
		Tool::Callgrind => {
			script.push_str(&format!(
				"valgrind --tool=callgrind --callgrind-out-file={out} --log-file={log} -- {command}\n"
			));
			script.push_str(&format!(
                "callgrind_annotate --threshold=100 --auto=no --tree=calling --inclusive=no --show=Ir {out} > {ann}\n"
            ));
		}
		Tool::Memcheck => {
			script.push_str(&format!(
                "valgrind --tool=memcheck --leak-check=full --show-leak-kinds=definite,indirect,possible --track-origins=yes --log-file={log} -- {command}\n"
            ));
		}
	}
	// Name the container before starting it: a killed docker client does not
	// stop a container. Cleanup is independent of client exit and cancellation.
	let container = format!("valgrind-job-{}-{}", std::process::id(), now_ms());
	let docker = vec![
		"docker".to_string(),
		"run".to_string(),
		"--rm".to_string(),
		"--name".to_string(),
		container.clone(),
		"--label".to_string(),
		"valgrind.owned=true".to_string(),
		// `--mount`, not `-v`: the drive colon in `C:/...:/x` is ambiguous
		// to docker's `-v` parser on Windows, `--mount` names both sides.
		"--mount".to_string(),
		format!(
			"type=bind,source={},target=/repo",
			root.to_string_lossy().replace('\\', "/")
		),
		"-w".to_string(),
		"/repo".to_string(),
		"--mount".to_string(),
		format!("type=bind,source={daemon_dir},target={CONTAINER_DAEMON_DIR}"),
		"-v".to_string(),
		format!("{CONTAINER_CARGO}:/usr/local/cargo"),
		"-e".to_string(),
		"CARGO_HOME=/usr/local/cargo".to_string(),
		"-e".to_string(),
		format!("CARGO_TARGET_DIR={CONTAINER_DAEMON_DIR}/target-linux"),
		IMAGE.to_string(),
		"sh".to_string(),
		"-c".to_string(),
		script,
	];
	let outcome = util::run_with_timeout(&docker, Duration::from_secs(req.timeout_s.max(1)), 2000);
	if outcome.is_err() {
		// --rm handles success. On failure explicitly remove only this run's
		// uniquely named container, even when the client already exited.
		let removed =
			util::run_cleanup(&["docker".into(), "rm".into(), "-f".into(), container.clone()]);
		if let Err(error) = removed {
			if !error.contains("No such container") {
				return Err(format!(
					"container cleanup failed for {container}: {error}; run: {outcome:?}"
				));
			}
		}
	}
	Ok(outcome.err())
}

/// Read the run's artifacts, parse them, and fill the record the caller
/// stores. Callgrind needs the annotate listing; memcheck the log.
#[doc(hidden)]
pub fn parse_and_store(
	req: &Request,
	root: &std::path::Path,
	record: &mut Value,
) -> Result<(), String> {
	parse_report(req, root, record).map_err(|error| match record["run_error"].as_str() {
		Some(run_error) => format!("{error}; run failed: {run_error}"),
		None => error,
	})
}

fn parse_report(req: &Request, root: &std::path::Path, record: &mut Value) -> Result<(), String> {
	let (out, log, ann) = artifacts(&req.name);
	match req.tool {
		Tool::Callgrind => {
			let text = std::fs::read_to_string(root.join(&ann)).map_err(|e| {
				format!("no annotate output at {ann}: {e} (was the run killed before it finished?)")
			})?;
			let raw = std::fs::read_to_string(root.join(&out))
				.map_err(|e| format!("no raw callgrind output at {out}: {e}"))?;
			fill_callgrind(record, &raw, &text)?;
			record["files"] = json!({"annotate": ann, "raw": out, "log": log});
		}
		Tool::Memcheck => {
			let text = std::fs::read_to_string(root.join(&log))
				.map_err(|e| format!("no memcheck log at {log}: {e}"))?;
			let report = memcheck_report(&text);
			if report["error_summary"].as_str().is_none_or(str::is_empty) {
				return Err("memcheck found nothing".into());
			}
			if !report["errors_total"].is_u64() {
				return Err("invalid Memcheck error summary".into());
			}
			for (key, value) in report.as_object().unwrap() {
				record[key] = value.clone();
			}
			record["files"] = json!({"log": log});
		}
	}
	Ok(())
}

/// Every stored run, one row each, oldest first.
pub fn list() -> Result<Value, String> {
	let dir = crate::valgrind_dir();
	let mut rows = Vec::new();
	let entries = match std::fs::read_dir(&dir) {
		Ok(entries) => entries,
		Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(json!({"runs":[]})),
		Err(e) => return Err(e.to_string()),
	};
	let mut names: Vec<_> = entries
		.flatten()
		.filter_map(|e| {
			let name = e.file_name().to_string_lossy().into_owned();
			name.strip_suffix(".json").map(|n| n.to_string())
		})
		.collect();
	names.sort();
	for name in names {
		if let Ok(record) = load(&name) {
			rows.push(json!({
					"name": name,
					"tool": record["tool"],
					"runner": record["runner"],
					"created_ms": record["created_ms"],
										"wall_p50_ms":record["wall_p50_ms"], "wall_p90_ms":record["wall_p90_ms"],
					"total_ir": record.get("total_ir").cloned().unwrap_or(Value::Null),
					"errors_total": record.get("errors_total").cloned().unwrap_or(Value::Null),
					"errors_scope": record.get("errors_scope").cloned().unwrap_or(Value::Null),
					"summary_complete": record.get("summary_complete").cloned().unwrap_or(Value::Null),
					"incomplete_processes": record.get("incomplete_processes").cloned().unwrap_or(Value::Null),
			}));
		}
	}
	Ok(json!({"runs": rows}))
}

/// Load one stored run by name.
pub fn load(name: &str) -> Result<Value, String> {
	crate::profiles::valid_name(name)?;
	let path = crate::valgrind_dir().join(format!("{name}.json"));
	let text = std::fs::read_to_string(&path)
		.map_err(|_| format!("no stored run `{name}`: list them with the runs tool"))?;
	let mut record: Value =
		serde_json::from_str(&text).map_err(|e| format!("{name}.json does not parse: {e}"))?;
	// Reinterpret legacy reports from their artifacts, so show/runs/rank never
	// silently keep using the old inflated totals. No process launch on reads.
	if record["tool"] == "callgrind" && record["parser_version"] != 3 {
		let (out, _, ann) = artifacts(name);
		let raw = std::fs::read_to_string(&out)
			.map_err(|e| format!("cannot repair legacy callgrind report `{name}`: {e}"))?;
		let annotated = std::fs::read_to_string(&ann)
			.map_err(|e| format!("cannot repair legacy callgrind report `{name}`: {e}"))?;
		fill_callgrind(&mut record, &raw, &annotated)?;
	}
	if record["tool"] == "memcheck" && record["memcheck_parser_version"] != 4 {
		let req = Request {
			name: name.into(),
			tool: Tool::Memcheck,
			command: vec![],
			build: None,
			timeout_s: 0,
		};
		let root = std::env::current_dir().map_err(|e| e.to_string())?;
		parse_and_store(&req, &root, &mut record)?;
	}
	Ok(record)
}

/// A docker daemon must answer; on a Mac, try OrbStack first.
fn docker_up() -> Result<(), String> {
	if util::succeeds(&["docker".into(), "info".into()]) {
		return Ok(());
	}
	let _ = util::run_with_timeout(&["orb".into(), "start".into()], Duration::from_secs(10), 0);
	for _ in 0..20 {
		if util::cancelled() {
			return Err("cancelled".into());
		}
		std::thread::sleep(Duration::from_secs(1));
		if util::succeeds(&["docker".into(), "info".into()]) {
			return Ok(());
		}
	}
	Err("no docker daemon answers `docker info` — start it (on a Mac: `orb start`), or install a native valgrind".into())
}

/// Build the valgrind image once; reuse it afterwards. The caller runs with
/// the project root as cwd, which is the `docker build` context.
fn ensure_image() -> Result<(), String> {
	if util::succeeds(&[
		"docker".into(),
		"image".into(),
		"inspect".into(),
		IMAGE.into(),
	]) {
		return Ok(());
	}
	eprintln!("valgrind: building image {IMAGE} (once)");
	let build = vec![
		"docker".to_string(),
		"build".to_string(),
		"-f".to_string(),
		"valgrind.Dockerfile".to_string(),
		"-t".to_string(),
		IMAGE.to_string(),
		".".to_string(),
	];
	util::run_with_timeout(&build, Duration::from_secs(900), 2000)
		.map(|_| ())
		.map_err(|e| format!("cannot build image {IMAGE} from valgrind.Dockerfile: {e}"))
}

/// Quote one argv word for the container's `sh -c`.
fn shell_word(word: &str) -> String {
	let safe = !word.is_empty()
		&& word
			.bytes()
			.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'-' | b'_' | b'=' | b':'));
	if safe {
		word.to_string()
	} else {
		format!("'{}'", word.replace('\'', r"'\''"))
	}
}

/// One row of `callgrind_annotate` output: `27,534,715 (44.85%) file:function`.
fn annotate_row(line: &str) -> Option<(u64, &str)> {
	let (number, rest) = line.split_once(" (")?;
	let (percent, tail) = rest.split_once("%)")?;
	if !percent
		.trim()
		.bytes()
		.all(|b| b.is_ascii_digit() || b == b'.' || b == b',')
	{
		return None;
	}
	let ir: u64 = number.replace(',', "").trim().parse().ok()?;
	let location = tail.trim();
	if location.is_empty() {
		return None;
	}
	Some((ir, location))
}

/// Only the self-cost function table, never totals or annotated source rows.
/// Tree output marks self rows with `*` and outgoing inclusive edges with `>`.
pub fn parse_annotate(text: &str) -> Vec<(String, u64)> {
	let (functions, _) = annotate_table(text);
	functions
}

fn annotate_table(text: &str) -> (Vec<(String, u64)>, Vec<Value>) {
	let mut in_table = false;
	let mut started = false;
	let mut caller = String::new();
	let mut functions = Vec::new();
	let mut edges = Vec::new();
	for line in text.lines() {
		if line.contains("file:function") {
			in_table = true;
			continue;
		}
		if !in_table {
			continue;
		}
		let trimmed = line.trim();
		if trimmed.starts_with("---") {
			if started {
				break;
			}
			continue;
		}
		let Some((ir, location)) = annotate_row(line) else {
			continue;
		};
		started = true;
		if let Some(target) = location.strip_prefix('>') {
			if caller.is_empty() {
				continue;
			}
			// The final `(Nx)` precedes an optional object path. Function names
			// themselves may contain parentheses, spaces, or Rust generics.
			let Some((callee, suffix)) = target.trim().rsplit_once(" (") else {
				continue;
			};
			let Some((count, object)) = suffix.split_once("x)") else {
				continue;
			};
			let Ok(calls) = count.replace(',', "").parse::<u64>() else {
				continue;
			};
			edges.push(json!({
					"caller": caller, "callee": format!("{callee}{object}"),
					"calls": calls, "inclusive_ir": ir,
					"ir_per_call": if calls == 0 { 0.0 } else { ir as f64 / calls as f64 }
			}));
		} else if !location.starts_with('<')
			&& !location.starts_with("=>")
			&& location != "PROGRAM TOTALS"
			&& location.contains(':')
		{
			caller = location
				.strip_prefix('*')
				.unwrap_or(location)
				.trim()
				.to_string();
			functions.push((caller.clone(), ir));
		}
	}
	edges.sort_by_key(|e| std::cmp::Reverse(e["calls"].as_u64().unwrap_or(0)));
	(functions, edges)
}

/// Ir counts executed instructions (including repetitions). Inclusive call
/// costs overlap with self costs and with each other; never sum them as work.
#[doc(hidden)]
pub fn fill_callgrind(record: &mut Value, raw: &str, annotated: &str) -> Result<(), String> {
	let events = raw
		.lines()
		.find_map(|l| l.strip_prefix("events:"))
		.ok_or("callgrind output has no events header")?;
	let ir_index = events
		.split_whitespace()
		.position(|e| e == "Ir")
		.ok_or("callgrind output has no Ir event")?;
	let total: u64 = raw
		.lines()
		.find_map(|l| l.strip_prefix("summary:"))
		.and_then(|s| s.split_whitespace().nth(ir_index))
		.and_then(|s| s.parse().ok())
		.ok_or("callgrind output has no valid Ir summary")?;
	if total == 0 {
		return Err("callgrind recorded no instructions".into());
	}
	let (functions, mut edges) = annotate_table(annotated);
	let self_total: u64 = functions.iter().map(|(_, ir)| ir).sum();
	if self_total != total {
		return Err(format!("callgrind self costs ({self_total}) do not match raw Ir summary ({total}); annotation may be incomplete or use inclusive costs"));
	}
	let mut rows: Vec<Value> = functions
		.iter()
		.map(
			|(location, ir)| json!({"location": location, "ir": ir, "share": *ir as f64 / total as f64}),
		)
		.collect();
	rows.sort_by_key(|r| std::cmp::Reverse(r["ir"].as_u64().unwrap_or(0)));
	for edge in &mut edges {
		let mut flags = Vec::new();
		if edge["inclusive_ir"].as_u64().unwrap_or(0) > total {
			// callgrind_annotate prints cycle members (`name'2`) with inclusive
			// costs far above the run total; the number is not a cost. Keep
			// the call count, drop the number, keep the flag.
			flags.push("inclusive_cost_exceeds_run_total");
			edge["inclusive_ir"] = Value::Null;
			edge["ir_per_call"] = Value::Null;
		}
		if edge["caller"] == edge["callee"] {
			flags.push("recursive_edge");
		}
		edge["attribution_flags"] = json!(flags);
	}
	record["parser_version"] = json!(3);
	record["total_ir"] = json!(total);
	record["functions_total"] = json!(rows.len());
	record["functions"] = json!(rows);
	record["call_edges"] = json!(edges);
	record["call_graph_available"] = json!(
		annotated.contains("Auto-annotation:  off")
			&& annotated
				.lines()
				.any(|l| annotate_row(l).is_some_and(|(_, loc)| loc.starts_with('*')))
	);
	record["call_graph_note"] = json!("Call counts and inclusive edge costs are profiler attribution, not additional instructions. Edges overlap; cycles, inlining and ARM call detection can distort attribution. Counts alone do not prove redundant work.");
	Ok(())
}

struct MemcheckProcess {
	pid: u64,
	parent: Option<u64>,
	startup: bool,
	command: Option<String>,
	text: String,
}

/// Keep PID streams separate before grouping findings or attaching frames.
fn memcheck_processes(text: &str) -> Vec<MemcheckProcess> {
	let mut processes: Vec<MemcheckProcess> = Vec::new();
	for line in text.lines() {
		let Some((pid, body)) = line.strip_prefix("==").and_then(|s| s.split_once("==")) else {
			continue;
		};
		let Ok(pid) = pid.parse::<u64>() else {
			continue;
		};
		let index = match processes.iter().position(|p| p.pid == pid) {
			Some(index) => index,
			None => {
				processes.push(MemcheckProcess {
					pid,
					parent: None,
					startup: false,
					command: None,
					text: String::new(),
				});
				processes.len() - 1
			}
		};
		let process = &mut processes[index];
		let body = body.trim();
		process.startup |= body.starts_with("Memcheck,") || body.starts_with("Command:");
		if let Some(parent) = body.strip_prefix("Parent PID:") {
			process.parent = parent.trim().parse().ok();
		}
		if let Some(command) = body.strip_prefix("Command:") {
			process.command = Some(command.trim().to_owned());
		}
		process.text.push_str(line);
		process.text.push('\n');
	}
	processes.sort_by_key(|p| p.pid);
	processes
}

fn memcheck_root(processes: &[MemcheckProcess]) -> (Option<u64>, &'static str) {
	if processes.len() == 1 {
		return (Some(processes[0].pid), "single_reported_pid");
	}
	let candidates: Vec<_> = processes
		.iter()
		.filter(|p| {
			p.startup
				&& p
					.parent
					.is_none_or(|parent| !processes.iter().any(|other| other.pid == parent))
		})
		.collect();
	match candidates.as_slice() {
		[root] => (Some(root.pid), "startup_headers_and_parent_relationships"),
		_ => (None, "ambiguous_or_missing_startup_headers"),
	}
}

/// Memory counters belong to the selected root. Error counters and findings
/// cover every reported PID; summed counters are not unique events across forks.
#[doc(hidden)]
pub fn memcheck_report(text: &str) -> Value {
	let processes = memcheck_processes(text);
	let (root, selection) = memcheck_root(&processes);
	let mut records = Vec::new();
	let mut findings = Vec::new();
	let mut missing = Vec::new();
	let mut errors = Some(0_u64);
	let mut summaries = 0;
	let mut allocations = Value::Null;
	let mut leak = Value::Null;
	for process in &processes {
		let (mut own_findings, own_leak, summary) = parse_process_memcheck(&process.text);
		let own_errors = memcheck_error_total(&summary);
		if let Some(count) = own_errors {
			summaries += 1;
			errors = errors.and_then(|total| total.checked_add(count));
		} else {
			missing.push(process.pid);
		}
		for finding in &mut own_findings {
			finding["pid"] = json!(process.pid);
		}
		findings.extend(own_findings.iter().cloned());
		let own_allocations = parse_process_allocations(&process.text);
		if Some(process.pid) == root {
			allocations = own_allocations.clone();
			leak = own_leak.clone();
		}
		records.push(json!({
				"pid":process.pid, "parent_pid":process.parent, "command":process.command,
				"allocations":own_allocations, "leak_summary":own_leak,
				"error_summary":summary, "errors_total":own_errors,
				"findings_total":own_findings.len(), "findings":own_findings,
				"summary_complete":own_errors.is_some(),
		}));
	}
	let mut summary = if records.len() == 1 {
		records[0]["error_summary"]
			.as_str()
			.unwrap_or_default()
			.to_owned()
	} else if summaries > 0 {
		format!("{} errors reported across {} process summaries (summed counters; not deduplicated across forks){}",
            errors.map_or_else(|| "overflow".into(), |n| n.to_string()), summaries,
            if missing.is_empty() { String::new() } else { format!("; {} processes lack a valid error summary", missing.len()) })
	} else {
		String::new()
	};
	if root.is_none() && !summary.is_empty() {
		summary.push_str("; root memory attribution unknown");
	}
	json!({
			"memcheck_parser_version":4,
			"summary_pid":root, "summary_scope":if root.is_some() {"root_process"} else {"unknown_root"},
			"root_selection":selection, "allocations":allocations, "leak_summary":leak,
			"errors_total":if summaries > 0 {errors} else {None}, "error_summary":summary,
			"errors_scope":"sum_of_reported_process_error_counters",
			"findings_total":findings.len(), "findings":findings,
			"findings_scope":"all_reported_processes", "processes":records,
			"summary_complete":missing.is_empty() && summaries > 0 && errors.is_some(),
			"process_accounting":"PID streams, not exec images: findings combined, first heap counters and last leak/error summaries retained",
			"incomplete_processes":missing,
	})
}

/// Test seam: the allocation counters of a memcheck log, single- or multi-process.
#[doc(hidden)]
pub fn parse_allocations(text: &str) -> Value {
	if memcheck_processes(text).is_empty() {
		return parse_process_allocations(text);
	}
	memcheck_report(text)["allocations"].clone()
}

/// PID-attributed findings for every process, root leak metrics, and an
/// explicitly aggregate error summary for multi-process reports.
pub fn parse_memcheck(text: &str) -> (Vec<Value>, Value, String) {
	if memcheck_processes(text).is_empty() {
		return (vec![], json!({}), String::new());
	}
	let mut report = memcheck_report(text);
	let findings = report["findings"]
		.take()
		.as_array()
		.cloned()
		.unwrap_or_default();
	let leak = report["leak_summary"].take();
	let summary = report["error_summary"]
		.as_str()
		.unwrap_or_default()
		.to_owned();
	(findings, leak, summary)
}

fn parse_process_allocations(text: &str) -> Value {
	for line in text.lines() {
		if let Some((_, numbers)) = line.split_once("total heap usage:") {
			let values: Vec<u64> = numbers
				.replace(',', "")
				.split_whitespace()
				.filter_map(|s| s.parse().ok())
				.collect();
			if values.len() == 3 {
				return json!({"allocs": values[0], "frees": values[1], "bytes": values[2]});
			}
		}
	}
	Value::Null
}

const ERROR_STARTS: &[&str] = &[
	"Invalid read",
	"Invalid write",
	"Invalid free",
	"Mismatched free",
	"Use of uninitialised value",
	"Conditional jump or move",
	"Syscall param",
	"Unaddressable",
	"jump to the invalid address",
	"Bad permissions",
	"Invalid alignment",
];

const LOST_KINDS: &[&str] = &[
	"definitely lost",
	"indirectly lost",
	"possibly lost",
	"still reachable",
];

/// Memcheck log → grouped findings, the leak summary and the error summary.
/// Findings come in two shapes: live error lines ("Invalid read of size 8",
/// grouped by kind, message and stack, count accumulating) and leak records
/// ("20,480 bytes in 5 blocks are definitely lost in loss record 2 of 2",
/// counted in blocks). Notes ("Uninitialised value was created by …",
/// "Address 0x… is …") are not findings; stack locations are retained.
fn parse_process_memcheck(text: &str) -> (Vec<Value>, Value, String) {
	let mut findings: Vec<Value> = Vec::new();
	let mut leak = Map::new();
	let mut summary = String::new();
	let mut active = false;
	let push = |findings: &mut Vec<Value>, kind: &str, message: String, count: u64| {
		findings
			.push(json!({"kind": kind, "message": message, "count": count, "frame": "", "stack": []}));
	};
	for line in text.lines() {
		let Some(body) = line
			.strip_prefix("==")
			.and_then(|rest| rest.split_once("=="))
			.map(|(_, body)| body.trim())
		else {
			continue;
		};
		if let Some(kind) = ERROR_STARTS.iter().find(|k| body.starts_with(*k)) {
			push(&mut findings, kind, body.to_string(), 1);
			active = true;
			continue;
		}
		// "20,480 bytes in 5 blocks are definitely lost in loss record 2 of 2"
		if body.contains(" in loss record") && body.contains(" are ") {
			if let Some(kind) = LOST_KINDS.iter().find(|k| body.contains(*k)) {
				// Mixed losses include parenthesized direct/indirect byte
				// counts before "bytes in". They are not the block count.
				let normalized = body.replace(',', "");
				let first_number = |text: &str| text.split_whitespace().next()?.parse::<u64>().ok();
				let bytes = first_number(&normalized).unwrap_or(0);
				let blocks = normalized
					.split_once(" bytes in ")
					.and_then(|(_, rest)| first_number(rest))
					.unwrap_or(1);
				let record = normalized
					.split_once(" in loss record ")
					.and_then(|(_, rest)| first_number(rest));
				let message = match record {
					Some(record) => {
						format!("{bytes} bytes in {blocks} blocks are {kind} in loss record {record}")
					}
					None => body.to_string(),
				};
				push(&mut findings, kind, message, blocks.max(1));
				active = true;
				continue;
			}
		}
		if let Some(rest) = body
			.strip_prefix("at 0x")
			.or_else(|| body.strip_prefix("by 0x"))
		{
			let frame = rest
				.split_once(": ")
				.and_then(|(_, tail)| tail.split_once(" (").map(|(name, _)| name.to_string()))
				.unwrap_or_default();
			if active && !frame.is_empty() {
				if let Some(last) = findings.last_mut() {
					last["stack"].as_array_mut().unwrap().push(json!(rest
						.split_once(": ")
						.map(|(_, tail)| tail)
						.unwrap_or(rest)));
					if last["frame"].as_str() == Some("") {
						last["frame"] = json!(frame);
					}
				}
			}
			continue;
		}
		if body.is_empty() || body.ends_with("SUMMARY:") {
			active = false;
		}
		if body == "All heap blocks were freed -- no leaks are possible" {
			for kind in [
				"definitely_lost",
				"indirectly_lost",
				"possibly_lost",
				"still_reachable",
			] {
				leak.insert(format!("{kind}_bytes"), json!(0));
				leak.insert(format!("{kind}_blocks"), json!(0));
			}
		}
		for (prefix, field) in [
			("definitely lost:", "definitely_lost"),
			("indirectly lost:", "indirectly_lost"),
			("possibly lost:", "possibly_lost"),
			("still reachable:", "still_reachable"),
		] {
			if let Some(rest) = body.strip_prefix(prefix) {
				let mut numbers = rest
					.replace(',', "")
					.split_whitespace()
					.filter_map(|w| w.parse::<u64>().ok())
					.take(2)
					.collect::<Vec<_>>()
					.into_iter();
				if let Some(bytes) = numbers.next() {
					leak.insert(format!("{field}_bytes"), json!(bytes));
				}
				if let Some(blocks) = numbers.next() {
					leak.insert(format!("{field}_blocks"), json!(blocks));
				}
				break;
			}
		}
		if let Some(rest) = body.strip_prefix("ERROR SUMMARY:") {
			summary = rest.trim().to_string();
		}
	}
	let mut grouped: Vec<Value> = Vec::new();
	for finding in findings {
		if let Some(existing) = grouped.iter_mut().find(|f| {
			f["kind"] == finding["kind"]
				&& f["message"] == finding["message"]
				&& f["stack"] == finding["stack"]
		}) {
			existing["count"] =
				json!(existing["count"].as_u64().unwrap() + finding["count"].as_u64().unwrap());
		} else {
			grouped.push(finding);
		}
	}
	(grouped, Value::Object(leak), summary)
}

#[doc(hidden)]
pub fn memcheck_error_total(summary: &str) -> Option<u64> {
	summary
		.split_whitespace()
		.next()?
		.replace(',', "")
		.parse()
		.ok()
}
