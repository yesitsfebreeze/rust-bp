//! usage — what the agent has in hand: which tools it calls, which files
//! and symbols those calls touch, which runs they read, and what followed
//! what. Folded from `events.jsonl` (compiler events plus the `tool_call`
//! and `jev_call` events the MCP appends). `graph` is the node/edge fold
//! the `usage` tool summarises; `stream` and `render` are the terminal
//! picture the `viz` binary paints: one column per tool, crate, model,
//! file, symbol and run, its calls stacked newest at the bottom, every
//! cell coloured by the call's speed against its stream's median.

use crate::events;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Two consecutive events further apart than this start a new chain: no
/// sequence edge between them.
pub const CHAIN_GAP_MS: u64 = 10 * 60 * 1000;
/// A symbol name and, when known, the file it sits in.
pub type Symbol = (String, Option<String>);
/// Diagnostics may be cut; only whole source and manifest paths count.
const SOURCE_SUFFIXES: [&str; 2] = [".rs", ".toml"];

/// The graph node an event folds into: `tool:<name>` for a tool call,
/// `event:<kind>` for everything else in the log.
pub fn node_of(event: &Value) -> String {
	if event["kind"] == "tool_call" {
		format!("tool:{}", event["tool"].as_str().unwrap_or("?"))
	} else if event["kind"] == "jev_call" {
		format!("jev:{}", event["model"].as_str().unwrap_or("jev"))
	} else {
		format!("event:{}", event["kind"].as_str().unwrap_or("?"))
	}
}

pub fn failed(event: &Value) -> bool {
	event["ok"] == json!(false)
		|| matches!(
			event["kind"].as_str(),
			Some("compile_failed" | "notification_failed")
		)
}

/// `src/x.rs` for a path under `root`, with `..` segments folded (Cargo
/// reports generated-manifest sources as `target/gen/<crate>/../../../src/…`).
pub fn normalize(path: &str, root: &Path) -> String {
	let full = if Path::new(path).is_absolute() {
		PathBuf::from(path)
	} else {
		root.join(path)
	};
	let mut folded = PathBuf::new();
	for part in full.components() {
		match part {
			std::path::Component::ParentDir => {
				folded.pop();
			}
			std::path::Component::CurDir => {}
			other => folded.push(other),
		}
	}
	folded
		.strip_prefix(root)
		.map(|p| p.display().to_string())
		.unwrap_or_else(|_| folded.display().to_string())
}

/// Files a compiler event names: every `--> path:line` in its diagnostics,
/// relative to `root`.
pub fn diagnostic_files(diagnostics: &str, root: &Path) -> Vec<String> {
	let mut files: Vec<String> = diagnostics
		.lines()
		.filter_map(|line| line.trim().strip_prefix("-->"))
		.filter_map(|rest| rest.trim().split(':').next())
		.filter(|path| SOURCE_SUFFIXES.iter().any(|s| path.ends_with(s)))
		.map(|path| normalize(path, root))
		.collect();
	files.sort();
	files.dedup();
	files
}

/// What one event touches: `(files, symbols)`; a symbol is `(name, file)`.
/// Tool calls carry both as recorded by the MCP; compiler events name the
/// files their diagnostics point at.
pub fn touches(event: &Value) -> (Vec<String>, Vec<Symbol>, Vec<String>) {
	let strings = |key: &str| -> Vec<String> {
		event[key]
			.as_array()
			.map(|items| {
				items
					.iter()
					.filter_map(Value::as_str)
					.map(str::to_string)
					.collect()
			})
			.unwrap_or_default()
	};
	let mut files = strings("files");
	if event["kind"] != "tool_call" {
		let root = event["cwd"]
			.as_str()
			.map(PathBuf::from)
			.unwrap_or_else(crate::project_root);
		files.extend(diagnostic_files(
			event["diagnostics"].as_str().unwrap_or(""),
			&root,
		));
	}
	let symbols = event["symbols"]
		.as_array()
		.map(|items| {
			items
				.iter()
				.filter_map(|s| {
					Some((
						s["name"].as_str()?.to_string(),
						s["file"].as_str().map(str::to_string),
					))
				})
				.collect()
		})
		.unwrap_or_default();
	(files, symbols, strings("runs"))
}

/// One source file of the project: its path relative to the root, the
/// functions and methods it defines, the files its `use crate::…`,
/// `use <crate>::…` and `mod …;` lines point at.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SourceFile {
	pub path: String,
	pub symbols: Vec<String>,
	pub uses: Vec<String>,
}

/// The project's code as nodes with no activity on them. Ceiling: a line
/// scanner, not a parser; symbols are plain names, so two functions with
/// one name share a node (both files get an `in` edge).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Skeleton {
	pub files: Vec<SourceFile>,
}

impl Skeleton {
	/// Every `.rs` under `<root>/src` and `<root>/tests`.
	pub fn scan(root: &Path) -> Skeleton {
		let mut paths = vec![];
		for top in ["src", "tests"] {
			walk(&root.join(top), &mut paths);
		}
		paths.sort();
		let rel: Vec<String> = paths
			.iter()
			.map(|p| normalize(&p.display().to_string(), root))
			.collect();
		let files = paths
			.iter()
			.zip(&rel)
			.map(|(path, name)| {
				let text = std::fs::read_to_string(path).unwrap_or_default();
				let (symbols, modules) = declarations(&text);
				let uses = modules
					.iter()
					.filter_map(|m| resolve(name, m, &rel))
					.filter(|target| target != name)
					.collect::<std::collections::BTreeSet<_>>()
					.into_iter()
					.collect();
				SourceFile {
					path: name.clone(),
					symbols,
					uses,
				}
			})
			.collect();
		Skeleton { files }
	}
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
	let Ok(entries) = std::fs::read_dir(dir) else {
		return;
	};
	for entry in entries.flatten() {
		let path = entry.path();
		if path.is_dir() {
			walk(&path, out);
		} else if path.extension().is_some_and(|e| e == "rs") {
			out.push(path);
		}
	}
}

/// Function names and module references in one file. A reference is
/// `crate::x`, `<name>::x` or a bare `x` from `mod x;`; `resolve` maps it.
pub fn declarations(text: &str) -> (Vec<String>, Vec<String>) {
	let mut symbols = vec![];
	let mut modules = vec![];
	let ident = |s: &str| -> String {
		s.chars()
			.take_while(|c| c.is_alphanumeric() || *c == '_')
			.collect()
	};
	for raw in text.lines() {
		let mut line = raw.trim();
		for prefix in [
			"pub(crate) ",
			"pub(super) ",
			"pub ",
			"async ",
			"unsafe ",
			"const ",
			"extern \"C\" ",
		] {
			line = line.strip_prefix(prefix).unwrap_or(line);
		}
		if let Some(rest) = line.strip_prefix("fn ") {
			let name = ident(rest);
			if !name.is_empty() {
				symbols.push(name);
			}
		} else if let Some(rest) = line.strip_prefix("mod ") {
			let name = ident(rest);
			if rest[name.len()..].trim_start().starts_with(';') {
				modules.push(name);
			}
		} else if let Some(rest) = line.strip_prefix("use ") {
			let rest = rest.trim_end_matches(';');
			let (head, tail) = rest.split_once("::").unwrap_or((rest, ""));
			if head == "crate"
				|| head.chars().all(|c| c.is_alphanumeric() || c == '_') && !tail.is_empty()
			{
				let heads = if let Some(group) = tail.strip_prefix('{') {
					group
						.trim_end_matches('}')
						.split(',')
						.map(|part| ident(part.trim()))
						.collect()
				} else {
					vec![ident(tail)]
				};
				modules.extend(
					heads
						.into_iter()
						.filter(|m| !m.is_empty() && m != "self")
						.map(|m| format!("{head}::{m}")),
				);
			}
		}
	}
	symbols.sort();
	symbols.dedup();
	(symbols, modules)
}

/// The project file a module reference from `from` points at, if one exists
/// in `files`. `crate::x` → `x.rs` beside the crate root of `from`; `d::x` →
/// `x.rs` in crate `d` (`src/runtime/d/`, or `src/` for the app), falling
/// back to that crate's `lib.rs` for a non-module item; a bare `x` → a
/// sibling `x.rs` or `x/mod.rs`.
pub fn resolve(from: &str, reference: &str, files: &[String]) -> Option<String> {
	let exists = |p: String| files.contains(&p).then_some(p);
	let dir = from.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
	let crate_dir = |name: &str| -> String {
		let under_runtime = format!("src/runtime/{name}");
		if files
			.iter()
			.any(|f| f.starts_with(&format!("{under_runtime}/")))
		{
			under_runtime
		} else {
			"src".to_string()
		}
	};
	match reference.split_once("::") {
		Some(("crate", item)) => {
			let root = if from.starts_with("src/runtime/") {
				from.splitn(4, '/').take(3).collect::<Vec<_>>().join("/")
			} else {
				"src".to_string()
			};
			exists(format!("{root}/{item}.rs")).or_else(|| exists(format!("{root}/{item}/mod.rs")))
		}
		Some((name, item)) => {
			let root = crate_dir(name);
			if !files
				.iter()
				.any(|f| f == &format!("{root}/lib.rs") || f == &format!("{root}/main.rs"))
			{
				return None;
			}
			exists(format!("{root}/{item}.rs"))
				.or_else(|| exists(format!("{root}/{item}/mod.rs")))
				.or_else(|| exists(format!("{root}/lib.rs")))
		}
		None => exists(format!("{dir}/{reference}.rs"))
			.or_else(|| exists(format!("{dir}/{reference}/mod.rs")))
			.or_else(|| exists(format!("{dir}/{reference}/mod.rs"))),
	}
}

#[derive(Default)]
struct Node {
	kind: &'static str,
	calls: u64,
	errors: u64,
	ms: u64,
	last_ms: u64,
}

/// Fold events into `{nodes, edges}`. Nodes: `tool:`, `event:`, `file:`,
/// `sym:`, `run:` with calls, errors, ms and the last time touched. Edges:
/// `seq` (one call followed another within `gap_ms`), `touch` (a call or
/// compiler event named a file, symbol or run), `in` (a symbol sits in a
/// file), plus the `skeleton`'s files and functions (`in`, `use`) with no
/// calls on them. Events before `since_ms` are left out.
pub fn graph(events: &[Value], gap_ms: u64, since_ms: u64, skeleton: &Skeleton) -> Value {
	let mut nodes: BTreeMap<String, Node> = BTreeMap::new();
	let mut edges: BTreeMap<(String, String, &'static str), u64> = BTreeMap::new();
	let mut previous: Option<(String, u64)> = None;
	let kind_of = |id: &str| -> &'static str {
		match id.split(':').next() {
			Some("tool") => "tool",
			Some("file") => "file",
			Some("sym") => "symbol",
			Some("run") => "run",
			Some("jev") => "jev",
			_ => "event",
		}
	};
	let touch = |nodes: &mut BTreeMap<String, Node>, id: String, time: u64| {
		let node = nodes.entry(id.clone()).or_default();
		node.kind = kind_of(&id);
		node.last_ms = node.last_ms.max(time);
		id
	};
	for file in &skeleton.files {
		let file_id = touch(&mut nodes, format!("file:{}", file.path), 0);
		for symbol in &file.symbols {
			let sym = touch(&mut nodes, format!("sym:{symbol}"), 0);
			edges.entry((sym, file_id.clone(), "in")).or_insert(1);
		}
		for target in &file.uses {
			let target = touch(&mut nodes, format!("file:{target}"), 0);
			edges.entry((file_id.clone(), target, "use")).or_insert(1);
		}
	}
	for event in events {
		let time = event["time_ms"].as_u64().unwrap_or(0);
		if time < since_ms {
			continue;
		}
		let id = touch(&mut nodes, node_of(event), time);
		{
			let node = nodes.get_mut(&id).unwrap();
			node.calls += 1;
			node.errors += u64::from(failed(event));
			node.ms += event["ms"].as_u64().unwrap_or(0);
		}
		if let Some((from, at)) = previous.take() {
			if time.saturating_sub(at) <= gap_ms {
				*edges.entry((from, id.clone(), "seq")).or_insert(0) += 1;
			}
		}
		previous = Some((id.clone(), time));
		let (files, symbols, runs) = touches(event);
		for file in files {
			let file = touch(&mut nodes, format!("file:{file}"), time);
			nodes.get_mut(&file).unwrap().calls += 1;
			*edges.entry((id.clone(), file, "touch")).or_insert(0) += 1;
		}
		for (name, file) in symbols {
			let sym = touch(&mut nodes, format!("sym:{name}"), time);
			nodes.get_mut(&sym).unwrap().calls += 1;
			*edges.entry((id.clone(), sym.clone(), "touch")).or_insert(0) += 1;
			if let Some(file) = file {
				let file = touch(&mut nodes, format!("file:{file}"), time);
				*edges.entry((sym, file, "in")).or_insert(0) += 1;
			}
		}
		for run in runs {
			let run = touch(&mut nodes, format!("run:{run}"), time);
			nodes.get_mut(&run).unwrap().calls += 1;
			*edges.entry((id.clone(), run, "touch")).or_insert(0) += 1;
		}
	}
	json!({
		"nodes": nodes.iter().map(|(id, n)| json!({
			"id": id, "kind": n.kind, "calls": n.calls, "errors": n.errors, "ms": n.ms, "last_ms": n.last_ms,
		})).collect::<Vec<_>>(),
		"edges": edges.iter().map(|((from, to, kind), count)| json!({
			"from": from, "to": to, "kind": kind, "count": count,
		})).collect::<Vec<_>>(),
	})
}

/// Every event in the log, in file order.
pub fn load() -> Vec<Value> {
	std::fs::read_to_string(events::log_path())
		.unwrap_or_default()
		.lines()
		.filter_map(|line| serde_json::from_str(line).ok())
		.collect()
}

/// Fold a `tool_call` event's data for the MCP: the fields `graph` reads.
pub fn tool_call_data(
	tool: &str,
	files: &[String],
	symbols: &[Symbol],
	runs: &[String],
) -> Map<String, Value> {
	let mut data = Map::new();
	data.insert("tool".into(), json!(tool));
	if !files.is_empty() {
		data.insert("files".into(), json!(files));
	}
	if !symbols.is_empty() {
		data.insert(
			"symbols".into(),
			Value::Array(
				symbols
					.iter()
					.map(|(name, file)| match file {
						Some(file) => json!({"name": name, "file": file}),
						None => json!({"name": name}),
					})
					.collect(),
			),
		);
	}
	if !runs.is_empty() {
		data.insert("runs".into(), json!(runs));
	}
	data
}

/// The column groups of the stream picture, left to right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Group {
	Tool,
	Build,
	Jev,
	Other,
	File,
	Symbol,
	Run,
}

/// One cell of the picture: a call (or a touch by one) in a column.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
	pub time_ms: u64,
	/// 0 fast … 0.5 at the stream's median … 1 slow; `None` when the call
	/// has no duration.
	pub heat: Option<f32>,
	pub err: bool,
	pub label: String,
}

/// One column: its calls, oldest first.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
	pub group: Group,
	pub name: String,
	pub cells: Vec<Cell>,
}

fn num(v: &Value) -> Option<f64> {
	v.as_f64().or_else(|| v.as_str()?.trim().parse().ok())
}

/// The stream an event belongs to: `(group, name, duration, failed)`.
fn stream_of(event: &Value) -> (Group, String, Option<f64>, bool) {
	let kind = event["kind"].as_str().unwrap_or("?");
	match kind {
		"tool_call" => (
			Group::Tool,
			event["tool"].as_str().unwrap_or("?").to_string(),
			num(&event["ms"]),
			event["ok"] == json!(false),
		),
		"compile_finished" | "compile_failed" => (
			Group::Build,
			event["crate"].as_str().unwrap_or("build").to_string(),
			num(&event["duration_ms"]),
			kind == "compile_failed",
		),
		"jev_call" => (
			Group::Jev,
			event["model"].as_str().unwrap_or("jev").to_string(),
			num(&event["ms"]),
			event["ok"] == json!(false),
		),
		_ => (Group::Other, kind.to_string(), None, false),
	}
}

fn label_of(event: &Value) -> String {
	match event["kind"].as_str().unwrap_or("") {
		"tool_call" => {
			let args = event["args"]
				.as_object()
				.map(|o| {
					o.iter()
						.map(|(k, v)| match v.as_str() {
							Some(s) => format!("{k}={s}"),
							None => format!("{k}={v}"),
						})
						.collect::<Vec<_>>()
						.join(" ")
				})
				.unwrap_or_default();
			format!("{} {args}", event["tool"].as_str().unwrap_or("?"))
				.trim_end()
				.to_string()
		}
		"jev_call" => format!(
			"jev {} {}",
			event["model"].as_str().unwrap_or(""),
			event["questions"]
				.as_array()
				.map(|q| q
					.iter()
					.filter_map(Value::as_str)
					.collect::<Vec<_>>()
					.join(", "))
				.unwrap_or_default()
		),
		kind => {
			let first = event["diagnostics"]
				.as_str()
				.and_then(|d| {
					d.lines()
						.find(|l| l.starts_with("error") || l.starts_with("warning"))
				})
				.unwrap_or("");
			format!("{kind} {} {first}", event["crate"].as_str().unwrap_or(""))
				.trim()
				.to_string()
		}
	}
}

/// One event on the timeline: its column, its cell and what it touched.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
	pub operation: String,
	pub signature: String,
	/// Recorded outcome, independent of whether duration was captured.
	pub ok: Option<bool>,
	pub duration_ms: Option<f64>,
	pub detail: String,
	pub column: usize,
	pub cell: Cell,
	/// The files, symbols and runs the call touched, else its arguments.
	pub things: String,
}

/// The timeline: tools, crates and models as columns, every event a row,
/// newest first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Timeline {
	pub health: crate::health::Health,
	pub columns: Vec<Column>,
	pub entries: Vec<Entry>,
}

/// Fold the log: one column per stream (tool, crate, model, other kind)
/// holding its calls oldest first, and every event as an entry, newest
/// first, labelled by the files, symbols and runs it touched. Heat is the
/// call's duration against its stream's median on a log2 scale (half the
/// median → 0.25, double → 0.75).
pub fn timeline(events: &[Value]) -> Timeline {
	let health = crate::health::Health::from_events(events);
	let mut events: Vec<&Value> = events
		.iter()
		.filter(|e| num(&e["time_ms"]).is_some())
		.collect();
	events.sort_by_key(|e| num(&e["time_ms"]).unwrap_or(0.0) as u64);
	let mut medians: BTreeMap<(Group, String), Vec<f64>> = BTreeMap::new();
	for e in &events {
		let (group, name, ms, _) = stream_of(e);
		if let Some(ms) = ms {
			medians.entry((group, name)).or_default().push(ms);
		}
	}
	let medians: BTreeMap<_, f64> = medians
		.into_iter()
		.map(|(k, mut v)| {
			v.sort_by(|a, b| a.total_cmp(b));
			(k, v[v.len() / 2])
		})
		.collect();
	let mut streams: BTreeMap<(Group, String), Vec<Cell>> = BTreeMap::new();
	let mut tagged = vec![];
	for e in &events {
		let (group, name, ms, err) = stream_of(e);
		let heat = ms.and_then(|ms| {
			let median = *medians.get(&(group, name.clone()))?;
			Some((0.5 + (ms.max(1.0) / median.max(1.0)).log2() as f32 / 4.0).clamp(0.0, 1.0))
		});
		let cell = Cell {
			time_ms: num(&e["time_ms"]).unwrap_or(0.0) as u64,
			heat,
			err: err || failed(e),
			label: label_of(e),
		};
		let (files, symbols, runs) = touches(e);
		let things: Vec<String> = files
			.into_iter()
			.chain(symbols.into_iter().map(|(name, _)| name))
			.chain(runs)
			.collect();
		let things = if things.is_empty() {
			match cell.label.split_once(' ') {
				Some((_, "workspace")) => "whole build".into(),
				Some((_, rest)) if !rest.is_empty() => rest.to_string(),
				_ => cell.label.clone(),
			}
		} else {
			things.join(" ")
		};
		streams
			.entry((group, name.clone()))
			.or_default()
			.push(cell.clone());
		tagged.push((
			(group, name),
			cell,
			things,
			operation(e),
			crate::health::signature(e),
			duration_ms(e),
			event_detail(e),
			e["ok"]
				.as_bool()
				.or_else(|| (e["kind"] == "compile_finished").then_some(true)),
		));
	}
	let mut columns: Vec<Column> = streams
		.into_iter()
		.map(|((group, name), cells)| Column { group, name, cells })
		.collect();
	columns.sort_by(|a, b| {
		a.group
			.cmp(&b.group)
			.then(b.cells.len().cmp(&a.cells.len()))
			.then(a.name.cmp(&b.name))
	});
	let index: BTreeMap<(Group, String), usize> = columns
		.iter()
		.enumerate()
		.map(|(i, c)| ((c.group, c.name.clone()), i))
		.collect();
	let entries = tagged
		.into_iter()
		.rev()
		.map(
			|(stream, cell, things, operation, signature, duration_ms, detail, ok)| Entry {
				operation,
				signature,
				ok: if cell.err { Some(false) } else { ok },
				duration_ms,
				detail,
				column: index[&stream],
				cell,
				things,
			},
		)
		.collect();
	Timeline {
		health,
		columns,
		entries,
	}
}

/// Human-readable operation shared by the terminal and the MCP audit.
pub fn operation(event: &Value) -> String {
	match event["kind"].as_str().unwrap_or("unknown") {
		"tool_call" => format!("MCP {}", event["tool"].as_str().unwrap_or("?")),
		"jev_call" => format!("Jev {}", event["model"].as_str().unwrap_or("?")),
		"compile_finished" | "compile_failed" => {
			if let Some(command) = event["build"].as_str() {
				format!("Build command: {command}")
			} else {
				format!(
					"Compile crate: {}",
					event["crate"].as_str().unwrap_or("unknown")
				)
			}
		}
		kind => kind.replace('_', " "),
	}
}

pub fn duration_ms(event: &Value) -> Option<f64> {
	num(&event["ms"])
		.or_else(|| num(&event["duration_ms"]))
		.filter(|ms| ms.is_finite() && *ms >= 0.0)
}

/// Details are recorded evidence, never executable instructions.
pub fn event_detail(event: &Value) -> String {
	let mut parts = vec![];
	for key in ["error", "diagnostics", "detail", "args", "cwd"] {
		if let Some(value) = event.get(key).filter(|v| !v.is_null()) {
			let text = value
				.as_str()
				.map(str::to_string)
				.unwrap_or_else(|| value.to_string());
			if !text.is_empty() {
				parts.push(format!("{key}: {text}"));
			}
		}
	}
	if event["crate"] == "workspace" {
		parts.push("Whole build command; no individual crate was identified.".into());
	}
	parts.join(" | ")
}

// Replace control characters from log data before painting terminal output.
fn fit(text: &str, width: usize) -> String {
	let clean: String = text
		.chars()
		.map(|c| if c.is_control() { ' ' } else { c })
		.take(width)
		.collect();
	format!(
		"{clean}{}",
		" ".repeat(width.saturating_sub(clean.chars().count()))
	)
}

/// Local wall-clock time, including the host timezone's daylight-saving rules.
pub fn local_clock(time_ms: u64) -> String {
	let seconds = (time_ms / 1000) as libc::time_t;
	// SAFETY: an all-zero tm is valid storage for localtime_r to fill.
	let mut local: libc::tm = unsafe { std::mem::zeroed() };
	// SAFETY: localtime_r reads a valid time_t and writes the supplied tm.
	if unsafe { libc::localtime_r(&seconds, &mut local) }.is_null() {
		return "--:--:--".into();
	}
	format!(
		"{:02}:{:02}:{:02}",
		local.tm_hour, local.tm_min, local.tm_sec
	)
}

/// A changing clock distinguishes a live but quiet viewer from a frozen one.
pub fn live_status(t: &Timeline, now: u64, following: bool, width: usize) -> String {
	let mode = if following {
		"LIVE"
	} else {
		"HISTORY (g follows)"
	};
	let age = t
		.entries
		.first()
		.map(|e| {
			format!(
				"last event {}s ago",
				now.saturating_sub(e.cell.time_ms) / 1000
			)
		})
		.unwrap_or_else(|| "waiting for events".into());
	let required = t.health.required(now).len();
	let line = fit(
		&format!(
			"{mode} {} local | errors {} / failed calls {} | audit_required {required} | {age}",
			local_clock(now),
			t.health.errors,
			t.health.failed_calls
		),
		width,
	);
	if width == 0 {
		return line;
	}
	let colour = if required > 0 {
		31
	} else if t.health.errors > 0 {
		33
	} else {
		32
	};
	format!("\x1b[{colour}m{line}\x1b[0m")
}

/// Readable default view. The heatmap remains available with --heatmap.
pub fn render_activity(
	t: &Timeline,
	width: usize,
	height: usize,
	top: usize,
	cursor: usize,
) -> Vec<String> {
	if height == 0 {
		return vec![];
	}
	let now = crate::health::now_ms();
	let mut rows = vec![
		live_status(t, now, cursor == 0, width),
		fit(
			"LOCAL     RESULT   DURATION  OPERATION / TOUCHED FILES",
			width,
		),
	];
	let body = height.saturating_sub(5);
	for i in top..top.saturating_add(body) {
		let text = if let Some(e) = t.entries.get(i) {
			let duration = e
				.duration_ms
				.map(|ms| format!("{ms:.0}ms"))
				.unwrap_or_else(|| "-".into());
			format!(
				"{} {:<7} {:>9}  {} | {}",
				local_clock(e.cell.time_ms),
				if e.cell.err {
					"FAIL"
				} else if e.ok == Some(true) {
					"OK"
				} else {
					"EVENT"
				},
				duration,
				e.operation,
				e.things
			)
		} else if t.entries.is_empty() && i == top {
			"No events yet. Run an MCP tool or build the app to record activity.".into()
		} else {
			String::new()
		};
		let line = fit(&text, width);
		if let Some(entry) = t.entries.get(i).filter(|_| width > 0) {
			let colour = if entry.cell.err {
				31
			} else if entry.ok == Some(true) {
				32
			} else {
				33
			};
			let head: String = line.chars().take(9).collect();
			let state: String = line.chars().skip(9).take(7).collect();
			let tail: String = line.chars().skip(16).collect();
			let reverse = if i == cursor { "\x1b[7m" } else { "" };
			rows.push(format!(
				"{reverse}{head}\x1b[{colour}m{state}\x1b[39m{tail}\x1b[0m"
			));
		} else {
			rows.push(line);
		}
	}
	let selected = t.entries.get(cursor);
	rows.push(fit(
		&selected
			.map(|e| {
				let state = t.health.problems.get(&e.signature);
				let (failures, calls, recent) = state
					.map(|p| (p.failures, p.calls, p.recent_failures(now)))
					.unwrap_or_default();
				format!(
					"Selected: {} | {failures}/{calls} failed | recent {recent}/{}{}",
					e.operation,
					crate::health::FAILURE_THRESHOLD,
					if recent >= crate::health::FAILURE_THRESHOLD {
						" audit_required"
					} else {
						""
					}
				)
			})
			.unwrap_or_default(),
		width,
	));
	rows.push(fit(
		&selected.map(|e| e.detail.clone()).unwrap_or_default(),
		width,
	));
	rows.push(fit(
		"j/k or arrows move | g/G newest/oldest | q quit | MCP audit: failures and slow calls",
		width,
	));
	rows.truncate(height);
	rows
}

const RAMP: [[u8; 3]; 5] = [
	[24, 56, 255],
	[24, 200, 255],
	[34, 216, 90],
	[242, 224, 32],
	[255, 42, 16],
];
const RAIL: [u8; 3] = [18, 18, 22];
const RAIL_LIT: [u8; 3] = [44, 44, 52];
const DIM: [u8; 3] = [108, 112, 122];
const BRIGHT: [u8; 3] = [230, 232, 236];
/// Column labels this tall; longer names are cut.
pub const LABEL_ROWS: usize = 10;
/// Row labels this wide at most; longer names keep their tail.
pub const LABEL_COLS: usize = 28;

/// The cell's colour: the ramp blue · cyan · green · yellow · red by heat,
/// white for an error, slate for a call without duration; older cells
/// fade (a minute old full, an hour old 70 %, a day old 50 %).
pub fn colour(cell: &Cell, now_ms: u64) -> [u8; 3] {
	let minutes = (now_ms.saturating_sub(cell.time_ms) as f32 / 60_000.0).max(1.0);
	let k = (1.0 - minutes.log10() / 5.0).clamp(0.4, 1.0);
	let base = if cell.err {
		[255, 255, 255]
	} else if let Some(h) = cell.heat {
		let t = h.clamp(0.0, 1.0) * 4.0;
		let i = (t.floor() as usize).min(3);
		let f = t - i as f32;
		let mut c = [0u8; 3];
		for (j, slot) in c.iter_mut().enumerate() {
			*slot = (RAMP[i][j] as f32 + (RAMP[i + 1][j] as f32 - RAMP[i][j] as f32) * f).round() as u8;
		}
		c
	} else {
		[96, 108, 150]
	};
	base.map(|v| (v as f32 * k).round() as u8)
}

/// Column x positions: one char per column, one gap between columns, two
/// more between groups.
fn positions(columns: &[Column]) -> Vec<usize> {
	let mut x = 0;
	let mut out = Vec::with_capacity(columns.len());
	let mut group = None;
	for c in columns {
		if group.is_some_and(|g| g != c.group) {
			x += 2;
		}
		group = Some(c.group);
		out.push(x);
		x += 2;
	}
	out
}

fn fg(c: [u8; 3]) -> String {
	format!("\x1b[38;2;{};{};{}m", c[0], c[1], c[2])
}
fn bg(c: [u8; 3]) -> String {
	format!("\x1b[48;2;{};{};{}m", c[0], c[1], c[2])
}

/// The width of the row labels in a frame `width` wide.
fn label_width(t: &Timeline, width: usize) -> usize {
	t.entries
		.iter()
		.map(|e| e.things.chars().count())
		.max()
		.unwrap_or(0)
		.clamp(4, LABEL_COLS)
		.min(width / 2)
		+ 1
}

/// How many columns a frame `width` wide shows from `scroll_x` on.
pub fn visible_columns(t: &Timeline, width: usize, scroll_x: usize) -> usize {
	let left = label_width(t, width);
	let xs = positions(&t.columns);
	let offset = xs.get(scroll_x).copied().unwrap_or(0);
	xs.iter()
		.skip(scroll_x)
		.take_while(|x| left + *x - offset < width)
		.count()
}

/// How many event rows a frame `height` tall shows.
pub fn visible_rows(t: &Timeline, height: usize) -> usize {
	height.saturating_sub(label_rows(t, height) + 1)
}

fn label_rows(t: &Timeline, height: usize) -> usize {
	t.columns
		.iter()
		.map(|c| c.name.chars().count().min(LABEL_ROWS))
		.max()
		.unwrap_or(0)
		.min(height.saturating_sub(3))
}

/// Paint `height` terminal rows of `width` cells: the column names written
/// downwards on top, then from row `top` one row per event, newest first,
/// labelled on the left by what it touched, its cell a block in its
/// column coloured by speed, the rest of the row the dark rail; the cursor
/// row is lit and its cell marked; the status line names the cursor's
/// event. `scroll_x` is in columns, `cursor` is `(column, row)` in the
/// timeline. Rows carry ANSI 24-bit colours and end with a reset.
pub fn render(
	t: &Timeline,
	width: usize,
	height: usize,
	scroll_x: usize,
	top: usize,
	cursor: (usize, usize),
	now_ms: u64,
) -> Vec<String> {
	let label_rows = label_rows(t, height);
	let left = label_width(t, width);
	let body_rows = height.saturating_sub(label_rows + 1);
	let xs = positions(&t.columns);
	let offset = xs.get(scroll_x).copied().unwrap_or(0);
	let visible: Vec<(usize, usize)> = xs
		.iter()
		.enumerate()
		.skip(scroll_x)
		.map(|(i, x)| (i, left + x - offset))
		.take_while(|(_, x)| *x < width)
		.collect();
	let mut rows = Vec::with_capacity(height);
	for r in 0..label_rows {
		let mut row = String::from("\x1b[0m");
		let mut at = 0;
		for (i, x) in &visible {
			if let Some(ch) = t.columns[*i].name.chars().nth(r) {
				row.push_str(&" ".repeat(x - at));
				row.push_str(&fg(if *i == cursor.0 { BRIGHT } else { DIM }));
				row.push(ch);
				at = x + 1;
			}
		}
		row.push_str(&" ".repeat(width.saturating_sub(at)));
		row.push_str("\x1b[0m");
		rows.push(row);
	}
	for r in top..top + body_rows {
		let Some(e) = t.entries.get(r) else {
			rows.push(format!("\x1b[0m{}", " ".repeat(width)));
			continue;
		};
		let on = r == cursor.1;
		let chars: Vec<char> = e.things.chars().collect();
		let cut: String = if chars.len() > left - 1 {
			std::iter::once('…')
				.chain(chars[chars.len() + 2 - left..].iter().copied())
				.collect()
		} else {
			chars.iter().collect()
		};
		let mut row = format!(
			"\x1b[0m{}{cut:<w$}",
			fg(if on { BRIGHT } else { DIM }),
			w = left
		);
		let mut at = left;
		for (i, x) in &visible {
			if *x > at {
				row.push_str(&" ".repeat(x - at));
			}
			let lit = *i == e.column;
			let here = on && *i == cursor.0;
			let c = if lit {
				colour(&e.cell, now_ms)
			} else if on {
				RAIL_LIT
			} else {
				RAIL
			};
			if here {
				row.push_str(&format!(
					"{}{}◆\x1b[0m",
					bg(if lit { RAIL_LIT } else { BRIGHT }),
					fg(if lit { c } else { [0, 0, 0] })
				));
			} else {
				row.push_str(&format!("{}█\x1b[0m", fg(c)));
			}
			at = x + 1;
		}
		row.push_str(&" ".repeat(width.saturating_sub(at)));
		row.push_str("\x1b[0m");
		rows.push(row);
	}
	let status = match t.entries.get(cursor.1) {
		Some(e) => {
			let c = &e.cell;
			let secs = c.time_ms / 1000 % 86_400;
			let heat = match c.heat {
				Some(h) => format!(
					" · {}",
					if h < 0.4 {
						"fast"
					} else if h > 0.6 {
						"slow"
					} else {
						"typical"
					}
				),
				None => String::new(),
			};
			format!(
				"{:02}:{:02}:{:02}Z {}{heat}{} · {} of {} · ↑↓←→ hjkl wasd move · q quit",
				secs / 3600,
				secs / 60 % 60,
				secs % 60,
				c.label,
				if c.err { " · error" } else { "" },
				cursor.1 + 1,
				t.entries.len()
			)
		}
		None => "no events yet · the log fills as the agent works · q quit".to_string(),
	};
	let status: String = status.chars().take(width).collect();
	let pad = width.saturating_sub(status.chars().count());
	rows.push(format!(
		"\x1b[0m{}{status}{}\x1b[0m",
		fg(DIM),
		" ".repeat(pad)
	));
	rows.truncate(height);
	rows
}
