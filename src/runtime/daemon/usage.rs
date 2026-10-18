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
/// reports generated-manifest sources as `target/gen/<crate>/../../../src/…`)
/// and `/` separators on every OS, so names match across platforms.
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
	let shown = folded
		.strip_prefix(root)
		.map(|p| p.display().to_string())
		.unwrap_or_else(|_| folded.display().to_string());
	if cfg!(windows) {
		shown.replace('\\', "/")
	} else {
		shown
	}
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
	match crate::platform::local_hms(time_ms) {
		Some((h, m, s)) => format!("{h:02}:{m:02}:{s:02}"),
		None => "--:--:--".into(),
	}
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

#[cfg(test)]
mod tests {
	use crate as daemon;
	fn viz_command() -> std::process::Command {
		crate::test_command("viz")
	}

	// The usage fold, the diagnostic file parser, the skeleton scan and the
	// terminal stream picture.

	use daemon::usage::{
		colour, declarations, diagnostic_files, graph, node_of, normalize, render, resolve, timeline,
		touches, visible_columns, visible_rows, Group, Skeleton, SourceFile, CHAIN_GAP_MS,
	};
	use serde_json::{json, Value};
	use std::path::Path;

	fn call(tool: &str, time_ms: u64, ok: bool, extra: Value) -> Value {
		let mut e = json!({"kind": "tool_call", "tool": tool, "time_ms": time_ms, "ok": ok, "ms": 7});
		for (k, v) in extra.as_object().cloned().unwrap_or_default() {
			e[k] = v;
		}
		e
	}

	#[test]
	fn fold_links_tools_to_files_symbols_runs_and_sequence() {
		let events = vec![
			json!({"kind": "compile_failed", "time_ms": 1_000, "diagnostics": "error[E0308]\n  --> src/main.rs:4:2\n"}),
			call("events", 2_000, true, json!({})),
			call(
				"lsp",
				3_000,
				true,
				json!({"files": ["src/a.rs", "src/b.rs"], "symbols": [{"name": "read_frame", "file": "src/a.rs"}, {"name": "read_frame", "file": "src/b.rs"}]}),
			),
			call("show", 4_000, false, json!({"runs": ["bench-1"]})),
			// a new chain: no sequence edge into it
			call("status", 4_000 + CHAIN_GAP_MS + 1, true, json!({})),
		];
		let g = graph(&events, CHAIN_GAP_MS, 0, &Skeleton::default());
		let nodes = g["nodes"].as_array().unwrap();
		let node = |id: &str| {
			nodes
				.iter()
				.find(|n| n["id"] == id)
				.cloned()
				.unwrap_or_else(|| panic!("no node {id}"))
		};
		assert_eq!(node("tool:lsp")["kind"], "tool");
		assert_eq!(node("tool:lsp")["last_ms"], 3_000);
		assert_eq!(node("file:src/a.rs")["kind"], "file");
		assert_eq!(node("file:src/main.rs")["kind"], "file");
		assert_eq!(node("sym:read_frame")["kind"], "symbol");
		assert_eq!(node("sym:read_frame")["calls"], 2);
		assert_eq!(node("run:bench-1")["kind"], "run");
		assert_eq!(node("tool:show")["errors"], 1);
		assert_eq!(node("event:compile_failed")["errors"], 1);
		let edges = g["edges"].as_array().unwrap();
		let edge = |from: &str, to: &str, kind: &str| {
			edges
				.iter()
				.find(|e| e["from"] == from && e["to"] == to && e["kind"] == kind)
				.map(|e| e["count"].as_u64().unwrap())
		};
		assert_eq!(
			edge("event:compile_failed", "file:src/main.rs", "touch"),
			Some(1)
		);
		assert_eq!(edge("event:compile_failed", "tool:events", "seq"), Some(1));
		assert_eq!(edge("tool:events", "tool:lsp", "seq"), Some(1));
		assert_eq!(edge("tool:lsp", "file:src/a.rs", "touch"), Some(1));
		assert_eq!(edge("tool:lsp", "sym:read_frame", "touch"), Some(2));
		assert_eq!(edge("sym:read_frame", "file:src/b.rs", "in"), Some(1));
		assert_eq!(edge("tool:show", "run:bench-1", "touch"), Some(1));
		assert_eq!(edge("tool:show", "tool:status", "seq"), None);
		// since_ms drops the older events and everything only they touched
		let recent = graph(&events, CHAIN_GAP_MS, 3_500, &Skeleton::default());
		let ids: Vec<&str> = recent["nodes"]
			.as_array()
			.unwrap()
			.iter()
			.filter_map(|n| n["id"].as_str())
			.collect();
		assert_eq!(ids, ["run:bench-1", "tool:show", "tool:status"]);
	}

	#[test]
	fn diagnostics_name_their_files_once() {
		let root = Path::new("/w");
		let text = "error: x\n --> src/a.rs:1:2\nwarning: y\n  --> src/a.rs:9:1\n  --> tests/t.rs:3:3\n  --> target/gen/mcp/../../../src/m.rs:1:1\n  --> /w/target/gen/app/Cargo.toml:2:1\n  --> /elsewhere/x.rs:1:1\n  --> src/cu\nnot --> a path";
		assert_eq!(
			diagnostic_files(text, root),
			[
				"/elsewhere/x.rs",
				"src/a.rs",
				"src/m.rs",
				"target/gen/app/Cargo.toml",
				"tests/t.rs"
			]
		);
		assert_eq!(normalize("a/../b/./c.rs", root), "b/c.rs");
		let (files, symbols, runs) = touches(
			&json!({"kind": "compile_finished", "cwd": "/w", "diagnostics": "  --> src/a.rs:1:1\n  --> target/gen/mcp/../../../tests/t.rs:3:3"}),
		);
		assert_eq!(files, ["src/a.rs", "tests/t.rs"]);
		assert!(symbols.is_empty() && runs.is_empty());
		assert_eq!(
			node_of(&json!({"kind": "compile_finished"})),
			"event:compile_finished"
		);
	}

	#[test]
	fn skeleton_scan_finds_files_functions_and_module_links() {
		let dir = std::env::temp_dir().join(format!("usage-skel-{}", std::process::id()));
		let _ = std::fs::remove_dir_all(&dir);
		let w = |rel: &str, text: &str| {
			let p = dir.join(rel);
			std::fs::create_dir_all(p.parent().unwrap()).unwrap();
			std::fs::write(p, text).unwrap();
		};
		w(
			"src/main.rs",
			"use daemon::{hot::Daemon, util};\nuse daemon::project_root;\nfn main() {}\n",
		);
		w(
			"src/runtime/daemon/lib.rs",
			"pub mod hot;\npub mod util;\npub fn project_root() {}\n",
		);
		w("src/runtime/daemon/hot.rs", "use crate::{events, util};\nuse crate::util::ProcessTree;\npub struct Daemon;\nimpl Daemon {\n\tpub fn run(self) {}\n\tasync fn tick() {}\n}\npub(crate) fn spawn() {}\n");
		w("src/runtime/daemon/util.rs", "pub fn now() {}\n");
		w(
			"tests/daemon/hot.rs",
			"use daemon::hot::Daemon;\n#[test]\nfn it_runs() {}\n",
		);
		w("src/notes.txt", "fn not_rust() {}");
		let skeleton = Skeleton::scan(&dir);
		let by_path = |p: &str| {
			skeleton
				.files
				.iter()
				.find(|f| f.path == p)
				.cloned()
				.unwrap()
		};
		assert_eq!(
			skeleton
				.files
				.iter()
				.map(|f| f.path.as_str())
				.collect::<Vec<_>>(),
			[
				"src/main.rs",
				"src/runtime/daemon/hot.rs",
				"src/runtime/daemon/lib.rs",
				"src/runtime/daemon/util.rs",
				"tests/daemon/hot.rs"
			]
		);
		assert_eq!(
			by_path("src/runtime/daemon/hot.rs"),
			SourceFile {
				path: "src/runtime/daemon/hot.rs".into(),
				symbols: vec!["run".into(), "spawn".into(), "tick".into()],
				// `events` has no file; `util` twice folds to one link
				uses: vec!["src/runtime/daemon/util.rs".into()],
			}
		);
		assert_eq!(
			by_path("src/main.rs").uses,
			[
				"src/runtime/daemon/hot.rs",
				"src/runtime/daemon/lib.rs",
				"src/runtime/daemon/util.rs"
			]
		);
		assert_eq!(
			by_path("src/runtime/daemon/lib.rs").uses,
			["src/runtime/daemon/hot.rs", "src/runtime/daemon/util.rs"]
		);
		assert_eq!(by_path("tests/daemon/hot.rs").symbols, ["it_runs"]);
		// the fold: idle nodes carry no calls, edges `in` and `use` exist, activity lands on the same nodes
		let touched = call(
			"lsp",
			1_000,
			true,
			json!({"symbols": [{"name": "run", "file": "src/runtime/daemon/hot.rs"}]}),
		);
		let g = graph(&[touched], CHAIN_GAP_MS, 0, &skeleton);
		let node = |id: &str| {
			g["nodes"]
				.as_array()
				.unwrap()
				.iter()
				.find(|n| n["id"] == id)
				.cloned()
		};
		assert_eq!(node("sym:now").unwrap()["calls"], 0);
		assert_eq!(node("sym:run").unwrap()["calls"], 1);
		assert_eq!(node("sym:run").unwrap()["kind"], "symbol");
		let edge = |from: &str, to: &str, kind: &str| {
			g["edges"]
				.as_array()
				.unwrap()
				.iter()
				.any(|e| e["from"] == from && e["to"] == to && e["kind"] == kind)
		};
		assert!(edge("sym:now", "file:src/runtime/daemon/util.rs", "in"));
		assert!(edge(
			"file:src/main.rs",
			"file:src/runtime/daemon/hot.rs",
			"use"
		));
		assert!(edge("tool:lsp", "sym:run", "touch"));
		let _ = std::fs::remove_dir_all(&dir);
	}

	#[test]
	fn declarations_and_resolve_read_the_common_shapes() {
		let (symbols, modules) = declarations("pub async fn a() {}\n\tfn b<T>(x: T) {}\nconst fn c() {}\nmod d;\nmod e { }\nuse std::io;\nuse crate::f::G;\nuse mcp::{self, h, i::J};\nlet fn_count = 1;\n");
		assert_eq!(symbols, ["a", "b", "c"]);
		assert_eq!(modules, ["d", "std::io", "crate::f", "mcp::h", "mcp::i"]);
		let files: Vec<String> = [
			"src/main.rs",
			"src/runtime/mcp/lib.rs",
			"src/runtime/mcp/h.rs",
			"src/runtime/mcp/i/mod.rs",
		]
		.iter()
		.map(|s| s.to_string())
		.collect();
		assert_eq!(
			resolve("src/main.rs", "mcp::h", &files).as_deref(),
			Some("src/runtime/mcp/h.rs")
		);
		assert_eq!(
			resolve("src/main.rs", "mcp::i", &files).as_deref(),
			Some("src/runtime/mcp/i/mod.rs")
		);
		assert_eq!(
			resolve("src/main.rs", "mcp::tool", &files).as_deref(),
			Some("src/runtime/mcp/lib.rs")
		);
		assert_eq!(resolve("src/main.rs", "std::io", &files), None);
		assert_eq!(
			resolve("src/runtime/mcp/h.rs", "crate::i", &files).as_deref(),
			Some("src/runtime/mcp/i/mod.rs")
		);
		assert_eq!(
			resolve("src/runtime/mcp/lib.rs", "h", &files).as_deref(),
			Some("src/runtime/mcp/h.rs")
		);
	}

	#[test]
	fn jev_calls_fold_into_one_model_node() {
		let e = json!({"kind": "jev_call", "model": "jev-latest", "time_ms": 1_000, "ok": false, "ms": 3, "questions": ["pressing"]});
		assert_eq!(node_of(&e), "jev:jev-latest");
		let g = graph(
			&[e, call("rank", 1_100, true, json!({}))],
			CHAIN_GAP_MS,
			0,
			&Skeleton::default(),
		);
		let jev = g["nodes"]
			.as_array()
			.unwrap()
			.iter()
			.find(|n| n["id"] == "jev:jev-latest")
			.cloned()
			.unwrap();
		assert_eq!(jev["kind"], "jev");
		assert_eq!(jev["errors"], 1);
		assert!(g["edges"]
			.as_array()
			.unwrap()
			.iter()
			.any(|e| e["from"] == "jev:jev-latest" && e["to"] == "tool:rank" && e["kind"] == "seq"));
	}

	fn plain(row: &str) -> String {
		let mut out = String::new();
		let mut chars = row.chars();
		while let Some(c) = chars.next() {
			if c == '\x1b' {
				for c in chars.by_ref() {
					if c == 'm' {
						break;
					}
				}
			} else {
				out.push(c);
			}
		}
		out
	}

	#[test]
	fn live_status_distinguishes_idle_following_and_history() {
		let t =
			timeline(&[json!({"kind":"tool_call","time_ms":1000,"tool":"mcp__git_fs__read","ok":true})]);
		let live = daemon::usage::live_status(&t, 6000, true, 120);
		assert!(live.contains("LIVE"));
		assert!(live.contains("last event 5s ago"));
		let history = daemon::usage::live_status(&t, 7000, false, 120);
		assert!(history.contains("HISTORY (g follows)"));
		assert!(history.contains("last event 6s ago"));
		assert_ne!(live, daemon::usage::live_status(&t, 8000, true, 120));
		assert!(
			daemon::usage::live_status(&Default::default(), 0, true, 120).contains("waiting for events")
		);
	}

	#[test]
	fn activity_outcomes_do_not_depend_on_timing() {
		for (event, expected) in [
			(json!({"kind":"tool_call","tool":"untimed","ok":true}), "OK"),
			(
				json!({"kind":"tool_call","tool":"failed","ok":false}),
				"FAIL",
			),
			(
				json!({"kind":"tool_call","tool":"unknown","ms":12}),
				"EVENT",
			),
			(json!({"kind":"compile_finished","crate":"app"}), "OK"),
			(json!({"kind":"compile_failed","crate":"app"}), "FAIL"),
			(
				json!({"kind":"compile_failed","crate":"app","ok":true}),
				"FAIL",
			),
		] {
			let mut event = event;
			event["time_ms"] = json!(1000);
			let t = timeline(&[event.clone()]);
			let rows = daemon::usage::render_activity(&t, 160, 6, 0, 0);
			assert_eq!(
				plain(&rows[2]).split_whitespace().nth(1),
				Some(expected),
				"{event}: {}",
				rows[2]
			);
		}
	}

	#[test]
	fn activity_selection_reverses_only_active_row_and_keeps_error_totals() {
		let now = daemon::health::now_ms();
		let mut events: Vec<_> = (0..3)
			.map(|n| {
				json!({
						"kind":"tool_call","tool":"lsp","args":{"method":"hover"},
						"ok":false,"time_ms":now-10+n,"id":n.to_string()
				})
			})
			.collect();
		events.push(json!({"kind":"tool_call","tool":"docs","ok":true,"time_ms":now}));
		let t = timeline(&events);
		let rows = daemon::usage::render_activity(&t, 140, 10, 0, 1);
		assert!(!rows[2].contains("\x1b[7m"));
		assert!(rows[3].starts_with("\x1b[7m"));
		assert!(rows[3].ends_with("\x1b[0m"));
		assert!(rows[2].contains("\x1b[32mOK"));
		assert!(rows[3].contains("\x1b[31mFAIL"));
		assert!(rows[0].contains("errors 3 / failed calls 3"));
		assert!(rows[0].contains("audit_required 1"));
		assert!(rows[0].starts_with("\x1b[31m"));
		assert!(rows[7].contains("3/3 failed"));
		assert!(rows.iter().all(|row| plain(row).chars().count() == 140));
		let expired = daemon::usage::live_status(&t, now + daemon::health::WINDOW_MS + 1, true, 140);
		assert!(expired.contains("audit_required 0"));
		assert!(expired.contains("errors 3 / failed calls 3"));
		assert!(expired.starts_with("\x1b[33m"));

		// viz replaces the first row with the live clock; counters must survive.
		let dir = std::env::temp_dir().join(format!("viz-health-{}", std::process::id()));
		std::fs::create_dir(&dir).unwrap();
		std::fs::write(
			dir.join("events.jsonl"),
			events.iter().map(|e| format!("{e}\n")).collect::<String>(),
		)
		.unwrap();
		let output = viz_command()
			.args(["--once", "--width", "140", "--height", "10"])
			.env("DAEMON_DIR", &dir)
			.output()
			.unwrap();
		std::fs::remove_dir_all(dir).unwrap();
		assert!(output.status.success());
		let text = String::from_utf8(output.stdout).unwrap();
		assert!(text
			.lines()
			.next()
			.unwrap()
			.contains("errors 3 / failed calls 3"));
		assert!(text.lines().next().unwrap().contains("audit_required 1"));
	}

	#[test]
	fn activity_clock_uses_local_timezone_instead_of_utc() {
		let dir = std::env::temp_dir().join(format!("viz-timezone-{}", std::process::id()));
		std::fs::create_dir(&dir).unwrap();
		std::fs::write(
			dir.join("events.jsonl"),
			format!(
				"{}\n",
				json!({
					"kind":"tool_call","tool":"mcp__git_fs__read","time_ms":1791747540000u64,"ok":true,"ms":1
				})
			),
		)
		.unwrap();
		// TZ with IANA zone names is POSIX; Windows takes the zone from the system settings.
		if cfg!(windows) {
			return;
		}
		for (zone, expected) in [("Europe/Berlin", "21:39:00"), ("UTC", "19:39:00")] {
			let output = viz_command()
				.args(["--once", "--width", "140", "--height", "8"])
				.env("DAEMON_DIR", &dir)
				.env("TZ", zone)
				.output()
				.unwrap();
			assert!(output.status.success());
			let text = String::from_utf8(output.stdout).unwrap();
			assert!(text.contains(expected), "{zone}: {text}");
			assert!(text.contains("LOCAL"));
			assert!(text.contains("MCP mcp__git_fs__read"));
		}
		std::fs::remove_dir_all(dir).unwrap();
	}

	#[test]
	fn activity_explains_builds_errors_and_handles_small_terminals() {
		let t = timeline(&[
			json!({"kind":"compile_finished","time_ms":1000,"crate":"workspace","build":"cargo build -p app","duration_ms":120}),
			json!({"kind":"tool_call","time_ms":2000,"tool":"lsp","ok":false,"ms":20,"error":"socket unavailable\u{001b}[2J"}),
		]);
		let rows = daemon::usage::render_activity(&t, 140, 10, 0, 1);
		assert_eq!(rows.len(), 10);
		assert!(rows.iter().all(|r| plain(r).chars().count() == 140));
		assert!(rows
			.iter()
			.any(|r| r.contains("Build command: cargo build -p app")));
		assert!(rows.iter().any(|r| r.contains("120ms")));
		assert!(rows.iter().any(|r| r.contains("Whole build command")));
		assert!(rows.iter().any(|r| r.contains("FAIL")));
		let error_rows = daemon::usage::render_activity(&t, 140, 10, 0, 0);
		assert!(error_rows.iter().any(|r| r.contains("socket unavailable")));
		assert!(!error_rows.iter().any(|r| r.contains("\x1b[2J")));
		for height in 0..6 {
			let rows = daemon::usage::render_activity(&t, 0, height, 0, 0);
			assert_eq!(rows.len(), height);
			assert!(rows.iter().all(String::is_empty));
		}
		let notification = timeline(&[json!({"kind":"notification_failed","time_ms":1})]);
		assert!(notification.entries[0].cell.err);
	}

	#[test]
	fn timeline_puts_now_on_top_and_render_paints_labels_cells_cursor_and_status() {
		let events = vec![
			call(
				"lsp",
				1_000,
				true,
				json!({"ms": 10, "files": ["src/a.rs"], "symbols": [{"name": "f", "file": "src/a.rs"}]}),
			),
			call(
				"lsp",
				2_000,
				true,
				json!({"ms": 1000, "files": ["src/a.rs"]}),
			),
			call(
				"lsp",
				3_000,
				false,
				json!({"ms": 100, "args": {"method": "hover"}}),
			),
			json!({"kind": "compile_failed", "time_ms": "4000", "crate": "app", "duration_ms": "250", "diagnostics": "error[E0308]: mismatched\n --> src/a.rs:3:1"}),
			json!({"kind": "jev_call", "time_ms": 5000, "model": "jev-latest", "ms": 40, "ok": true, "questions": ["q"]}),
		];
		let t = timeline(&events);
		let columns: Vec<(Group, &str, usize)> = t
			.columns
			.iter()
			.map(|c| (c.group, c.name.as_str(), c.cells.len()))
			.collect();
		assert_eq!(
			columns,
			[
				(Group::Tool, "lsp", 3),
				(Group::Build, "app", 1),
				(Group::Jev, "jev-latest", 1)
			]
		);
		// newest first, labelled by what it touched, else by its arguments
		let rows: Vec<(usize, &str)> = t
			.entries
			.iter()
			.map(|e| (e.column, e.things.as_str()))
			.collect();
		assert_eq!(
			rows,
			[
				(2, "jev-latest q"),
				(1, "src/a.rs"),
				(0, "method=hover"),
				(0, "src/a.rs"),
				(0, "src/a.rs f")
			]
		);
		// heat against the lsp median (100 ms): 10 ms fast, 1000 ms slow, the failed one white
		let lsp = &t.columns[0].cells;
		assert!(lsp[0].heat.unwrap() < 0.5 && lsp[1].heat.unwrap() > 0.5);
		assert_eq!(lsp[2].heat, Some(0.5));
		assert!(lsp[2].err);
		assert_eq!(colour(&lsp[2], 3_000), [255, 255, 255]);
		assert!(
			colour(&lsp[0], 3_000)[2] > colour(&lsp[1], 3_000)[2],
			"fast is bluer than slow"
		);
		assert!(
			colour(&lsp[0], 3_000 + 86_400_000)[2] < colour(&lsp[0], 3_000)[2],
			"a day later it faded"
		);
		assert!(
			t.entries[1].cell.err
				&& t.entries[1]
					.cell
					.label
					.starts_with("compile_failed app error")
		);
		// ten label rows, five event rows, two empty, a status line; labels 13 wide
		let rows = render(&t, 30, 18, 0, 0, (0, 0), 5_000);
		assert_eq!(rows.len(), 18);
		assert_eq!(visible_rows(&t, 18), 7);
		assert!(
			rows.iter().all(|r| plain(r).chars().count() == 30),
			"{:?}",
			rows
				.iter()
				.map(|r| plain(r).chars().count())
				.collect::<Vec<_>>()
		);
		let at = |col: usize| -> String {
			rows[..10]
				.iter()
				.map(|r| plain(r).chars().nth(col).unwrap())
				.collect()
		};
		assert_eq!(at(13).trim_end(), "lsp");
		assert_eq!(at(17).trim_end(), "app");
		assert_eq!(at(21), "jev-latest");
		assert!(
			plain(&rows[10]).starts_with("jev-latest q "),
			"{}",
			plain(&rows[10])
		);
		assert!(plain(&rows[14]).starts_with("src/a.rs f "));
		// three cells per row: the cursor cell is the diamond, the others blocks
		assert_eq!(plain(&rows[10]).chars().filter(|c| *c == '█').count(), 2);
		assert_eq!(plain(&rows[10]).chars().nth(13), Some('◆'));
		assert_eq!(plain(&rows[11]).chars().filter(|c| *c == '█').count(), 3);
		// the failed build is white; the cursor's event names the status line
		assert!(rows[11].contains("\x1b[38;2;255;255;255m█"));
		let wide = render(&t, 80, 18, 0, 0, (0, 0), 5_000);
		assert!(
			plain(&wide[17]).starts_with("00:00:05Z jev jev-latest q · typical · 1 of 5"),
			"{}",
			plain(&wide[17])
		);
		// cursor on the failed build: its row is bright and its own cell marked in its colour
		let on = render(&t, 80, 18, 0, 0, (1, 1), 5_000);
		assert!(on[11].starts_with("\x1b[0m\x1b[38;2;230;232;236msrc/a.rs"));
		assert!(on[11].contains("\x1b[38;2;255;255;255m◆"));
		assert!(plain(&on[17]).contains("· error · 2 of 5"));
		// scrolled: top row 2 leads; a narrow frame cuts a long label from the left and shows fewer columns
		let scrolled = render(&t, 30, 18, 0, 2, (0, 2), 5_000);
		assert!(plain(&scrolled[10]).starts_with("method=hover "));
		assert_eq!(visible_columns(&t, 30, 0), 3);
		assert_eq!(visible_columns(&t, 16, 0), 2);
		let narrow = render(&t, 16, 18, 1, 0, (1, 0), 5_000);
		assert!(
			plain(&narrow[10]).starts_with("…atest q "),
			"{}",
			plain(&narrow[10])
		);
		assert_eq!(plain(&narrow[0]).chars().nth(9), Some('a'));
		assert!(render(&Default::default(), 20, 5, 0, 0, (0, 0), 0)
			.iter()
			.any(|r| plain(r).starts_with("no events yet")));
	}
}
