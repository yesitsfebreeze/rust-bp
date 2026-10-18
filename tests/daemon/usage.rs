//! The usage fold, the diagnostic file parser, the skeleton scan and the
//! terminal stream picture.

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
	let output = std::process::Command::new(env!("CARGO_BIN_EXE_viz"))
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
		let output = std::process::Command::new(env!("CARGO_BIN_EXE_viz"))
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
