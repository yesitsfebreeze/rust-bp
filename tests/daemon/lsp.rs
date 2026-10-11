use daemon::lsp::{prefer_name_matches, read_frame, relative, send, uri_for};
use serde_json::json;
use std::io::Cursor;
use std::path::Path;

#[test]
fn frames_round_trip_and_end_at_eof() {
	let mut wire = Vec::new();
	send(&mut wire, &json!({"id": 1, "method": "a"})).unwrap();
	send(&mut wire, &json!({"id": 2, "params": {"ü": "ß"}})).unwrap();
	assert!(wire.starts_with(b"Content-Length: "));
	let mut reader = Cursor::new(wire);
	assert_eq!(read_frame(&mut reader).unwrap().unwrap()["id"], 1);
	assert_eq!(
		read_frame(&mut reader).unwrap().unwrap()["params"]["ü"],
		"ß"
	);
	assert!(read_frame(&mut reader).unwrap().is_none());
	let mut bad = Cursor::new(b"Content-Type: x\r\n\r\n{}".to_vec());
	assert!(read_frame(&mut bad).is_err());
}

#[test]
fn paths_relativize_under_the_root_only() {
	let root = Path::new("/w/p");
	assert_eq!(uri_for(root), "file:///w/p");
	assert_eq!(relative("file:///w/p/src/a.rs", root), "src/a.rs");
	assert_eq!(relative("file:///usr/lib/x.rs", root), "/usr/lib/x.rs");
}

/// Needs `rust-analyzer` on PATH and indexes this workspace (10–60 s):
/// `cargo test -p daemon --test lsp -- --ignored`.
#[test]
#[ignore]
fn live_rust_analyzer_answers_definition() {
	let root = daemon::project_root(); // Cargo runs tests from target/gen/<crate>
	let client = daemon::lsp::Client::start(&root).unwrap();
	let line = std::fs::read_to_string(root.join("src/runtime/daemon/lib.rs"))
		.unwrap()
		.lines()
		.position(|l| l.starts_with("pub fn dir_for"))
		.unwrap()
		+ 1;
	let reply = client
		.ask(&json!({"method": "hover", "file": "src/runtime/daemon/lib.rs", "line": line, "col": 8}))
		.unwrap();
	assert!(reply["hover"].as_str().unwrap().contains("pub fn dir_for"));
	let reply = client
		.ask(&json!({"method": "symbols", "query": "ensure_running"}))
		.unwrap();
	assert_eq!(reply["symbols"][0]["file"], "src/runtime/daemon/hot.rs");
	let rename = client.ask(&json!({"method":"rename","file":"src/runtime/daemon/lib.rs","line":line,"col":8,"new_name":"dir_for_preview"})).unwrap();
	assert_eq!(rename["applied"], false);
	assert!(!rename["workspace_edit"].is_null());
	let hierarchy = client
		.ask(&json!({"method":"call_hierarchy","file":"src/runtime/daemon/lib.rs","line":line,"col":8}))
		.unwrap();
	assert!(hierarchy["hierarchy"]
		.as_array()
		.is_some_and(|a| !a.is_empty()));
	let actions = client
		.ask(&json!({"method":"code_actions","file":"src/runtime/daemon/lib.rs","line":line,"col":8}))
		.unwrap();
	assert_eq!(actions["applied"], false);
}

#[test]
fn symbols_rank_exact_then_substring_and_drop_fuzzy_noise() {
	let list = vec![
		json!({"name": "compares_costs_and_rejects_different_workloads"}),
		json!({"name": "parse_workload"}),
		json!({"name": "parse_work"}),
	];
	let ranked = prefer_name_matches(list.clone(), "parse_work");
	assert_eq!(ranked[0]["name"], "parse_work");
	assert_eq!(ranked[1]["name"], "parse_workload");
	assert_eq!(ranked.len(), 2);
	assert_eq!(prefer_name_matches(list.clone(), "PARSE_WORK").len(), 2);
	assert_eq!(prefer_name_matches(list, "prs_wrk").len(), 3);
}
