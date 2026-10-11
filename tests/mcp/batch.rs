use mcp::batch;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

struct Fixture(PathBuf);
impl Fixture {
	fn new() -> Self {
		let path = std::env::temp_dir().join(format!("batch-{}", daemon::events::fresh_id()));
		std::fs::create_dir(&path).unwrap();
		Self(path)
	}
	fn command(&self) -> Command {
		let mut command = Command::new(env!("CARGO_BIN_EXE_mcp"));
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
fn single_item_keeps_the_original_response() {
	assert_eq!(
		batch::execute("show", &json!({"name":"one"}), |args| {
			assert_eq!(args["name"], "one");
			Ok("plain text".into())
		})
		.unwrap(),
		"plain text"
	);
}

#[test]
fn invalid_envelopes_never_execute_any_item() {
	for args in [
		Value::Null,
		json!([]),
		json!({"items":null}),
		json!({"items":[]}),
		json!({"items":vec![json!({});33]}),
		json!({"items":[{},null]}),
		json!({"items":[{"items":[{}]}]}),
		json!({"items":[{"concurrency":1}]}),
		json!({"items":[{}],"name":"mixed"}),
		json!({"concurrency":2}),
		json!({"items":[{}],"concurrency":0}),
		json!({"items":[{}],"concurrency":9}),
		json!({"items":[{}],"concurrency":1.5}),
		json!({"items":[{}],"concurrency":"2"}),
	] {
		assert!(
			batch::execute("reload", &args, |_| panic!("invalid batch executed")).is_err(),
			"{args}"
		);
	}
}

#[test]
fn read_batches_overlap_with_a_bound_and_preserve_order_and_errors() {
	let active = AtomicUsize::new(0);
	let peak = AtomicUsize::new(0);
	let gate = (Mutex::new(false), Condvar::new());
	let args = json!({"items":(0..10).map(|n| json!({"n":n})).collect::<Vec<_>>(),"concurrency":3});
	let reply = batch::execute("show", &args, |item| {
		let current = active.fetch_add(1, Ordering::SeqCst) + 1;
		peak.fetch_max(current, Ordering::SeqCst);
		let mut opened = gate.0.lock().unwrap();
		if current == 3 {
			*opened = true;
			gate.1.notify_all();
		}
		let (opened, _) = gate
			.1
			.wait_timeout_while(opened, Duration::from_secs(3), |open| !*open)
			.unwrap();
		assert!(*opened, "items did not overlap");
		drop(opened);
		let n = item["n"].as_u64().unwrap();
		std::thread::sleep(Duration::from_millis(10 - n));
		active.fetch_sub(1, Ordering::SeqCst);
		if n == 3 {
			Err("missing run".into())
		} else {
			Ok(n.to_string())
		}
	})
	.unwrap_err();
	let reply: Value = serde_json::from_str(&reply).unwrap();
	assert_eq!(peak.load(Ordering::SeqCst), 3);
	assert_eq!(reply["failed"], 1);
	assert_eq!(reply["skipped"], 0);
	for (n, result) in reply["results"].as_array().unwrap().iter().enumerate() {
		assert_eq!(result["index"], n);
		assert_eq!(result["result"]["error"], n == 3);
		assert_eq!(
			result["result"]["content"],
			if n == 3 {
				"missing run".into()
			} else {
				n.to_string()
			}
		);
	}
}

#[test]
fn mutations_run_in_order_and_stop_without_rolling_back() {
	let seen = Mutex::new(Vec::new());
	let reply = batch::execute(
		"ast_grep",
		&json!({"items":[
        {"n":0}, {"n":1,"apply":true}, {"n":2}
    ],"concurrency":8}),
		|item| {
			let n = item["n"].as_u64().unwrap();
			seen.lock().unwrap().push(n);
			if n == 1 {
				Err("rewrite failed".into())
			} else {
				Ok("done".into())
			}
		},
	)
	.unwrap_err();
	assert_eq!(*seen.lock().unwrap(), vec![0, 1]);
	let reply: Value = serde_json::from_str(&reply).unwrap();
	assert_eq!(reply["failed"], 1);
	assert_eq!(reply["skipped"], 1);
	assert_eq!(reply["results"][0]["result"]["content"], "done");
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
		let content: Value = serde_json::from_str(item["result"]["content"].as_str().unwrap()).unwrap();
		assert_eq!(content["daemon_running"], false);
	}
	let output = fixture
		.command()
		.args([
			"--call",
			"events",
			r#"{"items":[{"limit":1},{"subscribe_thread":"invalid"},{"unsubscribe":true}]} "#,
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
		.contains("subagent"));
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
fn repeated_failures_escalate_across_workers_and_recovery_clears_only_matching_alert() {
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
		let body: Value = serde_json::from_str(reply["content"][0]["text"].as_str().unwrap()).unwrap();
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

#[test]
fn daemon_socket_services_concurrent_requests() {
	let fixture = Fixture::new();
	// Keep socket paths below macOS's 104-byte limit.
	let path = fixture.0.join("d.sock").to_string_lossy().into_owned();
	let listener = daemon::wire::bind(&path).unwrap();
	let gate = (Mutex::new(0), Condvar::new());
	daemon::wire::serve_concurrent(listener, move |request| {
		let mut arrived = gate.0.lock().unwrap();
		*arrived += 1;
		gate.1.notify_all();
		let (arrived, _) = gate
			.1
			.wait_timeout_while(arrived, Duration::from_secs(3), |n| *n < 3)
			.unwrap();
		json!({"overlapped":*arrived >= 3,"n":request["n"]})
	});
	std::thread::scope(|scope| {
		let calls: Vec<_> = (0..3)
			.map(|n| {
				let path = &path;
				scope.spawn(move || daemon::wire::ask(path, json!({"n":n})).unwrap())
			})
			.collect();
		for (n, call) in calls.into_iter().enumerate() {
			let reply = call.join().unwrap();
			assert_eq!(reply["n"], n);
			assert_eq!(reply["overlapped"], true);
		}
	});
}
