use daemon::profiles::{compare, delta, percentile, source_context, valid_name};
use daemon::valgrind;
use serde_json::{json, Value};
use std::path::Path;

fn record(ir: u64) -> Value {
	json!({"name":"x", "tool":"callgrind", "runner":"docker", "environment":{"image":"fixed"},
        "command":["app","3"], "run_error":null, "total_ir":ir,
        "functions":[{"location":"main.rs:main", "ir":ir}]})
}
#[test]
fn rejects_same_median_but_different_repetition_workloads() {
	fn benchmark(counts: &[u64]) -> Value {
		json!({"tool":"benchmark","command":["app"],"runner":"native","environment":{},
            "repeats":3,"wall_p50_ms":1,"wall_p90_ms":2,"channel_metrics":{"work/count":2},
            "channel_snapshots": counts.iter().map(|c| json!({"channels":[{"name":"work","count":c}]})).collect::<Vec<_>>()})
	}
	let b = benchmark(&[1, 2, 100]);
	assert!(compare(&b, &benchmark(&[2, 2, 2]))
		.unwrap_err()
		.contains("counts differ"));
	assert!(compare(&b, &b).is_ok());
	assert!(compare(&b, &benchmark(&[1, 2])).is_err());
}

#[test]
fn shell_metacharacters_are_not_profile_names() {
	for name in [
		"",
		"../escape",
		"audit$(echo X)",
		"a b",
		"a;b",
		"a\n",
		"a'",
		"a`id`",
	] {
		assert!(valid_name(name).is_err(), "{name:?}");
		assert!(valgrind::run(valgrind::Request {
			name: name.into(),
			tool: valgrind::Tool::Memcheck,
			command: vec!["true".into()],
			build: None,
			timeout_s: 1
		})
		.is_err());
	}
	assert!(valid_name("audit-clean_123").is_ok());
}

#[test]
fn ambiguous_source_is_not_misattributed() {
	let root = std::env::temp_dir().join(format!("daemon-source-{}", std::process::id()));
	std::fs::create_dir_all(&root).unwrap();
	std::fs::write(
		root.join("sample.rs"),
		"struct A;\nimpl A {\nfn work() {}\n}\nstruct B;\nimpl B {\nfn work() {}\n}\n",
	)
	.unwrap();
	assert!(source_context("sample.rs:B::work", &root).is_none());
	std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn compares_costs_and_rejects_different_workloads_and_failed_runs() {
	let b = record(100);
	let mut a = record(125);
	assert_eq!(
		compare(&b, &a).unwrap()["metrics"]["total_ir"]["percent"],
		25.0
	);
	a["command"] = json!(["app", "4"]);
	assert!(compare(&b, &a).is_err());
	a = record(125);
	a["run_error"] = json!("timeout");
	assert!(compare(&b, &a).is_err());
	a = record(125);
	a["environment"] = Value::Null;
	assert!(compare(&b, &a).is_err());
}
#[test]
fn new_cost_has_no_infinite_percentage() {
	assert_eq!(delta(0.0, 12.0)["percent"], Value::Null);
	assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.5), 2.0);
}
#[test]
fn source_context_resolves_local_definition_and_rejects_outside_files() {
	let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
	let context = source_context("/repo/src/main.rs:app::main'2 [app]", &root).unwrap();
	assert!(context["snippet"].as_str().unwrap().contains("fn main()"));
	assert!(source_context("/etc/passwd:main", &root).is_none());
}

#[test]
fn comparison_omits_unchanged_functions_and_spread_describes_noise() {
	assert_eq!(
		compare(&record(100), &record(100)).unwrap()["functions"],
		json!([])
	);
	let spread = daemon::profiles::spread(&[9.0, 10.0, 11.0]);
	assert_eq!(spread["median"], 10.0);
	assert_eq!(spread["mad"], 1.0);
	assert_eq!(spread["min"], 9.0);
	assert!(daemon::profiles::spread(&[]).is_null());
}

#[test]
fn grouping_preserves_total_self_cost_and_distinct_clones() {
	let grouped = daemon::profiles::group_functions(&[
		json!({"location":"a.rs:foo","ir":10,"share":0.1}),
		json!({"location":"b.rs:foo","ir":20,"share":0.2}),
		json!({"location":"b.rs:foo'2","ir":5,"share":0.05}),
	]);
	assert_eq!(grouped.len(), 2);
	assert_eq!(grouped[0]["ir"], 30);
	assert_eq!(grouped[0]["source_entries"].as_array().unwrap().len(), 2);
}
