use daemon::valgrind::{
	container_word, fill_callgrind, memcheck_error_total, memcheck_report, now_ms, parse_allocations,
	parse_and_store, parse_annotate, parse_memcheck, Request, Tool,
};
use serde_json::json;

#[test]
fn container_word_maps_host_target_paths() {
	let rt = "/tmp/rt";
	assert_eq!(
		container_word("./target/release/leaky", rt),
		"/tmp/rt/target-linux/release/leaky"
	);
	assert_eq!(
		container_word("target/release/app", rt),
		"/tmp/rt/target-linux/release/app"
	);
	assert_eq!(
		container_word("target/gen/app/Cargo.toml", rt),
		"target/gen/app/Cargo.toml"
	);
	assert_eq!(container_word("3", rt), "3");
	assert_eq!(
		container_word("/repo/target/release/x", rt),
		"/repo/target/release/x"
	);
}

#[test]
fn missing_report_keeps_runner_failure() {
	let root = std::env::temp_dir().join(format!(
		"kern-missing-report-{}-{}",
		std::process::id(),
		now_ms()
	));
	assert!(!root.exists());
	for tool in [Tool::Memcheck, Tool::Callgrind] {
		let request = Request {
			name: "failed-build".into(),
			tool,
			command: vec!["true".into()],
			build: None,
			timeout_s: 1,
		};
		let mut record = json!({"run_error":"compiler exited 37: diagnostic-marker"});
		let error = parse_and_store(&request, &root, &mut record).unwrap_err();
		assert!(
			error.contains("compiler exited 37: diagnostic-marker"),
			"{error}"
		);
		assert!(
			error.contains("no memcheck log") || error.contains("no annotate output"),
			"{error}"
		);
		assert!(record["errors_total"].is_null());
	}
}

#[test]
fn clean_memcheck_compares_zero_leaks() {
	let text = "==1== total heap usage: 0 allocs, 0 frees, 0 bytes allocated\n==1== All heap blocks were freed -- no leaks are possible\n==1== ERROR SUMMARY: 0 errors from 0 contexts\n";
	let (_, leaks, _) = parse_memcheck(text);
	let record = json!({"name":"clean","tool":"memcheck","command":["true"],
        "runner":"native","environment":{},"allocations":parse_allocations(text),"leak_summary":leaks});
	assert_eq!(
		daemon::profiles::compare(&record, &record).unwrap()["metrics"]
			["leak_summary/definitely_lost_bytes"]["delta"],
		0.0
	);
	// An absent/truncated leak report is not evidence of zero leaks.
	assert!(
		parse_memcheck("==1== ERROR SUMMARY: 0 errors from 0 contexts").1["definitely_lost_bytes"]
			.is_null()
	);
}

#[test]
fn different_stacks_remain_separate_and_summary_counts_errors() {
	let text = "==1== Invalid read of size 8\n==1== at 0x1: work (a.rs:1)\n==1== Invalid read of size 8\n==1== at 0x2: work (b.rs:2)\n==1== ERROR SUMMARY: 30 errors from 2 contexts\n";
	let (findings, _, summary) = parse_memcheck(text);
	assert_eq!(findings.len(), 2);
	assert_eq!(findings[1]["stack"][0], "work (b.rs:2)");
	assert_eq!(memcheck_error_total(&summary), Some(30));
}

#[test]
fn annotate_rows_parse_and_noise_does_not() {
	let text = "\
--------------------------------------------------------------------------------
 I         file:function
--------------------------------------------------------------------------------
27,534,715 (44.85%)  ????:???:???: ???
6,872 (0.01%)  /repo/src/main.rs:app::parse_work
  .    ???:the end

--------------------------------------------------------------------------------
Profile data file 'x.out' (creator: callgrind-3.22.0)
--------------------------------------------------------------------------------
";
	let rows = parse_annotate(text);
	assert_eq!(rows.len(), 2, "{rows:?}");
	assert_eq!(rows[0], ("????:???:???: ???".to_string(), 27_534_715));
	assert_eq!(
		rows[1],
		("/repo/src/main.rs:app::parse_work".to_string(), 6_872)
	);
}

#[test]
fn self_costs_exclude_totals_edges_and_source_annotation() {
	let text = "Ir
---
442,692 (100.0%) PROGRAM TOTALS
---
Ir file:function
---
400,000 (90.36%) a.rs:main
42,692 ( 9.64%) b.rs:work
---
400,000 (90.36%) => b.rs:work (3x)
442,692 (100.0%) source totals
";
	let mut record = json!({});
	fill_callgrind(&mut record, "events: Dr Ir\nsummary: 12 442692\n", text).unwrap();
	assert_eq!(record["total_ir"], 442_692);
	assert_eq!(record["functions_total"], 2);
	assert_eq!(record["functions"][1]["ir"], 42_692);
	assert_eq!(record["call_graph_available"], false);
}

#[test]
fn call_edges_preserve_counts_without_inflating_self_totals() {
	let text = "Auto-annotation:  off
Ir file:function
---
90 (90.0%) * a.rs:main
1,500 (1500.0%) > b.rs:work (1,000x) [app]

10 (10.0%) * b.rs:work [app]
20 (20.0%) > b.rs:work (2x) [app]

0 ( 0.0%) * c.rs:wrapper
10 (10.0%) > b.rs:work (1x) [app]
";
	let mut record = json!({});
	fill_callgrind(&mut record, "events: Ir\nsummary: 100\n", text).unwrap();
	assert_eq!(record["total_ir"], 100);
	assert_eq!(record["functions_total"], 3);
	assert_eq!(record["call_edges"][0]["calls"], 1000);
	assert!(record["call_edges"][0]["inclusive_ir"].is_null());
	assert!(record["call_edges"][0]["ir_per_call"].is_null());
	assert_eq!(record["call_edges"][2]["inclusive_ir"], 10);
	assert_eq!(record["call_edges"][2]["ir_per_call"], 10.0);
	assert_eq!(record["call_edges"][0]["callee"], "b.rs:work [app]");
	assert_eq!(record["call_edges"][2]["caller"], "c.rs:wrapper");
	assert_eq!(record["call_graph_available"], true);
	assert_eq!(
		record["call_edges"][0]["attribution_flags"],
		json!(["inclusive_cost_exceeds_run_total"])
	);
	assert_eq!(
		record["call_edges"][1]["attribution_flags"],
		json!(["recursive_edge"])
	);
}

#[test]
fn clean_memcheck_records_allocations_without_findings() {
	let text = "==17== total heap usage: 1,000 allocs, 1,000 frees, 16,000 bytes allocated\n==17== ERROR SUMMARY: 0 errors from 0 contexts\n";
	let (findings, _, summary) = parse_memcheck(text);
	assert!(findings.is_empty());
	assert_eq!(summary, "0 errors from 0 contexts");
	assert_eq!(
		parse_allocations(text),
		json!({"allocs":1000,"frees":1000,"bytes":16000})
	);
}

#[test]
fn incomplete_or_inclusive_annotations_are_rejected() {
	let mut record = json!({});
	for ir in [99, 101] {
		let text = format!("Ir file:function\n---\n{ir} (99.0%) a.rs:main\n");
		assert!(fill_callgrind(&mut record, "events: Ir\nsummary: 100\n", &text).is_err());
	}
	assert!(fill_callgrind(&mut record, "events: Dr\nsummary: 100\n", "").is_err());
}

#[test]
fn mixed_loss_breakdown_does_not_become_block_count() {
	let text = "\
==861== 7,536 (768 direct, 6,768 indirect) bytes in 1 blocks are definitely lost in loss record 11 of 12
==861==    by 0x123: fixture::handoff (handoff.rs:12)
==861== LEAK SUMMARY:
==861==    definitely lost: 1,024 bytes in 2 blocks
==861==    indirectly lost: 6,768 bytes in 6 blocks
";
	let (findings, leak, _) = parse_memcheck(text);
	assert_eq!(findings.len(), 1);
	assert_eq!(findings[0]["count"], 1);
	assert_eq!(
		findings[0]["message"],
		"7536 bytes in 1 blocks are definitely lost in loss record 11"
	);
	assert_eq!(findings[0]["frame"], "fixture::handoff");
	assert_eq!(leak["definitely_lost_bytes"], 1024);
	assert_eq!(leak["indirectly_lost_blocks"], 6);
}

#[test]
fn leak_records_become_findings_with_blocks_as_count() {
	let text = "\
==17== 20,480 bytes in 5 blocks are definitely lost in loss record 2 of 2
==17==    by 0x12B713: leaky::main (leaky.rs:24)
==17==        by 0x110D0B: main (in /repo/leaky)
==17== 
==17== LEAK SUMMARY:
==17==    definitely lost: 20,480 bytes in 5 blocks
==17== ERROR SUMMARY: 1 errors from 1 contexts
";
	let (findings, leak, summary) = parse_memcheck(text);
	assert_eq!(findings.len(), 1, "{findings:?}");
	assert_eq!(findings[0]["kind"], "definitely lost");
	assert_eq!(findings[0]["count"], 5);
	assert_eq!(findings[0]["frame"], "leaky::main");
	assert_eq!(leak["definitely_lost_bytes"], 20_480);
	assert_eq!(summary, "1 errors from 1 contexts");
}

#[test]
fn memcheck_errors_group_and_summaries_parse() {
	let text = "\
==11== Invalid read of size 8
==11==    at 0x4005A1: parse_work (/tmp/app/target-linux/release/app)
==11==  Address 0x52bbe0 is 0 bytes inside a block of size 12 free'd
==11== Invalid read of size 8
==11==    at 0x4005A1: parse_work (/tmp/app/target-linux/release/app)
==11== Use of uninitialised value of size 8
==11==    by 0x400601: main (/tmp/app/target-linux/release/app)
==11==
==11== LEAK SUMMARY:
==11==    definitely lost: 10,240 bytes in 4 blocks
==11==    indirectly lost: 0 bytes in 0 blocks
==11==    still reachable: 8,192 bytes in 2 blocks
==11== ERROR SUMMARY: 10 errors from 3 contexts
";
	let (findings, leak, summary) = parse_memcheck(text);
	assert_eq!(findings.len(), 2, "{findings:?}");
	assert_eq!(findings[0]["kind"], "Invalid read");
	assert_eq!(findings[0]["count"], 2);
	assert_eq!(findings[0]["frame"], "parse_work");
	assert_eq!(findings[1]["kind"], "Use of uninitialised value");
	assert_eq!(leak["definitely_lost_bytes"], 10_240);
	assert_eq!(leak["definitely_lost_blocks"], 4);
	assert_eq!(leak["indirectly_lost_bytes"], 0);
	assert_eq!(leak["still_reachable_bytes"], 8_192);
	assert_eq!(summary, "10 errors from 3 contexts");
}
#[test]
fn root_allocations_do_not_come_from_a_child_summary() {
	let text = "==133== Memcheck, a memory error detector\n==133== Command: fixture\n==133== Parent PID: 1\n==136== HEAP SUMMARY:\n==136== total heap usage: 134 allocs, 41 frees, 54,903 bytes allocated\n==136== ERROR SUMMARY: 1 errors from 1 contexts\n==133== total heap usage: 157 allocs, 90 frees, 89,932 bytes allocated\n==133== ERROR SUMMARY: 3 errors from 3 contexts\n";
	assert_eq!(
		parse_allocations(text),
		json!({"allocs":157,"frees":90,"bytes":89932})
	);
}
#[test]
fn memcheck_interleaved_processes_keep_frames_and_root_metrics_separate() {
	let text = "==20== Memcheck, a memory error detector\n==20== Command: child\n==20== Parent PID: 10\n==10== Memcheck, a memory error detector\n==10== Command: root\n==10== Parent PID: 1\n==10== Invalid read of size 8\n==20== Invalid write of size 4\n==10== at 0x10: root_work (root.rs:2)\n==20== at 0x20: child_work (child.rs:3)\n==10== total heap usage: 10 allocs, 10 frees, 100 bytes allocated\n==10== All heap blocks were freed -- no leaks are possible\n==10== ERROR SUMMARY: 2 errors from 1 contexts\n==20== total heap usage: 20 allocs, 19 frees, 200 bytes allocated\n==20== definitely lost: 8 bytes in 1 blocks\n==20== ERROR SUMMARY: 3 errors from 1 contexts\n";
	let report = memcheck_report(text);
	assert_eq!(report["summary_pid"], 10);
	assert_eq!(report["allocations"]["bytes"], 100);
	assert_eq!(report["leak_summary"]["definitely_lost_bytes"], 0);
	assert_eq!(report["errors_total"], 5);
	assert_eq!(report["findings"][0]["pid"], 10);
	assert_eq!(
		report["findings"][0]["stack"],
		json!(["root_work (root.rs:2)"])
	);
	assert_eq!(report["findings"][1]["pid"], 20);
	assert_eq!(
		report["findings"][1]["stack"],
		json!(["child_work (child.rs:3)"])
	);
	assert_eq!(
		report["processes"][1]["leak_summary"]["definitely_lost_bytes"],
		8
	);
}

#[test]
fn memcheck_child_only_errors_survive_either_summary_order() {
	let header = "==20== Memcheck, a memory error detector\n==20== Parent PID: 10\n==10== Memcheck, a memory error detector\n==10== Parent PID: 1\n==20== Invalid read of size 8\n==20== at 0x20: child_work (child.rs:3)\n";
	let root = "==10== total heap usage: 10 allocs, 10 frees, 100 bytes allocated\n==10== All heap blocks were freed -- no leaks are possible\n==10== ERROR SUMMARY: 0 errors from 0 contexts\n";
	let child = "==20== total heap usage: 20 allocs, 19 frees, 200 bytes allocated\n==20== definitely lost: 8 bytes in 1 blocks\n==20== ERROR SUMMARY: 7 errors from 1 contexts\n";
	let first = memcheck_report(&format!("{header}{root}{child}"));
	let second = memcheck_report(&format!("{header}{child}{root}"));
	assert_eq!(first, second);
	assert_eq!(first["errors_total"], 7);
	assert_eq!(first["findings_total"], 1);
	assert_eq!(first["summary_pid"], 10);
	assert_eq!(first["processes"][0]["errors_total"], 0);
	assert!(first["error_summary"]
		.as_str()
		.unwrap()
		.contains("not deduplicated"));
}

#[test]
fn memcheck_ambiguous_roots_do_not_select_arbitrary_memory_metrics() {
	for header in ["", "==10== Memcheck, a memory error detector\n==10== Parent PID: 1\n==20== Memcheck, a memory error detector\n==20== Parent PID: 1\n"] {
        let report = memcheck_report(&format!("{header}==10== total heap usage: 10 allocs, 10 frees, 100 bytes allocated\n==10== ERROR SUMMARY: 0 errors from 0 contexts\n==20== total heap usage: 20 allocs, 19 frees, 200 bytes allocated\n==20== ERROR SUMMARY: 1 errors from 1 contexts\n"));
        assert!(report["summary_pid"].is_null());
        assert!(report["allocations"].is_null());
        assert!(report["leak_summary"].is_null());
        assert_eq!(report["errors_total"], 1);
        assert_eq!(report["summary_scope"], "unknown_root");
    }
}

#[test]
fn memcheck_incomplete_child_is_visible_without_an_invented_error_count() {
	let report = memcheck_report("==10== Memcheck, a memory error detector\n==10== Parent PID: 1\n==20== Invalid read of size 8\n==20== at 0x20: child_work (child.rs:3)\n==10== ERROR SUMMARY: 0 errors from 0 contexts\n");
	assert_eq!(report["errors_total"], 0);
	assert_eq!(report["findings_total"], 1);
	assert_eq!(report["incomplete_processes"], json!([20]));
	assert!(report["error_summary"]
		.as_str()
		.unwrap()
		.contains("lack a valid error summary"));
	assert!(report["processes"][1]["errors_total"].is_null());
}
#[test]
fn unprefixed_legacy_heap_counter_and_ignored_findings_stay_compatible() {
	let text = "total heap usage: 2 allocs, 2 frees, 10 bytes allocated\nERROR SUMMARY: 0 errors from 0 contexts\n";
	assert_eq!(
		parse_allocations(text),
		json!({"allocs":2,"frees":2,"bytes":10})
	);
	assert_eq!(parse_memcheck(text), (vec![], json!({}), String::new()));
}
