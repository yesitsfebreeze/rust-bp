use daemon::native::summarize;

#[test]
fn a_leaks_reply_compresses_to_summary_and_stacks() {
	let text = "\
Process 1234: 4 nodes malloced for 1024 bytes
Process 1234: 2 leaks for 320 total leaked bytes.

Stack backtrace for leak 1: 320 bytes
0   app                                0x1000 _foo + 32
1   app                                0x2000 _main + 48

Stack backtrace for leak 2: 64 bytes
0   app                                0x1000 _bar + 32
";
	let summary = summarize(text, 1);
	assert!(summary.contains("2 leaks for 320"), "{summary}");
	assert!(summary.contains("leak 1"), "{summary}");
	assert!(summary.contains("2 leak call stacks total"), "{summary}");
	assert!(!summary.contains("_bar"), "{summary}");
	assert!(!summary.contains("nodes malloced"), "{summary}");
}

#[test]
fn a_clean_leaks_reply_says_so() {
	let text = "Process 1234: 100 nodes malloced for 4096 bytes\nProcess 1234: 0 leaks for 0 total leaked bytes.\n";
	let summary = summarize(text, 5);
	assert!(summary.contains("0 leaks"), "{summary}");
	assert!(summary.contains("no leaks"), "{summary}");
}
