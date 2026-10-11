use daemon::util::run_with_timeout;
use std::time::Duration;

#[test]
fn deadline_keeps_bounded_diagnostic_output() {
	let error = run_with_timeout(
		&[
			"sh".into(),
			"-c".into(),
			"printf timeout-marker >&2; sleep 10".into(),
		],
		Duration::from_secs(1),
		64,
	)
	.unwrap_err();
	assert!(error.contains("timed out"), "{error}");
	assert!(error.contains("timeout-marker"), "{error}");
	assert!(error.len() < 100, "{error}");
}

#[test]
fn verbose_child_cannot_fill_pipes_and_hang() {
	let result = run_with_timeout(
		&[
			"sh".into(),
			"-c".into(),
			"i=0; while [ $i -lt 10000 ]; do echo stdout; echo stderr >&2; i=$((i+1)); done".into(),
		],
		Duration::from_secs(5),
		100,
	)
	.unwrap();
	assert_eq!(result.len(), 100);
}
