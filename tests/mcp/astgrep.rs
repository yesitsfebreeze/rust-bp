use mcp::astgrep::render;
use serde_json::json;

#[test]
fn renders_one_line_per_hit_with_one_based_positions() {
	let matches = json!([
		{"file": "src/a.rs", "text": "foo()\n.bar()", "range": {"start": {"line": 4, "column": 2}}},
		{"file": "src/b.rs", "text": "baz()", "range": {"start": {"line": 0, "column": 0}}, "replacement": "qux()"},
	]);
	let text = render(&matches, 100);
	assert_eq!(
		text,
		"src/a.rs:5:3  foo() …\nsrc/b.rs:1:1  baz()\n  → qux()\n2 match(es)"
	);
	assert_eq!(
		render(&matches, 1),
		"src/a.rs:5:3  foo() …\n… 1 more\n2 match(es)"
	);
	assert_eq!(render(&json!([]), 10), "0 match(es)");
}

#[test]
fn cli_rewrites_change_the_file_and_failures_are_errors() {
	// Optional CLI is a runtime dependency; CI with ast-grep exercises real writes.
	if std::process::Command::new("ast-grep")
		.arg("--version")
		.output()
		.is_err()
	{
		return;
	}
	let dir = std::env::temp_dir().join(format!("astgrep-write-{}", std::process::id()));
	std::fs::create_dir_all(&dir).unwrap();
	let file = dir.join("fixture.rs");
	let original = "fn main() { old_name(42); }\n";
	std::fs::write(&file, original).unwrap();
	let mut args =
		json!({"pattern":"old_name($X)","rewrite":"new_name($X)","paths":[file],"apply":false});
	assert!(mcp::astgrep::tool(&args).unwrap().contains("new_name(42)"));
	assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
	args["apply"] = json!(true);
	mcp::astgrep::tool(&args).unwrap();
	assert!(std::fs::read_to_string(&file)
		.unwrap()
		.contains("new_name(42)"));
	assert!(mcp::astgrep::tool(&json!({"rule":"not: [valid","paths":[file],"apply":true})).is_err());
	let rule = "id: rename\nlanguage: Rust\nrule:\n  pattern: new_name($X)\nfix: final_name($X)";
	mcp::astgrep::tool(&json!({"rule":rule,"paths":[file],"apply":true})).unwrap();
	assert!(std::fs::read_to_string(&file)
		.unwrap()
		.contains("final_name(42)"));
	std::fs::remove_dir_all(dir).unwrap();
}
