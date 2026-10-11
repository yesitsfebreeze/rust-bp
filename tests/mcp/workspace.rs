//! Fresh-checkout workflows must work without target/ or a bootstrap generator.
use std::path::{Path, PathBuf};
use std::process::Command;

struct Fixture(PathBuf);

impl Drop for Fixture {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.0);
	}
}

fn copy_tree(from: &Path, to: &Path) {
	std::fs::create_dir_all(to).unwrap();
	for entry in std::fs::read_dir(from).unwrap() {
		let entry = entry.unwrap();
		if entry.file_type().unwrap().is_dir() {
			copy_tree(&entry.path(), &to.join(entry.file_name()));
		} else {
			std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
		}
	}
}

fn run(root: &Path, command: &[&str]) {
	let output = Command::new(command[0])
		.args(&command[1..])
		.current_dir(root)
		.env_remove("CARGO_TARGET_DIR")
		.env_remove("RUSTC_WRAPPER")
		.output()
		.unwrap();
	assert!(
		output.status.success(),
		"{command:?}: {}\n{}",
		String::from_utf8_lossy(&output.stdout),
		String::from_utf8_lossy(&output.stderr)
	);
}

#[test]
fn fresh_checkout_and_repeated_clean_need_no_bootstrap() {
	let root = daemon::project_root();
	let fixture =
		Fixture(std::env::temp_dir().join(format!("cargo-workflow-{}", std::process::id())));
	std::fs::create_dir(&fixture.0).unwrap();
	for name in ["src", "tests"] {
		copy_tree(&root.join(name), &fixture.0.join(name));
	}
	for name in ["Cargo.toml", "Cargo.lock", "rustfmt.toml", "justfile"] {
		std::fs::copy(root.join(name), fixture.0.join(name)).unwrap();
	}
	assert!(!fixture.0.join("target").exists());
	run(
		&fixture.0,
		&[
			"cargo",
			"metadata",
			"--locked",
			"--no-deps",
			"--format-version",
			"1",
		],
	);
	// Check workspace discovery without coupling this test to formatting of an
	// unrelated edit. CI separately checks the actual formatting.
	run(
		&fixture.0,
		&["cargo", "fmt", "--all", "--", "--emit", "stdout"],
	);
	for _ in 0..2 {
		run(&fixture.0, &["cargo", "clean"]);
		run(
			&fixture.0,
			&[
				"cargo",
				"metadata",
				"--locked",
				"--no-deps",
				"--format-version",
				"1",
			],
		);
	}
	if Command::new("just").arg("--version").output().is_ok() {
		run(&fixture.0, &["just", "fmt", "--", "--emit", "stdout"]);
		run(&fixture.0, &["just", "clean"]);
		run(&fixture.0, &["just", "clean"]);
	}
}
