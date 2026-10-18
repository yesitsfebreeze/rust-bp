use std::time::Duration;

fn main() -> Result<(), String> {
	let mut host = reload::Host::new()?;
	host.load("counter", reload::library_path("counter"), &[])?;
	let mut ticks = match host.restored("ticks") {
		Some(bytes) => u64::from_le_bytes(bytes.try_into().map_err(|_| "invalid app tick state")?),
		None => 0,
	};
	daemon::ensure_dir().map_err(|e| e.to_string())?;
	daemon::channels::serve(&daemon::app_socket())?;
	while !daemon::util::cancelled() {
		if host.persist_requested() {
			if let Err(error) = host.persist(&[("ticks", ticks.to_le_bytes().to_vec())]) {
				eprintln!("persist: {error}");
			}
		}
		if !host.quiesced() {
			host.poll();
			ticks += 1;
			let bytes = host.call("counter", "tick", &[]);
			let count = u64::from_le_bytes(bytes.try_into().map_err(|_| "invalid counter result")?);
			daemon::channels::set("app_ticks", ticks as f64);
			daemon::channels::set("module_count", count as f64);
			daemon::channels::set("module_generation", host.generation("counter") as f64);
		}
		std::thread::sleep(Duration::from_millis(100));
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	// Fresh-checkout workflows must work without target/ or a bootstrap generator.
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
		// CreateProcess searches System32 before PATH and can select WSL's bash
		// instead of Git Bash. Resolve explicitly, as an interactive shell does.
		let program = std::env::var_os("PATH")
			.and_then(|path| {
				std::env::split_paths(&path)
					.map(|dir| dir.join(format!("{}{}", command[0], std::env::consts::EXE_SUFFIX)))
					.find(|file| file.is_file())
			})
			.unwrap_or_else(|| PathBuf::from(command[0]));
		let output = Command::new(program)
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
		copy_tree(&root.join("src"), &fixture.0.join("src"));
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

	#[test]
	fn setup_can_rename_and_rebuild_the_stateful_workspace_twice() {
		let root = daemon::project_root();
		let fixture =
			Fixture(std::env::temp_dir().join(format!("cargo-rename-{}", std::process::id())));
		std::fs::create_dir(&fixture.0).unwrap();
		copy_tree(&root.join("src"), &fixture.0.join("src"));
		for name in ["Cargo.toml", "Cargo.lock", "README.md", "setup.sh"] {
			std::fs::copy(root.join(name), fixture.0.join(name)).unwrap();
		}
		// Git's normal Windows checkout may use CRLF; setup must rename it too.
		let manifest = fixture.0.join("Cargo.toml");
		let text = std::fs::read_to_string(&manifest)
			.unwrap()
			.replace("\r\n", "\n");
		std::fs::write(&manifest, text.replace('\n', "\r\n")).unwrap();
		for name in ["renamed-demo", "renamed-again"] {
			run(&fixture.0, &["bash", "setup.sh", name]);
			assert!(fixture
				.0
				.join(format!(
					"target/release/{name}{}",
					std::env::consts::EXE_SUFFIX
				))
				.is_file());
			let hot = std::fs::read_to_string(fixture.0.join("src/runtime/daemon/hot.rs")).unwrap();
			assert!(hot.contains(&format!("./target/release/{name}")));
		}
	}
}
