//! The actual starter under the daemon, using the integration harness's
//! isolated checkout and process ownership helpers.
use super::*;

fn status(dir: &Path) -> Value {
	daemon::wire::ask(
		dir.join("daemon.sock").to_str().unwrap(),
		json!({"op":"status"}),
	)
	.unwrap()
}

fn sample(dir: &Path) -> Option<Value> {
	daemon::channels::read(dir.join("app.sock").to_str()?).ok()
}

fn gauge(sample: &Value, name: &str) -> f64 {
	sample["channels"]
		.as_array()
		.unwrap()
		.iter()
		.find(|c| c["name"] == name)
		.unwrap()["value"]
		.as_f64()
		.unwrap()
}

fn loaded(dir: &Path) -> Value {
	wait_for(Duration::from_secs(90), || {
		let s = sample(dir)?;
		s["channels"]
			.as_array()?
			.iter()
			.any(|c| c["name"] == "module_count")
			.then_some(s)
	})
}

fn reload(dir: &Path, ok: bool) -> Value {
	let request = daemon::wire::ask(
		dir.join("daemon.sock").to_str().unwrap(),
		json!({"op":"reload"}),
	)
	.unwrap();
	wait_for(Duration::from_secs(90), || {
		let s = status(dir);
		(s["building"] == false
			&& s["reload_completed"].as_u64() >= request["ticket"].as_u64()
			&& s["last_build"]["ok"] == ok)
			.then_some(s)
	})
}

pub fn run() {
	let root = root();
	let enclosure =
		Enclosure(std::env::temp_dir().join(format!("state-check-{}", std::process::id())));
	let directory = enclosure.0.join("boilerplate");
	std::fs::create_dir_all(&directory).unwrap();
	copy_fixture_sources(&root, &directory);
	std::fs::copy(root.join("src/main.rs"), directory.join("src/main.rs")).unwrap();
	let env = Env {
		set: vec![],
		remove: vec![
			"RUSTC_WRAPPER",
			"CARGO_TARGET_DIR",
			"DAEMON_DIR",
			"DAEMON_RESTORE_SNAPSHOT",
		],
	};
	env.run(
		&directory,
		&["cargo", "build", "--release", "--locked", "--workspace"],
	);
	let dir = daemon::dir_for(&directory);
	std::fs::create_dir_all(&dir).unwrap();
	let log = std::fs::File::create(dir.join("daemon.log")).unwrap();
	let mut command = Command::new(directory.join(format!(
		"target/release/daemon{}",
		std::env::consts::EXE_SUFFIX
	)));
	command
		.arg("serve")
		.current_dir(&directory)
		.stdout(log.try_clone().unwrap())
		.stderr(log);
	env.apply(&mut command);
	let mut process = Owned::spawn(&mut command).unwrap();
	let first = loaded(&dir);
	let first_pid = first["process"]["pid"].as_u64().unwrap();
	assert_eq!(gauge(&first, "module_count"), gauge(&first, "app_ticks"));

	let module = directory.join("src/modules/counter/lib.rs");
	let source = std::fs::read_to_string(&module).unwrap();
	std::fs::write(
		&module,
		source.replace("const STEP: u64 = 1;", "const STEP: u64 = 10;"),
	)
	.unwrap();
	let swapped = wait_for(Duration::from_secs(90), || {
		let s = sample(&dir)?;
		(gauge(&s, "module_generation") == 2.0 && gauge(&s, "module_count") > gauge(&s, "app_ticks"))
			.then_some(s)
	});
	assert_eq!(
		swapped["process"]["pid"].as_u64(),
		Some(first_pid),
		"module edit must keep the process"
	);
	let offset = gauge(&swapped, "module_count") - 10.0 * gauge(&swapped, "app_ticks");
	pass(
		true,
		"module rebuild changes behavior, retains state and keeps app PID",
	);

	let app = directory.join("src/main.rs");
	let source = std::fs::read_to_string(&app).unwrap();
	std::fs::write(&app, source.replace("from_millis(100)", "from_millis(120)")).unwrap();
	let restarted = wait_for(Duration::from_secs(90), || {
		let s = sample(&dir)?;
		(s["process"]["pid"].as_u64()? != first_pid && s["channels"].as_array()?.len() >= 3)
			.then_some(s)
	});
	assert!(gauge(&restarted, "app_ticks") >= gauge(&swapped, "app_ticks"));
	assert_eq!(
		gauge(&restarted, "module_count") - 10.0 * gauge(&restarted, "app_ticks"),
		offset,
		"both states must survive together"
	);
	pass(
		true,
		"app rebuild changes PID and restores module and app state",
	);

	// Force a write error without relying on OS permission semantics.
	let snapshot = dir.join("state/snapshot.bin");
	std::fs::rename(&snapshot, dir.join("state/backup.bin")).unwrap();
	std::fs::create_dir(&snapshot).unwrap();
	let rejected = reload(&dir, false);
	assert_eq!(rejected["child_pid"], restarted["process"]["pid"]);
	let live = loaded(&dir);
	std::thread::sleep(Duration::from_millis(300));
	assert!(
		gauge(&loaded(&dir), "app_ticks") > gauge(&live, "app_ticks"),
		"rejected save must resume simulation"
	);
	std::fs::remove_dir(&snapshot).unwrap();
	let accepted = reload(&dir, true);
	assert_ne!(accepted["child_pid"], rejected["child_pid"]);
	let after = loaded(&dir);
	assert_eq!(
		gauge(&after, "module_count") - 10.0 * gauge(&after, "app_ticks"),
		offset
	);
	pass(
		true,
		"save failure keeps old process running; repaired destination permits reload",
	);
	assert_eq!(
		std::fs::read_dir(dir.join("run")).unwrap().count(),
		1,
		"old executable copies are removed"
	);
	daemon::wire::ask(
		dir.join("daemon.sock").to_str().unwrap(),
		json!({"op":"stop"}),
	)
	.unwrap();
	assert!(wait_for(Duration::from_secs(10), || process
		.0
		.try_wait()
		.ok()
		.flatten())
	.success());
}
