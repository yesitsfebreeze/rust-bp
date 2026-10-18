use daemon::{hot::Daemon, util, wire};
use serde_json::json;
use std::{
	path::PathBuf,
	process::{Child, Command},
	time::{Duration, Instant},
};

struct Temp(PathBuf);
impl Temp {
	fn new(label: &str) -> Self {
		let dir = std::env::temp_dir().join(format!("rt-{}-{label}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		Self(dir)
	}
	fn text(&self, name: &str) -> String {
		std::fs::read_to_string(self.0.join(name)).unwrap_or_default()
	}
}
impl Drop for Temp {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.0);
	}
}
fn wait(mut condition: impl FnMut() -> bool) {
	let start = Instant::now();
	while !condition() {
		assert!(
			start.elapsed() < Duration::from_secs(10),
			"condition timed out"
		);
		std::thread::sleep(Duration::from_millis(20));
	}
}
#[cfg(windows)]
fn running(pid: i32) -> bool {
	daemon::platform::alive(pid as u32)
}
#[cfg(unix)]
fn running(pid: i32) -> bool {
	// Zombies cannot execute and may await the OS reaper briefly on Linux.
	let output = Command::new("ps")
		.args(["-o", "stat=", "-p", &pid.to_string()])
		.output()
		.unwrap();
	let state = String::from_utf8_lossy(&output.stdout);
	!state.trim().is_empty() && !state.trim().starts_with('Z')
}
// Git Bash's $! is an MSYS pid, not the native pid OpenProcess queries.
fn record_pid(path: &std::path::Path) -> String {
	let command = if cfg!(windows) {
		"cat /proc/$!/winpid"
	} else {
		"echo $!"
	};
	format!("{command} > '{}'", path.display())
}
struct Fixture(Child);
impl Drop for Fixture {
	fn drop(&mut self) {
		let _ = self.0.kill();
		let _ = self.0.wait();
	}
}
fn fixture(temp: &Temp) -> Fixture {
	fixture_with(temp, &[])
}
fn fixture_with(temp: &Temp, env: &[(&str, &str)]) -> Fixture {
	std::fs::write(temp.0.join("watched"), "v1").unwrap();
	Fixture(
		Command::new(std::env::current_exe().unwrap())
			.args(["--exact", "daemon_fixture", "--ignored", "--nocapture"])
			.env("LIFECYCLE_TEST_DIR", &temp.0)
			.env("DAEMON_DIR", temp.0.join("sv"))
			.envs(env.iter().copied())
			.spawn()
			.unwrap(),
	)
}
fn ask(temp: &Temp, op: &str) -> serde_json::Value {
	wire::ask(
		temp.0.join("control.sock").to_str().unwrap(),
		json!({"op":op}),
	)
	.unwrap()
}

#[test]
#[ignore = "subprocess fixture, run only via lifecycle tests"]
fn daemon_fixture() {
	let Some(dir) = std::env::var_os("LIFECYCLE_TEST_DIR") else {
		return;
	};
	let root = PathBuf::from(dir);
	std::env::set_current_dir(&root).unwrap();
	let daemon = Daemon {
        build_command: vec![
            "sh".into(),
            "-c".into(),
            format!("cat watched > built; echo build >> builds; sleep ${{LIFECYCLE_TEST_BUILD_DELAY:-0.4}} & {}; wait", record_pid(std::path::Path::new("build-descendant"))),
        ],
        child_command: vec![
            "sh".into(),
            "-c".into(),
            format!("sleep 120 & {}; wait", record_pid(std::path::Path::new("descendant"))),
        ],
        watch: vec![root.join("watched")],
        poll: Duration::from_millis(20),
        debounce: Duration::from_millis(20),
        control_socket: root.join("control.sock").to_str().unwrap().into(),
        idle: std::env::var("LIFECYCLE_TEST_IDLE_MS")
            .ok()
            .map(|ms| Duration::from_millis(ms.parse().unwrap())),
        ..Daemon::default()
    };
	assert_eq!(daemon.run(), 0);
}

#[test]
fn idle_daemon_exits_after_last_contact_and_cleans_up() {
	let temp = Temp::new("idle");
	let mut child = fixture_with(&temp, &[("LIFECYCLE_TEST_IDLE_MS", "700")]);
	wait(|| !temp.text("descendant").trim().is_empty());
	let descendant: i32 = temp.text("descendant").trim().parse().unwrap();
	let app = ask(&temp, "status")["child_pid"].as_i64().unwrap() as i32;
	// Requests reset the idle clock: two of them 400 ms apart span more than the limit.
	std::thread::sleep(Duration::from_millis(400));
	assert_eq!(ask(&temp, "status")["idle_limit_s"].as_u64(), Some(0));
	std::thread::sleep(Duration::from_millis(400));
	assert!(
		child.0.try_wait().unwrap().is_none(),
		"exited while being asked"
	);
	// A source edit resets it too.
	std::fs::write(temp.0.join("watched"), "v2").unwrap();
	wait(|| temp.text("built") == "v2");
	std::thread::sleep(Duration::from_millis(400));
	assert!(
		child.0.try_wait().unwrap().is_none(),
		"exited right after an edit"
	);
	// Silence: it leaves on its own, with the app and its descendants.
	wait(|| child.0.try_wait().unwrap().is_some());
	assert_eq!(child.0.try_wait().unwrap().unwrap().code(), Some(0));
	wait(|| !running(descendant) && !running(app));
	assert!(!temp.0.join("control.sock").exists());
}

#[test]
fn edits_during_build_are_not_lost_and_reload_builds_once() {
	let temp = Temp::new("watch");
	let mut child = fixture(&temp);
	wait(|| temp.text("built") == "v1");
	std::fs::write(temp.0.join("watched"), "v2").unwrap();
	wait(|| temp.text("built") == "v2" && ask(&temp, "status")["swaps"].as_u64() == Some(2));
	assert_eq!(temp.text("builds").lines().count(), 2);
	ask(&temp, "reload");
	wait(|| ask(&temp, "status")["swaps"].as_u64() == Some(3));
	std::thread::sleep(Duration::from_millis(200));
	assert_eq!(
		temp.text("builds").lines().count(),
		3,
		"reload ran more than one build"
	);
	wait(|| !temp.text("descendant").trim().is_empty());
	let descendant: i32 = temp.text("descendant").trim().parse().unwrap();
	ask(&temp, "stop");
	wait(|| child.0.try_wait().unwrap().is_some());
	wait(|| !running(descendant));
	assert!(!temp.0.join("control.sock").exists());
}

// Another process can be sent SIGTERM only on Unix; Windows stops a daemon through `stop`.
#[cfg(unix)]
#[test]
fn sigterm_cleans_daemon_descendants() {
	let temp = Temp::new("signal");
	let mut child = fixture(&temp);
	wait(|| !temp.text("descendant").trim().is_empty());
	let descendant: i32 = temp.text("descendant").trim().parse().unwrap();
	let app = ask(&temp, "status")["child_pid"].as_i64().unwrap() as i32;
	unsafe {
		libc::kill(child.0.id() as i32, libc::SIGTERM);
	}
	wait(|| child.0.try_wait().unwrap().is_some());
	wait(|| !running(descendant) && !running(app));
	assert!(!temp.0.join("control.sock").exists());
}

#[test]
fn command_deadline_includes_inherited_pipes_and_kills_descendants() {
	let temp = Temp::new("pipes");
	let script = format!("sleep 120 & {}; exit 0", record_pid(&temp.0.join("pid")));
	let started = Instant::now();
	let result = util::run_with_timeout(
		&["sh".into(), "-c".into(), script],
		Duration::from_millis(150),
		100,
	);
	assert!(result.unwrap_err().contains("timed out"));
	assert!(started.elapsed() < Duration::from_secs(2));
	let pid = temp.text("pid").trim().parse().unwrap();
	wait(|| !running(pid));
}

#[test]
fn benchmark_timeout_cleans_process_group() {
	let temp = Temp::new("benchmark");
	let result = daemon::profiles::benchmark(
		&json!({"name":"lifecycle-timeout", "command":["sh", "-c",
        format!("sleep 120 & {}; wait", record_pid(&temp.0.join("pid")))],"repeats":2,"timeout_s":1}),
	);
	assert!(result.unwrap_err().contains("timed out"));
	let pid = temp.text("pid").trim().parse().unwrap();
	wait(|| !running(pid));
	let log = daemon::valgrind_dir().join(format!("lifecycle-timeout-{}-0.log", std::process::id()));
	std::fs::remove_file(&log).unwrap();
}

#[test]
fn stop_during_build_cancels_compiler_group() {
	let temp = Temp::new("stop-build");
	std::fs::write(temp.0.join("watched"), "v1").unwrap();
	let mut child = Fixture(
		Command::new(std::env::current_exe().unwrap())
			.args(["--exact", "daemon_fixture", "--ignored", "--nocapture"])
			.env("LIFECYCLE_TEST_DIR", &temp.0)
			.env("DAEMON_DIR", temp.0.join("sv"))
			.env("LIFECYCLE_TEST_BUILD_DELAY", "120")
			.spawn()
			.unwrap(),
	);
	wait(|| !temp.text("build-descendant").trim().is_empty());
	let pid = temp.text("build-descendant").trim().parse().unwrap();
	ask(&temp, "stop");
	wait(|| child.0.try_wait().unwrap().is_some());
	wait(|| !running(pid));
}
