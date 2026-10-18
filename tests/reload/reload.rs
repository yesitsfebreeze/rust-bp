use reload::{snapshot, Host};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

struct Temp(PathBuf);
impl Temp {
	fn new(label: &str) -> Self {
		let path = std::env::temp_dir().join(format!("reload-{}-{label}", std::process::id()));
		std::fs::create_dir_all(&path).unwrap();
		Self(path)
	}
}
impl Drop for Temp {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.0);
	}
}

fn compile(output: &Path, cfg: &str) {
	let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
	let result = Command::new("rustc")
		.arg(root.join("tests/fixtures/reload_module.rs"))
		.args([
			"--crate-type",
			"cdylib",
			"--edition",
			"2021",
			"--cfg",
			cfg,
			"-o",
		])
		.arg(output)
		.output()
		.unwrap();
	assert!(
		result.status.success(),
		"{}",
		String::from_utf8_lossy(&result.stderr)
	);
}

fn count(host: &mut Host, method: &str) -> u64 {
	u64::from_le_bytes(host.call("counter", method, &[]).try_into().unwrap())
}

fn swap(host: &mut Host) -> usize {
	assert_eq!(host.poll(), 0, "first poll waits for stable build");
	host.poll()
}

#[test]
fn module_reload_and_process_handoff_preserve_state_and_reject_bad_generations() {
	let temp = Temp::new("transactions");
	std::env::set_var("DAEMON_DIR", &temp.0);
	std::env::remove_var("DAEMON_RESTORE_SNAPSHOT");
	let library = temp
		.0
		.join(format!("counter{}", std::env::consts::DLL_SUFFIX));
	compile(&library, "v1");
	let mut host = Host::new().unwrap();
	assert!(host.load("../escape", &library, &[]).is_err());
	host.load("counter", &library, &[]).unwrap();
	assert!(host.load("counter", &library, &[]).is_err());
	assert_eq!(count(&mut host, "tick"), 1);
	assert_eq!(count(&mut host, "tick"), 2);
	compile(&library, "v2");
	assert_eq!(swap(&mut host), 1);
	assert_eq!(host.generation("counter"), 2);
	assert_eq!(count(&mut host, "tick"), 12);
	for variant in ["restore_fail", "create_fail", "bad_abi"] {
		compile(&library, variant);
		assert_eq!(swap(&mut host), 0, "{variant} must keep old module");
		assert_eq!(host.generation("counter"), 2);
		assert_eq!(count(&mut host, "get"), 12);
		assert_eq!(
			std::fs::read_dir(temp.0.join("modules")).unwrap().count(),
			1,
			"failed libraries must be unloaded before deletion"
		);
	}
	std::fs::write(&library, b"incomplete build").unwrap();
	assert_eq!(swap(&mut host), 0);
	assert_eq!(count(&mut host, "get"), 12);
	compile(&library, "save_fail");
	assert_eq!(swap(&mut host), 1);
	compile(&library, "v1");
	assert_eq!(swap(&mut host), 0, "save panic must reject swap");
	assert_eq!(count(&mut host, "get"), 12);
	drop(host);
	assert_eq!(
		std::fs::read_dir(temp.0.join("modules")).unwrap().count(),
		0
	);

	// Real channel handshake, persisted module and app state, then a new process.
	let mut host = Host::new().unwrap();
	host.load("counter", &library, &[]).unwrap();
	assert_eq!(count(&mut host, "tick"), 1);
	daemon::channels::serve(&daemon::app_socket()).unwrap();
	let socket = daemon::app_socket();
	let request = std::thread::spawn(move || {
		daemon::wire::ask_within(
			&socket,
			serde_json::json!({"op":"persist"}),
			Duration::from_secs(7),
		)
		.unwrap()
	});
	let deadline = Instant::now() + Duration::from_secs(5);
	while !host.persist_requested() {
		assert!(Instant::now() < deadline);
		std::thread::sleep(Duration::from_millis(5));
	}
	let path = host
		.persist(&[("score", 42u64.to_le_bytes().to_vec())])
		.unwrap();
	assert!(host.quiesced());
	assert!(host.call("counter", "tick", &[]).is_empty());
	assert_eq!(request.join().unwrap()["ok"], true);
	let status = Command::new(std::env::current_exe().unwrap())
		.args(["--ignored", "--exact", "restore_fixture"])
		.env("DAEMON_RESTORE_SNAPSHOT", path)
		.env("TEST_LIBRARY", &library)
		.status()
		.unwrap();
	assert!(status.success());
	assert!(snapshot::path().exists(), "retain recovery snapshot");
	assert!(
		Host::new().unwrap().restored("score").is_none(),
		"fresh sessions ignore old snapshot"
	);
}

#[test]
#[ignore = "subprocess fixture"]
fn restore_fixture() {
	let Ok(library) = std::env::var("TEST_LIBRARY") else {
		return;
	};
	let mut host = Host::new().unwrap();
	host.load("counter", library, &[]).unwrap();
	assert_eq!(host.restored("score"), Some(42u64.to_le_bytes().to_vec()));
	assert_eq!(count(&mut host, "tick"), 2);
}

#[test]
fn snapshot_validates_lengths_names_and_duplicate_entries() {
	let entries = BTreeMap::from([
		("app:score".into(), vec![1, 2, 3]),
		("module:empty".into(), vec![]),
	]);
	let bytes = snapshot::encode(&entries);
	assert_eq!(snapshot::decode(&bytes).unwrap(), entries);
	assert!(snapshot::decode(b"wrong format").is_err());
	assert!(snapshot::decode(&bytes[..bytes.len() - 1]).is_err());
	let one = snapshot::encode(&BTreeMap::from([("same".into(), vec![7])]));
	let mut duplicate = one.clone();
	duplicate.extend_from_slice(&one[b"reload-snapshot-1\n".len()..]);
	assert!(snapshot::decode(&duplicate).is_err());
}
