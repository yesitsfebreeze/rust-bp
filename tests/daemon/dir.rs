use daemon::{dir_for, project_root};

#[test]
fn daemon_dir_is_stable_short_and_per_project() {
	let here = project_root();
	assert!(here.join("Cargo.toml").is_file());
	assert!(std::fs::read_to_string(here.join("Cargo.toml"))
		.unwrap()
		.contains("[workspace]"));
	let a = dir_for(&here);
	assert_eq!(a, dir_for(&here));
	assert_ne!(a, dir_for(&std::env::temp_dir()));
	assert!(a.starts_with(std::env::temp_dir()));
	assert!(
		a.join("daemon.sock").as_os_str().len() < 104,
		"socket path too long for macOS: {}",
		a.display()
	);
}
