use daemon::watch::newest;

#[test]
fn a_change_moves_the_newest_time() {
	let root = std::env::temp_dir().join(format!("daemon-watch-{}", std::process::id()));
	let sub = root.join("src/inner");
	std::fs::create_dir_all(&sub).unwrap();
	let file = sub.join("main.rs");
	std::fs::write(&file, "fn main() {}").unwrap();
	let dirs = vec![root.clone()];
	let first = newest(&dirs);
	assert!(first.is_some());
	std::thread::sleep(std::time::Duration::from_millis(20));
	std::fs::write(&file, "fn main() { changed }").unwrap();
	let second = newest(&dirs);
	assert!(second > first, "second {second:?} first {first:?}");
	let _ = std::fs::remove_dir_all(&root);
}
