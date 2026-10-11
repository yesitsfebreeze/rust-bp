use daemon::events::{fresh_id, Tail};
use std::fs::OpenOptions;
use std::io::Write;

#[test]
fn ids_are_32_hex_and_distinct() {
	let (a, b) = (fresh_id(), fresh_id());
	assert_eq!(a.len(), 32);
	assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
	assert_ne!(a, b);
}

#[test]
fn tail_only_yields_complete_lines_and_survives_truncation() {
	let dir = std::env::temp_dir().join(format!("rt-events-{}", std::process::id()));
	std::fs::create_dir_all(&dir).unwrap();
	let path = dir.join("events.jsonl");
	std::fs::write(&path, "").unwrap();
	let mut tail = Tail::at(path.clone());
	std::fs::write(&path, "{\"kind\":\"a\"}\n{\"kind\":\"b\"").unwrap();
	let first = tail.read_new().unwrap();
	assert_eq!(first.len(), 1);
	assert_eq!(first[0]["kind"], "a");
	let mut file = OpenOptions::new().append(true).open(&path).unwrap();
	file.write_all(b"}\n").unwrap();
	let second = tail.read_new().unwrap();
	assert_eq!(second.len(), 1);
	assert_eq!(second[0]["kind"], "b");
	std::fs::write(&path, "{\"kind\":\"c\"}\n").unwrap();
	assert_eq!(tail.read_new().unwrap()[0]["kind"], "c");
	let _ = std::fs::remove_dir_all(dir);
}
