//! Source watching for the daemon: a recursive mtime scan, no
//! dependencies. Dirs named `target`, `.git` and `node_modules`
//! are skipped, so builds and framework output never trigger a swap.

use std::path::Path;
use std::time::SystemTime;

fn skip(name: &str) -> bool {
	matches!(name, "target" | ".git" | "node_modules")
}

fn newest_under(path: &Path, newest: &mut Option<SystemTime>) {
	let Ok(meta) = std::fs::metadata(path) else {
		return;
	};
	if let Ok(modified) = meta.modified() {
		if newest.is_none_or(|n| modified > n) {
			*newest = Some(modified);
		}
	}
	if !meta.is_dir() {
		return;
	}
	let Ok(entries) = std::fs::read_dir(path) else {
		return;
	};
	for entry in entries.flatten() {
		let name = entry.file_name();
		let name = name.to_string_lossy();
		if skip(&name) {
			continue;
		}
		newest_under(&entry.path(), newest);
	}
}

/// The newest modification time anywhere under the watched dirs/files.
pub fn newest(dirs: &[std::path::PathBuf]) -> Option<SystemTime> {
	let mut newest = None;
	for dir in dirs {
		newest_under(dir, &mut newest);
	}
	newest
}

/// Compare every watched path, including deletion and edits older than another
/// file's mtime. Do not follow directory symlinks (which may form cycles).
pub fn snapshot(
	paths: &[std::path::PathBuf],
) -> std::collections::BTreeMap<std::path::PathBuf, (Option<SystemTime>, u64)> {
	fn scan(
		path: &Path,
		result: &mut std::collections::BTreeMap<std::path::PathBuf, (Option<SystemTime>, u64)>,
	) {
		let Ok(meta) = std::fs::symlink_metadata(path) else {
			return;
		};
		result.insert(path.to_owned(), (meta.modified().ok(), meta.len()));
		if meta.is_dir() {
			if let Ok(entries) = std::fs::read_dir(path) {
				for entry in entries.flatten() {
					if !skip(&entry.file_name().to_string_lossy()) {
						scan(&entry.path(), result);
					}
				}
			}
		}
	}
	let mut result = std::collections::BTreeMap::new();
	for path in paths {
		scan(path, &mut result);
	}
	result
}

#[cfg(test)]
mod tests {
	use crate as daemon;
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
}
