//! The state one process hands to the next: named byte blobs in one file,
//! `<daemon dir>/state/snapshot.bin`. Layout: the magic line, then per entry
//! a u32 name length, the UTF-8 name, a u64 data length and the data (all
//! little-endian). Written to a temporary file and renamed, so a reader never
//! sees half a snapshot.

use std::collections::BTreeMap;
use std::path::PathBuf;

const MAGIC: &[u8] = b"reload-snapshot-1\n";

pub fn path() -> PathBuf {
	daemon::dir().join("state").join("snapshot.bin")
}

pub fn encode(entries: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
	let mut out = MAGIC.to_vec();
	for (name, data) in entries {
		out.extend_from_slice(&(name.len() as u32).to_le_bytes());
		out.extend_from_slice(name.as_bytes());
		out.extend_from_slice(&(data.len() as u64).to_le_bytes());
		out.extend_from_slice(data);
	}
	out
}

pub fn decode(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
	let mut rest = bytes.strip_prefix(MAGIC).ok_or("not a reload snapshot")?;
	let mut entries = BTreeMap::new();
	let take = |n: usize, rest: &mut &[u8]| -> Result<Vec<u8>, String> {
		if rest.len() < n {
			return Err("truncated snapshot".into());
		}
		let (head, tail) = rest.split_at(n);
		*rest = tail;
		Ok(head.to_vec())
	};
	while !rest.is_empty() {
		let name_len = u32::from_le_bytes(take(4, &mut rest)?.try_into().unwrap()) as usize;
		let name =
			String::from_utf8(take(name_len, &mut rest)?).map_err(|_| "snapshot name is not UTF-8")?;
		let data_len = usize::try_from(u64::from_le_bytes(take(8, &mut rest)?.try_into().unwrap()))
			.map_err(|_| "state too large")?;
		if entries.insert(name, take(data_len, &mut rest)?).is_some() {
			return Err("duplicate snapshot entry".into());
		}
	}
	Ok(entries)
}

pub fn write(entries: &BTreeMap<String, Vec<u8>>) -> Result<PathBuf, String> {
	let path = path();
	let dir = path.parent().unwrap();
	std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
	let temp = dir.join(format!("snapshot-{}.tmp", std::process::id()));
	let result = (|| {
		use std::io::Write;
		let mut file = std::fs::File::create(&temp)?;
		file.write_all(&encode(entries))?;
		file.sync_all()?;
		drop(file);
		std::fs::rename(&temp, &path)
	})();
	if let Err(error) = result {
		let _ = std::fs::remove_file(&temp);
		return Err(format!("{}: {error}", path.display()));
	}
	Ok(path)
}

/// Only read an explicitly handed-off snapshot. Keep the file for recovery
/// if startup fails; a fresh launch never restores an unrelated old session.
pub fn read() -> Result<BTreeMap<String, Vec<u8>>, String> {
	let Some(path) = std::env::var_os("DAEMON_RESTORE_SNAPSHOT") else {
		return Ok(BTreeMap::new());
	};
	let path = PathBuf::from(path);
	let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
	decode(&bytes)
}
