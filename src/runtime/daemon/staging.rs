//! Per-process executable copies keep Cargo outputs writable on Windows.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Executable(PathBuf);

impl Executable {
	pub fn copy(source: &Path, directory: &Path) -> std::io::Result<Self> {
		static NEXT: AtomicU64 = AtomicU64::new(0);
		std::fs::create_dir_all(directory)?;
		let stem = source.file_stem().unwrap_or_default().to_string_lossy();
		let serial = NEXT.fetch_add(1, Ordering::Relaxed);
		let path = directory.join(format!(
			"{stem}-{}-{serial}{}",
			std::process::id(),
			std::env::consts::EXE_SUFFIX
		));
		std::fs::copy(source, &path)?;
		Ok(Self(path))
	}

	pub fn path(&self) -> &Path {
		&self.0
	}
}

impl Drop for Executable {
	fn drop(&mut self) {
		let _ = std::fs::remove_file(&self.0);
	}
}
