//! One loaded generation of a module library. The built library is copied
//! to its own file first — Windows locks a loaded DLL, and the build must
//! keep writing the original — then loaded, checked against the ABI version
//! and given its state. Dropping it destroys the state, unloads the library
//! and removes the copy.

use crate::abi::{ModuleApi, ABI_VERSION, ENTRY};
use libloading::Library;
use std::ffi::c_void;
use std::path::{Path, PathBuf};

pub struct Loaded {
	library: Option<Library>,
	api: *const ModuleApi,
	state: *mut c_void,
	file: PathBuf,
}

impl Loaded {
	/// Copy `source` to `copy`, load it and create the module from `config`.
	pub fn open(source: &Path, copy: PathBuf, config: &[u8]) -> Result<Loaded, String> {
		std::fs::copy(source, &copy).map_err(|e| format!("copy {}: {e}", source.display()))?;
		let result = Self::initialize(source, &copy, config);
		if result.is_err() {
			// initialize has returned: its Library has been unloaded first.
			let _ = std::fs::remove_file(&copy);
		}
		result
	}

	fn initialize(source: &Path, copy: &Path, config: &[u8]) -> Result<Loaded, String> {
		// SAFETY: loading runs the library's initializers; module libraries are
		// built from this workspace for this ABI, which the version check guards.
		unsafe {
			let library = Library::new(copy).map_err(|e| format!("load {}: {e}", copy.display()))?;
			let entry: libloading::Symbol<extern "C" fn() -> *const ModuleApi> = library
				.get(ENTRY)
				.map_err(|e| format!("{}: no module_api: {e}", source.display()))?;
			let api = entry();
			if api.is_null() || (*api).abi != ABI_VERSION {
				return Err(format!(
					"{}: module ABI differs from the host's ({ABI_VERSION})",
					source.display()
				));
			}
			let state = ((*api).create)(config.as_ptr(), config.len());
			if state.is_null() {
				return Err(format!(
					"{}: module create failed (see stderr)",
					source.display()
				));
			}
			Ok(Loaded {
				library: Some(library),
				api,
				state,
				file: copy.to_path_buf(),
			})
		}
	}

	pub fn call(&mut self, method: &str, args: &[u8]) -> Vec<u8> {
		// SAFETY: api and state are live while self is; the returned bytes are
		// copied out and handed back to the library that allocated them.
		unsafe {
			let api = &*self.api;
			self.take((api.call)(
				self.state,
				method.as_ptr(),
				method.len(),
				args.as_ptr(),
				args.len(),
			))
		}
	}

	pub fn save(&self) -> Result<Vec<u8>, String> {
		// SAFETY: as in call.
		unsafe {
			let saved = ((*self.api).save)(self.state);
			let bytes = self.take(saved.bytes);
			if saved.ok {
				Ok(bytes)
			} else {
				Err("module save failed".into())
			}
		}
	}

	pub fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
		// SAFETY: the module reads `bytes` only during the call.
		match unsafe { ((*self.api).restore)(self.state, bytes.as_ptr(), bytes.len()) } {
			0 => Ok(()),
			code => Err(format!("restore returned {code}")),
		}
	}

	unsafe fn take(&self, bytes: crate::abi::Bytes) -> Vec<u8> {
		let copy = if bytes.len == 0 {
			Vec::new()
		} else {
			std::slice::from_raw_parts(bytes.ptr, bytes.len).to_vec()
		};
		((*self.api).free)(bytes);
		copy
	}
}

impl Drop for Loaded {
	fn drop(&mut self) {
		// SAFETY: state was created by this library and is destroyed once.
		unsafe { ((*self.api).destroy)(self.state) };
		drop(self.library.take());
		let _ = std::fs::remove_file(&self.file);
	}
}
