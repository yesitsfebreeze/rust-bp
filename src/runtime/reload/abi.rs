//! The stable C ABI between the app ([`crate::Host`]) and module crates. A
//! module is a `cdylib` that implements [`Module`] and calls
//! [`export_module!`], which exports one symbol, `module_api`, returning a
//! static [`ModuleApi`]. Calls exchange byte buffers whose encoding each
//! module documents, so module crates need no shared types with the app.
//! `save` and `restore` carry the module's state across a rebuild and a
//! process restart.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Bumped on any layout change; the host refuses a module built for another.
pub const ABI_VERSION: u32 = 1;
/// The exported entry symbol.
pub const ENTRY: &[u8] = b"module_api\0";

/// Bytes owned by the library that returned them; released through
/// [`ModuleApi::free`] of that same library.
#[repr(C)]
pub struct Bytes {
	pub ptr: *mut u8,
	pub len: usize,
	pub cap: usize,
}

/// A failed save must never be mistaken for a valid empty state.
#[repr(C)]
pub struct Saved {
	pub ok: bool,
	pub bytes: Bytes,
}

impl Bytes {
	pub fn from_vec(mut v: Vec<u8>) -> Self {
		let out = Bytes {
			ptr: v.as_mut_ptr(),
			len: v.len(),
			cap: v.capacity(),
		};
		std::mem::forget(v);
		out
	}
	/// # Safety
	/// Only for `Bytes` made by [`Bytes::from_vec`] in this same library.
	pub unsafe fn into_vec(self) -> Vec<u8> {
		Vec::from_raw_parts(self.ptr, self.len, self.cap)
	}
}

#[repr(C)]
pub struct ModuleApi {
	pub abi: u32,
	/// Configuration bytes in, module state out; null on failure.
	pub create: unsafe extern "C" fn(config: *const u8, len: usize) -> *mut c_void,
	pub destroy: unsafe extern "C" fn(state: *mut c_void),
	/// A named method with argument bytes; result bytes (empty for an unknown method).
	pub call: unsafe extern "C" fn(
		state: *mut c_void,
		method: *const u8,
		method_len: usize,
		args: *const u8,
		args_len: usize,
	) -> Bytes,
	/// The state to carry into the next build or process.
	pub save: unsafe extern "C" fn(state: *mut c_void) -> Saved,
	/// Take over saved state; 0 on success.
	pub restore: unsafe extern "C" fn(state: *mut c_void, bytes: *const u8, len: usize) -> i32,
	pub free: unsafe extern "C" fn(bytes: Bytes),
}

/// What a module crate implements; [`export_module!`] exports it.
pub trait Module: Sized {
	fn create(config: &[u8]) -> Result<Self, String>;
	fn call(&mut self, method: &str, args: &[u8]) -> Vec<u8>;
	/// State to keep across reloads; stateless modules keep the default.
	fn save(&self) -> Result<Vec<u8>, String> {
		Ok(Vec::new())
	}
	/// Adopt state saved by an earlier build. An error rejects the replacement
	/// and leaves the previous generation running.
	fn restore(&mut self, _state: &[u8]) -> Result<(), String> {
		Ok(())
	}
}

// Entry-point bodies the macro wraps; panics never cross the C boundary.

#[doc(hidden)]
/// # Safety
/// `config` must reference `len` readable bytes (or be null when empty).
pub unsafe fn create<M: Module>(config: *const u8, len: usize) -> *mut c_void {
	let bytes = slice(config, len);
	match catch_unwind(|| M::create(bytes)) {
		Ok(Ok(state)) => Box::into_raw(Box::new(state)) as *mut c_void,
		Ok(Err(error)) => {
			eprintln!("module create failed: {error}");
			std::ptr::null_mut()
		}
		Err(_) => std::ptr::null_mut(),
	}
}

#[doc(hidden)]
/// # Safety
/// `state` must be null or an unfreed value returned by `create::<M>`.
pub unsafe fn destroy<M: Module>(state: *mut c_void) {
	if !state.is_null() {
		let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(state as *mut M))));
	}
}

#[doc(hidden)]
/// # Safety
/// `state` must be live and exclusively borrowed; the byte pointers must
/// reference their stated lengths, or be null when their lengths are zero.
pub unsafe fn call<M: Module>(
	state: *mut c_void,
	method: *const u8,
	method_len: usize,
	args: *const u8,
	args_len: usize,
) -> Bytes {
	let module = &mut *(state as *mut M);
	let method = std::str::from_utf8(slice(method, method_len)).unwrap_or("");
	let args = slice(args, args_len);
	Bytes::from_vec(catch_unwind(AssertUnwindSafe(|| module.call(method, args))).unwrap_or_default())
}

#[doc(hidden)]
/// # Safety
/// `state` must reference a live `M` with no concurrent mutable access.
pub unsafe fn save<M: Module>(state: *mut c_void) -> Saved {
	let module = &*(state as *mut M);
	match catch_unwind(AssertUnwindSafe(|| module.save())) {
		Ok(Ok(bytes)) => Saved {
			ok: true,
			bytes: Bytes::from_vec(bytes),
		},
		_ => Saved {
			ok: false,
			bytes: Bytes::from_vec(Vec::new()),
		},
	}
}

#[doc(hidden)]
/// # Safety
/// `state` must be live and exclusively borrowed; `bytes` must reference
/// `len` readable bytes, or be null when empty.
pub unsafe fn restore<M: Module>(state: *mut c_void, bytes: *const u8, len: usize) -> i32 {
	let module = &mut *(state as *mut M);
	let bytes = slice(bytes, len);
	match catch_unwind(AssertUnwindSafe(|| module.restore(bytes))) {
		Ok(Ok(())) => 0,
		Ok(Err(error)) => {
			eprintln!("module restore failed: {error}");
			1
		}
		Err(_) => 2,
	}
}

#[doc(hidden)]
/// # Safety
/// `bytes` must have been returned by this library and not yet freed.
pub unsafe extern "C" fn free(bytes: Bytes) {
	drop(bytes.into_vec());
}

unsafe fn slice<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
	if len == 0 {
		&[]
	} else {
		std::slice::from_raw_parts(ptr, len)
	}
}

/// `reload::export_module!(MyModule);` exports `module_api` for the host.
#[macro_export]
macro_rules! export_module {
	($module:ty) => {
		unsafe extern "C" fn __create(config: *const u8, len: usize) -> *mut ::std::ffi::c_void {
			$crate::abi::create::<$module>(config, len)
		}
		unsafe extern "C" fn __destroy(state: *mut ::std::ffi::c_void) {
			$crate::abi::destroy::<$module>(state)
		}
		unsafe extern "C" fn __call(
			state: *mut ::std::ffi::c_void,
			method: *const u8,
			method_len: usize,
			args: *const u8,
			args_len: usize,
		) -> $crate::abi::Bytes {
			$crate::abi::call::<$module>(state, method, method_len, args, args_len)
		}
		unsafe extern "C" fn __save(state: *mut ::std::ffi::c_void) -> $crate::abi::Saved {
			$crate::abi::save::<$module>(state)
		}
		unsafe extern "C" fn __restore(
			state: *mut ::std::ffi::c_void,
			bytes: *const u8,
			len: usize,
		) -> i32 {
			$crate::abi::restore::<$module>(state, bytes, len)
		}
		static __API: $crate::abi::ModuleApi = $crate::abi::ModuleApi {
			abi: $crate::abi::ABI_VERSION,
			create: __create,
			destroy: __destroy,
			call: __call,
			save: __save,
			restore: __restore,
			free: $crate::abi::free,
		};
		#[no_mangle]
		pub extern "C" fn module_api() -> *const $crate::abi::ModuleApi {
			&__API
		}
	};
}
