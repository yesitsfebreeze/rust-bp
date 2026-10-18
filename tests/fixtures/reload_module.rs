//! Compiled as a real shared library by reload's transaction tests.
#[path = "../../src/runtime/reload/abi.rs"]
pub mod abi;

struct Counter(u64);
impl abi::Module for Counter {
	fn create(_: &[u8]) -> Result<Self, String> {
		if cfg!(create_fail) { return Err("intentional create rejection".into()); }
		Ok(Self(0))
	}
	fn call(&mut self, method: &str, _: &[u8]) -> Vec<u8> {
		if method == "tick" { self.0 += if cfg!(v2) { 10 } else { 1 }; }
		self.0.to_le_bytes().to_vec()
	}
	fn save(&self) -> Result<Vec<u8>, String> {
		assert!(!cfg!(save_fail), "intentional save panic");
		Ok(self.0.to_le_bytes().to_vec())
	}
	fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
		if cfg!(restore_fail) { return Err("intentional restore rejection".into()); }
		self.0 = u64::from_le_bytes(bytes.try_into().map_err(|_| "bad state")?);
		Ok(())
	}
}

#[cfg(not(bad_abi))]
export_module!(Counter);

#[cfg(bad_abi)]
#[no_mangle]
pub extern "C" fn module_api() -> *const abi::ModuleApi {
	// The loader must read only the version before rejecting this pointer.
	static VERSION: u32 = 999;
	&VERSION as *const u32 as *const abi::ModuleApi
}
