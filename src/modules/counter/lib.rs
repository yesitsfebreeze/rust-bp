//! Minimal stateful module. Change STEP while `just serve` runs: the
//! counter keeps its value, the next tick uses the new implementation.
const STEP: u64 = 1;

struct Counter(u64);

impl reload::Module for Counter {
	fn create(_config: &[u8]) -> Result<Self, String> {
		Ok(Self(0))
	}

	fn call(&mut self, method: &str, _args: &[u8]) -> Vec<u8> {
		if method == "tick" {
			self.0 = self.0.saturating_add(STEP);
		}
		self.0.to_le_bytes().to_vec()
	}

	fn save(&self) -> Result<Vec<u8>, String> {
		Ok(self.0.to_le_bytes().to_vec())
	}

	fn restore(&mut self, state: &[u8]) -> Result<(), String> {
		self.0 = u64::from_le_bytes(
			state
				.try_into()
				.map_err(|_| "expected an eight-byte counter")?,
		);
		Ok(())
	}
}

reload::export_module!(Counter);
