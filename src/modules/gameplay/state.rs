//! The game state carried across module swaps and app restarts: a small
//! byte format packed by hand. Every value is plain little-endian; raw
//! `Copy` structs go through [`raw`] with their byte size recorded, so a
//! struct that changed layout between builds fails the restore and keeps
//! the previous generation running instead of being reinterpreted. Heap
//! bearing structs (Armed, Shot) pack field by field, never raw.

/// Prefix of every saved state; the digit is the format version.
pub const MAGIC: &[u8] = b"game-state-1\n";

pub struct Packer {
	out: Vec<u8>,
}

impl Packer {
	pub fn new() -> Packer {
		Packer {
			out: MAGIC.to_vec(),
		}
	}

	pub fn u8(&mut self, v: u8) {
		self.out.push(v);
	}

	pub fn bool(&mut self, v: bool) {
		self.out.push(v as u8);
	}

	pub fn u32(&mut self, v: u32) {
		self.out.extend_from_slice(&v.to_le_bytes());
	}

	pub fn f32(&mut self, v: f32) {
		self.out.extend_from_slice(&v.to_le_bytes());
	}

	pub fn str(&mut self, v: &str) {
		self.u32(v.len() as u32);
		self.out.extend_from_slice(v.as_bytes());
	}

	/// One `Copy` struct as its size and raw bytes; see the module docs.
	pub fn raw<T: Copy>(&mut self, v: &T) {
		self.u32(std::mem::size_of::<T>() as u32);
		// SAFETY: T is Copy plain data; size_of bytes are its whole image.
		self.out.extend_from_slice(unsafe {
			std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>())
		});
	}

	pub fn bytes(&mut self, v: &[u8]) {
		self.u32(v.len() as u32);
		self.out.extend_from_slice(v);
	}

	pub fn finish(self) -> Vec<u8> {
		self.out
	}
}

pub struct Unpacker<'a> {
	rest: &'a [u8],
}

/// Every error keeps the previous generation running, so the message only
/// has to identify the field that failed.
pub type Error = String;

impl<'a> Unpacker<'a> {
	pub fn new(bytes: &'a [u8]) -> Result<Unpacker<'a>, Error> {
		Ok(Unpacker {
			rest: bytes
				.strip_prefix(MAGIC)
				.ok_or("not a game state snapshot")?,
		})
	}

	fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
		if self.rest.len() < n {
			return Err("truncated game state".into());
		}
		let (head, tail) = self.rest.split_at(n);
		self.rest = tail;
		Ok(head)
	}

	pub fn u8(&mut self) -> Result<u8, Error> {
		Ok(self.take(1)?[0])
	}

	pub fn bool(&mut self) -> Result<bool, Error> {
		Ok(self.u8()? != 0)
	}

	pub fn u32(&mut self) -> Result<u32, Error> {
		Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
	}

	pub fn f32(&mut self) -> Result<f32, Error> {
		Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
	}

	pub fn str(&mut self) -> Result<String, Error> {
		let len = self.u32()? as usize;
		let bytes = self.take(len)?;
		String::from_utf8(bytes.to_vec()).map_err(|_| "state string is not UTF-8".into())
	}

	/// The counterpart of [`Packer::raw`]; the size must match exactly.
	pub fn raw<T: Copy>(&mut self) -> Result<T, Error> {
		let size = self.u32()? as usize;
		if size != std::mem::size_of::<T>() {
			return Err(format!(
				"state struct size differs: saved {size}, this build {}",
				std::mem::size_of::<T>()
			));
		}
		let bytes = self.take(size)?;
		// SAFETY: the size check above matched this build's layout.
		Ok(unsafe { std::ptr::read(bytes.as_ptr() as *const T) })
	}

	pub fn bytes(&mut self) -> Result<Vec<u8>, Error> {
		let len = self.u32()? as usize;
		Ok(self.take(len)?.to_vec())
	}

	pub fn end(&mut self) -> Result<(), Error> {
		if self.rest.is_empty() {
			Ok(())
		} else {
			Err("trailing bytes in game state".into())
		}
	}
}
