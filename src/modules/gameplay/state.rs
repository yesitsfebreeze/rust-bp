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

/// What one game generation hands the next when the game hot-reloads in an
/// open window: where the player is and looks, what they carry, and the
/// cvars by name, so a cvar added or renamed since keeps its default.
pub struct Carried {
	pub player: crate::player::Player,
	pub previous: crate::player::Player,
	pub yaw: f32,
	pub pitch: f32,
	pub fly: bool,
	pub fly_eye: engine::linalg::Vec3,
	pub slot: usize,
	pub money: u32,
	pub owned: Vec<bool>,
	pub mags: Vec<(u32, u32)>,
	/// [`crate::weapons::Armed::save`] of the gun in hand.
	pub armed: Vec<u8>,
	pub now: f32,
	pub console_open: bool,
	pub cvars: Vec<(String, f32)>,
}

impl Carried {
	pub fn save(&self) -> Vec<u8> {
		let mut p = Packer::new();
		p.raw(&self.player);
		p.raw(&self.previous);
		p.f32(self.yaw);
		p.f32(self.pitch);
		p.bool(self.fly);
		p.raw(&self.fly_eye);
		p.u32(self.slot as u32);
		p.u32(self.money);
		p.u32(self.owned.len() as u32);
		for owned in &self.owned {
			p.bool(*owned);
		}
		p.u32(self.mags.len() as u32);
		for (clip, reserve) in &self.mags {
			p.u32(*clip);
			p.u32(*reserve);
		}
		p.bytes(&self.armed);
		p.f32(self.now);
		p.bool(self.console_open);
		p.u32(self.cvars.len() as u32);
		for (name, value) in &self.cvars {
			p.str(name);
			p.f32(*value);
		}
		p.finish()
	}

	pub fn restore(bytes: &[u8]) -> Result<Carried, Error> {
		let mut u = Unpacker::new(bytes)?;
		let player = u.raw()?;
		let previous = u.raw()?;
		let (yaw, pitch, fly) = (u.f32()?, u.f32()?, u.bool()?);
		let fly_eye = u.raw()?;
		let (slot, money) = (u.u32()? as usize, u.u32()?);
		let owned = (0..u.u32()?).map(|_| u.bool()).collect::<Result<_, _>>()?;
		let mags = (0..u.u32()?)
			.map(|_| Ok((u.u32()?, u.u32()?)))
			.collect::<Result<_, Error>>()?;
		let armed = u.bytes()?;
		let (now, console_open) = (u.f32()?, u.bool()?);
		let cvars = (0..u.u32()?)
			.map(|_| Ok((u.str()?, u.f32()?)))
			.collect::<Result<_, Error>>()?;
		u.end()?;
		Ok(Carried {
			player,
			previous,
			yaw,
			pitch,
			fly,
			fly_eye,
			slot,
			money,
			owned,
			mags,
			armed,
			now,
			console_open,
			cvars,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::player::Player;
	use engine::linalg::vec3;

	#[test]
	fn carried_state_survives_the_round_trip() {
		let mut player = Player::spawn(vec3(10.0, 20.0, 30.0));
		player.velocity = vec3(250.0, 0.0, -40.0);
		let carried = Carried {
			player,
			previous: Player::spawn(vec3(9.0, 20.0, 30.0)),
			yaw: 105.0,
			pitch: -12.5,
			fly: true,
			fly_eye: vec3(1.0, 2.0, 3.0),
			slot: 5,
			money: 3150,
			owned: vec![true, true, false, false, false, true, false, false],
			mags: vec![(30, 90), (20, 120)],
			armed: vec![1, 2, 3],
			now: 42.0,
			console_open: true,
			cvars: vec![("fov".into(), 100.0), ("sv_gravity".into(), 600.0)],
		};
		let back = Carried::restore(&carried.save()).expect("restore");
		assert_eq!(back.player, carried.player);
		assert_eq!(back.previous, carried.previous);
		assert_eq!((back.yaw, back.pitch, back.fly), (105.0, -12.5, true));
		assert_eq!(back.fly_eye, carried.fly_eye);
		assert_eq!((back.slot, back.money, back.now), (5, 3150, 42.0));
		assert_eq!(back.owned, carried.owned);
		assert_eq!(back.mags, carried.mags);
		assert_eq!(back.armed, carried.armed);
		assert!(back.console_open);
		assert_eq!(back.cvars, carried.cvars);
	}

	#[test]
	fn a_truncated_state_is_refused() {
		let bytes = Carried {
			player: Player::spawn(vec3(0.0, 0.0, 0.0)),
			previous: Player::spawn(vec3(0.0, 0.0, 0.0)),
			yaw: 0.0,
			pitch: 0.0,
			fly: false,
			fly_eye: vec3(0.0, 0.0, 0.0),
			slot: 1,
			money: 800,
			owned: Vec::new(),
			mags: Vec::new(),
			armed: Vec::new(),
			now: 0.0,
			console_open: false,
			cvars: Vec::new(),
		}
		.save();
		assert!(Carried::restore(&bytes[..bytes.len() - 1]).is_err());
	}
}
