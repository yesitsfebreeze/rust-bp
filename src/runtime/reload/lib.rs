//! Hot reload that keeps state: module crates swap in while the app runs and
//! carry their state across the swap; the app's own state survives the
//! process restart the daemon performs when the executable itself changes.
//!
//! - Module crates (`crate-type = ["cdylib", "rlib"]`, depending on `reload`
//!   with `default-features = false`) implement [`Module`] and call
//!   [`export_module!`].
//! - The app holds one [`Host`] (feature `host`, on by default), loads modules
//!   with [`Host::load`]`(name, `[`library_path`]`(crate), config)`, calls
//!   [`Host::poll`] once per frame or tick, and answers
//!   [`Host::persist_requested`] with [`Host::persist`].

pub mod abi;
pub use abi::{Module, ABI_VERSION};

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
mod library;
#[cfg(feature = "host")]
pub mod snapshot;
#[cfg(feature = "host")]
pub use host::{library_path, Host};
