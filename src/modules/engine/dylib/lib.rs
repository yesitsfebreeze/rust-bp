//! `engine` and `audio` linked once, as a dylib. A crate that names this one
//! links both from here: see `src/modules/hot`.

pub use audio;
pub use engine;
