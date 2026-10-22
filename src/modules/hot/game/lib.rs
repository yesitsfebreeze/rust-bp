//! `gameplay` as the library the `hot` host loads, unloads and loads again.
//! Only gameplay's own code lives here; the engine and audio link from the
//! shared `engine_dylib` the host already holds, so nothing the game leaves
//! in the window or the GPU points into this library once it unloads.

extern crate engine_dylib;

use engine_dylib::engine::gpu::Window;

/// The host's entry point: [`gameplay::play`] under a name it can look up.
#[no_mangle]
pub fn hot_play(
	window: &mut Window,
	carried: Option<&[u8]>,
	generation: u32,
	reload: &mut dyn FnMut() -> bool,
) -> Option<Vec<u8>> {
	gameplay::play(window, carried, generation, reload)
}
