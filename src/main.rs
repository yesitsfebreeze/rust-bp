//! The player demo: `gameplay` runs the game; `daemon` reads its live
//! channels through the socket every session shares.

fn main() {
	// The daemon harness reads live channel data from this socket; serving it
	// once at startup is the framework contract, and the game loop continues.
	if let Err(error) = daemon::channels::serve(&daemon::app_socket()) {
		eprintln!("app: {error}");
	}
	gameplay::run();
}
