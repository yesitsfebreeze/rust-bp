use std::time::Duration;

fn main() -> Result<(), String> {
	let mut host = reload::Host::new()?;
	host.load("counter", reload::library_path("counter"), &[])?;
	let mut ticks = match host.restored("ticks") {
		Some(bytes) => u64::from_le_bytes(bytes.try_into().map_err(|_| "invalid app tick state")?),
		None => 0,
	};
	daemon::ensure_dir().map_err(|e| e.to_string())?;
	daemon::channels::serve(&daemon::app_socket())?;
	while !daemon::util::cancelled() {
		if host.persist_requested() {
			if let Err(error) = host.persist(&[("ticks", ticks.to_le_bytes().to_vec())]) {
				eprintln!("persist: {error}");
			}
		}
		if !host.quiesced() {
			host.poll();
			ticks += 1;
			let bytes = host.call("counter", "tick", &[]);
			let count = u64::from_le_bytes(bytes.try_into().map_err(|_| "invalid counter result")?);
			daemon::channels::set("app_ticks", ticks as f64);
			daemon::channels::set("module_count", count as f64);
			daemon::channels::set("module_generation", host.generation("counter") as f64);
		}
		std::thread::sleep(Duration::from_millis(100));
	}
	Ok(())
}
