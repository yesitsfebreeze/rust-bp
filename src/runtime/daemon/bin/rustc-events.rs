//! `rustc-events` — a Cargo `RUSTC_WRAPPER`: runs the compiler unchanged
//! (stdout, stdin, jobserver fds and exit status pass through), appends one
//! `compile_finished` / `compile_failed` event per crate to `events.jsonl`
//! in the project's daemon dir, for the runtime's MCP notification stream.
//!
//! Opt-in, and then for every build: Cargo fingerprints include the wrapper,
//! so builds with and without it recompile each other's work. Install it once
//! (`cargo install --path src/runtime/daemon --bin rustc-events`) and set
//! `[build] rustc-wrapper = "rustc-events"` in a `.cargo/config.toml` that is
//! not checked in, or export `RUSTC_WRAPPER=rustc-events`. Without it the
//! daemon still records one event per build it runs. Cargo invokes the
//! wrapper as `<wrapper> <rustc> <args…>` from the compiled crate's directory
//! (a registry path for dependencies), so the project root comes from
//! `--out-dir` (`<root>/target/<profile>/deps`), not from the working dir.

use serde_json::{json, Map};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let Some((compiler, rest)) = args.split_first() else {
		eprintln!("usage: rustc-events <rustc> <args…> (a Cargo RUSTC_WRAPPER)");
		std::process::exit(2);
	};
	if std::env::var_os("DAEMON_DIR").is_none() {
		let root = rest
			.iter()
			.position(|a| a == "--out-dir")
			.and_then(|at| rest.get(at + 1))
			.and_then(|out| {
				std::path::Path::new(out)
					.ancestors()
					.filter(|p| p.join("Cargo.toml").is_file())
					.last()
			});
		if let Some(root) = root {
			std::env::set_var("DAEMON_DIR", daemon::dir_for(root));
		}
	}
	let started = Instant::now();
	let output = match Command::new(compiler)
		.args(rest)
		.stdin(Stdio::inherit())
		.stdout(Stdio::inherit())
		.stderr(Stdio::piped())
		.output()
	{
		Ok(output) => output,
		Err(error) => {
			eprintln!("rustc-events: cannot run {compiler}: {error}");
			std::process::exit(127);
		}
	};
	let mut stderr = std::io::stderr();
	let _ = stderr.write_all(&output.stderr);
	let _ = stderr.flush();
	let code = daemon::platform::exit_code(output.status);
	// Compiler discovery/probes are not builds; real crate invocations name a crate.
	if let Some(at) = rest.iter().position(|a| a == "--crate-name") {
		if let Some(crate_name) = rest.get(at + 1) {
			let mut data = Map::new();
			data.insert("source".into(), json!("rustc-wrapper"));
			data.insert("crate".into(), json!(crate_name));
			data.insert("exit_code".into(), json!(code));
			data.insert(
				"duration_ms".into(),
				json!(started.elapsed().as_millis() as u64),
			);
			data.insert(
				"diagnostics".into(),
				json!(String::from_utf8_lossy(&output.stderr)),
			);
			data.insert(
				"cwd".into(),
				json!(std::env::current_dir()
					.map(|p| p.to_string_lossy().into_owned())
					.unwrap_or_default()),
			);
			let kind = if code == 0 {
				"compile_finished"
			} else {
				"compile_failed"
			};
			match daemon::events::publish(kind, data) {
				Ok(_) => (),
				Err(error) => eprintln!("event recording failed: {error}"),
			}
		}
	}
	std::process::exit(code);
}
