//! `daemon` — the CLI. `serve` runs the hot-reload loop in the
//! foreground of a terminal, `serve --detach` as a per-project daemon;
//! `status`/`reload`/`channels` start that daemon when none answers, then
//! talk to it and to the app's channel socket.

use daemon::{app_socket, channels, daemon_socket, hot, wire};
use serde_json::json;
use std::process::Command;

/// The daemon `status`/`reload`/`channels` start when no daemon answers.
fn launcher() -> Command {
	let mut command = Command::new(std::env::current_exe().expect("own path"));
	command.args(["serve", "--idle", "30m"]);
	command
}

/// Ask the daemon, starting it first when needed.
fn ask(op: &str) -> String {
	match hot::ensure_running(launcher).and_then(|_| wire::ask(&daemon_socket(), json!({"op": op}))) {
		Ok(reply) => serde_json::to_string_pretty(&reply).unwrap_or_default(),
		Err(error) => error,
	}
}

fn usage() -> i32 {
	println!(
		"daemon — the hot-reload + profiler runtime\n\n\
         \x20 daemon mcp (also the default with no arguments)\n\
         \x20     serve MCP on stdio and own supervision in this process\n\
         \x20 daemon --call <tool> '<json>'\n\
         \x20     call one tool from the shell\n\
         \x20 daemon serve [--detach] [--idle 30m|0] [-- <child command>]\n\
         \x20     watch the sources, rebuild, swap the executable\n\
         \x20     (default child: ./target/release/app); --detach runs it as a\n\
         \x20     daemon that exits after --idle without requests or edits\n\
         \x20 daemon status | reload | stop\n\
         \x20     talk to the daemon socket (status/reload start a daemon\n\
         \x20     when none answers; stop never starts one)\n\
         \x20 daemon channels\n\
         \x20     read the app's live channel snapshot (starts the daemon too)\n\
         \x20 daemon logs [-f]\n\
         \x20     the detached daemon's output\n\
         \x20 daemon dir\n\
         \x20     print the daemon dir (sockets, runs, events; DAEMON_DIR overrides)"
	);
	2
}

fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	match args.first().map(String::as_str) {
		None => std::process::exit(daemon::mcp::run(&[])),
		Some("mcp") => std::process::exit(daemon::mcp::run(&args[1..])),
		Some("--call") => std::process::exit(daemon::mcp::run(&args)),
		Some("serve") => {
			let (flags, child): (&[String], &[String]) = match args.iter().position(|a| a == "--") {
				Some(at) => (&args[1..at], &args[at + 1..]),
				None => (&args[1..], &[]),
			};
			let mut detach = false;
			let mut idle: Option<String> = None;
			let mut flags = flags.iter();
			while let Some(flag) = flags.next() {
				match flag.as_str() {
					"--detach" => detach = true,
					"--idle" => idle = flags.next().cloned(),
					other => {
						eprintln!("serve: unknown flag `{other}`");
						std::process::exit(usage());
					}
				}
			}
			if detach {
				let mut command = Command::new(std::env::current_exe().expect("own path"));
				command.args(["serve", "--idle", idle.as_deref().unwrap_or("30m")]);
				if !child.is_empty() {
					command.arg("--").args(child);
				}
				match hot::ensure_running(move || command) {
					Ok(reply) => println!(
						"{}",
						serde_json::to_string_pretty(&reply).unwrap_or_default()
					),
					Err(error) => {
						eprintln!("{error}");
						std::process::exit(1);
					}
				}
				return;
			}
			let mut daemon = hot::Daemon::default();
			if let Some(idle) = idle {
				daemon.idle = hot::parse_idle(&idle).unwrap_or_else(|e| {
					eprintln!("serve: {e}");
					std::process::exit(usage());
				});
			}
			if !child.is_empty() {
				daemon.child_command = child.to_vec();
			}
			std::process::exit(daemon.run());
		}
		Some("status") => println!("{}", ask("status")),
		Some("reload") => println!("{}", ask("reload")),
		Some("stop") => println!(
			"{}",
			wire::ask_or_guide(&daemon_socket(), json!({"op": "stop"}), "nothing to stop")
		),
		Some("channels") => {
			if let Err(error) = hot::ensure_running(launcher) {
				println!("{error}");
			}
			match channels::read(&app_socket()) {
				Ok(snapshot) => println!("{}", channels::render_table(&snapshot)),
				Err(error) => {
					println!("{error}");
					println!("the executable serves its channels by calling daemon::channels::serve(&daemon::app_socket()) once at startup; right after a start the first build may still be running");
				}
			}
		}
		Some("logs") => {
			let log = hot::log_path();
			if args.get(1).map(String::as_str) == Some("-f") {
				let status = Command::new("tail")
					.args(["-n", "50", "-f"])
					.arg(&log)
					.status()
					.map(|s| s.code().unwrap_or(1))
					.unwrap_or(1);
				std::process::exit(status);
			}
			match std::fs::read_to_string(&log) {
				Ok(text) => print!("{text}"),
				Err(_) => println!("no log at {} — nothing detached has run yet", log.display()),
			}
		}
		Some("dir") => println!("{}", daemon::dir().display()),
		_ => {
			std::process::exit(usage());
		}
	}
}
