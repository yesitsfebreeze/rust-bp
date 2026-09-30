//! `boiler` — the supervisor CLI. `serve` runs the hot-reload loop in the
//! foreground of a terminal; `status`/`reload`/`stop`/`channels` talk to
//! the running supervisor and the app's channel socket.

use boiler::{channels, hot, wire, APP_SOCKET, SUPERVISOR_SOCKET};
use serde_json::json;

const START_HINT: &str = "start one with `cargo run --release -p boiler -- serve`";

fn usage() -> i32 {
    println!(
        "boiler — the hot-reload + profiler supervisor\n\n\
         \x20 boiler serve [-- <child command>]\n\
         \x20     watch the sources, rebuild, swap the executable\n\
         \x20     (default child: ./target/release/app)\n\
         \x20 boiler status | reload | stop\n\
         \x20     talk to the supervisor socket\n\
         \x20 boiler channels\n\
         \x20     read the app's live channel snapshot"
    );
    2
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => {
            let mut supervisor = hot::Supervisor::default();
            if let Some(at) = args.iter().position(|a| a == "--") {
                let child: Vec<String> = args[at + 1..].to_vec();
                if !child.is_empty() {
                    supervisor.child_command = child;
                }
            }
            std::process::exit(supervisor.run());
        }
        Some("status") => println!("{}", wire::ask_or_guide(SUPERVISOR_SOCKET, json!({"op": "status"}), START_HINT)),
        Some("reload") => println!("{}", wire::ask_or_guide(SUPERVISOR_SOCKET, json!({"op": "reload"}), START_HINT)),
        Some("stop") => println!("{}", wire::ask_or_guide(SUPERVISOR_SOCKET, json!({"op": "stop"}), START_HINT)),
        Some("channels") => match channels::read(APP_SOCKET) {
            Ok(snapshot) => println!("{}", channels::render_table(&snapshot)),
            Err(error) => {
                println!("{error}");
                println!("the executable serves its channels by calling boiler::channels::serve(boiler::APP_SOCKET) once at startup");
            }
        },
        _ => {
            std::process::exit(usage());
        }
    }
}