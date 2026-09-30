//! The hot-reload supervisor: watch the sources, rebuild, swap the
//! executable — the old build keeps serving until the new one is ready.
//!
//! `boiler serve` runs this loop in the foreground of a terminal:
//!
//! 1. build once, spawn the executable (default `./target/release/app`);
//! 2. poll the watched paths every `poll`; on change (or a `reload`
//!    request through the control socket) run `cargo build --release`;
//! 3. a good build SIGKILLs the old child and spawns the new one; a failed
//!    build keeps the old child serving and records the error;
//! 4. a child that exits on its own is respawned (counted as a restart).
//!
//! The control socket answers `status`, `reload` and `stop` — the `boiler`
//! CLI and the MCP `status`/`reload`/`stop` tools speak it.

use crate::{util, watch, wire};
use serde_json::json;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct Supervisor {
    /// Rebuild command; run with the project root as cwd.
    pub build_command: Vec<String>,
    /// The executable to supervise, argv from the project root.
    pub child_command: Vec<String>,
    /// What triggers a rebuild: a change anywhere under these paths.
    pub watch: Vec<PathBuf>,
    pub poll: Duration,
    pub debounce: Duration,
    pub build_timeout: Duration,
    pub control_socket: String,
}

impl Default for Supervisor {
    fn default() -> Self {
        Self {
            build_command: vec!["cargo".into(), "build".into(), "--release".into()],
            child_command: vec!["./target/release/app".into()],
            watch: vec![
                PathBuf::from("crates"),
                PathBuf::from("app"),
                PathBuf::from("Cargo.toml"),
            ],
            poll: Duration::from_millis(250),
            debounce: Duration::from_millis(300),
            build_timeout: Duration::from_secs(600),
            control_socket: crate::SUPERVISOR_SOCKET.into(),
        }
    }
}

struct Shared {
    child_pid: Option<u32>,
    child_started_ms: u64,
    swaps: u32,
    restarts: u32,
    last_build_ok: bool,
    last_build: String,
    last_build_ms: u64,
    last_swap_ms: u64,
}

struct State {
    shared: Mutex<Shared>,
    reload: AtomicBool,
    stop: AtomicBool,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn spawn(command: &[String]) -> Option<Child> {
    let (head, tail) = command.split_first()?;
    Command::new(head).args(tail).spawn().ok()
}

impl Supervisor {
    /// Run the loop until a `stop` request; returns the exit code.
    pub fn run(self) -> i32 {
        std::fs::create_dir_all(".boiler").ok();
        let state = Arc::new(State {
            shared: Mutex::new(Shared {
                child_pid: None,
                child_started_ms: 0,
                swaps: 0,
                restarts: 0,
                last_build_ok: false,
                last_build: "never built".into(),
                last_build_ms: 0,
                last_swap_ms: 0,
            }),
            reload: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let listener = match wire::bind(&self.control_socket) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("supervisor: {e}");
                return 2;
            }
        };
        let control = state.clone();
        let supervisor = self.clone_shallow();
        wire::serve_loop(listener, move |req| {
            let op = req["op"].as_str().unwrap_or("");
            match op {
                "status" => {
                    let sh = control.shared.lock().unwrap();
                    json!({
                        "op": "status", "ok": true,
                        "running": sh.child_pid.is_some(),
                        "child_pid": sh.child_pid,
                        "child_started_ms": sh.child_started_ms,
                        "swaps": sh.swaps, "restarts": sh.restarts,
                        "last_build": {"ok": sh.last_build_ok, "detail": sh.last_build, "took_ms": sh.last_build_ms},
                        "last_swap_ms": sh.last_swap_ms,
                        "build_command": supervisor.build_command,
                        "child_command": supervisor.child_command,
                        "watch": supervisor.watch.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                        "app_socket": crate::APP_SOCKET,
                    })
                }
                "reload" => {
                    control.reload.store(true, Ordering::Relaxed);
                    json!({"op": "reload", "ok": true, "queued": true})
                }
                "stop" => {
                    control.stop.store(true, Ordering::Relaxed);
                    json!({"op": "stop", "ok": true})
                }
                other => json!({"ok": false, "error": format!("unknown op `{other}`: status, reload, stop")}),
            }
        });

        println!("supervisor: first build, then watching {:?} (edit sources to swap)", self.watch);
        let mut child = self.build_and_swap(None, &state);
        let mut baseline = watch::newest(&self.watch);
        println!("supervisor: serving {} — reload through the control socket", self.child_command.join(" "));

        loop {
            if state.stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some(live) = child.as_mut() {
                if let Ok(Some(status)) = live.try_wait() {
                    println!("supervisor: child exited on its own ({status}); respawning");
                    {
                        let mut sh = state.shared.lock().unwrap();
                        sh.restarts += 1;
                    }
                    child = spawn(&self.child_command);
                    self.note_child(&state, child.as_ref());
                }
            }
            if child.is_none() {
                child = spawn(&self.child_command);
                self.note_child(&state, child.as_ref());
                std::thread::sleep(Duration::from_secs(2));
            }
            let changed = match (watch::newest(&self.watch), baseline) {
                (Some(newest), Some(base)) => newest > base,
                _ => false,
            };
            let requested = state.reload.swap(false, Ordering::Relaxed);
            if changed || requested {
                if changed {
                    std::thread::sleep(self.debounce); // editors write in bursts
                }
                let started = Instant::now();
                match util::run_with_timeout(&self.build_command, self.build_timeout, 4000) {
                    Ok(_) => {
                        println!("supervisor: build ok in {:.1}s — swapping", started.elapsed().as_secs_f32());
                        child = self.build_and_swap(child.take(), &state);
                    }
                    Err(error) => {
                        println!("supervisor: build failed, the old build keeps serving: {error}");
                        let mut sh = state.shared.lock().unwrap();
                        sh.last_build_ok = false;
                        sh.last_build = error.chars().take(1500).collect();
                        sh.last_build_ms = started.elapsed().as_millis() as u64;
                    }
                }
                baseline = watch::newest(&self.watch);
            }
            std::thread::sleep(self.poll);
        }
        if let Some(mut live) = child {
            let _ = live.kill();
            let _ = live.wait();
        }
        let _ = std::fs::remove_file(&self.control_socket);
        println!("supervisor: stopped");
        0
    }

    fn build_and_swap(&self, old: Option<Child>, state: &State) -> Option<Child> {
        let started = Instant::now();
        match util::run_with_timeout(&self.build_command, self.build_timeout, 4000) {
            Ok(_) => {
                let mut sh = state.shared.lock().unwrap();
                sh.last_build_ok = true;
                sh.last_build = "cargo build --release".into();
                sh.last_build_ms = started.elapsed().as_millis() as u64;
            }
            Err(error) => {
                println!("supervisor: build failed, the old build keeps serving: {error}");
                let mut sh = state.shared.lock().unwrap();
                sh.last_build_ok = false;
                sh.last_build = error.chars().take(1500).collect();
                sh.last_build_ms = started.elapsed().as_millis() as u64;
                return old;
            }
        }
        if let Some(mut live) = old {
            let _ = live.kill();
            let _ = live.wait();
        }
        let child = spawn(&self.child_command);
        {
            let mut sh = state.shared.lock().unwrap();
            sh.swaps += 1;
            sh.last_swap_ms = now_ms();
        }
        if let Some(pid) = child.as_ref().map(|c| c.id()) {
            println!("supervisor: swapped to pid {pid}");
        } else {
            println!("supervisor: cannot spawn {} — fix the child command", self.child_command.join(" "));
        }
        self.note_child(state, child.as_ref());
        child
    }

    fn note_child(&self, state: &State, child: Option<&Child>) {
        let mut sh = state.shared.lock().unwrap();
        sh.child_pid = child.map(|c| c.id());
        sh.child_started_ms = now_ms();
    }

    fn clone_shallow(&self) -> Supervisor {
        Supervisor {
            build_command: self.build_command.clone(),
            child_command: self.child_command.clone(),
            watch: self.watch.clone(),
            poll: self.poll,
            debounce: self.debounce,
            build_timeout: self.build_timeout,
            control_socket: self.control_socket.clone(),
        }
    }
}