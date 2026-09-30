//! The demo executable — replace this crate with your project. It only
//! shows the framework contract: serve your channels once, then work; the
//! supervisor swaps you on every source change, the `channels` tool reads
//! you live, and `valgrind_run` builds and runs you under callgrind or
//! memcheck inside the Linux container.
//!
//! One optional argument: the number of work iterations before exiting
//! (absent = until killed — the supervisor's normal mode; bounded = what a
//! valgrind run wants, since the run needs a program that terminates).

use std::hint::black_box;
use std::time::Duration;

fn main() {
    let iterations: Option<u64> = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok());
    if let Err(error) = boiler::channels::serve(boiler::APP_SOCKET) {
        eprintln!("app: {error}");
    }
    let mut done = 0;
    loop {
        if let Some(limit) = iterations {
            if done >= limit {
                break;
            }
        }
        let parsed = boiler::channels::timer("app.work.parse");
        let items = parse_work(done);
        parsed.done();
        boiler::channels::add("app.work.items", items as u64);
        boiler::channels::set("app.work.last", items as f64);

        let copied = boiler::channels::timer("app.work.alloc");
        let held: Vec<u8> = (0..4096u16).map(|i| i as u8).collect();
        black_box(&held);
        copied.done();
        boiler::channels::inc("app.work.alloc.rounds");

        std::thread::sleep(Duration::from_millis(50));
        done += 1;
    }
}

fn parse_work(round: u64) -> usize {
    let text = format!("round {round} alpha beta gamma delta epsilon zeta").repeat(64);
    black_box(text.split_whitespace().filter(|word| word.len() > 3).count())
}