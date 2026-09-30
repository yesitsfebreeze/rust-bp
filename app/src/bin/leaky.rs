//! `leaky` — the intentionally broken demo target. The framework's memcheck
//! loop needs a program that actually trips valgrind, and safe Rust never
//! does: this binary is a catalog of the classic three, behind `unsafe`,
//! each documented with the memcheck line it should produce. Run it with
//! one argument (the number of faulting rounds) — never fix it; it is
//! evidence, not code.

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1);
    for round in 0..rounds {
        use std::alloc::{alloc, Layout};

        // 1. use after free — "Invalid read of size 8". The payload goes
        //    through black_box so LLVM cannot constant-fold the read away
        //    (a known value makes the UB unobservable and memcheck silent).
        let payload = std::hint::black_box(41u64 + round as u64);
        let freed = Box::into_raw(Box::new(payload));
        unsafe { drop(Box::from_raw(freed)) };
        let read_after_free = unsafe { std::hint::black_box(freed).read() };
        println!("round {round}: use after free read {read_after_free}");

        // 2. uninitialized heap — "Conditional jump or move depends on
        //    uninitialised value"
        let raw = unsafe { alloc(Layout::new::<u64>()) };
        let uninit = unsafe { std::hint::black_box(raw).cast::<u64>().read() };
        if uninit > u64::MAX / 2 {
            println!("round {round}: the uninitialized heap said something");
        }

        // 3. a definite leak — "definitely lost: N bytes in M blocks"
        let leaked: &'static mut [u8; 4096] = Box::leak(Box::new([0u8; 4096]));
        leaked[0] = round as u8;
        std::hint::black_box(leaked);

        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    println!("leaky: done tripping all three");
}