//! Native memory debugging — on the machine, against the live process,
//! nothing containerized. On macOS that is the system's own tools (a
//! valgrind does not exist on Apple Silicon):
//!
//! - `leaks <pid>` — leaked bytes with the call stacks that allocated them;
//! - `heap <pid>`  — the live heap's allocation classes and sizes.
//!
//! For a graphics program this is the only honest view: Metal, the window
//! server and the driver's allocations all live in the same process the
//! tools read. The MCP `memory_report` tool wraps both; on other platforms
//! it names that it is macOS-only.

use crate::util;
use std::time::Duration;

/// `leaks` on the live process: the summary line plus the first
/// `backtraces` leak call stacks, truncated.
pub fn leaks_report(pid: u32, backtraces: usize) -> Result<String, String> {
	let text = util::run_with_timeout(
		&["leaks".into(), pid.to_string()],
		Duration::from_secs(180),
		4_000_000,
	)
	.map_err(|e| format!("leaks {pid}: {e} (macOS-only; same user, process not hardened)"))?;
	Ok(summarize(&text, backtraces))
}

/// `heap` on the live process: the first classes.
pub fn heap_summary(pid: u32) -> Result<String, String> {
	util::run_with_timeout(
		&["heap".into(), pid.to_string()],
		Duration::from_secs(180),
		40_000,
	)
	.map_err(|e| format!("heap {pid}: {e} (macOS-only)"))
}

/// The `leaks` reply, compressed: the "N leaks for X total leaked bytes"
/// line, then the callstack section, then a hint when the summary line is
/// missing.
#[doc(hidden)]
pub fn summarize(text: &str, backtraces: usize) -> String {
	let lines: Vec<&str> = text.lines().collect();
	let summary = lines
		.iter()
		.rev()
		.find(|l| l.contains("leaks for"))
		.map(|l| (*l).to_string())
		.unwrap_or_else(|| {
			format!(
				"leaks printed no summary line — first line: {:?}",
				lines.first().unwrap_or(&"")
			)
		});
	if summary.contains(" 0 leaks for") {
		return format!("{summary}\nno leaks");
	}
	let mut out = vec![summary];
	let mut stacks = 0;
	let mut in_stack = false;
	for line in &lines {
		if line.contains("leaks for") {
			in_stack = true;
			continue;
		}
		if !in_stack {
			continue;
		}
		if line.trim_start().starts_with("Stack backtrace for leak") {
			stacks += 1;
			if stacks > backtraces {
				break;
			}
		}
		out.push(line.trim_end().to_string());
	}
	if stacks > backtraces {
		out.push(format!(
			"… {stacks} leak call stacks total, first {backtraces} shown"
		));
	}
	out.join("\n")
}
