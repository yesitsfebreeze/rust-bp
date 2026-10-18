//! Live process memory accounting on each platform, plus macOS leaks/heap.

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

#[cfg(test)]
mod tests {
	use crate as daemon;
	use daemon::native::summarize;

	#[test]
	fn live_process_memory_is_readable() {
		let report = crate::platform::memory_usage(std::process::id()).unwrap();
		assert!(report["resident_bytes"].as_u64().unwrap() > 0, "{report}");
		assert!(crate::platform::memory_usage(u32::MAX).is_err());
	}

	#[test]
	fn a_leaks_reply_compresses_to_summary_and_stacks() {
		let text = "\
Process 1234: 4 nodes malloced for 1024 bytes
Process 1234: 2 leaks for 320 total leaked bytes.

Stack backtrace for leak 1: 320 bytes
0   app                                0x1000 _foo + 32
1   app                                0x2000 _main + 48

Stack backtrace for leak 2: 64 bytes
0   app                                0x1000 _bar + 32
";
		let summary = summarize(text, 1);
		assert!(summary.contains("2 leaks for 320"), "{summary}");
		assert!(summary.contains("leak 1"), "{summary}");
		assert!(summary.contains("2 leak call stacks total"), "{summary}");
		assert!(!summary.contains("_bar"), "{summary}");
		assert!(!summary.contains("nodes malloced"), "{summary}");
	}

	#[test]
	fn a_clean_leaks_reply_says_so() {
		let text = "Process 1234: 100 nodes malloced for 4096 bytes\nProcess 1234: 0 leaks for 0 total leaked bytes.\n";
		let summary = summarize(text, 5);
		assert!(summary.contains("0 leaks"), "{summary}");
		assert!(summary.contains("no leaks"), "{summary}");
	}
}
