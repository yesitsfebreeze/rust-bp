//! Small process helpers shared by the supervisor and the valgrind runner.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Run `command` to completion within `timeout`, capturing combined output.
/// Ok carries the output (trimmed to `keep` bytes); Err names the failure.
pub fn run_with_timeout(command: &[String], timeout: Duration, keep: usize) -> Result<String, String> {
    let (head, tail) = command.split_first().ok_or("empty command")?;
    let mut child = Command::new(head)
        .args(tail)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run {head}: {e}"))?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child.wait_with_output().map_err(|e| e.to_string())?;
                let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
                text.push_str(&String::from_utf8_lossy(&out.stderr));
                let text = text.trim().to_string();
                if status.success() {
                    return Ok(text.chars().take(keep).collect());
                } else {
                    return Err(format!(
                        "exit {status}: {}",
                        text.chars().take(keep.min(1500)).collect::<String>()
                    ));
                }
            }
            Ok(None) => {
                if started.elapsed() > timeout {
                    let _ = child.kill();
                    return Err(format!("timed out after {} s", timeout.as_secs()));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// Run `command`, return only whether it succeeded (used to probe docker).
pub fn succeeds(command: &[String]) -> bool {
    let (head, tail) = match command.split_first() {
        Some(split) => split,
        None => return false,
    };
    Command::new(head)
        .args(tail)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}