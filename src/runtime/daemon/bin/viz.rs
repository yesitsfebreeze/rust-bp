//! `viz` — readable live activity: local time, outcome, duration and operation.
//! The selected row shows the actual build command or tool arguments and error.
//! `workspace` in older events means a whole build with no identified crate.
//! `--once` prints a frame; `--heatmap` selects the original colour timeline.
//! Use MCP `audit {mode: local}` for evidence and repair-verification steps.
//!
//! The optional heatmap tails the daemon's
//! `events.jsonl` (the same log `usage` folds): tools, crates and Jev
//! models are columns with their names written downwards on top, every
//! event is a row with now at the top, labelled on the left by the files,
//! symbols and runs it touched, its cell a block in its column coloured by
//! the call's speed against its stream's median (blue fast, green typical,
//! red slow, white error). New events slide in at the top; the cursor
//! stays on its event. Arrows, `hjkl` or `wasd` move the cursor, `g`/`G`
//! jump to now/oldest, `q`, Esc or Ctrl-C quit. `--once [--width W
//! --height H]` prints one frame and exits, for a pipe or a test.

use daemon::usage::{self, Timeline};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_millis(300);

fn now_ms() -> u64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|d| d.as_millis() as u64)
		.unwrap_or(0)
}

fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let heatmap = args.iter().any(|a| a == "--heatmap");
	let arg = |key: &str| {
		args
			.iter()
			.position(|a| a == key)
			.and_then(|i| args.get(i + 1))
			.and_then(|v| v.parse::<usize>().ok())
	};
	if args.iter().any(|a| a == "--once") {
		let (w, h) = size().unwrap_or((120, 40));
		let t = usage::timeline(&usage::load());
		let mut out = std::io::stdout().lock();
		let (w, h) = (arg("--width").unwrap_or(w), arg("--height").unwrap_or(h));
		let mut rows = if heatmap {
			usage::render(&t, w, h, 0, 0, (0, 0), now_ms())
		} else {
			usage::render_activity(&t, w, h, 0, 0)
		};
		if !heatmap && !rows.is_empty() {
			rows[0] = usage::live_status(&t, now_ms(), true, w);
		}
		for row in rows {
			let _ = writeln!(out, "{row}\x1b[0m");
		}
		return;
	}
	let Some(_raw) = Raw::enter() else {
		eprintln!("viz: not a terminal; use --once");
		std::process::exit(2);
	};
	let log = daemon::events::log_path();
	let mut seen = None;
	let mut t = Timeline::default();
	let (mut cx, mut cy, mut sx, mut top) = (0usize, 0usize, 0usize, 0usize);
	let mut last_size = (0, 0);
	let mut last_paint = Instant::now() - POLL;
	let mut dirty = true;
	let mut stdin = std::io::stdin().lock();
	let mut buf = [0u8; 32];
	loop {
		// the log grew: refold; a cursor below the top stays on its event
		let len = std::fs::metadata(&log).map(|m| m.len()).ok();
		if len != seen && last_paint.elapsed() >= POLL {
			seen = len;
			let before = t.entries.len();
			t = usage::timeline(&usage::load());
			let added = t.entries.len().saturating_sub(before);
			if cy > 0 {
				cy += added;
				top += added;
			}
			dirty = true;
		}
		let size = size().unwrap_or((80, 24));
		if size != last_size {
			last_size = size;
			dirty = true;
		}
		if dirty || last_paint.elapsed() >= Duration::from_secs(1) {
			// keep the cursor in the frame
			let rows = if heatmap {
				usage::visible_rows(&t, size.1)
			} else {
				size.1.saturating_sub(5)
			}
			.max(1);
			cy = cy.min(t.entries.len().saturating_sub(1));
			if cy < top {
				top = cy;
			}
			if cy >= top + rows {
				top = cy + 1 - rows;
			}
			cx = cx.min(t.columns.len().saturating_sub(1));
			if cx < sx {
				sx = cx;
			}
			while cx >= sx + usage::visible_columns(&t, size.0, sx).max(1) {
				sx += 1;
			}
			paint(&t, size, sx, top, (cx, cy), heatmap);
			last_paint = Instant::now();
			dirty = false;
		}
		// keys: VMIN 0, VTIME 1 makes this read wait at most 100 ms
		let n = stdin.read(&mut buf).unwrap_or(0);
		let max_x = t.columns.len().saturating_sub(1);
		let max_y = t.entries.len().saturating_sub(1);
		let mut i = 0;
		while i < n {
			let key = match buf[i] {
				0x1b if i + 2 < n && buf[i + 1] == b'[' => {
					i += 2;
					match buf[i] {
						b'A' => b'k',
						b'B' => b'j',
						b'C' => b'l',
						b'D' => b'h',
						_ => 0,
					}
				}
				0x1b | 3 | b'q' => return,
				b => b,
			};
			i += 1;
			match key {
				b'h' | b'a' => cx = cx.saturating_sub(1),
				b'l' | b'd' => cx = (cx + 1).min(max_x),
				b'k' | b'w' => cy = cy.saturating_sub(1),
				b'j' | b's' => cy = (cy + 1).min(max_y),
				b'g' => cy = 0,
				b'G' => cy = max_y,
				_ => continue,
			}
			dirty = true;
		}
	}
}

fn paint(
	t: &Timeline,
	(w, h): (usize, usize),
	sx: usize,
	top: usize,
	cursor: (usize, usize),
	heatmap: bool,
) {
	let mut out = std::io::stdout().lock();
	let mut rows = if heatmap {
		usage::render(t, w, h, sx, top, cursor, now_ms())
	} else {
		usage::render_activity(t, w, h, top, cursor.1)
	};
	if !heatmap && !rows.is_empty() {
		rows[0] = usage::live_status(t, now_ms(), cursor.1 == 0, w);
	}
	for (i, row) in rows.iter().enumerate() {
		let _ = write!(out, "\x1b[{};1H{row}\x1b[0m\x1b[K", i + 1);
	}
	let _ = out.flush();
}

fn size() -> Option<(usize, usize)> {
	let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
	// SAFETY: TIOCGWINSZ fills a winsize; a failure leaves it zeroed.
	if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) } != 0 || ws.ws_col == 0 {
		return None;
	}
	Some((ws.ws_col as usize, ws.ws_row as usize))
}

/// Raw terminal mode on the alternate screen, restored on drop.
struct Raw(libc::termios);

impl Raw {
	fn enter() -> Option<Raw> {
		let mut t: libc::termios = unsafe { std::mem::zeroed() };
		// SAFETY: tcgetattr writes a termios for a tty fd and fails otherwise.
		if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut t) } != 0 {
			return None;
		}
		let saved = t;
		t.c_lflag &= !(libc::ECHO | libc::ICANON | libc::ISIG | libc::IEXTEN);
		t.c_iflag &= !(libc::IXON | libc::ICRNL);
		t.c_cc[libc::VMIN] = 0;
		t.c_cc[libc::VTIME] = 1;
		// SAFETY: a termios we read and adjusted, applied to the same fd.
		unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t) };
		print!("\x1b[?1049h\x1b[?25l\x1b[2J");
		let _ = std::io::stdout().flush();
		Some(Raw(saved))
	}
}

impl Drop for Raw {
	fn drop(&mut self) {
		print!("\x1b[0m\x1b[?25h\x1b[?1049l");
		let _ = std::io::stdout().flush();
		// SAFETY: the termios tcgetattr gave us, restored on the same fd.
		unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.0) };
	}
}
