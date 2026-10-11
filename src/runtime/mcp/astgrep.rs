//! The `ast_grep` tool: structural search and rewrite over the sources
//! through the `ast-grep` CLI (`brew install ast-grep`, `cargo install
//! ast-grep`). Stateless, so each call runs the binary from the project
//! root; the daemon has nothing to keep. A `pattern` is one `ast-grep run`,
//! a `rule` (YAML, one or more separated by `---`) is one `ast-grep scan`.
//! Rewrites are a dry run unless `apply` is set.

use serde_json::Value;
use std::process::Command;

const DEFAULT_PATHS: &[&str] = &["src", "tests"];

pub fn tool(args: &Value) -> Result<String, String> {
	let pattern = args["pattern"].as_str();
	let rule = args["rule"].as_str();
	let apply = args["apply"].as_bool().unwrap_or(false);
	let limit = args["limit"].as_u64().unwrap_or(100).clamp(1, 1000) as usize;
	if apply && pattern.is_some() && args["rewrite"].as_str().is_none() {
		return Err("apply requires a rewrite or a rule with a fix".into());
	}
	let paths: Vec<String> = match args["paths"].as_array() {
		Some(list) => list
			.iter()
			.map(|v| {
				v.as_str()
					.map(str::to_string)
					.ok_or("paths must contain strings")
			})
			.collect::<Result<Vec<_>, _>>()?,
		None => DEFAULT_PATHS.iter().map(|p| p.to_string()).collect(),
	};
	let mut command = Command::new("ast-grep");
	command.current_dir(daemon::project_root());
	match (pattern, rule) {
		(Some(pattern), None) => {
			command.args([
				"run",
				"-p",
				pattern,
				"-l",
				args["lang"].as_str().unwrap_or("rust"),
			]);
			if let Some(fix) = args["rewrite"].as_str() {
				command.args(["--rewrite", fix]);
			}
		}
		(None, Some(rule)) => {
			command.args(["scan", "--inline-rules", rule]);
		}
		_ => return Err("give exactly one of `pattern` or `rule`".into()),
	}
	if paths.is_empty() {
		return Err("paths cannot be empty".into());
	}
	if apply {
		command.arg("--update-all");
	} else {
		command.arg("--json=compact");
	}
	command.args(&paths);
	let output = command.output().map_err(|e| {
		format!("cannot run ast-grep: {e}; install it with `brew install ast-grep` or `cargo install ast-grep`")
	})?;
	if !output.status.success() && !(output.status.code() == Some(1) && output.stderr.is_empty()) {
		return Err(format!(
			"ast-grep failed ({}):\n{}{}",
			output.status,
			String::from_utf8_lossy(&output.stdout),
			String::from_utf8_lossy(&output.stderr)
		));
	}
	if apply {
		return Ok(format!(
			"rewrite completed in {}:\n{}{}",
			paths.join(" "),
			String::from_utf8_lossy(&output.stdout),
			String::from_utf8_lossy(&output.stderr)
		));
	}
	let stdout = String::from_utf8_lossy(&output.stdout);
	let matches: Value = serde_json::from_str(stdout.trim()).map_err(|_| {
		format!(
			"ast-grep failed:\n{}{}",
			stdout.trim(),
			String::from_utf8_lossy(&output.stderr).trim()
		)
	})?;
	Ok(render(&matches, limit))
}

/// `file:line:col  first line of the match` per hit, `→ replacement` under a
/// rewrite, the count last; `limit` caps the listing, not the count.
pub fn render(matches: &Value, limit: usize) -> String {
	let list = matches.as_array().cloned().unwrap_or_default();
	let mut out = String::new();
	for hit in list.iter().take(limit) {
		let start = &hit["range"]["start"];
		let text = hit["text"].as_str().unwrap_or("");
		let first = text.lines().next().unwrap_or("").trim();
		let more = if text.lines().count() > 1 { " …" } else { "" };
		out.push_str(&format!(
			"{}:{}:{}  {first}{more}\n",
			hit["file"].as_str().unwrap_or("?"),
			start["line"].as_u64().unwrap_or(0) + 1,
			start["column"].as_u64().unwrap_or(0) + 1,
		));
		if let Some(replacement) = hit["replacement"].as_str() {
			out.push_str(&format!(
				"  → {}\n",
				replacement.lines().next().unwrap_or("").trim()
			));
		}
	}
	if list.len() > limit {
		out.push_str(&format!("… {} more\n", list.len() - limit));
	}
	out.push_str(&format!("{} match(es)", list.len()));
	out
}
