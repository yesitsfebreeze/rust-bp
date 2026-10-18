//! Repeatable native timing, guarded comparisons, and local source context.
use crate::valgrind;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn output(program: &str, args: &[&str]) -> Option<String> {
	let out = Command::new(program).args(args).output().ok()?;
	out
		.status
		.success()
		.then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn environment(runner: &str) -> Value {
	json!({
			"host": output("hostname", &[]), "os": std::env::consts::OS,
			"arch": std::env::consts::ARCH, "runner": runner,
			"image": if runner == "docker" {
					output("docker", &["image", "inspect", "--format", "{{.Id}}", valgrind::IMAGE])
			} else { None },
			"valgrind": if runner == "native" { output("valgrind", &["--version"]) } else { None }
	})
}

pub fn provenance() -> Value {
	json!({"revision": output("git", &["rev-parse", "HEAD"]),
        "dirty": output("git", &["status", "--porcelain"]).map(|s| !s.is_empty())})
}

/// SHA-256 of the executable actually invoked (resolving PATH for bare names).
pub fn executable_fingerprint(program: &str) -> Value {
	let path = if program.contains('/') {
		std::path::PathBuf::from(program)
	} else {
		std::env::var_os("PATH")
			.and_then(|p| {
				std::env::split_paths(&p)
					.map(|d| d.join(program))
					.find(|p| p.is_file())
			})
			.unwrap_or_else(|| program.into())
	};
	let hash = path
		.to_str()
		.and_then(|p| output("shasum", &["-a", "256", p]).or_else(|| output("sha256sum", &[p])))
		.and_then(|s| s.split_whitespace().next().map(str::to_owned));
	json!({"path":path.canonicalize().unwrap_or(path),"sha256":hash})
}

pub fn spread(samples: &[f64]) -> Value {
	if samples.is_empty() {
		return Value::Null;
	}
	let median = percentile(samples, 0.5);
	let deviations: Vec<_> = samples.iter().map(|v| (v - median).abs()).collect();
	json!({"samples":samples.len(),"min":samples.iter().copied().reduce(f64::min),"max":samples.iter().copied().reduce(f64::max),"median":median,"mad":percentile(&deviations,0.5)})
}

pub fn valid_name(name: &str) -> Result<(), String> {
	if name.is_empty()
		|| !name
			.bytes()
			.all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
	{
		return Err("name must contain only letters, digits, hyphens or underscores".into());
	}
	Ok(())
}

#[doc(hidden)]
pub fn percentile(samples: &[f64], fraction: f64) -> f64 {
	let mut sorted = samples.to_vec();
	sorted.sort_by(f64::total_cmp);
	sorted[((sorted.len() as f64 * fraction).ceil() as usize).saturating_sub(1)]
}

pub fn benchmark(args: &Value) -> Result<Value, String> {
	let name = args["name"].as_str().ok_or("name is required")?;
	valid_name(name)?;
	let command: Vec<String> = serde_json::from_value(args["command"].clone())
		.map_err(|_| "command must be a nonempty argv array")?;
	let head = command.first().ok_or("command cannot be empty")?;
	let repeats = args["repeats"].as_u64().unwrap_or(5);
	if !(2..=30).contains(&repeats) {
		return Err("repeats must be between 2 and 30".into());
	}
	let warmups = args["warmups"].as_u64().unwrap_or(1);
	if warmups > 10 {
		return Err("warmups must be between 0 and 10".into());
	}
	let fingerprint = executable_fingerprint(head);
	let timeout = Duration::from_secs(args["timeout_s"].as_u64().unwrap_or(60).clamp(1, 600));
	let dir = crate::valgrind_dir();
	crate::ensure_dir().map_err(|e| e.to_string())?;
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
	let mut samples = Vec::new();
	let mut snapshots = Vec::new();
	let job_started = Instant::now();
	for index in 0..(warmups + repeats) {
		let prefix = format!("{name}-{}-{index}", std::process::id());
		let snapshot = dir.join(format!("{prefix}.channels"));
		let log = dir.join(format!("{prefix}.log"));
		let file = std::fs::File::create(&log).map_err(|e| e.to_string())?;
		let started = Instant::now();
		let mut child = crate::util::ProcessTree::spawn(
			Command::new(head)
				.args(&command[1..])
				.env("CHANNELS_OUT", &snapshot)
				.stdin(Stdio::null())
				.stdout(file.try_clone().map_err(|e| e.to_string())?)
				.stderr(file),
		)
		.map_err(|e| format!("cannot start benchmark: {e}"))?;
		loop {
			if crate::util::cancelled() {
				return Err("benchmark cancelled".into());
			}
			match child.0.try_wait().map_err(|e| e.to_string())? {
				Some(status) if status.success() => break,
				Some(status) => {
					return Err(format!(
						"benchmark exited with {status}; see {}",
						log.display()
					))
				}
				None if job_started.elapsed() > timeout => {
					return Err(format!("benchmark timed out; see {}", log.display()));
				}
				None => std::thread::sleep(Duration::from_millis(1)),
			}
		}
		drop(child);
		if index >= warmups {
			samples.push(started.elapsed().as_secs_f64() * 1000.0);
		}
		if snapshot.exists() {
			let text = std::fs::read_to_string(&snapshot).map_err(|e| e.to_string())?;
			let parsed: Value =
				serde_json::from_str(&text).map_err(|e| format!("invalid final channels: {e}"))?;
			if !parsed["channels"].is_array() {
				return Err("snapshot has no channels array".into());
			}
			if index >= warmups {
				snapshots.push(parsed);
			}
			let _ = std::fs::remove_file(snapshot);
		}
	}
	if !snapshots.is_empty() && snapshots.len() != repeats as usize {
		return Err(
			"only some benchmark iterations saved channels; refusing a partial distribution".into(),
		);
	}
	let mut metrics: BTreeMap<String, Vec<f64>> = BTreeMap::new();
	for snapshot in &snapshots {
		for channel in snapshot["channels"].as_array().unwrap() {
			let name = channel["name"].as_str().unwrap_or("?");
			for field in [
				"count", "mean_us", "p50_us", "p90_us", "p99_us", "max_us", "value",
			] {
				if let Some(value) = channel[field].as_f64() {
					metrics
						.entry(format!("{name}/{field}"))
						.or_default()
						.push(value);
				}
			}
		}
	}
	if metrics.values().any(|v| v.len() != repeats as usize) {
		return Err("channel set changed between repetitions; refusing incomparable samples".into());
	}
	let channel_spread: BTreeMap<_, _> = metrics
		.iter()
		.map(|(k, v)| (k.clone(), spread(v)))
		.collect();
	let channel_metrics: BTreeMap<String, f64> = metrics
		.into_iter()
		.map(|(k, v)| (k, percentile(&v, 0.5)))
		.collect();
	let record = json!({
			"name": name, "tool": "benchmark", "runner": "native", "command": command,
			"workload": args["workload"], "repeats": repeats, "warmups":warmups, "executable":fingerprint,
						"wall_spread_ms":spread(&samples), "channel_spread":channel_spread,
			"environment": environment("timing"), "provenance": provenance(),
			"created_ms": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
			"run_error": null, "samples_ms": samples,
			"wall_p50_ms": percentile(&samples, 0.5), "wall_p90_ms": percentile(&samples, 0.9),
			"channel_metrics": channel_metrics, "channel_snapshots": snapshots,
			"note": "Native wall times include process startup; small differences are noise. Channel metrics are medians of per-run summaries, not pooled quantiles. No final snapshots means channel metrics are unavailable."
	});
	std::fs::write(
		dir.join(format!("{name}.json")),
		serde_json::to_vec_pretty(&record).unwrap(),
	)
	.map_err(|e| e.to_string())?;
	Ok(record)
}

#[doc(hidden)]
pub fn delta(before: f64, after: f64) -> Value {
	json!({"before": before, "after": after, "delta": after - before,
        "percent": if before == 0.0 { None } else { Some(100.0 * (after - before) / before) }})
}

pub fn compare(before: &Value, after: &Value) -> Result<Value, String> {
	for field in ["tool", "command", "runner", "environment"] {
		if before[field].is_null() || after[field].is_null() || before[field] != after[field] {
			return Err(format!(
				"cannot compare: {field} differs or is missing; use the same workload and environment"
			));
		}
	}
	if before["workload"] != after["workload"] {
		return Err("cannot compare: workload labels differ".into());
	}
	if !before["run_error"].is_null() || !after["run_error"].is_null() {
		return Err("cannot compare incomplete or failed runs".into());
	}
	let mut metrics = serde_json::Map::new();
	let fields: &[&str] = match before["tool"].as_str() {
		Some("callgrind") => &["/total_ir"],
		Some("memcheck") => &[
			"/allocations/allocs",
			"/allocations/frees",
			"/allocations/bytes",
			"/leak_summary/definitely_lost_bytes",
		],
		Some("benchmark") => &["/wall_p50_ms", "/wall_p90_ms"],
		_ => return Err("unsupported run type".into()),
	};
	for field in fields {
		let b = before
			.pointer(field)
			.and_then(Value::as_f64)
			.ok_or_else(|| format!("baseline is missing {field}"))?;
		let a = after
			.pointer(field)
			.and_then(Value::as_f64)
			.ok_or_else(|| format!("candidate is missing {field}"))?;
		metrics.insert(field.trim_start_matches('/').into(), delta(b, a));
	}
	let mut functions = Vec::new();
	if before["tool"] == "callgrind" {
		let mut costs: BTreeMap<String, (u64, u64)> = BTreeMap::new();
		for (run, is_after) in [(before, false), (after, true)] {
			for f in run["functions"].as_array().ok_or("missing functions")? {
				let entry = costs
					.entry(f["location"].as_str().unwrap_or("?").into())
					.or_default();
				if is_after {
					entry.1 += f["ir"].as_u64().unwrap_or(0);
				} else {
					entry.0 += f["ir"].as_u64().unwrap_or(0);
				}
			}
		}
		for (location, (b, a)) in costs {
			if b != a {
				functions.push(json!({"location": location, "instructions": delta(b as f64, a as f64)}));
			}
		}
		functions.sort_by(|a, b| {
			b["instructions"]["delta"]
				.as_f64()
				.unwrap()
				.abs()
				.total_cmp(&a["instructions"]["delta"].as_f64().unwrap().abs())
		});
	}
	if before["tool"] == "benchmark" {
		if before["warmups"].as_u64().unwrap_or(0) != after["warmups"].as_u64().unwrap_or(0) {
			return Err("warmup counts differ".into());
		}
		if before["repeats"] != after["repeats"] {
			return Err("repeat counts differ".into());
		}
		if workload_counts(before)? != workload_counts(after)? {
			return Err("channel workload counts differ between repetitions".into());
		}
		let b = before["channel_metrics"]
			.as_object()
			.ok_or("missing channel metrics")?;
		let a = after["channel_metrics"]
			.as_object()
			.ok_or("missing channel metrics")?;
		if b.keys().ne(a.keys()) {
			return Err("channel sets differ".into());
		}
		for (key, value) in b {
			if key.ends_with("/count") && value != &a[key] {
				return Err(format!("channel workload counts differ: {key}"));
			}
			metrics.insert(
				key.clone(),
				delta(value.as_f64().unwrap(), a[key].as_f64().unwrap()),
			);
		}
	}
	Ok(
		json!({"before": before["name"], "after": after["name"], "metrics": metrics,
        "functions": functions, "evidence": {"before_executable":before["executable"],"after_executable":after["executable"],"before_wall_spread_ms":before["wall_spread_ms"],"after_wall_spread_ms":after["wall_spread_ms"],"before_channel_spread":before["channel_spread"],"after_channel_spread":after["channel_spread"]}, "note": "Positive deltas mean more cost. Commands and recorded environments match; keep input files and unrecorded environment variables identical. Timing changes require repeated confirmation. Spread reports min/max and median absolute deviation, not statistical confidence. Channel quantiles are log2 lower bounds; near-zero values and sub-bucket changes cannot establish a speedup."}),
	)
}

fn workload_counts(record: &Value) -> Result<Vec<BTreeMap<String, u64>>, String> {
	let snapshots = record["channel_snapshots"]
		.as_array()
		.ok_or("missing per-repetition channel snapshots")?;
	if !snapshots.is_empty() && Some(snapshots.len() as u64) != record["repeats"].as_u64() {
		return Err("incomplete channel snapshots".into());
	}
	snapshots
		.iter()
		.map(|snapshot| {
			let mut counts = BTreeMap::new();
			for channel in snapshot["channels"].as_array().ok_or("missing channels")? {
				let name = channel["name"].as_str().ok_or("missing channel name")?;
				if !channel["count"].is_null() {
					let count = channel["count"].as_u64().ok_or("invalid channel count")?;
					if counts.insert(name.to_owned(), count).is_some() {
						return Err("duplicate channel name".into());
					}
				}
			}
			Ok(counts)
		})
		.collect()
}

/// Locate a matching definition in the current checkout, not an invented
/// profiler line number. Never read paths outside the workspace.
pub fn source_context(location: &str, root: &Path) -> Option<Value> {
	let (file, function) = location.split_once(':')?;
	let relative = file.strip_prefix("/repo/").unwrap_or(file);
	let root = root.canonicalize().ok()?;
	let path = root.join(relative).canonicalize().ok()?;
	if !path.starts_with(&root) {
		return None;
	}
	let symbol = function
		.split(" [")
		.next()?
		.split('\'')
		.next()?
		.rsplit("::")
		.next()?;
	if symbol.is_empty() || !symbol.chars().all(|c| c.is_alphanumeric() || c == '_') {
		return None;
	}
	let text = std::fs::read_to_string(&path).ok()?;
	let lines: Vec<_> = text.lines().collect();
	let needle = format!("fn {symbol}(");
	let mut candidates = lines.iter().enumerate().filter(|(_, l)| {
		l.trim_start().starts_with(&needle)
			|| l.trim_start().starts_with(&format!("pub {needle}"))
			|| l.trim_start().starts_with(&format!("pub(crate) {needle}"))
	});
	let (line, _) = candidates.next()?;
	// A basename match cannot identify an impl when multiple definitions exist.
	if candidates.next().is_some() {
		return None;
	}
	Some(json!({"path": path.display().to_string(), "line": line + 1,
        "snippet": lines[line..(line + 4).min(lines.len())].join("\n"),
        "basis": "matching definition in current checkout; not a measured call site"}))
}

/// Compact display only; stored locations retain their full attribution.
pub fn display_location(location: &str) -> String {
	let root = crate::project_root();
	let local = location
		.strip_prefix(&format!("{}/", root.display()))
		.or_else(|| location.strip_prefix("/repo/"))
		.unwrap_or(location);
	if let Some(rest) = local
		.strip_prefix("/rustc/")
		.and_then(|s| s.split_once('/').map(|(_, rest)| rest))
	{
		format!("rustlib/{rest}")
	} else {
		local.to_string()
	}
}

/// Sum self costs for exactly equal function names across source entries.
/// Clone suffixes and object names are retained to avoid inventing identities.
pub fn group_functions(functions: &[Value]) -> Vec<Value> {
	let mut groups: BTreeMap<String, (u64, f64, Vec<String>)> = BTreeMap::new();
	for row in functions {
		let location = row["location"].as_str().unwrap_or("?");
		let name = location.split_once(':').map(|(_, f)| f).unwrap_or(location);
		let entry = groups.entry(name.into()).or_default();
		entry.0 += row["ir"].as_u64().unwrap_or(0);
		entry.1 += row["share"].as_f64().unwrap_or(0.0);
		entry.2.push(location.into());
	}
	let mut rows: Vec<_> = groups.into_iter().map(|(name,(ir,share,sources))| json!({"location":name,"ir":ir,"share":share,"source_entries":sources})).collect();
	rows.sort_by_key(|r| std::cmp::Reverse(r["ir"].as_u64().unwrap_or(0)));
	rows
}

#[cfg(test)]
mod tests {
	use crate as daemon;
	use daemon::profiles::{compare, delta, percentile, source_context, valid_name};
	use daemon::valgrind;
	use serde_json::{json, Value};
	use std::path::Path;

	fn record(ir: u64) -> Value {
		json!({"name":"x", "tool":"callgrind", "runner":"docker", "environment":{"image":"fixed"},
        "command":["app","3"], "run_error":null, "total_ir":ir,
        "functions":[{"location":"main.rs:main", "ir":ir}]})
	}
	#[test]
	fn rejects_same_median_but_different_repetition_workloads() {
		fn benchmark(counts: &[u64]) -> Value {
			json!({"tool":"benchmark","command":["app"],"runner":"native","environment":{},
            "repeats":3,"wall_p50_ms":1,"wall_p90_ms":2,"channel_metrics":{"work/count":2},
            "channel_snapshots": counts.iter().map(|c| json!({"channels":[{"name":"work","count":c}]})).collect::<Vec<_>>()})
		}
		let b = benchmark(&[1, 2, 100]);
		assert!(compare(&b, &benchmark(&[2, 2, 2]))
			.unwrap_err()
			.contains("counts differ"));
		assert!(compare(&b, &b).is_ok());
		assert!(compare(&b, &benchmark(&[1, 2])).is_err());
	}

	#[test]
	fn shell_metacharacters_are_not_profile_names() {
		for name in [
			"",
			"../escape",
			"audit$(echo X)",
			"a b",
			"a;b",
			"a\n",
			"a'",
			"a`id`",
		] {
			assert!(valid_name(name).is_err(), "{name:?}");
			assert!(valgrind::run(valgrind::Request {
				name: name.into(),
				tool: valgrind::Tool::Memcheck,
				command: vec!["true".into()],
				build: None,
				timeout_s: 1
			})
			.is_err());
		}
		assert!(valid_name("audit-clean_123").is_ok());
	}

	#[test]
	fn ambiguous_source_is_not_misattributed() {
		let root = std::env::temp_dir().join(format!("daemon-source-{}", std::process::id()));
		std::fs::create_dir_all(&root).unwrap();
		std::fs::write(
			root.join("sample.rs"),
			"struct A;\nimpl A {\nfn work() {}\n}\nstruct B;\nimpl B {\nfn work() {}\n}\n",
		)
		.unwrap();
		assert!(source_context("sample.rs:B::work", &root).is_none());
		std::fs::remove_dir_all(root).unwrap();
	}

	#[test]
	fn compares_costs_and_rejects_different_workloads_and_failed_runs() {
		let b = record(100);
		let mut a = record(125);
		assert_eq!(
			compare(&b, &a).unwrap()["metrics"]["total_ir"]["percent"],
			25.0
		);
		a["command"] = json!(["app", "4"]);
		assert!(compare(&b, &a).is_err());
		a = record(125);
		a["run_error"] = json!("timeout");
		assert!(compare(&b, &a).is_err());
		a = record(125);
		a["environment"] = Value::Null;
		assert!(compare(&b, &a).is_err());
	}
	#[test]
	fn new_cost_has_no_infinite_percentage() {
		assert_eq!(delta(0.0, 12.0)["percent"], Value::Null);
		assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.5), 2.0);
	}
	#[test]
	fn source_context_resolves_local_definition_and_rejects_outside_files() {
		let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
		let context = source_context("/repo/src/main.rs:app::main'2 [app]", &root).unwrap();
		assert!(context["snippet"].as_str().unwrap().contains("fn main()"));
		assert!(source_context("/etc/passwd:main", &root).is_none());
	}

	#[test]
	fn comparison_omits_unchanged_functions_and_spread_describes_noise() {
		assert_eq!(
			compare(&record(100), &record(100)).unwrap()["functions"],
			json!([])
		);
		let spread = daemon::profiles::spread(&[9.0, 10.0, 11.0]);
		assert_eq!(spread["median"], 10.0);
		assert_eq!(spread["mad"], 1.0);
		assert_eq!(spread["min"], 9.0);
		assert!(daemon::profiles::spread(&[]).is_null());
	}

	#[test]
	fn grouping_preserves_total_self_cost_and_distinct_clones() {
		let grouped = daemon::profiles::group_functions(&[
			json!({"location":"a.rs:foo","ir":10,"share":0.1}),
			json!({"location":"b.rs:foo","ir":20,"share":0.2}),
			json!({"location":"b.rs:foo'2","ir":5,"share":0.05}),
		]);
		assert_eq!(grouped.len(), 2);
		assert_eq!(grouped[0]["ir"], 30);
		assert_eq!(grouped[0]["source_entries"].as_array().unwrap().len(), 2);
	}
}
