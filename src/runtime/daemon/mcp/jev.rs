//! The Jev wire: typed questions over TypeSafe's system-one endpoint, the
//! same wire kern's profiler ranks its series with. `ask` is the transport;
//! `rank_callgrind` / `rank_memcheck` are the two preset judgments, ordering
//! a run's functions or findings by `value/4 × P(real gain)`, ×0.2 when the
//! cost is startup by design.

use crate as daemon;
use daemon::events::publish;
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

const ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";
const MODEL: &str = "jev-latest";

/// Ask Jev `questions` about `state`; answers keyed by question id.
/// Needs `TYPESAFE_API_KEY`; `JEV_URL` overrides the endpoint. Every ask,
/// answered or not, lands in the daemon log as a `jev_call` event.
pub fn ask(state: &Value, questions: &Value) -> Result<Map<String, Value>, String> {
	let started = std::time::Instant::now();
	let answers = ask_wire(state, questions);
	let mut data = Map::new();
	data.insert("model".into(), json!(MODEL));
	data.insert(
		"questions".into(),
		json!(questions
			.as_object()
			.map(|q| q.keys().cloned().collect::<Vec<_>>())
			.unwrap_or_default()),
	);
	data.insert("ok".into(), json!(answers.is_ok()));
	data.insert("ms".into(), json!(started.elapsed().as_millis() as u64));
	if let Err(error) = &answers {
		data.insert("error".into(), json!(error));
	}
	let _ = publish("jev_call", data);
	answers
}

fn ask_wire(state: &Value, questions: &Value) -> Result<Map<String, Value>, String> {
	let key = std::env::var("TYPESAFE_API_KEY")
		.map_err(|_| "TYPESAFE_API_KEY is not set; the rank tools need Jev".to_string())?;
	let url = std::env::var("JEV_URL").unwrap_or_else(|_| ENDPOINT.to_string());
	let body = json!({"model": MODEL, "state": state, "questions": questions});
	let reply = ureq::post(&url)
		.set("Authorization", &format!("Bearer {key}"))
		.timeout(Duration::from_secs(60))
		.send_json(body)
		.map_err(|e| format!("jev did not answer: {e}"))?
		.into_json::<Value>()
		.map_err(|e| format!("jev answered no JSON: {e}"))?;
	reply
		.get("answers")
		.and_then(Value::as_object)
		.cloned()
		.ok_or_else(|| "jev answered no answers object".to_string())
}

/// Questions for one callgrind function: how pressing its instruction count
/// is, what kind of work dominates it, whether a behavior-preserving
/// reduction is plausible.
pub fn callgrind_questions() -> Value {
	json!({
			"pressing": {
					"type": "score",
					"instructions": "`function` is one function of a program run measured under callgrind (Linux, valgrind). `self_ir` is the instruction count valgrind attributed to it, `share` its share of the whole run's instructions. How much could making this function cheaper speed up the whole run?",
					"criteria": [
							"Negligible: tiny share of the run's instructions",
							"Minor: small share, unlikely to be felt",
							"Moderate: a visible share of the run's instructions",
							"High: a clear hotspot, large share",
							"Critical: it dominates the run's instructions"
					]
			},
			"work": {
					"type": "choice",
					"instructions": "Judging by the function and file name, what kind of work most likely dominates its instruction count?",
					"criteria": {
							"hash": "hashing, comparing, sorting, searching",
							"copy": "allocating and copying data",
							"parse": "parsing, formatting, serializing",
							"math": "arithmetic, encoding, compression, crypto",
							"syscall": "system calls, file or socket io",
							"lock": "locks, atomics, synchronization",
							"by_design": "process or runtime startup, one-time initialization"
					}
			},
			"safe": {
					"type": "noul",
					"instructions": "Could this function's instruction count plausibly drop (cheaper algorithm, fewer allocations, caching, less copying) without changing its observable behavior?",
					"criteria": {
							"true": "a behavior-preserving reduction is plausible",
							"false": "the work is inherent or a reduction would change results"
					}
			}
	})
}

/// Questions for one memcheck finding: how severe, and whether it is a real
/// defect rather than allocator or runtime noise.
pub fn memcheck_questions() -> Value {
	json!({
			"severity": {
					"type": "score",
					"instructions": "`finding` is one memcheck finding from a run of a program: `kind` names the check it tripped, `message` the raw line, `frame` the first stack frame, `stack` retains locations, and `count` is displayed records (or leaked blocks), not the ERROR SUMMARY occurrence total. How severe is it?",
					"criteria": [
							"Noise: a benign allocator or language-runtime artifact",
							"Low: technically undefined but harmless in practice",
							"Moderate: wrong values are possible",
							"High: memory corruption, or a leak that grows with the input",
							"Critical: definite leak or corruption on the program's main path"
					]
			},
			"real": {
					"type": "noul",
					"instructions": "Is this a real defect in the program rather than valgrind noise from the allocator or a language runtime?",
					"criteria": {
							"true": "a real defect worth fixing",
							"false": "allocator or runtime noise, safe to suppress"
					}
			}
	})
}

/// Ask `questions` about each entry, 8 at a time; `state_of` builds one
/// entry's state. Ok carries one answer slot per entry (`None` where a
/// single ask failed); Err names the first error when every ask failed.
fn ask_all<F>(
	entries: &[Value],
	questions: &Value,
	state_of: F,
) -> Result<Vec<Option<Map<String, Value>>>, String>
where
	F: Fn(&Value) -> Value + Sync,
{
	let answers: Mutex<Vec<Option<Map<String, Value>>>> = Mutex::new(vec![None; entries.len()]);
	let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
	let next = AtomicUsize::new(0);
	let workers = entries.len().min(8);
	std::thread::scope(|scope| {
		for _ in 0..workers {
			scope.spawn(|| loop {
				let i = next.fetch_add(1, Ordering::Relaxed);
				if i >= entries.len() {
					break;
				}
				match ask(&state_of(&entries[i]), questions) {
					Ok(answer) => answers.lock().unwrap()[i] = Some(answer),
					Err(error) => errors.lock().unwrap().push(error),
				}
			});
		}
	});
	let answers = answers.into_inner().unwrap_or_default();
	let errors = errors.into_inner().unwrap_or_default();
	if !errors.is_empty() && answers.iter().all(Option::is_none) {
		return Err(errors[0].clone());
	}
	Ok(answers)
}

/// Rank a stored callgrind run's heaviest functions; a text table.
pub fn rank_callgrind(run: &Value, top: usize) -> Result<String, String> {
	let entries: Vec<Value> = run["functions"]
		.as_array()
		.map(|a| a.iter().take(top).cloned().collect())
		.unwrap_or_default();
	if entries.is_empty() {
		return Err(
			"this run has no callgrind functions (a memcheck run ranks with memcheck questions)".into(),
		);
	}
	let answers = ask_all(&entries, &callgrind_questions(), |f| {
		let edges: Vec<&Value> = run["call_edges"]
			.as_array()
			.into_iter()
			.flatten()
			.filter(|e| e["caller"] == f["location"] || e["callee"] == f["location"])
			.take(10)
			.collect();
		json!({"function": f["location"], "self_ir": f["ir"], "share": f["share"],
            "call_edges": edges, "call_graph_note": run["call_graph_note"],
            "source": std::env::current_dir().ok().and_then(|root| daemon::profiles::source_context(f["location"].as_str().unwrap_or(""), &root))})
	})?;
	let mut ranked: Vec<(f64, &Value, &Map<String, Value>)> = Vec::new();
	for (entry, answer) in entries.iter().zip(&answers) {
		let Some(a) = answer else { continue };
		let pressing = a["pressing"]["score"].as_f64().unwrap_or(0.0);
		let safe = a["safe"]["noul"].as_f64().unwrap_or(0.0);
		let work = a["work"]["choice"].as_str().unwrap_or("");
		let priority = pressing / 4.0 * safe * if work == "by_design" { 0.2 } else { 1.0 };
		ranked.push((priority, entry, a));
	}
	ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
	let mut lines = vec![format!(
		"{:>8}  {:>9}  {:<8}{:>6}  {:>6} {:>12}  function",
		"priority", "pressing", "work", "safe", "share", "ir"
	)];
	for (priority, entry, answer) in &ranked {
		lines.push(format!(
			"{priority:>8.2}  {:>4.1}/4  {:<8}{:>6.2}  {:>5.1}% {:>12}  {}",
			answer["pressing"]["score"].as_f64().unwrap_or(0.0),
			answer["work"]["choice"].as_str().unwrap_or("?"),
			answer["safe"]["noul"].as_f64().unwrap_or(0.0),
			entry["share"].as_f64().unwrap_or(0.0) * 100.0,
			entry["ir"].as_u64().unwrap_or(0),
			entry["location"].as_str().unwrap_or("?"),
		));
	}
	lines.push(
        "priority = pressing/4 × P(safe reduction), ×0.2 when the work is startup by design. Work the top row.".into(),
    );
	Ok(lines.join("\n"))
}

/// Rank a stored memcheck run's findings; a text table.
pub fn rank_memcheck(run: &Value, top: usize) -> Result<String, String> {
	let mut findings: Vec<Value> = run["findings"].as_array().cloned().unwrap_or_default();
	findings.sort_by_key(|f| std::cmp::Reverse(f["count"].as_u64().unwrap_or(0)));
	findings.truncate(top);
	if findings.is_empty() {
		return Err(
			"this run has no memcheck findings (a callgrind run ranks with callgrind questions)".into(),
		);
	}
	let answers = ask_all(&findings, &memcheck_questions(), |f| json!({"finding": f}))?;
	let mut ranked: Vec<(f64, &Value, &Map<String, Value>)> = Vec::new();
	for (finding, answer) in findings.iter().zip(&answers) {
		let Some(a) = answer else { continue };
		let severity = a["severity"]["score"].as_f64().unwrap_or(0.0);
		let real = a["real"]["noul"].as_f64().unwrap_or(0.0);
		ranked.push((severity / 4.0 * real, finding, a));
	}
	ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
	let mut lines = vec![format!(
		"{:>8}  {:>9}  {:>5}  {:>5}  finding",
		"priority", "severity", "real", "count"
	)];
	for (priority, finding, answer) in &ranked {
		let frame = finding["stack"][0]
			.as_str()
			.or_else(|| finding["frame"].as_str())
			.filter(|f| !f.is_empty());
		lines.push(format!(
			"{priority:>8.2}  {:>4.1}/4  {:>5.2}  {:>5}  {} — {} {}",
			answer["severity"]["score"].as_f64().unwrap_or(0.0),
			answer["real"]["noul"].as_f64().unwrap_or(0.0),
			finding["count"].as_u64().unwrap_or(0),
			finding["kind"].as_str().unwrap_or("?"),
			finding["message"].as_str().unwrap_or(""),
			frame.map(|f| format!("({f})")).unwrap_or_default(),
		));
	}
	lines.push("priority = severity/4 × P(real defect). Work the top row.".into());
	Ok(lines.join("\n"))
}

/// Evidence-only fallback; does not invent probabilities or claim a fix is safe.
pub fn rank_local(record: &Value, top: usize) -> Result<String, String> {
	let (mut rows, basis) = match record["tool"].as_str() {
		Some("callgrind") => {
			let mut rows = record["functions"].as_array().cloned().unwrap_or_default();
			rows.sort_by_key(|r| std::cmp::Reverse(r["ir"].as_u64().unwrap_or(0)));
			(
				rows,
				"descending measured self instructions; cost is not optimization feasibility",
			)
		}
		Some("memcheck") => {
			let mut rows = record["findings"].as_array().cloned().unwrap_or_default();
			rows.sort_by_key(|r| {
				let kind = r["kind"].as_str().unwrap_or("").to_lowercase();
				let severity = if kind.contains("invalid") {
					0
				} else if kind.contains("uninit") {
					1
				} else if kind.contains("definitely") {
					2
				} else {
					3
				};
				(
					severity,
					std::cmp::Reverse(r["count"].as_u64().unwrap_or(0)),
				)
			});
			(rows, "invalid accesses first, then uninitialized uses, definite leaks, other findings; ties by occurrence count; requires human triage")
		}
		_ => {
			return Err("rank supports Callgrind or Memcheck; use compare for native benchmarks".into())
		}
	};
	rows.truncate(top.clamp(1, 30));
	Ok(json!({"mode":"local","basis":basis,"ranking":rows,"note":"No model assessment; no API key needed."}).to_string())
}

#[cfg(test)]
mod tests {
	use crate::mcp;
	use mcp::jev::{callgrind_questions, memcheck_questions};

	#[test]
	fn the_question_sets_speak_jevs_shapes() {
		for questions in [callgrind_questions(), memcheck_questions()] {
			for (_, question) in questions.as_object().unwrap() {
				let kind = question["type"].as_str().unwrap();
				assert!(matches!(kind, "score" | "choice" | "noul"), "{kind}");
				assert!(question["instructions"].as_str().unwrap().len() > 40);
			}
		}
		let callgrind = callgrind_questions();
		assert!(callgrind["pressing"]["criteria"].as_array().unwrap().len() == 5);
		assert!(callgrind["work"]["criteria"]
			.as_object()
			.unwrap()
			.contains_key("by_design"));
	}

	#[test]
	fn local_ranking_uses_measured_cost_and_labels_its_basis() {
		let report = mcp::jev::rank_local(
			&serde_json::json!({"tool":"callgrind","functions":[
					{"location":"a:small","ir":1},{"location":"b:large","ir":100}
			]}),
			1,
		)
		.unwrap();
		let report: serde_json::Value = serde_json::from_str(&report).unwrap();
		assert_eq!(report["mode"], "local");
		assert_eq!(report["ranking"].as_array().unwrap().len(), 1);
		assert_eq!(report["ranking"][0]["ir"], 100);
		assert!(report["basis"]
			.as_str()
			.unwrap()
			.contains("not optimization feasibility"));
	}
}
