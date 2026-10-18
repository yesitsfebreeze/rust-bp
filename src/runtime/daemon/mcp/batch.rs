//! Per-tool batching shared by stdio MCP and the shell entry point.
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const MAX_ITEMS: usize = 32;
pub const MAX_CONCURRENCY: usize = 8;
pub const DEFAULT_CONCURRENCY: usize = 4;

pub const DESCRIPTION: &str = " Also accepts {items:[args],concurrency:4}; 1..32 items, concurrency 1..8. Inspect every result and failed/skipped. Reads parallelize; mutations run in order, stop on error, and do not roll back. See docs topic:batching.";

/// Keep the original properties visible to MCP hosts, and enforce the two
/// disjoint forms. Item schemas cannot recursively contain batch envelopes.
pub fn schema(mut single: Value) -> Value {
	single["properties"]["items"] = json!(false);
	single["properties"]["concurrency"] = json!(false);
	let batch = json!({
			"type": "object",
			"properties": {
					"items": {"type": "array", "items": single, "minItems": 1, "maxItems": MAX_ITEMS},
					"concurrency": {"type": "integer", "minimum": 1, "maximum": MAX_CONCURRENCY, "default": DEFAULT_CONCURRENCY}
			},
			"required": ["items"],
			"additionalProperties": false
	});
	let mut properties = single["properties"].clone();
	properties["items"] = batch["properties"]["items"].clone();
	properties["concurrency"] = batch["properties"]["concurrency"].clone();
	json!({"type": "object", "properties": properties, "oneOf": [single, batch]})
}

/// Explicit policy for every registered tool. Unknown tools fail closed.
/// Diagnostics trigger shared flycheck state; previews never apply edits.
pub fn sequential(tool: &str, args: &Value) -> bool {
	match tool {
		"lsp" => args["method"] == "diagnostics",
		"ast_grep" => args["apply"] == true,
		"events" | "docs" | "audit" | "usage" | "compare" | "status" | "channels" | "runs" | "show"
		| "rank" => false,
		"reload" | "stop" | "benchmark" | "valgrind_run" | "memory_report" => true,
		_ => true,
	}
}

/// Validate the entire envelope before running anything. Errors from executed
/// items remain in the JSON response even when the overall call is an error.
pub fn execute(
	tool: &str,
	arguments: &Value,
	run: impl Fn(&Value) -> Result<String, String> + Sync,
) -> Result<String, String> {
	let object = arguments
		.as_object()
		.ok_or("tool arguments must be an object")?;
	let Some(items) = object.get("items") else {
		if object.contains_key("concurrency") {
			return Err("concurrency requires an items batch".into());
		}
		return run(arguments);
	};
	if object
		.keys()
		.any(|key| key != "items" && key != "concurrency")
	{
		return Err(
			"batch envelopes accept only items and concurrency; put tool arguments inside each item"
				.into(),
		);
	}
	let items = items.as_array().ok_or("items must be an array")?;
	if items.is_empty() || items.len() > MAX_ITEMS {
		return Err(format!(
			"items must contain 1..{MAX_ITEMS} argument objects"
		));
	}
	for (index, item) in items.iter().enumerate() {
		if !item.is_object() || item.get("items").is_some() || item.get("concurrency").is_some() {
			return Err(format!(
				"items[{index}] must be a single argument object, not a nested batch"
			));
		}
	}
	let concurrency = match object.get("concurrency") {
		None => DEFAULT_CONCURRENCY,
		Some(value) => value
			.as_u64()
			.filter(|n| (1..=MAX_CONCURRENCY as u64).contains(n))
			.ok_or("concurrency must be an integer from 1 to 8")? as usize,
	};
	let serial = items.iter().any(|item| sequential(tool, item));
	let mut results = Vec::with_capacity(items.len());
	if serial {
		for (index, item) in items.iter().enumerate() {
			let result = run(item);
			let failed = result.is_err();
			results.push((index, result));
			if failed {
				break;
			}
		}
	} else {
		let next = AtomicUsize::new(0);
		std::thread::scope(|scope| {
			let workers: Vec<_> = (0..concurrency.min(items.len()))
				.map(|_| {
					let run = &run;
					let next = &next;
					scope.spawn(move || {
						let mut results = Vec::new();
						loop {
							let index = next.fetch_add(1, Ordering::Relaxed);
							let Some(item) = items.get(index) else { break };
							results.push((index, run(item)));
						}
						results
					})
				})
				.collect();
			for worker in workers {
				results.extend(worker.join().expect("batch worker panicked"));
			}
		});
		results.sort_by_key(|(index, _)| *index);
	}
	let failed = results.iter().filter(|(_, result)| result.is_err()).count();
	let skipped = items.len() - results.len();
	let results: Vec<_> = results
		.into_iter()
		.map(|(index, result)| {
			let error = result.is_err();
			let content = result.unwrap_or_else(|error| error);
			json!({"index": index, "result": {"content": content, "error": error}})
		})
		.collect();
	let text = json!({"results": results, "failed": failed, "skipped": skipped}).to_string();
	if failed > 0 {
		Err(text)
	} else {
		Ok(text)
	}
}

#[cfg(test)]
mod tests {
	use crate::mcp::batch;
	use serde_json::{json, Value};
	use std::sync::atomic::{AtomicUsize, Ordering};
	use std::sync::{Condvar, Mutex};
	use std::time::Duration;
	#[test]
	fn single_item_keeps_the_original_response() {
		assert_eq!(
			batch::execute("show", &json!({"name":"one"}), |args| {
				assert_eq!(args["name"], "one");
				Ok("plain text".into())
			})
			.unwrap(),
			"plain text"
		);
	}

	#[test]
	fn invalid_envelopes_never_execute_any_item() {
		for args in [
			Value::Null,
			json!([]),
			json!({"items":null}),
			json!({"items":[]}),
			json!({"items":vec![json!({});33]}),
			json!({"items":[{},null]}),
			json!({"items":[{"items":[{}]}]}),
			json!({"items":[{"concurrency":1}]}),
			json!({"items":[{}],"name":"mixed"}),
			json!({"concurrency":2}),
			json!({"items":[{}],"concurrency":0}),
			json!({"items":[{}],"concurrency":9}),
			json!({"items":[{}],"concurrency":1.5}),
			json!({"items":[{}],"concurrency":"2"}),
		] {
			assert!(
				batch::execute("reload", &args, |_| panic!("invalid batch executed")).is_err(),
				"{args}"
			);
		}
	}

	#[test]
	fn read_batches_overlap_with_a_bound_and_preserve_order_and_errors() {
		let active = AtomicUsize::new(0);
		let peak = AtomicUsize::new(0);
		let gate = (Mutex::new(false), Condvar::new());
		let args = json!({"items":(0..10).map(|n| json!({"n":n})).collect::<Vec<_>>(),"concurrency":3});
		let reply = batch::execute("show", &args, |item| {
			let current = active.fetch_add(1, Ordering::SeqCst) + 1;
			peak.fetch_max(current, Ordering::SeqCst);
			let mut opened = gate.0.lock().unwrap();
			if current == 3 {
				*opened = true;
				gate.1.notify_all();
			}
			let (opened, _) = gate
				.1
				.wait_timeout_while(opened, Duration::from_secs(3), |open| !*open)
				.unwrap();
			assert!(*opened, "items did not overlap");
			drop(opened);
			let n = item["n"].as_u64().unwrap();
			std::thread::sleep(Duration::from_millis(10 - n));
			active.fetch_sub(1, Ordering::SeqCst);
			if n == 3 {
				Err("missing run".into())
			} else {
				Ok(n.to_string())
			}
		})
		.unwrap_err();
		let reply: Value = serde_json::from_str(&reply).unwrap();
		assert_eq!(peak.load(Ordering::SeqCst), 3);
		assert_eq!(reply["failed"], 1);
		assert_eq!(reply["skipped"], 0);
		for (n, result) in reply["results"].as_array().unwrap().iter().enumerate() {
			assert_eq!(result["index"], n);
			assert_eq!(result["result"]["error"], n == 3);
			assert_eq!(
				result["result"]["content"],
				if n == 3 {
					"missing run".into()
				} else {
					n.to_string()
				}
			);
		}
	}

	#[test]
	fn mutations_run_in_order_and_stop_without_rolling_back() {
		let seen = Mutex::new(Vec::new());
		let reply = batch::execute(
			"ast_grep",
			&json!({"items":[
        {"n":0}, {"n":1,"apply":true}, {"n":2}
    ],"concurrency":8}),
			|item| {
				let n = item["n"].as_u64().unwrap();
				seen.lock().unwrap().push(n);
				if n == 1 {
					Err("rewrite failed".into())
				} else {
					Ok("done".into())
				}
			},
		)
		.unwrap_err();
		assert_eq!(*seen.lock().unwrap(), vec![0, 1]);
		let reply: Value = serde_json::from_str(&reply).unwrap();
		assert_eq!(reply["failed"], 1);
		assert_eq!(reply["skipped"], 1);
		assert_eq!(reply["results"][0]["result"]["content"], "done");
	}
}
