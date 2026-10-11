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
		"events" => args.get("subscribe_thread").is_some() || args["unsubscribe"] == true,
		"docs" | "audit" | "usage" | "compare" | "status" | "channels" | "runs" | "show" | "rank" => {
			false
		}
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
