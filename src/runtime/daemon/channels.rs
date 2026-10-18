//! The preset profiler channel: every interesting number of the executable
//! recorded in-process and read live through a local socket.
//!
//! Three kinds, one registry, zero configuration:
//!
//! - **counter** — `channels::inc("items")` / `add(name, n)`: how often.
//! - **latency** — `channels::timer("op.x")`: microsecond times in a log2
//!   histogram (bucket `i` counts calls under `2^i` µs), reported as
//!   mean/p50/p90/p99/max.
//! - **value** — `channels::set("depth", 1.5)`: the last value and its age.
//!
//! The executable calls [`serve`] once; readers (the `daemon channels` CLI,
//! the MCP `channels` tool) call [`read`] against the same path. Channels
//! are per-process: a hot swap starts them from zero again.

use crate::wire;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// log2 microseconds: bucket `i` holds calls under `2^i` µs, up to ~3.5 h.
const BUCKETS: usize = 48;

/// One latency histogram: total and max for the mean, log2 buckets for the
/// percentiles. Percentiles report the lower edge of the bucket they land in.
struct Latency {
	total_us: u64,
	max_us: u64,
	buckets: [u64; BUCKETS],
}

impl Default for Latency {
	fn default() -> Self {
		Self {
			total_us: 0,
			max_us: 0,
			buckets: [0; BUCKETS],
		}
	}
}

struct Summary {
	count: u64,
	mean_us: u64,
	p50_us: u64,
	p90_us: u64,
	p99_us: u64,
	max_us: u64,
}

impl Latency {
	fn record(&mut self, micros: u64) {
		// bucket i holds calls under 2^i µs: index = number of bits used
		let bucket = (64 - micros.leading_zeros() as usize).min(BUCKETS - 1);
		self.buckets[bucket] += 1;
		self.total_us += micros;
		self.max_us = self.max_us.max(micros);
	}

	fn percentile(&self, count: u64, p: f64) -> u64 {
		if count == 0 {
			return 0;
		}
		let target = (count as f64 * p).ceil() as u64;
		let mut seen = 0;
		for (i, &b) in self.buckets.iter().enumerate() {
			seen += b;
			if seen >= target {
				return if i == 0 { 0 } else { 1 << (i - 1) };
			}
		}
		1 << (BUCKETS - 1)
	}

	fn summary(&self) -> Summary {
		let count: u64 = self.buckets.iter().sum();
		Summary {
			count,
			mean_us: self.total_us.checked_div(count).unwrap_or(0),
			p50_us: self.percentile(count, 0.50),
			p90_us: self.percentile(count, 0.90),
			p99_us: self.percentile(count, 0.99),
			max_us: self.max_us,
		}
	}
}

fn started() -> Instant {
	static START: OnceLock<Instant> = OnceLock::new();
	*START.get_or_init(Instant::now)
}

// Keep histogram storage inline; see the before/after allocation evidence.
#[allow(clippy::large_enum_variant)]
enum Inner {
	Counter(AtomicU64),
	Latency(Mutex<Latency>),
	Value(Mutex<(f64, u64)>),
}

struct Channel {
	kind: &'static str,
	inner: Inner,
}

fn registry() -> &'static Mutex<BTreeMap<String, Channel>> {
	static REGISTRY: OnceLock<Mutex<BTreeMap<String, Channel>>> = OnceLock::new();
	REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn entry(name: &str, make: impl FnOnce() -> Inner, kind: &'static str) {
	started();
	let mut reg = registry().lock().unwrap();
	reg.entry(name.to_string()).or_insert_with(|| Channel {
		kind,
		inner: make(),
	});
}

/// Count one occurrence.
pub fn inc(name: &str) {
	add(name, 1);
}

/// Count `n` occurrences.
pub fn add(name: &str, n: u64) {
	entry(name, || Inner::Counter(AtomicU64::new(0)), "counter");
	let reg = registry().lock().unwrap();
	if let Some(Channel {
		inner: Inner::Counter(c),
		..
	}) = reg.get(name)
	{
		c.fetch_add(n, Ordering::Relaxed);
	}
}

/// Record the last value of a gauge, with the time it was set.
pub fn set(name: &str, value: f64) {
	entry(name, || Inner::Value(Mutex::new((0.0, 0))), "value");
	let reg = registry().lock().unwrap();
	if let Some(Channel {
		inner: Inner::Value(v),
		..
	}) = reg.get(name)
	{
		*v.lock().unwrap() = (value, now_ms());
	}
}

/// Record one duration in microseconds under `name` (a latency histogram).
pub fn record_us(name: &str, micros: u64) {
	entry(
		name,
		|| Inner::Latency(Mutex::new(Latency::default())),
		"latency",
	);
	let reg = registry().lock().unwrap();
	if let Some(Channel {
		inner: Inner::Latency(latency),
		..
	}) = reg.get(name)
	{
		latency.lock().unwrap().record(micros);
	}
}

/// Time a scope: `channels::timer("op.x").done()` records the elapsed time;
/// dropping the timer without `done` records it too.
pub fn timer(name: &str) -> Timer {
	Timer {
		name: name.to_string(),
		start: Instant::now(),
		armed: true,
	}
}

/// One measurement in flight.
pub struct Timer {
	name: String,
	start: Instant,
	armed: bool,
}

impl Timer {
	/// Record the elapsed time and stop the timer.
	pub fn done(mut self) {
		self.record();
	}

	fn record(&mut self) {
		if self.armed {
			self.armed = false;
			record_us(&self.name, self.start.elapsed().as_micros() as u64);
		}
	}
}

impl Drop for Timer {
	fn drop(&mut self) {
		self.record();
	}
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|d| d.as_millis() as u64)
		.unwrap_or(0)
}

/// Every channel as one JSON object: process info plus one entry per channel.
pub fn snapshot() -> Value {
	let reg = registry().lock().unwrap();
	let mut channels = Vec::new();
	for (name, ch) in &*reg {
		let mut entry = Map::new();
		entry.insert("name".into(), json!(name));
		entry.insert("kind".into(), json!(ch.kind));
		match &ch.inner {
			Inner::Counter(c) => {
				entry.insert("count".into(), json!(c.load(Ordering::Relaxed)));
			}
			Inner::Latency(latency) => {
				let summary = latency.lock().unwrap().summary();
				entry.insert("count".into(), json!(summary.count));
				entry.insert("mean_us".into(), json!(summary.mean_us));
				entry.insert("p50_us".into(), json!(summary.p50_us));
				entry.insert("p90_us".into(), json!(summary.p90_us));
				entry.insert("p99_us".into(), json!(summary.p99_us));
				entry.insert("max_us".into(), json!(summary.max_us));
			}
			Inner::Value(v) => {
				let (value, at) = *v.lock().unwrap();
				entry.insert("value".into(), json!(value));
				entry.insert("age_ms".into(), json!(now_ms().saturating_sub(at)));
			}
		}
		channels.push(Value::Object(entry));
	}
	json!({
			"process": {
					"pid": std::process::id(),
					"uptime_s": started().elapsed().as_secs(),
			},
			"channels_total": channels.len(),
			"channels": channels,
	})
}

type PersistHook = std::sync::Arc<dyn Fn(&str) -> Value + Send + Sync>;
static PERSIST: Mutex<Option<PersistHook>> = Mutex::new(None);

/// Answer `{"op":"persist"}` on the channel socket with `hook`: the daemon
/// sends it before restarting the executable so it can save its state
/// (`reload::Host` installs this). The latest hook wins.
pub fn on_persist(hook: impl Fn(&str) -> Value + Send + Sync + 'static) {
	*PERSIST.lock().unwrap() = Some(std::sync::Arc::new(hook));
}

static SERVER: Mutex<Option<wire::Server>> = Mutex::new(None);

/// Serve the channel snapshots at `path` (call once from the executable);
/// `persist` requests go to the [`on_persist`] hook.
pub fn serve(path: &str) -> Result<(), String> {
	let listener = wire::bind(path).map_err(|e| format!("serve {path}: {e}"))?;
	*SERVER.lock().unwrap() = Some(wire::serve_loop(listener, |request| {
		if request["op"] == "persist" || request["op"] == "resume" {
			let hook = PERSIST.lock().unwrap().clone();
			return match hook {
				Some(hook) => hook(request["op"].as_str().unwrap()),
				None => json!({"op": "persist", "ok": true, "supported": false}),
			};
		}
		snapshot()
	}));
	Ok(())
}

/// Read a snapshot from the executable's channel socket.
pub fn read(path: &str) -> Result<Value, String> {
	wire::ask(path, json!({"op": "snapshot"}))
}

/// Save final channels for a bounded native benchmark, when requested by the
/// runner. Call at normal program exit, after all measured scopes have ended.
pub fn save_requested_snapshot() -> Result<(), String> {
	if let Some(path) = std::env::var_os("CHANNELS_OUT") {
		std::fs::write(path, snapshot().to_string()).map_err(|e| e.to_string())?;
	}
	Ok(())
}

/// Render a snapshot as a table, latency rows first, then counters and values.
pub fn render_table(snapshot: &Value) -> String {
	render_filtered(snapshot, usize::MAX, "")
}

pub fn render_filtered(snapshot: &Value, top: usize, filter: &str) -> String {
	let mut lines = vec![format!(
		"{:<8}{:>10}{:>10}{:>9}{:>9}{:>9}{:>9}{:>9}  name",
		"kind", "count", "mean_us", "p50_us", "p90_us", "p99_us", "max_us", "value"
	)];
	let mut rows: Vec<&Value> = snapshot["channels"]
		.as_array()
		.map(|a| a.iter().collect())
		.unwrap_or_default();
	rows.sort_by_key(|r| {
		let rank = match r["kind"].as_str() {
			Some("latency") => 0,
			Some("counter") => 1,
			_ => 2,
		};
		(rank, std::cmp::Reverse(r["count"].as_u64().unwrap_or(0)))
	});
	let matched = rows
		.iter()
		.filter(|r| r["name"].as_str().unwrap_or("").contains(filter))
		.count();
	for r in rows
		.into_iter()
		.filter(|r| r["name"].as_str().unwrap_or("").contains(filter))
		.take(top)
	{
		let kind = r["kind"].as_str().unwrap_or("?");
		let name = r["name"].as_str().unwrap_or("?");
		let num = |k: &str| r[k].as_u64().unwrap_or(0);
		let value = r["value"]
			.as_f64()
			.map(|v| format!("{v:.3}"))
			.unwrap_or_else(|| "-".into());
		lines.push(format!(
			"{:<8}{:>10}{:>10}{:>9}{:>9}{:>9}{:>9}{:>9}  {}",
			kind,
			num("count"),
			if kind == "latency" {
				num("mean_us").to_string()
			} else {
				"-".into()
			},
			if kind == "latency" {
				num("p50_us").to_string()
			} else {
				"-".into()
			},
			if kind == "latency" {
				num("p90_us").to_string()
			} else {
				"-".into()
			},
			if kind == "latency" {
				num("p99_us").to_string()
			} else {
				"-".into()
			},
			if kind == "latency" {
				num("max_us").to_string()
			} else {
				"-".into()
			},
			value,
			name,
		));
	}
	let p = &snapshot["process"];
	lines.push(format!(
		"\nprocess pid {} uptime {} s · {} channels",
		p["pid"], p["uptime_s"], snapshot["channels_total"]
	));
	lines.push(format!("{} of {matched} matching channels shown. Percentiles are log2 bucket lower bounds in integer µs (sub-µs values round to zero). Uptime starts with channel instrumentation.", matched.min(top)));
	lines.join("\n")
}

#[cfg(test)]
mod tests {
	use crate as daemon;
	use daemon::channels::{add, inc, record_us, render_table, set, snapshot, timer};

	#[test]
	fn counters_latencies_and_values_answer_the_snapshot() {
		add("test.items", 2);
		inc("test.items");
		record_us("test.op", 3); // bucket 2 (< 4 µs)
		record_us("test.op", 1_000); // bucket 10
		record_us("test.op", 2_000); // bucket 11
		set("test.depth", 1.5);

		let snap = snapshot();
		let find = |name: &str| {
			snap["channels"]
				.as_array()
				.unwrap()
				.iter()
				.find(|c| c["name"] == name)
				.unwrap_or_else(|| panic!("channel {name} missing"))
				.clone()
		};
		let items = find("test.items");
		assert_eq!(items["kind"], "counter");
		assert_eq!(items["count"], 3);
		let op = find("test.op");
		assert_eq!(op["kind"], "latency");
		assert_eq!(op["count"], 3);
		assert!(op["mean_us"].as_u64().unwrap() >= 1000, "{}", op);
		assert!(op["p50_us"].as_u64().unwrap() >= 256, "{}", op);
		assert!(op["p90_us"].as_u64().unwrap() >= 1024, "{}", op);
		assert_eq!(op["max_us"], 2000);
		let depth = find("test.depth");
		assert_eq!(depth["kind"], "value");
		assert_eq!(depth["value"], 1.5);

		let table = render_table(&snap);
		assert!(table.contains("test.op"), "{table}");
		assert!(table.contains("latency"), "{table}");
	}

	#[test]
	fn timer_records_on_done_and_once_on_drop() {
		timer("test.drop").done();
		std::thread::sleep(std::time::Duration::from_micros(100));
		let armed = timer("test.armed");
		armed.done();
		let snap = snapshot();
		let count = |name: &str| {
			snap["channels"]
				.as_array()
				.unwrap()
				.iter()
				.find(|c| c["name"] == name)
				.map(|c| c["count"].as_u64().unwrap_or(0))
				.unwrap_or(0)
		};
		assert_eq!(count("test.drop"), 1);
		assert_eq!(count("test.armed"), 1);
	}

	#[test]
	fn table_limits_matching_channels() {
		let snapshot = serde_json::json!({"process":{"pid":1,"uptime_s":1},"channels_total":3,"channels":[
				{"name":"work/a","kind":"counter","count":1},
				{"name":"work/b","kind":"counter","count":2},
				{"name":"other","kind":"counter","count":3}
		]});
		let text = daemon::channels::render_filtered(&snapshot, 1, "work/");
		assert!(text.contains("work/b"));
		assert!(!text.contains("work/a"));
		assert!(!text.contains("other"));
		assert!(text.contains("1 of 2 matching"));
	}
}
