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
