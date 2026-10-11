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
