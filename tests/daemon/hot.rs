use daemon::hot::parse_idle;
use std::time::Duration;

#[test]
fn idle_durations_parse() {
	assert_eq!(parse_idle("0").unwrap(), None);
	assert_eq!(parse_idle("90s").unwrap(), Some(Duration::from_secs(90)));
	assert_eq!(parse_idle("30m").unwrap(), Some(Duration::from_secs(1800)));
	assert_eq!(parse_idle("2h").unwrap(), Some(Duration::from_secs(7200)));
	assert_eq!(parse_idle("45").unwrap(), Some(Duration::from_secs(45)));
	assert!(parse_idle("soon").is_err());
	assert!(parse_idle("5d").is_err());
}
