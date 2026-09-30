//! The valgrind runner: callgrind and memcheck runs of any command, inside
//! a Linux container — Apple Silicon macOS builds no valgrind, so the
//! project is mounted at `/repo` in the `boiler-valgrind` image (OrbStack or
//! any Docker), and container builds land in `.boiler/target-linux` so they
//! never clobber the host's `target/`.
//!
//! Every run is stored parsed under `.boiler/valgrind/<name>.json`
//! (callgrind: per-function instruction counts; memcheck: grouped findings
//! with leak and error summaries), next to the raw files for reference.

use crate::util;
use serde_json::{json, Map, Value};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const IMAGE: &str = "boiler-valgrind";
const CONTAINER_CARGO: &str = "boiler-cargo";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    Callgrind,
    Memcheck,
}

impl Tool {
    pub fn from_name(name: &str) -> Result<Tool, String> {
        match name {
            "callgrind" => Ok(Tool::Callgrind),
            "memcheck" => Ok(Tool::Memcheck),
            other => Err(format!("unknown valgrind tool `{other}`: callgrind or memcheck")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Tool::Callgrind => "callgrind",
            Tool::Memcheck => "memcheck",
        }
    }
}

pub struct Request {
    /// Stored as `.boiler/valgrind/<name>.json`; one path component.
    pub name: String,
    pub tool: Tool,
    /// The argv the container runs under valgrind, from `/repo`.
    pub command: Vec<String>,
    /// A shell line run in the container before valgrind (e.g.
    /// `cargo build --release -p app` — its target dir is .boiler/target-linux).
    pub build: Option<String>,
    /// Bound on the whole container run. Default 600.
    pub timeout_s: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Run one valgrind job; the reply is the stored run record.
pub fn run(req: Request) -> Result<Value, String> {
    if req.name.is_empty() || req.name.contains('/') || req.name.contains("..") {
        return Err("`name` must be one path component (it names .boiler/valgrind/<name>.json)".into());
    }
    if req.command.is_empty() {
        return Err("`command` is empty: give the argv the container runs under valgrind".into());
    }
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let dir = root.join(crate::VALGRIND_DIR);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let out = format!("{}/{}.out", crate::VALGRIND_DIR, req.name);
    let log = format!("{}/{}.log", crate::VALGRIND_DIR, req.name);
    let ann = format!("{}/{}.ann", crate::VALGRIND_DIR, req.name);
    docker_up()?;
    ensure_image()?;

    let command = req
        .command
        .iter()
        .map(|w| shell_word(w))
        .collect::<Vec<_>>()
        .join(" ");
    let mut script = String::from("set -e\ncd /repo\n");
    if let Some(build) = &req.build {
        script.push_str(build);
        script.push('\n');
    }
    match req.tool {
        Tool::Callgrind => {
            script.push_str(&format!(
                "valgrind --tool=callgrind --callgrind-out-file=/repo/{out} --log-file=/repo/{log} -- {command}\n"
            ));
            script.push_str(&format!(
                "callgrind_annotate --threshold=100 /repo/{out} > /repo/{ann}\n"
            ));
        }
        Tool::Memcheck => {
            script.push_str(&format!(
                "valgrind --tool=memcheck --leak-check=full --show-leak-kinds=definite,indirect,possible --track-origins=yes --log-file=/repo/{log} -- {command}\n"
            ));
        }
    }
    let docker = vec![
        "docker".to_string(),
        "run".to_string(),
        "--rm".to_string(),
        "-v".to_string(),
        format!("{}:/repo", root.display()),
        "-w".to_string(),
        "/repo".to_string(),
        "-v".to_string(),
        format!("{CONTAINER_CARGO}:/usr/local/cargo"),
        "-e".to_string(),
        "CARGO_HOME=/usr/local/cargo".to_string(),
        "-e".to_string(),
        "CARGO_TARGET_DIR=/repo/.boiler/target-linux".to_string(),
        IMAGE.to_string(),
        "sh".to_string(),
        "-c".to_string(),
        script,
    ];
    let started = Instant::now();
    let run_error = match util::run_with_timeout(&docker, Duration::from_secs(req.timeout_s.max(30)), 2000) {
        Ok(_) => None,
        Err(error) => Some(error),
    };

    let mut record = json!({
        "name": req.name,
        "tool": req.tool.name(),
        "command": req.command,
        "build": req.build,
        "created_ms": now_ms(),
        "run_seconds": started.elapsed().as_secs(),
        "run_error": run_error,
    });
    match req.tool {
        Tool::Callgrind => {
            let text = std::fs::read_to_string(root.join(&ann))
                .map_err(|e| format!("no annotate output at {ann}: {e} (was the run killed before it finished?)"))?;
            let functions = parse_annotate(&text);
            let total: u64 = functions.iter().map(|(_, ir)| ir).sum();
            if total == 0 {
                return Err(format!(
                    "callgrind recorded nothing in {out}{}",
                    run_error.map(|e| format!(" — the run failed: {e}")).unwrap_or_default()
                ));
            }
            let mut rows: Vec<Value> = functions
                .iter()
                .map(|(location, ir)| {
                    json!({"location": location, "ir": ir, "share": *ir as f64 / total as f64})
                })
                .collect();
            rows.sort_by(|a, b| b["ir"].as_u64().cmp(&a["ir"].as_u64()));
            record["total_ir"] = json!(total);
            record["functions_total"] = json!(rows.len());
            record["functions"] = Value::Array(rows);
            record["files"] = json!({"annotate": ann, "raw": out, "log": log});
        }
        Tool::Memcheck => {
            let text = std::fs::read_to_string(root.join(&log))
                .map_err(|e| format!("no memcheck log at {log}: {e}"))?;
            let (findings, leak, summary) = parse_memcheck(&text);
            let errors: u64 = findings.iter().map(|f| f["count"].as_u64().unwrap_or(0)).sum();
            if errors == 0 {
                return Err(format!(
                    "memcheck found nothing{}",
                    run_error.map(|e| format!(" — the run failed: {e}")).unwrap_or_default()
                ));
            }
            record["findings_total"] = json!(findings.len());
            record["errors_total"] = json!(errors);
            record["findings"] = Value::Array(findings);
            record["leak_summary"] = leak;
            record["error_summary"] = json!(summary);
            record["files"] = json!({"log": log});
        }
    }
    let path = dir.join(format!("{}.json", req.name));
    std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap())
        .map_err(|e| e.to_string())?;
    Ok(record)
}

/// Every stored run, one row each, newest last.
pub fn list() -> Result<Value, String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let dir = root.join(crate::VALGRIND_DIR);
    let mut rows = Vec::new();
    let entries = std::fs::read_dir(&dir).map_err(|_| format!("no .boiler/valgrind yet: no run stored"))?;
    let mut names: Vec<_> = entries.flatten().filter_map(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        name.strip_suffix(".json").map(|n| n.to_string())
    }).collect();
    names.sort();
    for name in names {
        if let Ok(record) = load(&name) {
            rows.push(json!({
                "name": name,
                "tool": record["tool"],
                "created_ms": record["created_ms"],
                "total_ir": record.get("total_ir").cloned().unwrap_or(Value::Null),
                "errors_total": record.get("errors_total").cloned().unwrap_or(Value::Null),
            }));
        }
    }
    Ok(json!({"runs": rows}))
}

/// Load one stored run by name.
pub fn load(name: &str) -> Result<Value, String> {
    if name.contains('/') || name.contains("..") {
        return Err("`name` must be one path component".into());
    }
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let path = root.join(crate::VALGRIND_DIR).join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .map_err(|_| format!("no stored run `{name}`: list them with the runs tool"))?;
    serde_json::from_str(&text).map_err(|e| format!("{name}.json does not parse: {e}"))
}

/// A docker daemon must answer; on a Mac, try OrbStack first.
fn docker_up() -> Result<(), String> {
    if util::succeeds(&["docker".into(), "info".into()]) {
        return Ok(());
    }
    let _ = std::process::Command::new("orb")
        .arg("start")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    for _ in 0..20 {
        std::thread::sleep(Duration::from_secs(1));
        if util::succeeds(&["docker".into(), "info".into()]) {
            return Ok(());
        }
    }
    Err("no docker daemon answers `docker info` — start it (on a Mac: `orb start`)".into())
}

/// Build the valgrind image once; reuse it afterwards. The caller runs with
/// the project root as cwd, which is the `docker build` context.
fn ensure_image() -> Result<(), String> {
    if util::succeeds(&["docker".into(), "image".into(), "inspect".into(), IMAGE.into()]) {
        return Ok(());
    }
    println!("valgrind: building image {IMAGE} (once)");
    let build = vec![
        "docker".to_string(),
        "build".to_string(),
        "-f".to_string(),
        "valgrind.Dockerfile".to_string(),
        "-t".to_string(),
        IMAGE.to_string(),
        ".".to_string(),
    ];
    util::run_with_timeout(&build, Duration::from_secs(900), 2000)
        .map(|_| ())
        .map_err(|e| format!("cannot build image {IMAGE} from valgrind.Dockerfile: {e}"))
}

/// Quote one argv word for the container's `sh -c`.
fn shell_word(word: &str) -> String {
    let safe = !word.is_empty()
        && word
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'-' | b'_' | b'=' | b':'));
    if safe {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', r"'\''"))
    }
}

/// One row of `callgrind_annotate` output: `27,534,715 (44.85%) file:function`.
fn annotate_row(line: &str) -> Option<(u64, &str)> {
    let (number, rest) = line.split_once(" (")?;
    let (percent, tail) = rest.split_once("%)")?;
    if !percent
        .bytes()
        .all(|b| b.is_ascii_digit() || b == b'.' || b == b',')
    {
        return None;
    }
    let ir: u64 = number.replace(',', "").trim().parse().ok()?;
    let location = tail.trim();
    if ir == 0 || location.is_empty() {
        return None;
    }
    Some((ir, location))
}

/// The whole annotate listing: every `(location, instructions)` row.
pub fn parse_annotate(text: &str) -> Vec<(String, u64)> {
    text.lines()
        .filter_map(annotate_row)
        .map(|(ir, location)| (location.to_string(), ir))
        .collect()
}

const ERROR_STARTS: &[&str] = &[
    "Invalid read",
    "Invalid write",
    "Invalid free",
    "Mismatched free",
    "Use of uninitialised value",
    "Conditional jump or move",
    "Syscall param",
    "Unaddressable",
    "jump to the invalid address",
    "Bad permissions",
    "Invalid alignment",
    "Uninit",
];

/// Memcheck log → grouped findings, the leak summary and the error summary.
pub fn parse_memcheck(text: &str) -> (Vec<Value>, Value, String) {
    let mut findings: Vec<Value> = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut leak = Map::new();
    let mut summary = String::new();
    for line in text.lines() {
        let Some(body) = line
            .strip_prefix("==")
            .and_then(|rest| rest.split_once("=="))
            .map(|(_, body)| body.trim())
        else {
            continue;
        };
        if let Some((kind, _)) = ERROR_STARTS
            .iter()
            .find(|k| body.starts_with(*k))
            .map(|k| (*k, body))
        {
            let message = body.to_string();
            let key = (kind.to_string(), message.clone());
            match seen.iter().position(|k| k == &key) {
                Some(i) => findings[i]["count"] = json!(findings[i]["count"].as_u64().unwrap_or(0) + 1),
                None => {
                    seen.push(key);
                    findings.push(json!({"kind": kind, "message": message, "count": 1, "frame": ""}));
                }
            }
            continue;
        }
        if let Some(rest) = body.strip_prefix("at 0x") {
            let frame = rest
                .split_once(": ")
                .and_then(|(_, tail)| tail.split_once(" (").map(|(name, _)| name.to_string()))
                .unwrap_or_default();
            if !frame.is_empty() {
                if let Some(last) = findings.last_mut() {
                    if last["frame"].as_str() == Some("") {
                        last["frame"] = json!(frame);
                    }
                }
            }
            continue;
        }
        for (prefix, field) in [
            ("definitely lost:", "definitely_lost"),
            ("indirectly lost:", "indirectly_lost"),
            ("possibly lost:", "possibly_lost"),
            ("still reachable:", "still_reachable"),
        ] {
            if let Some(rest) = body.strip_prefix(prefix) {
                let mut numbers = rest
                    .replace(',', "")
                    .split_whitespace()
                    .filter_map(|w| w.parse::<u64>().ok())
                    .take(2)
                    .collect::<Vec<_>>()
                    .into_iter();
                if let Some(bytes) = numbers.next() {
                    leak.insert(format!("{field}_bytes"), json!(bytes));
                }
                if let Some(blocks) = numbers.next() {
                    leak.insert(format!("{field}_blocks"), json!(blocks));
                }
                break;
            }
        }
        if let Some(rest) = body.strip_prefix("ERROR SUMMARY:") {
            summary = rest.trim().to_string();
        }
    }
    (findings, Value::Object(leak), summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annotate_rows_parse_and_noise_does_not() {
        let text = "\
--------------------------------------------------------------------------------
 I         file:function
--------------------------------------------------------------------------------
27,534,715 (44.85%)  ????:???:???: ???
6,872 (0.01%)  /repo/src/main.rs:app::parse_work
      .    ???:the end

--------------------------------------------------------------------------------
Profile data file 'x.out' (creator: callgrind-3.22.0)
--------------------------------------------------------------------------------
";
        let rows = parse_annotate(text);
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!(rows[0], ("????:???:???: ???".to_string(), 27_534_715));
        assert_eq!(rows[1], ("/repo/src/main.rs:app::parse_work".to_string(), 6_872));
    }

    #[test]
    fn memcheck_errors_group_and_summaries_parse() {
        let text = "\
==11== Invalid read of size 8
==11==    at 0x4005A1: parse_work (/repo/.boiler/target-linux/release/app)
==11==  Address 0x52bbe0 is 0 bytes inside a block of size 12 free'd
==11== Invalid read of size 8
==11==    at 0x4005A1: parse_work (/repo/.boiler/target-linux/release/app)
==11== Use of uninitialised value of size 8
==11==    by 0x400601: main (/repo/.boiler/target-linux/release/app)
==11==
==11== LEAK SUMMARY:
==11==    definitely lost: 10,240 bytes in 4 blocks
==11==    indirectly lost: 0 bytes in 0 blocks
==11==    still reachable: 8,192 bytes in 2 blocks
==11== ERROR SUMMARY: 10 errors from 3 contexts
";
        let (findings, leak, summary) = parse_memcheck(text);
        assert_eq!(findings.len(), 2, "{findings:?}");
        assert_eq!(findings[0]["kind"], "Invalid read");
        assert_eq!(findings[0]["count"], 2);
        assert_eq!(findings[0]["frame"], "parse_work");
        assert_eq!(findings[1]["kind"], "Use of uninitialised value");
        assert_eq!(leak["definitely_lost_bytes"], 10_240);
        assert_eq!(leak["definitely_lost_blocks"], 4);
        assert_eq!(leak["indirectly_lost_bytes"], 0);
        assert_eq!(leak["still_reachable_bytes"], 8_192);
        assert_eq!(summary, "10 errors from 3 contexts");
    }
}