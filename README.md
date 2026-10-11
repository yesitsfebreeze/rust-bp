# 🦀 rust-boilerplate

**A Rust starter your agent can build, reload and profile.**

[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![MCP](https://img.shields.io/badge/MCP-server-7C3AED)](https://modelcontextprotocol.io)
[![Stars](https://img.shields.io/github/stars/yesitsfebreeze/rust-bp?style=social)](https://github.com/yesitsfebreeze/rust-bp/stargazers)

Start with `src/main.rs`, which prints `hello world` and exits. The harness in `src/runtime/` gives your agent one
MCP server to run the app, inspect it and measure changes.

- **Reload on edit.** Rebuild and restart automatically; failed builds keep the last working app running.
- **Measure real work.** Read live counters and timings, inspect memory, and compare profiles before and after a change.
- **Navigate the code.** Find definitions and references with rust-analyzer, or search structurally with ast-grep.
- **See what happened.** Follow builds and tool activity with `just viz`.

## Get started

Requires Rust/Cargo on macOS or Linux. Install [just](https://github.com/casey/just)
to use the recipes below.

```sh
git clone https://github.com/yesitsfebreeze/rust-bp.git
cd rust-bp
cargo build --release --locked
just serve
```

Edit `src/main.rs` — the daemon rebuilds and restarts the app.
To name your project, run `./setup.sh my-project`.

Your app lives in `src/main.rs` and `src/bin/`; the reusable daemon and MCP
crates live in `src/runtime/`. All use ordinary Cargo manifests.

## Connect your agent

The checkout includes configuration for Claude Code and Codex. Start your
agent from the project root and approve or trust the project configuration.
For another MCP host, register `./mcp.sh` as a stdio server with this
directory as its working directory.

Try:

> Run the app and show me its output.

> Find the code behind this function and explain its callers.

> Profile this workload, improve the most expensive part, and compare the results.

Every tool also works from the shell:

```sh
just call status
just call channels
just viz
```

Code navigation needs rust-analyzer; structural search needs ast-grep.
Valgrind profiling needs native Valgrind or the Docker image.
Ranking works locally; Jev ranking optionally uses `TYPESAFE_API_KEY`.

## Check your changes

```sh
just ci       # formatting, Clippy and tests
just smoke    # real MCP and reload journey; no Docker or API key
```

For the tool list, agent configuration, profiling workflows and troubleshooting,
start with [llms.txt](llms.txt), which links to the focused guides in `docs/*.txt`.

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=yesitsfebreeze/rust-bp&type=Date)](https://star-history.com/#yesitsfebreeze/rust-bp&Date)
