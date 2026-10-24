# 🦀 rust-boilerplate

**A Rust starter your agent can build, reload and profile.**

[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![MCP](https://img.shields.io/badge/MCP-server-7C3AED)](https://modelcontextprotocol.io)
[![Stars](https://img.shields.io/github/stars/yesitsfebreeze/rust-bp?style=social)](https://github.com/yesitsfebreeze/rust-bp/stargazers)

Start with `src/main.rs` and the small counter module in `src/modules/counter/`.
The app keeps ticking while rebuilt modules swap in. The harness in
`src/runtime/` gives your agent one MCP server to run, inspect and profile it.

- **Reload with state.** Module edits keep the process and module state. App edits restart with both app and module state restored. Failed builds or state transfers keep the working generation.
- **Measure real work.** Read live counters and timings, inspect memory, and compare profiles before and after a change.
- **Navigate the code.** Find definitions and references with rust-analyzer, or search structurally with ast-grep.
- **See what happened.** Follow builds and tool activity with `just viz`.

## Get started

Requires Rust/Cargo on Windows 10 1803 or newer, macOS, or Linux. Windows
also needs the MSVC C++ build tools and Git for Windows (Bash on PATH for
the shell recipes/tests). Install [just](https://github.com/casey/just) for the recipes.

```sh
git clone https://github.com/yesitsfebreeze/rust-bp.git
cd rust-bp
cargo build --release --locked
just serve
```

Edit `STEP` in `src/modules/counter/lib.rs`, then inspect `just call channels`:
the count continues with the new step and the process stays alive. Edit
`src/main.rs` to rebuild the app; its tick count and the module count survive
the restart. To name your project, run `bash setup.sh my-project`.

Your app lives in `src/main.rs` and `src/bin/`; the reusable daemon and MCP
crates live in `src/runtime/`. All use ordinary Cargo manifests.

## Connect your agent

The checkout includes configuration for Claude Code and Codex. Start your
agent from the project root and approve or trust the project configuration.
For another MCP host, register `./mcp.sh` as a stdio server with this
directory as its working directory. On Windows, use
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./mcp.ps1`.

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
just smoke    # real MCP, module swaps and app state handoff; no Docker or API key
```

For the tool list, agent configuration, profiling workflows and troubleshooting,
start with [llms.txt](llms.txt), which links to the focused guides in `docs/*.txt`.

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=yesitsfebreeze/rust-bp&type=Date)](https://star-history.com/#yesitsfebreeze/rust-bp&Date)

## In-game console editing

- Console: arrows/Home/End move the caret, Shift selects, Ctrl+arrows and
  Ctrl+Backspace/Delete edit words. Click or drag to select; double-click selects
  a word and triple-click selects the line. Ctrl+A/C/X/V/Z/Y select all,
  copy/cut/paste and undo/redo (Command on macOS, Option for word movement).
  Right-click opens the edit menu. Up/Down browse history and restore your draft;
  Tab completes at the caret. F1 or Escape closes the console; backtick and caret
  characters can be typed while it is open. Pasting line breaks inserts spaces
  and never submits a command.
