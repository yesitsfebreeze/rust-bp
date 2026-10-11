# Recipes for this workspace. `just` lists them by group.
# Plain Cargo commands work too, including on a fresh checkout.

set positional-arguments

default:
    @just --list --unsorted

# ---- build ----

# Build; `just setup <name>` renames the demo crate and the MCP server.
[group('build')]
setup *name:
    ./setup.sh "$@"

# Release build: what the daemon swaps and the profilers inspect.
[group('build')]
build:
    cargo build --release

# The small binary for shipping (profile `distribution`).
[group('build')]
dist:
    cargo build --profile distribution

# Remove build artifacts; source manifests remain available.
[group('build')]
clean:
    cargo clean

# ---- check ----

# Type-check every target in the dev profile.
[group('check')]
check:
    cargo check --workspace --all-targets

# Clippy on every target; warnings fail.
[group('check')]
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Format the tree (`just fmt --check` only verifies).
[group('check')]
fmt *args:
    cargo fmt --all "$@"

# Unit and integration tests that need no daemon, profiler or Docker.
[group('check')]
test:
    cargo test --workspace --locked

# End-to-end check in an isolated harness (CHECK_LIVE_JEV=1: real Jev).
[group('check')]
e2e:
    cargo test -p mcp --test integration -- --ignored --nocapture check

# Fast end-to-end daemon, MCP and code-intelligence journey; no Docker or API key.
[group('check')]
smoke:
    CHECK_JOURNEY=lifecycle cargo test --locked -p mcp --test integration -- --ignored --nocapture --exact check

# Audit regressions, incl. Docker timeout and cancellation cleanup; needs a daemon and the image.
[group('check')]
regressions:
    cargo test -p mcp --test integration -- --ignored --nocapture regressions

# What CI runs: format check, clippy, tests.
[group('check')]
ci:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace --locked

# ---- run ----

# The hot-reload daemon in this terminal; `just serve --detach` runs it as the daemon the tools start on their own. `just serve -- <command>` supervises another child.
[group('run')]
serve *args:
    cargo run --release -p daemon -- serve "$@"

# Daemon control: status | reload | stop | dir | channels.
[group('run')]
daemon *args:
    cargo run --release -p daemon -- "$@"

# Call one MCP tool from the shell: `just call status`, `just call rank '{"name":"demo"}'`.
[group('run')]
call tool *json:
    cargo run --release -p mcp -- --call "$@"

# The MCP server on stdio, as .mcp.json starts it.
[group('run')]
mcp:
    ./mcp.sh

# The live session stream in the terminal: one column per tool, crate, model, file, symbol and run, calls stacked newest at the bottom (q quits).
[group('run')]
viz:
    cargo run --quiet --release -p daemon --bin viz

# Build the Linux valgrind image the container fallback uses.
[group('run')]
valgrind-image:
    docker build -f valgrind.Dockerfile -t valgrind-v2 .   # daemon::valgrind::IMAGE
