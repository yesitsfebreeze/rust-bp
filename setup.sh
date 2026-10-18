#!/usr/bin/env bash
# Bootstrap:         ./setup.sh
# Name the project:  ./setup.sh <name>
#
# Without a name: build the workspace.
# With a name: rename the app (the root package, binary target/release/app) to <name>,
# points the daemon, the checks and the README at it, and names the MCP
# server <name> in .mcp.json, so the agent sees mcp__<name>__status instead of
# mcp__app__status. The framework crates stay `daemon` and `mcp`.
# The current names are read from the tree, so running it again renames again.
# Needs bash, perl and cargo; ends with `cargo build --release` so Cargo.lock
# follows the new package name.
set -euo pipefail
cd "$(dirname "$0")"

new="${1:-}"
if [[ -z "$new" ]]; then
  cargo build --release
  echo "next: cargo run --release -p daemon -- serve    # or ./setup.sh <name> to name the project"
  exit 0
fi
if [[ ! "$new" =~ ^[a-z][a-z0-9_-]{0,63}$ ]]; then
  echo "usage: ./setup.sh [<name>]   (a-z 0-9 _ -, starting with a letter)" >&2
  exit 2
fi
case "$new" in
  daemon|mcp|mcp-launcher|mcp-usage-hook|rustc-events|viz|reload|counter)
    echo "setup.sh: '$new' is taken by a framework binary" >&2
    exit 2 ;;
esac

cur=$(perl -ne 'if (m{"\./target/release/([A-Za-z0-9_-]+)"}) { print $1; exit }' src/runtime/daemon/hot.rs)
key=$(perl -0ne 'print $1 if /"mcpServers":\s*\{\s*"([^"]+)"/' .mcp.json)
if [[ -z "$cur" || -z "$key" ]]; then
  echo "setup.sh: cannot read the current names from src/runtime/daemon/hot.rs and .mcp.json" >&2
  exit 1
fi
if [[ "$new" == "$cur" ]]; then
  echo "setup.sh: already named $cur"
  cargo build --release
  exit 0
fi

# sub FILE PATTERN REPLACEMENT: perl, so macOS and Linux behave the same.
# '#' delimits, so braces and quotes in the texts need no escaping.
sub() { perl -pi -e "s#$2#$3#g" "$1"; }
q=$(perl -e 'print quotemeta shift' "$cur")
kq=$(perl -e 'print quotemeta shift' "$key")

# The package (root Cargo.toml) and with it the bin; then every path and `-p` that names it.
perl -pi -e "s{^name = \"$q\"(?=\\r?\$)}{name = \"$new\"} if !\$done && (\$done = /^name = \"$q\"\\r?\$/)" Cargo.toml
for f in src/runtime/daemon/hot.rs src/runtime/daemon/bin/daemon.rs src/runtime/daemon/profiles.rs \
         src/runtime/mcp/main.rs tests/mcp/integration.rs README.md; do
  sub "$f" "release/$q\\b" "release/$new"   # ./target/release/ and <daemon dir>/target-linux/release/
  sub "$f" "-p $q\\b" "-p $new"
done
sub tests/mcp/integration.rs "\\(\"callgrind\", \"$q\"\\)" "(\"callgrind\", \"$new\")"
sub README.md "bin \`$q\`" "bin \`$new\`"
sub README.md "--bin $q\\b" "--bin $new"
perl -pi -e "s{^# .*\$}{# $new} if \$. == 1" README.md

# The MCP server name the agent sees.
if [[ -f .codex/config.toml ]]; then
  sub .codex/config.toml "\\[mcp_servers\\.$kq\\]" "[mcp_servers.$new]"
fi
perl -0pi -e "s{(\"mcpServers\":\\s*\\{\\s*\")$kq\"}{\${1}$new\"}" .mcp.json
sub src/runtime/mcp/main.rs "\"serverInfo\": \\{\"name\": \"$kq\"" "\"serverInfo\": {\"name\": \"$new\""
sub src/runtime/mcp/main.rs "as \`$kq\`;" "as \`$new\`;"
sub README.md "mcpServers\\.$kq\\b" "mcpServers.$new"
sub README.md "claude mcp get $kq\\b" "claude mcp get $new"
sub README.md "claude mcp add --scope local $kq\\b" "claude mcp add --scope local $new"
sub README.md "\\[mcp_servers\\.$kq\\]" "[mcp_servers.$new]"
sub README.md "mcp__${kq}__" "mcp__${new}__"
sub README.md "(supplies|wires) \`$kq\`" "\$1 \`$new\`"

cargo build --release
echo
echo "named the project $new (executable was $cur, MCP server was $key)"
echo "next: cargo run --release -p daemon -- serve    # then edit src/main.rs"
