#!/usr/bin/env bash
# Build the workspace, optionally renaming the app: bash tools/setup.sh [name].
# Needs Bash, Perl and Cargo. No agent configuration is read or changed.
set -euo pipefail
cd "$(dirname "$0")/.."

new="${1:-}"
if [[ -z "$new" ]]; then
  cargo build --release --workspace --locked
  exit 0
fi
if [[ ! "$new" =~ ^[a-z][a-z0-9_-]{0,63}$ ]]; then
  echo "usage: bash tools/setup.sh [name] (a-z 0-9 _ -, starting with a letter)" >&2
  exit 2
fi
case "$new" in
  daemon|rustc-events|viz|reload|counter)
    echo "tools/setup.sh: '$new' is taken by a runtime target" >&2
    exit 2 ;;
esac
cur=$(perl -ne 'if (m{"\./target/release/([A-Za-z0-9_-]+)"}) { print $1; exit }' src/runtime/daemon/hot.rs)
if [[ -z "$cur" ]]; then
  echo "tools/setup.sh: cannot read app path from src/runtime/daemon/hot.rs" >&2
  exit 1
fi
if [[ "$new" != "$cur" ]]; then
  # Names are validated above; Perl handles both LF and CRLF checkouts.
  q=$(perl -e 'print quotemeta shift' "$cur")
  sub() { perl -pi -e "s#$2#$3#g" "$1"; }
  sub Cargo.toml "^name = \"$q\"(?=\\r?\$)" "name = \"$new\""
  for f in src/runtime/daemon/hot.rs src/runtime/daemon/bin/daemon.rs \
           src/runtime/daemon/profiles.rs src/runtime/daemon/mcp/mod.rs README.md; do
    sub "$f" "release/$q\\b" "release/$new"
    sub "$f" "-p $q\\b" "-p $new"
  done
  sub src/runtime/daemon/mcp/mod.rs "\"$q\"" "\"$new\""
  sub README.md "--bin $q\\b" "--bin $new"
  perl -pi -e "s{^# .*\$}{# $new} if \$. == 1" README.md
fi
# Renaming changes Cargo.lock. Build app, module and runtime together.
cargo build --release --workspace
echo "App: $new. Register target/release/daemon (daemon.exe on Windows); see docs/install.txt."
