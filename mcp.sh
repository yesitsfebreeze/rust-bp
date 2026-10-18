#!/usr/bin/env bash
# Build launcher and worker together once, then keep stdout for JSON-RPC.
set -euo pipefail
cd "$(dirname "$0")"
case "$(uname -s)" in
  MINGW*|MSYS*) exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./mcp.ps1 "$@" ;;
esac
cargo build --quiet --release --locked -p mcp --bins >&2
exec "${CARGO_TARGET_DIR:-target}/release/mcp-launcher" --no-build "$@"
