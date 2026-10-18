# Native Windows MCP entry point. Keep stdout exclusively for JSON-RPC.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& cargo build --quiet --release --locked -p mcp --bins
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { 'target' }
$releaseDir = Join-Path $targetDir 'release'
$liveDir = Join-Path ([System.IO.Path]::GetTempPath()) "rust-bp-mcp-$PID"
New-Item -ItemType Directory -Force -Path $liveDir | Out-Null
$launcher = Join-Path $liveDir 'mcp-launcher.exe'
try {
    Copy-Item -LiteralPath (Join-Path $releaseDir 'mcp-launcher.exe') -Destination $launcher
    & $launcher --no-build @args
    $result = $LASTEXITCODE
} finally {
    Remove-Item -LiteralPath $launcher -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $liveDir -ErrorAction SilentlyContinue
}
exit $result
