# run_inspector.ps1 — launch MCP Inspector with sarvam-tools pre-configured.
# Run from anywhere: .\run_inspector.ps1
# Then just click Connect — Command and Args are pre-filled, SARVAM_API_KEY is in env.

$root = $PSScriptRoot

$apiKey = ((Get-Content "$root\.env") -match '^SARVAM_API_KEY=') -replace '^SARVAM_API_KEY=', ''
$apiKey = $apiKey.Trim()
if (-not $apiKey) {
    Write-Error 'SARVAM_API_KEY not found in .env — copy .env.example to .env and add your key.'
    exit 1
}

$env:SARVAM_API_KEY = $apiKey

# cd to project root so relative paths work and spaces in the directory name
# are never passed to the inspector (it uses CWD when spawning the server).
Set-Location $root

$pkg = '@modelcontextprotocol/inspector'

# start_server.bat reads SARVAM_API_KEY from .env itself before launching Python,
# so the key is always set regardless of what the inspector passes as env.
# Relative path — no spaces, no quoting issues.
npx.cmd --yes $pkg 'start_server.bat'
