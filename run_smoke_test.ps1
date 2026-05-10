$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Code = Join-Path $Root "code"
$Data = Join-Path $Root "data\SEMG_DB1"
$Out = Join-Path $Root "outputs\smoke"

New-Item -ItemType Directory -Force -Path $Out | Out-Null

Push-Location $Code
try {
    python run_tests.py
    if ($LASTEXITCODE -ne 0) { throw "run_tests.py failed with exit code $LASTEXITCODE" }
    python run_enhanced_publishable_experiment.py `
        --data-root $Data `
        --output-dir $Out `
        --subjects 1Nmar `
        --epochs 1 `
        --seeds 42 `
        --selection-seeds 42 `
        --no-plots `
        --device auto
    if ($LASTEXITCODE -ne 0) { throw "run_enhanced_publishable_experiment.py failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Smoke test finished. Outputs:"
Write-Host $Out
