$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Code = Join-Path $Root "code"
$Data = Join-Path $Root "data\SEMG_DB1"
$Out = Join-Path $Root "outputs\artifacts_enhanced_publishable_reproduced"

New-Item -ItemType Directory -Force -Path $Out | Out-Null

Push-Location $Code
try {
    python run_enhanced_publishable_experiment.py `
        --data-root $Data `
        --output-dir $Out `
        --subjects all `
        --epochs 160 `
        --seeds 42,3407,2026,2027,7 `
        --selection-seeds 42 `
        --device auto
    if ($LASTEXITCODE -ne 0) { throw "run_enhanced_publishable_experiment.py failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Main experiment finished. Key files:"
Write-Host (Join-Path $Out "metrics_summary.csv")
Write-Host (Join-Path $Out "metrics_by_subject.csv")
Write-Host (Join-Path $Out "leakage_audit.json")
Write-Host (Join-Path $Out "publication_readiness_report.json")
