$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Code = Join-Path $Root "code"
$Data = Join-Path $Root "data\SEMG_DB1"
$MainArtifacts = Join-Path $Root "outputs\artifacts_enhanced_publishable_reproduced"
$RevisionOut = Join-Path $Root "outputs\revision_package_reproduced"

if (-not (Test-Path (Join-Path $MainArtifacts "metrics_summary.csv"))) {
    throw "Main artifacts were not found. Run .\run_main_experiment.ps1 first."
}

Push-Location $Code
try {
    python run_reviewer_artifacts.py `
        --data-root $Data `
        --artifact-root $MainArtifacts `
        --strict-artifact-root $MainArtifacts `
        --output-dir $RevisionOut
    if ($LASTEXITCODE -ne 0) { throw "run_reviewer_artifacts.py failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Reviewer artifact generation finished:"
Write-Host $RevisionOut
