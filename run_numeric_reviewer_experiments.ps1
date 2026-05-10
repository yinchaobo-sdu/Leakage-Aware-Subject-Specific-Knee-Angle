$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Code = Join-Path $Root "code"
$Data = Join-Path $Root "data\SEMG_DB1"
$MainArtifacts = Join-Path $Root "outputs\artifacts_enhanced_publishable_reproduced"
$AblationOut = Join-Path $Root "outputs\artifacts_nomultiscale_reproduced"
$RevisionOut = Join-Path $Root "outputs\revision_package_reproduced"

if (-not (Test-Path (Join-Path $MainArtifacts "subjects"))) {
    throw "Main checkpoints were not found. Run .\run_main_experiment.ps1 first."
}

Push-Location $Code
try {
    python run_reviewer_numeric_experiments.py `
        --data-root $Data `
        --main-artifact-root $MainArtifacts `
        --ablation-output-dir $AblationOut `
        --revision-dir $RevisionOut `
        --subjects all `
        --run both `
        --epochs 160 `
        --seeds 42 `
        --device auto
    if ($LASTEXITCODE -ne 0) { throw "run_reviewer_numeric_experiments.py failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Reviewer numeric experiments finished:"
Write-Host $RevisionOut
