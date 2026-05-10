param(
    [double]$Tolerance = 0.25,
    [string]$ArtifactDir = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Code = Join-Path $Root "code"
if ([string]::IsNullOrWhiteSpace($ArtifactDir)) {
    $ArtifactDir = Join-Path $Root "outputs\artifacts_enhanced_publishable_reproduced"
}
elseif (-not [System.IO.Path]::IsPathRooted($ArtifactDir)) {
    $ArtifactDir = Join-Path $Root $ArtifactDir
}
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactDir)

Push-Location $Code
try {
    python verify_reproduction.py `
        --artifact-dir $ArtifactDir `
        --mae 6.296599355610934 `
        --rmse 9.033915257953181 `
        --tolerance $Tolerance
    if ($LASTEXITCODE -ne 0) { throw "verify_reproduction.py failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
