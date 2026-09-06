$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot "train_strength_stratified_trajectory.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$initializationCheckpoint = Join-Path $projectRoot "results\runs\P6F_aggregation_adaptation_v0\best.pt"
$calibrationDir = Join-Path $projectRoot "results\runs\P7A_blur_strength_calibration_v1"
$calibrationSummary = Join-Path $calibrationDir "summary.json"
$strengthManifest = Join-Path $calibrationDir "per_image_strength.csv"
$outputDir = Join-Path $projectRoot "results\runs\P7B_strength_stratified_joint_v0"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    $existingSummary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "P7-B is already complete. passed=$($existingSummary.passed)"
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $trainScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --initialization-checkpoint $initializationCheckpoint `
    --calibration-summary $calibrationSummary `
    --strength-manifest $strengthManifest `
    --output $outputDir `
    --steps 3000 `
    --batch-size 6 `
    --train-pairs 0 `
    --val-pairs 0 `
    --train-patch-size 256 `
    --learning-rate 0.0002 `
    --frequency-weight 0.1 `
    --heavy-endpoint-weight 0.5 `
    --light-safety-weight 0.5 `
    --val-interval 375 `
    --num-workers 2 `
    --seed 3407

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "P7-B completed, but one or more frozen scientific criteria were not met."
    Write-Host "This is a valid screening result, not a runtime failure."
} elseif ($experimentExitCode -ne 0) {
    throw "P7-B failed with exit code $experimentExitCode"
} else {
    Write-Host "P7-B passed all frozen screening criteria."
}
Write-Host "Summary: $summaryPath"
