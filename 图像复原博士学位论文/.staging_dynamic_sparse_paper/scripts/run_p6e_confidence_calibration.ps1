$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot "train_predicted_trajectory_geometry.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$oracleCheckpoint = Join-Path $projectRoot "results\runs\P6A_paired_oracle_trajectory_v0\best_aligned.pt"
$initializationCheckpoint = Join-Path $projectRoot "results\runs\P6C_wide_direction_geometry_v0\best.pt"
$outputDir = Join-Path $projectRoot "results\runs\P6E_confidence_calibration_v0"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    $existingSummary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "P6-E confidence-calibration experiment is already complete. passed=$($existingSummary.passed)"
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $trainScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --oracle-checkpoint $oracleCheckpoint `
    --initialization-checkpoint $initializationCheckpoint `
    --output $outputDir `
    --steps 3000 `
    --batch-size 4 `
    --train-pairs 0 `
    --val-pairs 0 `
    --train-patch-size 256 `
    --oracle-window 31 `
    --direction-residual-scale 2.0 `
    --training-mode confidence_calibration `
    --learning-rate 0.0005 `
    --geometry-weight 0.0 `
    --frequency-weight 0.1 `
    --val-interval 375 `
    --num-workers 2 `
    --seed 3407

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "P6-E completed, but one or more frozen scientific criteria were not met."
    Write-Host "This is a valid result, not a runtime failure."
} elseif ($experimentExitCode -ne 0) {
    throw "P6-E confidence-calibration experiment failed with exit code $experimentExitCode"
} else {
    Write-Host "P6-E completed and passed all frozen geometry-transfer criteria."
}

Write-Host "Summary: $summaryPath"
