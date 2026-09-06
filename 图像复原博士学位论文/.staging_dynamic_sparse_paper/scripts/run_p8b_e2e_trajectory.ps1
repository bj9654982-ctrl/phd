$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot "train_strength_stratified_trajectory.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$calibrationDir = Join-Path $projectRoot "results\runs\P7A_blur_strength_calibration_v1"
$outputDir = Join-Path $projectRoot "results\runs\P8B_e2e_trajectory_v0"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "P8-B is already complete. passed=$($summary.passed)"
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $trainScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --initialization-checkpoint $baselineCheckpoint `
    --calibration-summary (Join-Path $calibrationDir "summary.json") `
    --strength-manifest (Join-Path $calibrationDir "per_image_strength.csv") `
    --output $outputDir `
    --candidate-variant trajectory `
    --trajectory-initialization identity_from_baseline `
    --trainable-scope all `
    --trajectory-samples 5 `
    --max-feature-length 7.0 `
    --trajectory-fusion detail `
    --direction-residual-scale 2.0 `
    --steps 3000 `
    --batch-size 6 `
    --train-patch-size 256 `
    --learning-rate 0.0002 `
    --frequency-weight 0.1 `
    --heavy-endpoint-weight 2.0 `
    --light-safety-weight 0.1 `
    --val-interval 375 `
    --num-workers 2 `
    --seed 3407

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "P8-B completed without passing every absolute method gate; compare it directly with P8-A."
} elseif ($experimentExitCode -ne 0) {
    throw "P8-B failed with exit code $experimentExitCode"
}
Write-Host "Summary: $summaryPath"
