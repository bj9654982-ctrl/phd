$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$evaluationScript = Join-Path $PSScriptRoot "evaluate_p12_condition_interventions.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$adaptiveCheckpoint = Join-Path $projectRoot "results\runs\P12D_blur_aware_nonlocal_v0\best.pt"
$calibrationSummary = Join-Path $projectRoot "results\runs\P7A_blur_strength_calibration_v1\summary.json"
$outputDir = Join-Path $projectRoot "results\runs\P12E_condition_intervention_v0"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    Write-Host "P12 condition intervention audit is already complete."
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $evaluationScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --adaptive-checkpoint $adaptiveCheckpoint `
    --calibration-summary $calibrationSummary `
    --output $outputDir `
    --num-workers 2 `
    --seed 3407

if ($LASTEXITCODE -ne 0) {
    throw "P12 condition intervention audit failed with exit code $LASTEXITCODE"
}
Write-Host "Summary: $summaryPath"
