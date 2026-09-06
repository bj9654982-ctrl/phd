$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$evaluationScript = Join-Path $PSScriptRoot "evaluate_p10_complementary_composition.py"
$runs = Join-Path $projectRoot "results\runs"
$outputDir = Join-Path $runs "P10_complementary_composition_v0"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "P10 is already complete. method_passed=$($summary.method_passed)"
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $evaluationScript `
    --data-root (Join-Path $projectRoot "data\GOPRO_Large") `
    --baseline-checkpoint (Join-Path $runs "P3_baseline_group_split_v0\best.pt") `
    --scalar-checkpoint (Join-Path $runs "P4_scalar_residual_gate_v1_patch256\best_gate.pt") `
    --spatial-checkpoint (Join-Path $runs "P5_spatial_residual_gate_v1_anchored\best_gate.pt") `
    --local-checkpoint (Join-Path $runs "P8A_e2e_local_control_v0\best.pt") `
    --trajectory-checkpoint (Join-Path $runs "P8B_e2e_trajectory_v0\best.pt") `
    --calibration-summary (Join-Path $runs "P7A_blur_strength_calibration_v1\summary.json") `
    --output $outputDir `
    --batch-size 1 `
    --num-workers 2 `
    --seed 3407

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "P10 completed without passing every method gate."
} elseif ($experimentExitCode -ne 0) {
    throw "P10 failed with exit code $experimentExitCode"
}
Write-Host "Summary: $summaryPath"
