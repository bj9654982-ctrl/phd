$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot "train_spatial_residual_gate.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$scalarCheckpoint = Join-Path $projectRoot "results\runs\P4_scalar_residual_gate_v1_patch256\best_gate.pt"
$outputDir = Join-Path $projectRoot "results\runs\P5_spatial_residual_gate_v1_anchored"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    $existingSummary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "P5 v1 anchored experiment is already complete. passed=$($existingSummary.passed)"
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $trainScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --scalar-checkpoint $scalarCheckpoint `
    --output $outputDir `
    --steps 2000 `
    --batch-size 4 `
    --train-pairs 0 `
    --val-pairs 0 `
    --gate-train-patch-size 256 `
    --gate-width 16 `
    --spatial-mode anchored `
    --max-correction 0.15 `
    --oracle-window 31 `
    --learning-rate 0.001 `
    --frequency-weight 0.1 `
    --oracle-weight 1.0 `
    --restoration-weight 0.1 `
    --smoothness-weight 0.01 `
    --val-interval 250 `
    --num-workers 2 `
    --seed 3407

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "P5 v1 completed, but one or more frozen scientific criteria were not met."
    Write-Host "This is a valid negative experiment, not a runtime failure."
} elseif ($experimentExitCode -ne 0) {
    throw "P5 v1 anchored experiment failed with exit code $experimentExitCode"
} else {
    Write-Host "P5 v1 completed and passed all frozen screening criteria."
}

Write-Host "Summary: $summaryPath"
