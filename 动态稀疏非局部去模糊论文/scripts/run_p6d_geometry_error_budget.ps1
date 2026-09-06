$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$evaluationScript = Join-Path $PSScriptRoot "evaluate_p6d_geometry_error_budget.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$predictedCheckpoint = Join-Path $projectRoot "results\runs\P6C_wide_direction_geometry_v0\best.pt"
$outputDir = Join-Path $projectRoot "results\runs\P6D_geometry_error_budget_v0"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    Write-Host "P6-D geometry error-budget diagnostic is already complete."
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $evaluationScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --predicted-checkpoint $predictedCheckpoint `
    --output $outputDir `
    --val-pairs 0 `
    --batch-size 4 `
    --oracle-window 31 `
    --num-workers 2

if ($LASTEXITCODE -ne 0) {
    throw "P6-D geometry error-budget diagnostic failed with exit code $LASTEXITCODE"
}
Write-Host "P6-D geometry error-budget diagnostic completed."
Write-Host "Summary: $summaryPath"
