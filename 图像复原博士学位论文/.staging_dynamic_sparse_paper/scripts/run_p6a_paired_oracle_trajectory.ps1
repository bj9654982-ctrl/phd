$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot "train_paired_oracle_trajectory.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$outputDir = Join-Path $projectRoot "results\runs\P6A_paired_oracle_trajectory_v0"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    $existingSummary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "P6-A paired-oracle trajectory experiment is already complete. passed=$($existingSummary.passed)"
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $trainScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --output $outputDir `
    --steps 3000 `
    --batch-size 4 `
    --train-pairs 0 `
    --val-pairs 0 `
    --train-patch-size 256 `
    --trajectory-samples 5 `
    --max-feature-length 7.0 `
    --trajectory-fusion detail `
    --oracle-window 31 `
    --learning-rate 0.0005 `
    --frequency-weight 0.1 `
    --val-interval 375 `
    --num-workers 2 `
    --seed 3407

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "P6-A completed, but one or more frozen scientific criteria were not met."
    Write-Host "This is a valid upper-bound result, not a runtime failure."
} elseif ($experimentExitCode -ne 0) {
    throw "P6-A paired-oracle trajectory experiment failed with exit code $experimentExitCode"
} else {
    Write-Host "P6-A completed and passed all frozen upper-bound criteria."
}

Write-Host "Summary: $summaryPath"
