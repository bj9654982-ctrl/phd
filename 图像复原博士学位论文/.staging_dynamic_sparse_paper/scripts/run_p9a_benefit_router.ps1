$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$analysisScript = Join-Path $PSScriptRoot "analyze_p9_benefit_router.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$candidateCheckpoint = Join-Path $projectRoot "results\runs\P8B_e2e_trajectory_v0\best.pt"
$calibrationSummary = Join-Path $projectRoot "results\runs\P7A_blur_strength_calibration_v1\summary.json"
$outputDir = Join-Path $projectRoot "results\runs\P9A_benefit_router_v0"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "P9-A is already complete. passed=$($summary.passed)"
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $analysisScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --candidate-checkpoint $candidateCheckpoint `
    --calibration-summary $calibrationSummary `
    --output $outputDir `
    --patch-size 256 `
    --batch-size 8 `
    --num-workers 2 `
    --ridge 1.0 `
    --seed 3407

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "P9-A completed without passing every development gate."
} elseif ($experimentExitCode -ne 0) {
    throw "P9-A failed with exit code $experimentExitCode"
}
Write-Host "Summary: $summaryPath"
