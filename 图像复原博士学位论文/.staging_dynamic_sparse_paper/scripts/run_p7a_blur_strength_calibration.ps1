$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$analysisScript = Join-Path $PSScriptRoot "analyze_p7_blur_strength.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$outputDir = Join-Path $projectRoot "results\runs\P7A_blur_strength_calibration_v1"
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    Write-Host "P7-A blur-strength calibration is already complete."
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $analysisScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --output $outputDir `
    --patch-size 256 `
    --batch-size 8 `
    --max-pairs 0 `
    --num-workers 2 `
    --calibration-fraction 0.25 `
    --ridge 0.001 `
    --seed 3407

if ($LASTEXITCODE -ne 0) {
    throw "P7-A blur-strength calibration failed with exit code $LASTEXITCODE"
}
Write-Host "P7-A completed. Summary: $summaryPath"
