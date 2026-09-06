param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("local", "gate_adaptive")]
    [string]$Variant
)

$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot "train_p12_matched_nonlocal.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$calibrationSummary = Join-Path $projectRoot "results\runs\P7A_blur_strength_calibration_v1\summary.json"
$runNames = @{
    local = "P13A_local_control_seed7813_v0"
    gate_adaptive = "P13G_gate_adaptive_seed7813_v0"
}
$outputDir = Join-Path $projectRoot ("results\runs\" + $runNames[$Variant])
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "$Variant seed-7813 replicate is already complete."
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe $trainScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --calibration-summary $calibrationSummary `
    --output $outputDir `
    --variant $Variant `
    --scales 1 3 7 `
    --top-k 6 `
    --statistics-window 5 `
    --steps 3000 `
    --batch-size 6 `
    --train-patch-size 256 `
    --learning-rate 0.0002 `
    --frequency-weight 0.1 `
    --val-interval 375 `
    --num-workers 2 `
    --seed 7813

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "$Variant seed-7813 replicate completed but did not pass every absolute B0 gate."
} elseif ($experimentExitCode -ne 0) {
    throw "$Variant seed-7813 replicate failed with exit code $experimentExitCode"
}
Write-Host "Summary: $summaryPath"
exit 0
