param(
    [ValidateSet(3407, 7813)]
    [int]$Seed = 3407
)

$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot "train_p12_matched_nonlocal.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$baselineCheckpoint = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt"
$calibrationSummary = Join-Path $projectRoot "results\runs\P7A_blur_strength_calibration_v1\summary.json"
$outputDir = Join-Path $projectRoot ("results\runs\P15_protected_nonlocal_seed{0}_v0" -f $Seed)
$summaryPath = Join-Path $outputDir "summary.json"

if (Test-Path -LiteralPath $summaryPath) {
    Write-Host "P15 protected nonlocal seed-$Seed run is already complete."
    Write-Host "Summary: $summaryPath"
    exit 0
}

& $pythonExe -u $trainScript `
    --data-root $dataRoot `
    --baseline-checkpoint $baselineCheckpoint `
    --calibration-summary $calibrationSummary `
    --output $outputDir `
    --variant protected_gate_adaptive `
    --scales 1 3 7 `
    --top-k 6 `
    --statistics-window 5 `
    --steps 3000 `
    --batch-size 6 `
    --train-patch-size 256 `
    --learning-rate 0.0002 `
    --frequency-weight 0.1 `
    --gate-supervision-weight 0.01 `
    --val-interval 375 `
    --num-workers 2 `
    --sampling image_uniform `
    --seed $Seed

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "P15 completed but did not pass every absolute B0 gate."
} elseif ($experimentExitCode -ne 0) {
    throw "P15 failed with exit code $experimentExitCode"
}
Write-Host "Summary: $summaryPath"
exit 0
