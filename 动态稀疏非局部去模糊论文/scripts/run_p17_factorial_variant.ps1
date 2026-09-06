param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("C10", "C01")]
    [string]$Cell
)

$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($Cell -eq "C10") {
    $variant = "protected_gate_adaptive"
    $teacherWeight = 0.0
    $runName = "P17C10_outer_gate_only_seed7813_v0"
} else {
    $variant = "gate_adaptive"
    $teacherWeight = 0.01
    $runName = "P17C01_teacher_only_seed7813_v0"
}
$outputDir = Join-Path $projectRoot "results\runs\$runName"
$summaryPath = Join-Path $outputDir "summary.json"
if (Test-Path -LiteralPath $summaryPath) {
    Write-Host "Completed run already exists: $summaryPath"
    exit 0
}
if (Test-Path -LiteralPath $outputDir) {
    throw "Output exists without a summary; refusing to overwrite: $outputDir"
}

# Archive the current sources before training without modifying historical runs.
$snapshotRoot = Join-Path $outputDir "source_snapshot"
New-Item -ItemType Directory -Path (Join-Path $snapshotRoot "scripts") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot "src") -Destination (Join-Path $snapshotRoot "src") -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "train_p12_matched_nonlocal.py") -Destination (Join-Path $snapshotRoot "scripts\train_p12_matched_nonlocal.py")
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $snapshotRoot "scripts\run_p17_factorial_variant.ps1")

& $pythonExe -u (Join-Path $PSScriptRoot "train_p12_matched_nonlocal.py") `
    --data-root (Join-Path $projectRoot "data\GOPRO_Large") `
    --baseline-checkpoint (Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0\best.pt") `
    --calibration-summary (Join-Path $projectRoot "results\runs\P7A_blur_strength_calibration_v1\summary.json") `
    --output $outputDir `
    --variant $variant `
    --scales 1 3 7 --top-k 6 --statistics-window 5 `
    --steps 3000 --batch-size 6 --train-patch-size 256 `
    --learning-rate 0.0002 --frequency-weight 0.1 `
    --gate-supervision-weight $teacherWeight `
    --val-interval 375 --num-workers 2 --sampling image_uniform --seed 7813

$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "Training completed; at least one absolute B0 performance gate failed."
} elseif ($experimentExitCode -ne 0) {
    throw "Training failed with exit code $experimentExitCode"
}
if (-not (Test-Path -LiteralPath $summaryPath)) {
    throw "Training returned without its final summary: $summaryPath"
}
Write-Host "Summary: $summaryPath"
exit 0
