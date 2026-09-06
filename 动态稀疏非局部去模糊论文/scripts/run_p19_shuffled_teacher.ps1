param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(3407, 7813)]
    [int]$Seed
)
$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runName = 'P19C01_shuffled_teacher_seed' + $Seed + '_v0'
$outputDir = Join-Path $projectRoot "results\runs\$runName"
if (Test-Path -LiteralPath $outputDir) {
    throw "Refusing to overwrite an existing run: $outputDir"
}
$snapshotRoot = Join-Path $outputDir "source_snapshot"
New-Item -ItemType Directory -Path (Join-Path $snapshotRoot "scripts") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot "src") -Destination (Join-Path $snapshotRoot "src") -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "train_p12_matched_nonlocal.py") -Destination (Join-Path $snapshotRoot "scripts\train_p12_matched_nonlocal.py")
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $snapshotRoot "scripts\run_p19_shuffled_teacher.ps1")
$trainingArgs = @(
    '-u', (Join-Path $PSScriptRoot 'train_p12_matched_nonlocal.py'),
    '--data-root', (Join-Path $projectRoot 'data\GOPRO_Large'),
    '--baseline-checkpoint', (Join-Path $projectRoot 'results\runs\P3_baseline_group_split_v0\best.pt'),
    '--calibration-summary', (Join-Path $projectRoot 'results\runs\P7A_blur_strength_calibration_v1\summary.json'),
    '--output', $outputDir, '--variant', 'gate_adaptive',
    '--scales', '1', '3', '7', '--top-k', '6', '--statistics-window', '5',
    '--steps', '3000', '--batch-size', '6', '--train-patch-size', '256',
    '--learning-rate', '0.0002', '--frequency-weight', '0.1', '--gate-supervision-weight', '0.01',
    '--teacher-mode', 'shuffled', '--teacher-seed', ($Seed + 19000),
    '--val-interval', '375', '--num-workers', '2', '--sampling', 'image_uniform', '--seed', $Seed
)
& $pythonExe @trainingArgs
$experimentExitCode = $LASTEXITCODE
if ($experimentExitCode -eq 2) {
    Write-Host "Negative-control training completed; an absolute B0 gate was not met."
} elseif ($experimentExitCode -ne 0) {
    throw "Training failed with exit code $experimentExitCode"
}
if (-not (Test-Path -LiteralPath (Join-Path $outputDir 'summary.json'))) {
    throw 'Training returned without a summary.'
}
Write-Host "Completed: $outputDir"
