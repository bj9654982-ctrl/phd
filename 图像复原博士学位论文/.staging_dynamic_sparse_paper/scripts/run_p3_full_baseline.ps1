$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot "train_baseline_mini.py"
$dataRoot = Join-Path $projectRoot "data\GOPRO_Large"
$outputDir = Join-Path $projectRoot "results\runs\P3_baseline_group_split_v0"
$summaryPath = Join-Path $outputDir "summary.json"
$latestPath = Join-Path $outputDir "latest.pt"

if (Test-Path -LiteralPath $summaryPath) {
    $existingSummary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($existingSummary.training_complete -eq $true) {
        Write-Host "P3 baseline is already complete. Summary: $summaryPath"
        exit 0
    }
}

$resumeArguments = @()
if (Test-Path -LiteralPath $latestPath) {
    Write-Host "Found an incomplete checkpoint; resuming from $latestPath"
    $resumeArguments = @("--resume", $latestPath)
}

& $pythonExe $trainScript `
    --data-root $dataRoot `
    --output $outputDir `
    --steps 30000 `
    --batch-size 4 `
    --patch-size 128 `
    --val-patch-size 256 `
    --train-pairs 0 `
    --val-pairs 0 `
    --validation-source train_holdout `
    --validation-groups GOPR0372_07 GOPR0378_13 GOPR0385_11 `
    --width 24 `
    --variant local `
    --learning-rate 0.0002 `
    --frequency-weight 0.1 `
    --val-interval 2500 `
    --num-workers 2 `
    --seed 3407 `
    @resumeArguments

if ($LASTEXITCODE -ne 0) {
    throw "P3 group-split baseline failed with exit code $LASTEXITCODE"
}

Write-Host "P3 group-split baseline completed. Summary: $outputDir\summary.json"
