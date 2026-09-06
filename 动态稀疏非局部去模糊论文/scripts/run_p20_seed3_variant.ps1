param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Local', 'C00', 'C01')]
    [string]$Role
)
$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe'
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $PSScriptRoot 'train_p12_matched_nonlocal.py'
$baseline = Join-Path $projectRoot 'results\runs\P3_baseline_group_split_v0\best.pt'
$calibration = Join-Path $projectRoot 'results\runs\P7A_blur_strength_calibration_v1\summary.json'
$fingerprints = @{
    $trainer = '37FE90523CD9E50A8F3A3D07AECAE1E175680BDD2BC9846F41277CF2FA378B24'
    $baseline = '8538E91D8327DC6E17E08AB008D114FE33BE1C7A31E619352F62A5CEDF3A78C5'
    $calibration = 'DD704BBF6707CEBFB418078DA7C2D1F649090DCBC8C25193A45D3CB092CFFC99'
}
foreach ($path in $fingerprints.Keys) {
    if ((Get-FileHash -LiteralPath $path).Hash -ne $fingerprints[$path]) {
        throw "Frozen input changed: $path"
    }
}
$variant = 'gate_adaptive'
$weight = '0'
switch ($Role) {
    'Local' { $variant = 'local'; $runName = 'P20A_local_seed12011_v0' }
    'C00' { $runName = 'P20G_unsupervised_seed12011_v0' }
    'C01' { $weight = '0.01'; $runName = 'P20C01_paired_teacher_seed12011_v0' }
}
$outputDir = Join-Path $projectRoot "results\runs\$runName"
if (Test-Path -LiteralPath $outputDir) { throw "Refusing to overwrite: $outputDir" }
$snapshot = Join-Path $outputDir 'source_snapshot'
New-Item -ItemType Directory -Path (Join-Path $snapshot 'scripts') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'src') -Destination (Join-Path $snapshot 'src') -Recurse
Copy-Item -LiteralPath $trainer -Destination (Join-Path $snapshot 'scripts\train_p12_matched_nonlocal.py')
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $snapshot 'scripts\run_p20_seed3_variant.ps1')
# Find the fixed protocol by its unambiguous stage prefix (avoids script-file encoding issues).
$protocol = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'docs') -File -Filter '17_P20*.md')
if ($protocol.Count -ne 1) { throw 'Expected exactly one preregistered P20 protocol' }
Copy-Item -LiteralPath $protocol[0].FullName -Destination (Join-Path $snapshot 'protocol.md')
$trainingArgs = @(
    '-u', $trainer, '--data-root', (Join-Path $projectRoot 'data\GOPRO_Large'),
    '--baseline-checkpoint', $baseline, '--calibration-summary', $calibration,
    '--output', $outputDir, '--variant', $variant,
    '--scales', '1', '3', '7', '--top-k', '6', '--statistics-window', '5',
    '--steps', '3000', '--batch-size', '6', '--train-patch-size', '256',
    '--learning-rate', '0.0002', '--frequency-weight', '0.1',
    '--gate-supervision-weight', $weight, '--teacher-mode', 'paired',
    '--val-interval', '375', '--num-workers', '2', '--sampling', 'image_uniform',
    '--seed', '12011'
)
$quotedArgs = @($trainingArgs | ForEach-Object { '"' + [string]$_ + '"' })
Write-Host "Starting $Role, seed 12011: $outputDir"
$process = Start-Process -FilePath $pythonExe -ArgumentList $quotedArgs -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $outputDir 'train_stdout.log') -RedirectStandardError (Join-Path $outputDir 'train_stderr.log')
if ($process.ExitCode -notin @(0, 2)) { throw "Training failed, exit $($process.ExitCode); logs preserved" }
if (-not (Test-Path -LiteralPath (Join-Path $outputDir 'summary.json'))) { throw 'Missing final summary' }
Write-Host "Completed $Role (exit $($process.ExitCode)); compare matched controls, not only B0 gates."
exit 0
