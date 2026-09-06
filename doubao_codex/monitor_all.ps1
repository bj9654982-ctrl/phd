# 增强版监控脚本：监控 to_doubao_2 到 to_doubao_10
# 每分钟检查一次，发现新文件后写入信号日志
$monitorDir = "J:\phd\doubao_codex"
$logFile = Join-Path $monitorDir "monitor_all.log"
$signalFile = Join-Path $monitorDir "NEW_REVIEW_SIGNAL.log"

# 已发现的文件集合（防止重复记录）
$found = @{}

$checkCount = 0
while ($true) {
    $checkCount++
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # 检查 to_doubao_2 到 to_doubao_10
    $newFiles = @()
    for ($i = 2; $i -le 10; $i++) {
        $target = Join-Path $monitorDir "to_doubao_$i"
        if (Test-Path $target) {
            if (-not $found.ContainsKey("to_doubao_$i")) {
                $found["to_doubao_$i"] = $true
                $newFiles += "to_doubao_$i"
                $msg = "[$timestamp] 发现新文件: to_doubao_$i!"
                Write-Host $msg
                Add-Content -Path $signalFile -Value $msg -Encoding UTF8
            }
        }
    }

    # 定期写入心跳日志（每5次检查写一次）
    if ($checkCount % 5 -eq 0) {
        $heartbeat = "[$timestamp] 第${checkCount}次检查: 运行中, 已发现$($found.Count)个文件"
        Add-Content -Path $logFile -Value $heartbeat -Encoding UTF8
    }

    Start-Sleep -Seconds 60
}
