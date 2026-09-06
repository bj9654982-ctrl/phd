# to_doubao_1 持续监控脚本
# 每分钟检查一次，发现文件后写入信号日志
$monitorDir = "J:\phd\doubao_codex"
$targetFile = Join-Path $monitorDir "to_doubao_1"
$signalFile = Join-Path $monitorDir "to_doubao_1_detected.log"
$logFile = Join-Path $monitorDir "monitor.log"

$checkCount = 0
while ($true) {
    $checkCount++
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if (Test-Path $targetFile) {
        $msg = "[$timestamp] 第${checkCount}次检查: 发现 to_doubao_1!"
        Write-Host $msg
        Add-Content -Path $signalFile -Value $msg -Encoding UTF8
        Add-Content -Path $logFile -Value $msg -Encoding UTF8
        # 发现后继续监控（防止文件被重新创建）
    } else {
        $msg = "[$timestamp] 第${checkCount}次检查: 未发现 to_doubao_1"
        Write-Host $msg
        Add-Content -Path $logFile -Value $msg -Encoding UTF8
    }
    Start-Sleep -Seconds 60
}
