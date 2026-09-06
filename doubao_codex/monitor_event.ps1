# 信号文件监控（2秒轮询，简单可靠）
# 用法：powershell -ExecutionPolicy Bypass -File monitor_event.ps1

$watchPath = "J:\phd\doubao_codex"
$logFile = Join-Path $watchPath "monitor_event.log"
$pollInterval = 2  # 秒

if (-not (Test-Path $watchPath)) {
    Write-Host "[错误] 目录不存在: $watchPath" -ForegroundColor Red
    exit 1
}

# 设置窗口标题（明显标识，避免误关）
$host.ui.RawUI.WindowTitle = "=== 信号监控中 - 请勿关闭 ==="

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  信号文件监控已启动" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  监控目录: $watchPath" -ForegroundColor White
Write-Host "  轮询间隔: $pollInterval 秒" -ForegroundColor White
Write-Host "  日志文件: $logFile" -ForegroundColor White
Write-Host "  启动时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
Write-Host ""
Write-Host "  说明:" -ForegroundColor Yellow
Write-Host "  - 每 $pollInterval 秒检查一次，检测到 to_doubao_* 立即提示" -ForegroundColor Gray
Write-Host "  - 响铃 + 红色醒目提示" -ForegroundColor Gray
Write-Host "  - 按 Ctrl+C 停止" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 记录已知文件
$known = @{}
Get-ChildItem $watchPath -Force | Where-Object { $_.Name -like "to_doubao_*" } | ForEach-Object {
    $known[$_.Name] = $_.LastWriteTime
    Write-Host "  [已知] $($_.Name)  ($($_.LastWriteTime))" -ForegroundColor Gray
}
Write-Host ""

$heartbeatCount = 0

try {
    while ($true) {
        Start-Sleep -Seconds $pollInterval
        $heartbeatCount++

        # 检查新文件
        $newSignals = @()
        Get-ChildItem $watchPath -Force | Where-Object { $_.Name -like "to_doubao_*" } | ForEach-Object {
            if (-not $known.ContainsKey($_.Name)) {
                $newSignals += $_
                $known[$_.Name] = $_.LastWriteTime
            }
        }

        # 处理新信号
        foreach ($file in $newSignals) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $name = $file.Name
            $roundNum = $name -replace 'to_doubao_', ''

            Write-Host ""
            Write-Host "============================================================" -ForegroundColor Yellow
            Write-Host "  !! 新审查信号到达: $name !!" -ForegroundColor Red
            Write-Host "  时间: $timestamp" -ForegroundColor Yellow
            Write-Host "  请通知 Doubao 端进行第 $roundNum 轮整改" -ForegroundColor Cyan
            Write-Host "============================================================" -ForegroundColor Yellow
            Write-Host ""

            # 写入日志
            $msg = "[$timestamp] [信号] 检测到新文件: $name"
            Add-Content -Path $logFile -Value $msg -Encoding UTF8

            # 响铃
            [console]::beep(1000, 500)
            Start-Sleep -Milliseconds 200
            [console]::beep(1200, 500)
        }

        # 每30秒打印心跳
        if ($heartbeatCount % 15 -eq 0) {
            $hb = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Write-Host "[$hb] [心跳] 监控运行中，已知信号 $($known.Count) 个..." -ForegroundColor DarkGray
        }
    }
}
finally {
    Write-Host "`n[停止] 监控已终止" -ForegroundColor Yellow
}
