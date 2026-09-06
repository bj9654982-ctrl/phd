# ============================================================
# 前台监控脚本 - 持续监控 to_doubao_3 到 to_doubao_10
# 前台窗口运行，不会被后台杀掉
# ============================================================

$monitorDir = "J:\phd\doubao_codex"
$logFile = Join-Path $monitorDir "monitor_foreground.log"
$signalFile = Join-Path $monitorDir "NEW_REVIEW_SIGNAL.log"

# 已发现的文件集合
$found = @{}
# 已生成的codex文件集合（用于显示进度）
$codexFound = @{}

$checkCount = 0
$startTime = Get-Date

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  图像复原项目 - 审查整改循环监控" -ForegroundColor Cyan
Write-Host "  监控范围: to_doubao_3 ~ to_doubao_10" -ForegroundColor Cyan
Write-Host "  启动时间: $startTime" -ForegroundColor Cyan
Write-Host "  检查频率: 每 60 秒一次" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "当前信号文件状态:" -ForegroundColor Yellow
for ($i = 1; $i -le 10; $i++) {
    $doubao = Join-Path $monitorDir "to_doubao_$i"
    $codex = Join-Path $monitorDir "to_codex_$i"
    $dStatus = if (Test-Path $doubao) { "[OK]" } else { "[--]" }
    $cStatus = if (Test-Path $codex) { "[OK]" } else { "[--]" }
    Write-Host "  to_doubao_$i $dStatus    to_codex_$i $cStatus"
}
Write-Host ""
Write-Host "开始监控... (按 Ctrl+C 停止)" -ForegroundColor Green
Write-Host ""

while ($true) {
    $checkCount++
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $elapsed = (Get-Date) - $startTime
    $elapsedStr = "$($elapsed.Hours)h $($elapsed.Minutes)m $($elapsed.Seconds)s"

    # 检查 to_doubao_3 到 to_doubao_10
    $newFiles = @()
    for ($i = 3; $i -le 10; $i++) {
        $target = Join-Path $monitorDir "to_doubao_$i"
        if (Test-Path $target) {
            if (-not $found.ContainsKey("to_doubao_$i")) {
                $found["to_doubao_$i"] = $true
                $newFiles += "to_doubao_$i"
                $msg = "[$timestamp] 发现新审查信号: to_doubao_$i !"
                Write-Host ""
                Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
                Write-Host "  $msg" -ForegroundColor Red
                Write-Host "  请通知 MainAgent 进行第 $i 轮整改!" -ForegroundColor Yellow
                Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
                Write-Host ""
                Add-Content -Path $signalFile -Value $msg -Encoding UTF8
                Add-Content -Path $logFile -Value $msg -Encoding UTF8
            }
        }
    }

    # 检查 to_codex 文件（显示整改进度）
    for ($i = 1; $i -le 10; $i++) {
        $codex = Join-Path $monitorDir "to_codex_$i"
        if (Test-Path $codex) {
            if (-not $codexFound.ContainsKey("to_codex_$i")) {
                $codexFound["to_codex_$i"] = $true
                $msg = "[$timestamp] 整改完成信号: to_codex_$i 已生成"
                Write-Host $msg -ForegroundColor Green
                Add-Content -Path $logFile -Value $msg -Encoding UTF8
            }
        }
    }

    # 心跳显示（每10次检查更新一次状态行）
    if ($checkCount % 10 -eq 0) {
        $foundCount = $found.Count
        $codexCount = $codexFound.Count
        Write-Host "[$timestamp] 心跳: 第${checkCount}次检查, 已运行$elapsedStr, 发现审查信号$foundCount个, 整改完成$codexCount个" -ForegroundColor Gray
        Add-Content -Path $logFile -Value "[$timestamp] 心跳: 第${checkCount}次检查, 已运行$elapsedStr" -Encoding UTF8
    }

    # 检查是否已经到to_codex_10（全部完成）
    if (Test-Path (Join-Path $monitorDir "to_codex_10")) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "  to_codex_10 已生成! 全部10轮整改循环完成!" -ForegroundColor Green
        Write-Host "  监控将在 10 秒后自动退出..." -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
        Start-Sleep -Seconds 10
        break
    }

    Start-Sleep -Seconds 60
}

Write-Host "监控结束。" -ForegroundColor Cyan
Read-Host "按回车键关闭窗口"
