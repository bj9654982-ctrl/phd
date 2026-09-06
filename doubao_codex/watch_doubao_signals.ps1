<#
.SYNOPSIS
    Doubao端信号文件监听器 - 监控 to_doubao_* 文件，检测到新信号时播放提示音并写入日志。

.DESCRIPTION
    参考 codex 端监听器实现，使用 PowerShell FileSystemWatcher + Wait-Event 真正事件驱动。
    单例锁文件防止重复启动。检测到 to_doubao_N 时播放两声提示音并写入日志。

.NOTES
    用法: powershell -ExecutionPolicy Bypass -File watch_doubao_signals.ps1
    停止: 关闭窗口或 Ctrl+C
#>

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$signalDir = [System.IO.Path]::GetFullPath('J:\phd\doubao_codex')
$stateDir = Join-Path $scriptDir 'state'
$logDir = Join-Path $stateDir 'logs'

[System.IO.Directory]::CreateDirectory($stateDir) | Out-Null
[System.IO.Directory]::CreateDirectory($logDir) | Out-Null

function Write-ListenerLog {
    param([string]$Message, [string]$Level = 'INFO')
    $stamp = [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss.fff')
    $line = "[$stamp] [$Level] $Message" + [Environment]::NewLine
    $logPath = Join-Path $logDir ('watcher-' + [DateTime]::Now.ToString('yyyy-MM-dd') + '.log')
    [System.IO.File]::AppendAllText($logPath, $line, [System.Text.UTF8Encoding]::new($false))
    Write-Host $line -NoNewline
}

function Test-RegularZeroByteFile {
    param([string]$Path)
    try {
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        return (-not $item.PSIsContainer) -and ($item.Length -eq 0)
    }
    catch {
        return $false
    }
}

function Get-NextPendingNumber {
    $number = 1
    while ($true) {
        $donePath = Join-Path $signalDir ("to_doubao_{0}" -f $number)
        if (-not (Test-Path -LiteralPath $donePath)) {
            return $number
        }
        if ($number -eq [long]::MaxValue) {
            throw 'Signal sequence exceeded the Int64 numbering range.'
        }
        $number++
    }
}

# 脚本级已提醒信号集合（避免重复提醒）
$script:notifiedSignals = @{}

function Invoke-SignalCheck {
    <# 检查是否有新的 to_doubao_N 信号，有则提醒一次并返回 #>
    # 从1开始扫描所有可能的信号
    $number = 1
    while ($true) {
        $signalPath = Join-Path $signalDir ("to_doubao_{0}" -f $number)
        if (-not (Test-RegularZeroByteFile -Path $signalPath)) {
            # 这个信号不存在，说明前面的都已检查过
            return 'waiting'
        }

        $signalName = "to_doubao_{0}" -f $number
        # 检查是否已经提醒过
        if (-not $script:notifiedSignals.ContainsKey($signalName)) {
            # 新信号，提醒一次
            $script:notifiedSignals[$signalName] = $true
            Write-ListenerLog "!! 新审查信号到达: $signalName !!" 'SIGNAL'
            Write-Host ""
            Write-Host "============================================================" -ForegroundColor Yellow
            Write-Host "  !! 新审查信号到达: $signalName !!" -ForegroundColor Red
            Write-Host "  时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
            Write-Host "  请通知 Doubao 端进行第 $number 轮整改" -ForegroundColor Cyan
            Write-Host "============================================================" -ForegroundColor Yellow
            Write-Host ""
            # 弹出窗口提醒（独立进程，不阻塞监听器）
            try {
                $msg = "新审查信号到达: $signalName`n时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n请通知 Doubao 端进行第 $number 轮整改"
                $cmd = "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('$msg', '信号监控提醒', 'OK', 'Information')"
                Start-Process powershell -ArgumentList "-NoProfile", "-Command", $cmd -WindowStyle Hidden
            } catch {}
            # 提醒后返回，不继续循环（避免一次事件弹出多个窗口）
            return 'signaled'
        }
        $number++
        if ($number -gt 10000) { return 'waiting' }  # 安全上限
    }
}

# 检查目录
if (-not (Test-Path -LiteralPath $signalDir -PathType Container)) {
    throw "Required directory not found: $signalDir"
}

# 单例锁文件
$singletonPath = Join-Path $stateDir 'watcher.lock'
$singleton = $null
try {
    $singleton = [System.IO.File]::Open(
        $singletonPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch [System.IO.IOException] {
    Write-Host "另一个监听器实例正在运行，退出。" -ForegroundColor Yellow
    exit 0
}

$watcher = $null
$subscriptions = @()
try {
    Write-ListenerLog "Doubao信号监听器启动，监控目录: $signalDir"
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Doubao 信号监听器已启动" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  监控目录: $signalDir" -ForegroundColor White
    Write-Host "  触发方式: FileSystemWatcher 事件驱动 (非轮询)" -ForegroundColor White
    Write-Host "  监控文件: to_doubao_*" -ForegroundColor White
    Write-Host "  日志目录: $logDir" -ForegroundColor White
    Write-Host "  启动时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
    Write-Host ""
    Write-Host "  说明:" -ForegroundColor Yellow
    Write-Host "  - 检测到 to_doubao_N 时弹出窗口提醒（无声音）" -ForegroundColor Gray
    Write-Host "  - 按 Ctrl+C 或关闭窗口停止监控" -ForegroundColor Gray
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    # 初始化：把已存在的旧信号标记为已提醒，避免启动时重复弹窗
    $initNum = 1
    while ($initNum -lt 10000) {
        $initPath = Join-Path $signalDir ("to_doubao_{0}" -f $initNum)
        if (Test-RegularZeroByteFile -Path $initPath) {
            $script:notifiedSignals["to_doubao_{0}" -f $initNum] = $true
            $initNum++
        } else {
            break
        }
    }
    if ($initNum -gt 1) {
        Write-ListenerLog "初始化：已标记 $($initNum - 1) 个旧信号为已提醒"
    }

    # 初始检查（检查是否有启动后新出现的信号）
    Invoke-SignalCheck | Out-Null

    # 创建 FileSystemWatcher
    $watcher = New-Object System.IO.FileSystemWatcher
    $watcher.Path = $signalDir
    $watcher.Filter = 'to_doubao_*'
    $watcher.IncludeSubdirectories = $false
    $watcher.NotifyFilter = [System.IO.NotifyFilters]'FileName, CreationTime, Size, LastWrite'
    $watcher.InternalBufferSize = 16384
    $watcher.EnableRaisingEvents = $true

    # 注册事件
    foreach ($eventName in @('Created', 'Changed', 'Renamed', 'Error')) {
        $subscriptions += Register-ObjectEvent -InputObject $watcher -EventName $eventName
    }

    Write-ListenerLog "FileSystemWatcher 已启动，等待 to_doubao_* 信号..."

    # 主循环：Wait-Event 阻塞等待事件（真正的事件驱动，不是轮询）
    while ($true) {
        $event = Wait-Event
        if ($null -ne $event) {
            Remove-Event -EventIdentifier $event.EventIdentifier -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 250
            Invoke-SignalCheck | Out-Null
        }
    }
}
catch {
    Write-ListenerLog "监听器错误停止: $($_.Exception.Message)" 'ERROR'
    throw
}
finally {
    foreach ($subscription in $subscriptions) {
        Unregister-Event -SubscriptionId $subscription.Id -ErrorAction SilentlyContinue
    }
    if ($null -ne $watcher) {
        $watcher.EnableRaisingEvents = $false
        $watcher.Dispose()
    }
    if ($null -ne $singleton) {
        $singleton.Dispose()
    }
    Write-ListenerLog '监听器已停止。'
}
