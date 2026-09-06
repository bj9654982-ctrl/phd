@echo off
chcp 65001 >nul
title 信号文件事件监控 (FileSystemWatcher)
echo ============================================================
echo   启动事件驱动信号监控...
echo   监控目录: J:\phd\doubao_codex
echo ============================================================
echo.
powershell -ExecutionPolicy Bypass -NoExit -File "%~dp0monitor_event.ps1"
