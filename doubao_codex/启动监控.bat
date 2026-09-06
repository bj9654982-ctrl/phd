@echo off
chcp 65001 >nul
title === 信号监控自动重启器 - 请勿关闭 ===
cd /d "%~dp0"
echo ============================================================
echo   信号监控自动重启器已启动
echo   监控脚本: monitor.py
echo   如果监控进程崩溃或被关闭，会自动重启
echo   关闭此窗口 = 停止监控
echo ============================================================
echo.

:loop
echo [%time%] 启动监控进程...
"C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe" monitor.py
echo [%time%] 监控进程已退出，3秒后自动重启...
timeout /t 3 /nobreak >nul
goto loop
