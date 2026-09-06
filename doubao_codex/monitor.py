"""
信号文件事件驱动监控（watchdog + Windows ReadDirectoryChangesW）
用法: pythonw monitor.py  (后台运行，无窗口)
  或: python monitor.py    (前台运行，有输出)

真正的事件驱动：只有目录下出现新文件时才触发，不轮询。
"""
import os
import sys
import time
import winsound
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_PATH = r"J:\phd\doubao_codex"
LOG_FILE = os.path.join(WATCH_PATH, "monitor_event.log")

def log(msg):
    """写入日志并打印"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"写入日志失败: {e}", flush=True)

def alert_signal(filename):
    """检测到信号时的提醒"""
    log(f"[信号] 检测到新文件: {filename}")
    # 播放提示音（两声）
    try:
        winsound.Beep(1000, 500)
        time.sleep(0.2)
        winsound.Beep(1200, 500)
    except Exception:
        pass

class SignalHandler(FileSystemEventHandler):
    """文件系统事件处理器"""

    def __init__(self):
        super().__init__()
        # 记录已知的信号文件（避免重复触发）
        self.known = set()
        # 初始化时记录已存在的信号文件
        try:
            for f in os.listdir(WATCH_PATH):
                if f.startswith("to_doubao_"):
                    self.known.add(f)
                    log(f"[已知] {f}")
        except Exception as e:
            log(f"初始化失败: {e}")

    def on_created(self, event):
        """文件创建事件（真正的事件驱动，不是轮询）"""
        if event.is_directory:
            return
        filename = os.path.basename(event.src_path)
        if filename.startswith("to_doubao_") and filename not in self.known:
            self.known.add(filename)
            alert_signal(filename)

    def on_moved(self, event):
        """文件重命名事件（有些编辑器保存时先创建临时文件再重命名）"""
        if event.is_directory:
            return
        filename = os.path.basename(event.dest_path)
        if filename.startswith("to_doubao_") and filename not in self.known:
            self.known.add(filename)
            alert_signal(filename)


def main():
    if not os.path.exists(WATCH_PATH):
        print(f"[错误] 目录不存在: {WATCH_PATH}", flush=True)
        sys.exit(1)

    log("=" * 60)
    log("事件驱动信号监控已启动 (watchdog + ReadDirectoryChangesW)")
    log(f"监控目录: {WATCH_PATH}")
    log(f"触发方式: 文件创建/重命名事件（非轮询）")
    log(f"日志文件: {LOG_FILE}")
    log("=" * 60)
    log("等待新信号文件...")

    # 创建事件处理器和观察者
    handler = SignalHandler()
    observer = Observer()
    observer.schedule(handler, WATCH_PATH, recursive=False)
    observer.start()

    heartbeat_count = 0
    try:
        # 主循环只是保持进程存活，等待事件通知（不是轮询文件系统）
        while True:
            time.sleep(30)
            heartbeat_count += 1
            # 每30秒心跳，确认进程还在运行
            if observer.is_alive():
                log(f"[心跳] 监控运行中，已知信号 {len(handler.known)} 个")
            else:
                log("[错误] Observer线程已停止，尝试重启...")
                observer.stop()
                observer = Observer()
                observer.schedule(handler, WATCH_PATH, recursive=False)
                observer.start()
                log("[恢复] Observer已重启")
    except KeyboardInterrupt:
        log("[停止] 收到中断信号，正在停止...")
        observer.stop()
    except Exception as e:
        log(f"[错误] 监控异常: {e}")
        import traceback
        log(traceback.format_exc())
        observer.stop()

    observer.join()
    log("[停止] 监控已终止")


if __name__ == "__main__":
    main()
