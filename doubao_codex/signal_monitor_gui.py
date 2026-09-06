"""
信号文件监控 GUI 应用（持久运行，不弹窗，不会被杀进程）

功能:
- 实时显示 to_doubao_N 和 to_codex_N 信号文件列表
- 显示监控日志和整改过程
- 检测到新信号时界面高亮提醒（红色闪烁，不弹窗）
- 显示当前状态（等待哪个信号、已完成几轮）
- 手动刷新、打开目录、复制信号名等操作

用法:
    python signal_monitor_gui.py
    (用 pythonw.exe 后台运行无控制台窗口)
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
import threading
import time

# 配置
SIGNAL_DIR = r"J:\phd\doubao_codex"
LOG_DIR = os.path.join(SIGNAL_DIR, "state", "logs")
POLL_INTERVAL = 2  # 秒（GUI轮询，轻量）


class SignalMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("信号文件监控 - Doubao/Codex 审查整改循环")
        self.root.geometry("1000x700")
        self.root.minsize(800, 500)

        # 状态
        self.notified_signals = set()  # 已提醒过的信号
        self.running = True
        self.flash_state = False
        self.flash_count = 0

        # 初始化已存在的信号为已提醒
        self._init_existing_signals()

        # 创建界面
        self._create_widgets()

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        # 启动UI刷新定时器
        self._schedule_refresh()

        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._log("GUI 监控应用已启动", "INFO")
        self._log(f"监控目录: {SIGNAL_DIR}", "INFO")
        self._log(f"轮询间隔: {POLL_INTERVAL}秒", "INFO")
        self._update_status()

    def _init_existing_signals(self):
        """初始化已存在的信号为已提醒，避免启动时重复提醒"""
        try:
            for f in os.listdir(SIGNAL_DIR):
                if f.startswith("to_doubao_") or f.startswith("to_codex_"):
                    self.notified_signals.add(f)
        except Exception:
            pass

    def _create_widgets(self):
        """创建界面组件"""
        # 顶部状态栏
        self.status_frame = ttk.Frame(self.root, padding="10")
        self.status_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(
            self.status_frame,
            text="初始化中...",
            font=("Microsoft YaHei", 12, "bold"),
            foreground="blue"
        )
        self.status_label.pack(side=tk.LEFT)

        # 按钮区
        btn_frame = ttk.Frame(self.status_frame)
        btn_frame.pack(side=tk.RIGHT)

        ttk.Button(btn_frame, text="刷新", command=self._manual_refresh).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="打开目录", command=self._open_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="清除日志", command=self._clear_log).pack(side=tk.LEFT, padx=2)

        # 主内容区（左右分栏）
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧：信号文件列表
        left_frame = ttk.LabelFrame(main_frame, text="信号文件列表", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # to_doubao 列表
        ttk.Label(left_frame, text="to_doubao_N (Codex审查信号)", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.doubao_listbox = tk.Listbox(left_frame, height=10, font=("Consolas", 10))
        self.doubao_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.doubao_listbox.bind("<Double-Button-1>", self._on_signal_double_click)

        # to_codex 列表
        ttk.Label(left_frame, text="to_codex_N (Doubao整改信号)", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.codex_listbox = tk.Listbox(left_frame, height=10, font=("Consolas", 10))
        self.codex_listbox.pack(fill=tk.BOTH, expand=True)
        self.codex_listbox.bind("<Double-Button-1>", self._on_signal_double_click)

        # 右侧：日志区
        right_frame = ttk.LabelFrame(main_frame, text="监控日志 / 整改过程", padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            right_frame,
            font=("Consolas", 9),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 日志标签颜色
        self.log_text.tag_configure("INFO", foreground="gray")
        self.log_text.tag_configure("SIGNAL", foreground="red", font=("Consolas", 9, "bold"))
        self.log_text.tag_configure("SUCCESS", foreground="green")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("WARN", foreground="orange")

        # 底部进度条（新信号提醒闪烁）
        self.alert_frame = ttk.Frame(self.root, padding="5")
        self.alert_frame.pack(fill=tk.X)
        self.alert_label = ttk.Label(
            self.alert_frame,
            text="",
            font=("Microsoft YaHei", 11, "bold"),
            anchor=tk.CENTER
        )
        self.alert_label.pack(fill=tk.X)

    def _monitor_loop(self):
        """后台监控线程：轮询信号文件目录"""
        while self.running:
            try:
                self._check_signals()
            except Exception as e:
                self._log(f"监控异常: {e}", "ERROR")
            time.sleep(POLL_INTERVAL)

    def _check_signals(self):
        """检查信号文件，发现新信号时提醒"""
        try:
            files = os.listdir(SIGNAL_DIR)
        except Exception:
            return

        for f in sorted(files):
            if f.startswith("to_doubao_") and f not in self.notified_signals:
                self.notified_signals.add(f)
                self._on_new_signal(f, "doubao")
            elif f.startswith("to_codex_") and f not in self.notified_signals:
                self.notified_signals.add(f)
                self._on_new_signal(f, "codex")

    def _on_new_signal(self, signal_name, signal_type):
        """新信号到达时的处理"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] [新信号] {signal_name} 已到达！"

        if signal_type == "doubao":
            round_num = signal_name.replace("to_doubao_", "")
            msg += f" 请进行第 {round_num} 轮整改"
            self._log(msg, "SIGNAL")
            self._flash_alert(f"!! 新审查信号 to_doubao_{round_num} 到达！请进行第 {round_num} 轮整改 !!")
        else:
            round_num = signal_name.replace("to_codex_", "")
            msg += f" 第 {round_num} 轮整改已提交"
            self._log(msg, "SUCCESS")
            self._flash_alert(f"第 {round_num} 轮整改信号 to_codex_{round_num} 已提交，等待 Codex 审查...")

        # 线程安全更新UI
        self.root.after(0, self._refresh_signal_lists)
        self.root.after(0, self._update_status)

    def _flash_alert(self, message):
        """闪烁提醒（界面高亮，不弹窗）"""
        self.flash_count = 10  # 闪烁10次
        self._do_flash(message)

    def _do_flash(self, message):
        """执行一次闪烁"""
        if self.flash_count <= 0:
            self.alert_label.config(text="", background=self.root.cget("background"))
            return

        self.flash_state = not self.flash_state
        if self.flash_state:
            self.alert_label.config(text=message, foreground="white", background="red")
        else:
            self.alert_label.config(text=message, foreground="red", background="yellow")

        self.flash_count -= 1
        self.root.after(300, lambda: self._do_flash(message))

    def _refresh_signal_lists(self):
        """刷新信号文件列表"""
        try:
            files = os.listdir(SIGNAL_DIR)
        except Exception:
            return

        # to_doubao
        self.doubao_listbox.delete(0, tk.END)
        doubao_files = sorted([f for f in files if f.startswith("to_doubao_")],
                               key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
        for f in doubao_files:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(SIGNAL_DIR, f)))
                self.doubao_listbox.insert(tk.END, f"  {f:20s}  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                self.doubao_listbox.insert(tk.END, f"  {f}")

        # to_codex
        self.codex_listbox.delete(0, tk.END)
        codex_files = sorted([f for f in files if f.startswith("to_codex_")],
                              key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
        for f in codex_files:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(SIGNAL_DIR, f)))
                self.codex_listbox.insert(tk.END, f"  {f:20s}  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                self.codex_listbox.insert(tk.END, f"  {f}")

    def _update_status(self):
        """更新顶部状态栏"""
        try:
            files = os.listdir(SIGNAL_DIR)
            doubao_nums = [int(f.split("_")[-1]) for f in files
                           if f.startswith("to_doubao_") and f.split("_")[-1].isdigit()]
            codex_nums = [int(f.split("_")[-1]) for f in files
                          if f.startswith("to_codex_") and f.split("_")[-1].isdigit()]

            max_doubao = max(doubao_nums) if doubao_nums else 0
            max_codex = max(codex_nums) if codex_nums else 0

            if max_codex > max_doubao:
                status = f"等待 Codex 审查 (to_codex_{max_codex} 已提交，to_doubao_{max_codex} 未到达)"
                color = "orange"
            elif max_doubao > max_codex:
                status = f"!! 等待 Doubao 整改 (to_doubao_{max_doubao} 已到达，to_codex_{max_doubao} 未提交) !!"
                color = "red"
            else:
                status = f"当前进度: 第 {max_doubao} 轮已完成，等待下一轮信号..."
                color = "green"

            self.status_label.config(text=status, foreground=color)
        except Exception as e:
            self.status_label.config(text=f"状态更新失败: {e}", foreground="red")

    def _schedule_refresh(self):
        """定时刷新UI"""
        if self.running:
            self._refresh_signal_lists()
            self._update_status()
            self.root.after(POLL_INTERVAL * 1000, self._schedule_refresh)

    def _log(self, message, level="INFO"):
        """添加日志（线程安全）"""
        self.root.after(0, lambda: self._append_log(message, level))

    def _append_log(self, message, level):
        """实际追加日志到文本框"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n", level)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _manual_refresh(self):
        """手动刷新"""
        self._refresh_signal_lists()
        self._update_status()
        self._log("手动刷新完成", "INFO")

    def _open_dir(self):
        """打开信号文件目录"""
        try:
            os.startfile(SIGNAL_DIR)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录: {e}")

    def _clear_log(self):
        """清除日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self._log("日志已清除", "INFO")

    def _on_signal_double_click(self, event):
        """双击信号文件时复制文件名到剪贴板"""
        listbox = event.widget
        selection = listbox.curselection()
        if selection:
            text = listbox.get(selection[0])
            # 提取文件名（去掉前面的空格和时间）
            filename = text.strip().split()[0]
            self.root.clipboard_clear()
            self.root.clipboard_append(filename)
            self._log(f"已复制到剪贴板: {filename}", "INFO")

    def _on_close(self):
        """窗口关闭时清理"""
        self.running = False
        self._log("GUI 监控应用正在关闭...", "WARN")
        time.sleep(0.5)
        self.root.destroy()


def main():
    root = tk.Tk()
    # 设置中文字体（使用tkinter字体元组格式）
    default_font = ("Microsoft YaHei", 9)
    root.option_add("*Font", default_font)
    app = SignalMonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
