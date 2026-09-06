import os
import time
import subprocess
import sys

# 配置
TEST_SCRIPT = r"J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\TMT\code\test_TMT_static.py"
DATA_PATH = r"J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\turb_syn_static\syn_static\test_100"
RESULT_PATH = r"J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\test_results_static"
MODEL_PATH = r"J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\TMT-20260807T140132Z-1-001\TMT\TMT_static.pth"
PYTHON_PATH = r"C:\Users\Administrator\miniconda3\envs\tmt\python.exe"
WORK_DIR = r"J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\TMT\code"

CHECK_INTERVAL = 60  # 检查间隔（秒）

def is_running():
    """检查GPU上有没有python进程在跑"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return 'python.exe' in result.stdout
    except:
        return False

def get_progress():
    """获取当前进度"""
    try:
        result_dir = RESULT_PATH
        dirs = [d for d in os.listdir(result_dir) if d.isdigit() and os.path.isdir(os.path.join(result_dir, d))]
        return len(dirs)
    except:
        return 0

def start_test():
    """启动测试"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 启动测试...")
    cmd = [
        PYTHON_PATH, TEST_SCRIPT,
        '--data_path', DATA_PATH,
        '--result_path', RESULT_PATH,
        '--model_path', MODEL_PATH,
        '--total_frames', '48',
        '--temp_window', '12'
    ]
    subprocess.Popen(cmd, cwd=WORK_DIR)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试已启动")

def main():
    print("=" * 60)
    print("TMT静态测试自动监控重启脚本")
    print("=" * 60)
    print(f"检查间隔: {CHECK_INTERVAL}秒")
    print(f"结果目录: {RESULT_PATH}")
    print("=" * 60)
    
    last_progress = get_progress()
    print(f"初始进度: {last_progress} / 100 个序列")
    
    # 先检查是不是已经在跑了
    if is_running():
        print("检测到测试已经在运行中")
    else:
        print("检测到测试未运行，启动中...")
        start_test()
        time.sleep(30)  # 等启动
    
    while True:
        time.sleep(CHECK_INTERVAL)
        
        current_progress = get_progress()
        running = is_running()
        
        if current_progress != last_progress:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 进度: {current_progress} / 100")
            last_progress = current_progress
        
        if current_progress >= 100:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 全部完成！共100个序列")
            break
        
        if not running:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测到测试停止，自动重启中...")
            start_test()
            time.sleep(30)  # 等启动

if __name__ == '__main__':
    main()
