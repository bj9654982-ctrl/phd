"""
GPU 功耗采样脚本
每 2 秒读取一次 nvidia-smi，持续指定时长，最后统计平均/最大/最小功耗。
"""
import subprocess
import time
import re
import sys

DURATION = 180  # 采样时长（秒）
INTERVAL = 2    # 采样间隔（秒）
NVIDIA_SMI = r"C:\Windows\System32\nvidia-smi.exe"

def read_gpu():
    """读取一次 GPU 状态，返回字典"""
    result = subprocess.run(
        [NVIDIA_SMI,
         "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    line = result.stdout.strip()
    parts = [p.strip() for p in line.split(",")]
    return {
        "power_w": float(parts[0]),
        "temp_c": float(parts[1]),
        "util_pct": float(parts[2]),
        "mem_used_mib": float(parts[3]),
        "mem_total_mib": float(parts[4]),
    }

def main():
    print(f"开始采样：时长 {DURATION}s，间隔 {INTERVAL}s，共约 {DURATION//INTERVAL} 个样本")
    print("=" * 70)
    print(f"{'时间':<10} {'功耗(W)':<10} {'温度(°C)':<10} {'利用率(%)':<10} {'显存(MiB)':<12}")
    print("-" * 70)

    samples = []
    start = time.time()
    count = 0

    try:
        while time.time() - start < DURATION:
            try:
                gpu = read_gpu()
                samples.append(gpu)
                count += 1
                elapsed = int(time.time() - start)
                print(f"{elapsed:>5}s    {gpu['power_w']:<10.2f} {gpu['temp_c']:<10.0f} "
                      f"{gpu['util_pct']:<10.0f} {gpu['mem_used_mib']:<12.0f}",
                      end="\r", flush=True)
            except Exception as e:
                print(f"读取失败: {e}", file=sys.stderr)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n\n用户中断，使用已采集数据统计...")

    print("\n" + "=" * 70)

    if not samples:
        print("没有采集到数据")
        return

    powers = [s["power_w"] for s in samples]
    temps = [s["temp_c"] for s in samples]
    utils = [s["util_pct"] for s in samples]
    mems = [s["mem_used_mib"] for s in samples]

    avg_power = sum(powers) / len(powers)
    max_power = max(powers)
    min_power = min(powers)
    avg_temp = sum(temps) / len(temps)
    avg_util = sum(utils) / len(utils)
    avg_mem = sum(mems) / len(mems)

    # 计算耗电量
    duration_hours = DURATION / 3600
    energy_wh = avg_power * duration_hours
    energy_kwh = energy_wh / 1000

    print(f"采样总数: {len(samples)}")
    print(f"采样时长: {DURATION}s ({duration_hours:.3f}h)")
    print()
    print(f"{'指标':<15} {'最小':<12} {'平均':<12} {'最大':<12}")
    print("-" * 55)
    print(f"{'功耗 (W)':<15} {min_power:<12.2f} {avg_power:<12.2f} {max_power:<12.2f}")
    print(f"{'温度 (°C)':<15} {min(temps):<12.0f} {avg_temp:<12.1f} {max(temps):<12.0f}")
    print(f"{'利用率 (%)':<15} {min(utils):<12.0f} {avg_util:<12.1f} {max(utils):<12.0f}")
    print(f"{'显存 (MiB)':<15} {min(mems):<12.0f} {avg_mem:<12.0f} {max(mems):<12.0f}")
    print()
    print("=" * 70)
    print("耗电估算：")
    print(f"  平均功耗: {avg_power:.2f} W")
    print(f"  采样期间耗电: {energy_wh:.2f} Wh = {energy_kwh:.5f} 度")
    print(f"   extrapolate 1小时: {avg_power * 1 / 1000:.4f} 度")
    print(f"  extrapolate 24小时: {avg_power * 24 / 1000:.3f} 度")
    print(f"  按0.55元/度算，1小时电费: {avg_power / 1000 * 0.55:.4f} 元")
    print(f"  按0.55元/度算，24小时电费: {avg_power * 24 / 1000 * 0.55:.3f} 元")
    print("=" * 70)

if __name__ == "__main__":
    main()
