"""
GPU 最简 Demo：在显卡上跑矩阵乘法
"""
import torch
import time

print("=" * 50)
print("1. 检测 GPU")
print("=" * 50)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 数量:", torch.cuda.device_count())
print("GPU 名称:", torch.cuda.get_device_name(0))
print("当前设备:", torch.cuda.current_device())

print()
print("=" * 50)
print("2. 在 GPU 上创建张量")
print("=" * 50)
device = torch.device("cuda")
x = torch.randn(3, 3, device=device)
y = torch.randn(3, 3, device=device)
print("x 所在设备:", x.device)
print("y 所在设备:", y.device)
print()
print("x =")
print(x)
print()
print("y =")
print(y)

print()
print("=" * 50)
print("3. GPU 上做矩阵乘法 x @ y")
print("=" * 50)
z = x @ y
print("结果 z =")
print(z)
print("z 所在设备:", z.device)

print()
print("=" * 50)
print("4. GPU 显存使用")
print("=" * 50)
print("已分配显存: %.2f MB" % (torch.cuda.memory_allocated() / 1024**2))
print("已预留显存: %.2f MB" % (torch.cuda.memory_reserved() / 1024**2))

print()
print("=" * 50)
print("5. 速度对比：CPU vs GPU（大矩阵乘法）")
print("=" * 50)
size = 5000

# CPU
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print("CPU  %dx%d 矩阵乘法: %.3f 秒" % (size, size, cpu_time))

# GPU
a_gpu = torch.randn(size, size, device="cuda")
b_gpu = torch.randn(size, size, device="cuda")
torch.cuda.synchronize()  # 等待 GPU 完成
start = time.time()
c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()  # 等待 GPU 完成
gpu_time = time.time() - start
print("GPU  %dx%d 矩阵乘法: %.3f 秒" % (size, size, gpu_time))
print("加速比: %.1f 倍" % (cpu_time / gpu_time))

print()
print("Demo 完成！你的显卡正在工作。")
