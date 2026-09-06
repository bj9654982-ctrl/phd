"""
第1课：PyTorch 张量（Tensor）基础
====================================
张量就是 PyTorch 里的"多维数组"，类似 NumPy 的 ndarray，
但张量可以放在 GPU 上加速计算，并且支持自动求导。

运行方式：
    conda activate torch-gpu
    python 01_tensor_basics.py
"""

import torch
import numpy as np

print("=" * 60)
print("1. 创建张量")
print("=" * 60)

# 从 Python 列表创建
a = torch.tensor([1, 2, 3])
print(f"从列表创建: {a}, 形状: {a.shape}, 类型: {a.dtype}")

# 创建全 0 / 全 1 / 随机张量
b = torch.zeros(2, 3)       # 2行3列全0
c = torch.ones(2, 3)        # 2行3列全1
d = torch.rand(2, 3)        # 2行3列，[0,1) 均匀分布
e = torch.randn(2, 3)       # 2行3列，标准正态分布
print(f"\nzeros:\n{b}")
print(f"\nrand:\n{d}")

# 指定数据类型和设备
f = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"\n指定 float32: {f}, dtype={f.dtype}")

print("\n" + "=" * 60)
print("2. 张量的属性")
print("=" * 60)

x = torch.randn(3, 4, 5)
print(f"形状 shape: {x.shape}")
print(f"维度 ndim: {x.ndim}")       # 3维
print(f"元素个数 numel: {x.numel()}")  # 3*4*5=60
print(f"数据类型 dtype: {x.dtype}")
print(f"设备 device: {x.device}")    # cpu 或 cuda

print("\n" + "=" * 60)
print("3. 基本运算（逐元素）")
print("=" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"加法: {a + b}")
print(f"减法: {a - b}")
print(f"乘法(逐元素): {a * b}")
print(f"除法: {a / b}")
print(f"幂: {a ** 2}")

# 常用数学函数
print(f"\nsin: {torch.sin(a)}")
print(f"exp: {torch.exp(a)}")
print(f"mean: {a.mean()}")
print(f"sum: {a.sum()}")
print(f"max: {a.max()}, argmax: {a.argmax()}")

print("\n" + "=" * 60)
print("4. 矩阵乘法（深度学习最常用）")
print("=" * 60)

A = torch.randn(2, 3)   # 2行3列
B = torch.randn(3, 4)   # 3行4列
C = A @ B                # 结果 2行4列，等价于 torch.matmul(A, B)
print(f"A({A.shape}) @ B({B.shape}) = C({C.shape})")
print(C)

# 向量点积
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
print(f"\n点积 v1·v2 = {torch.dot(v1, v2)}")  # 1*4+2*5+3*6=32

print("\n" + "=" * 60)
print("5. 形状变换（reshape / view / transpose）")
print("=" * 60)

x = torch.arange(12)  # [0,1,2,...,11]
print(f"原始: {x}, shape={x.shape}")

x2 = x.reshape(3, 4)   # 变成 3行4列
print(f"\nreshape(3,4):\n{x2}")

x3 = x.reshape(2, -1)  # -1 表示自动推算，这里变成 2行6列
print(f"\nreshape(2,-1): shape={x3.shape}")

# 转置
print(f"\n转置 x2.T:\n{x2.T}")  # 4行3列

# 增加/减少维度
x4 = torch.randn(3)
print(f"\nunsqueeze(0): {x4.unsqueeze(0).shape}")  # [1, 3]
print(f"unsqueeze(1): {x4.unsqueeze(1).shape}")  # [3, 1]

print("\n" + "=" * 60)
print("6. 索引与切片（和 NumPy 一样）")
print("=" * 60)

x = torch.arange(12).reshape(3, 4)
print(f"原始:\n{x}")
print(f"\n第0行: {x[0]}")
print(f"第1列: {x[:, 1]}")
print(f"第0-1行, 第2-3列:\n{x[0:2, 2:4]}")

# 布尔索引
mask = x > 5
print(f"\n大于5的元素: {x[mask]}")

print("\n" + "=" * 60)
print("7. GPU 加速（核心优势）")
print("=" * 60)

print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    import time

    # ---- 关键：GPU 预热 ----
    # 第一次调用 CUDA 时有上下文初始化、kernel 编译等开销，必须先跑几次
    print("\n[预热] 初始化 CUDA 上下文...")
    warmup = torch.randn(1000, 1000, device=device)
    for _ in range(10):
        _ = warmup @ warmup
    torch.cuda.synchronize()
    print("预热完成")

    # ---- 不同尺寸矩阵对比 ----
    # GPU 有几千个核心，小矩阵无法充分并行；大矩阵才能体现 GPU 优势
    sizes = [1000, 4000, 8000]
    n_runs = 50  # 每个尺寸跑50次取平均

    print(f"\n{'矩阵尺寸':<12} {'CPU耗时':<12} {'GPU耗时':<12} {'加速比':<10} {'说明'}")
    print("-" * 70)

    for size in sizes:
        x_cpu = torch.randn(size, size)
        x_gpu = x_cpu.to(device)

        # CPU 计时
        start = time.time()
        for _ in range(n_runs):
            _ = x_cpu @ x_cpu
        cpu_time = (time.time() - start) / n_runs

        # GPU 计时（数据已在 GPU 上，只算纯计算时间）
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_runs):
            _ = x_gpu @ x_gpu
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / n_runs

        speedup = cpu_time / gpu_time
        note = "GPU太慢！传输+启动开销 > 计算时间" if speedup < 1 else \
               "GPU开始领先" if speedup < 5 else "GPU大幅领先！"

        print(f"{size}x{size:<8} {cpu_time*1000:>8.1f}ms   {gpu_time*1000:>8.1f}ms   "
              f"{speedup:>6.1f}x    {note}")

    # ---- 数据传输开销演示 ----
    print(f"\n[数据传输开销] 8000x8000 矩阵 CPU->GPU 传输时间:")
    x_big = torch.randn(8000, 8000)
    torch.cuda.synchronize()
    start = time.time()
    x_big_gpu = x_big.to(device)
    torch.cuda.synchronize()
    transfer_time = time.time() - start
    print(f"  传输耗时: {transfer_time*1000:.1f}ms（矩阵大小: {8000*8000*4/1024/1024:.1f} MB）")

    print(f"""
[为什么小矩阵 GPU 比 CPU 慢？三个原因]

1. 启动开销（Kernel Launch）
   GPU 每次执行计算都要启动一个 kernel，有固定开销（~10-50μs）。
   小矩阵计算本身只要几微秒，启动开销占了大头。

2. 数据传输开销（PCIe 传输）
   数据要从 CPU 内存通过 PCIe 总线传到 GPU 显存。
   1000x1000 float32 = 4MB，传输就要几毫秒，比计算还慢。
   深度学习中数据一直在 GPU 上，所以没有这个问题。

3. GPU 利用率不足
   RTX 3060 有 3584 个 CUDA 核心。1000x1000 矩阵乘法
   只能用到一小部分核心，大部分核心在闲置。
   矩阵越大，并行度越高，GPU 优势越明显。

[结论]
   - 小矩阵（< 2000x2000）：CPU 更快
   - 中等矩阵（2000-5000）：GPU 开始领先
   - 大矩阵（> 5000）：GPU 大幅领先（10x+）
   - 深度学习的 batch 训练本质是大矩阵运算，所以 GPU 有巨大优势
""")

print("\n" + "=" * 60)
print("8. 与 NumPy 互转")
print("=" * 60)

# Tensor -> NumPy
t = torch.tensor([1.0, 2.0, 3.0])
n = t.numpy()
print(f"Tensor: {t} -> NumPy: {n}")

# NumPy -> Tensor
n2 = np.array([4.0, 5.0, 6.0])
t2 = torch.from_numpy(n2)
print(f"NumPy: {n2} -> Tensor: {t2}")

print("\n" + "=" * 60)
print("第1课完成！你已经掌握了张量的核心操作。")
print("下一课：自动求导机制（autograd）—— 深度学习的基石")
print("=" * 60)
