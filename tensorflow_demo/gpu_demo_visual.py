"""
GPU Demo 可视化版
运行后生成图表，展示 GPU 计算结果和速度对比
"""
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)

# ============================================================
# 1. 小矩阵乘法 + 结果热力图
# ============================================================
x = torch.randn(3, 3, device=device)
y = torch.randn(3, 3, device=device)
z = x @ y

x_np = x.cpu().numpy()
y_np = y.cpu().numpy()
z_np = z.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, mat, title in zip(axes, [x_np, y_np, z_np], ['矩阵 x (GPU)', '矩阵 y (GPU)', 'x @ y 结果 (GPU)']):
    im = ax.imshow(mat, cmap='coolwarm', aspect='auto')
    ax.set_title(title, fontsize=13)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, '%.2f' % mat[i, j], ha='center', va='center',
                    color='white' if abs(mat[i, j]) > 0.5 else 'black', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle('GPU 矩阵乘法演示 (%s)' % gpu_name, fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\gpu_matrix_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("图表1已保存: gpu_matrix_result.png")

# ============================================================
# 2. CPU vs GPU 速度对比（多种矩阵尺寸）
# ============================================================
sizes = [500, 1000, 2000, 3000, 4000, 5000]
cpu_times = []
gpu_times = []
speedups = []

print("正在 benchmark...")
for size in sizes:
    # CPU
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    start = time.time()
    _ = a_cpu @ b_cpu
    cpu_t = time.time() - start
    cpu_times.append(cpu_t)

    # GPU (warmup + 计时)
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    _ = a_gpu @ b_gpu  # warmup
    torch.cuda.synchronize()
    start = time.time()
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_t = time.time() - start
    gpu_times.append(gpu_t)
    speedups.append(cpu_t / gpu_t)
    print("  %dx%d: CPU=%.3fs, GPU=%.3fs, 加速=%.1fx" % (size, size, cpu_t, gpu_t, cpu_t/gpu_t))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：耗时对比
x_pos = np.arange(len(sizes))
width = 0.35
axes[0].bar(x_pos - width/2, cpu_times, width, label='CPU', color='#ff7f0e', alpha=0.85)
axes[0].bar(x_pos + width/2, gpu_times, width, label='GPU', color='#1f77b4', alpha=0.85)
axes[0].set_xlabel('矩阵尺寸', fontsize=12)
axes[0].set_ylabel('耗时 (秒)', fontsize=12)
axes[0].set_title('CPU vs GPU 矩阵乘法耗时', fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(['%d' % s for s in sizes])
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# 右图：加速比曲线
axes[1].plot(sizes, speedups, 'o-', color='#2ca02c', linewidth=2, markersize=8)
axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='GPU=CPU (无加速)')
axes[1].set_xlabel('矩阵尺寸', fontsize=12)
axes[1].set_ylabel('加速比 (CPU时间/GPU时间)', fontsize=12)
axes[1].set_title('GPU 加速比随矩阵尺寸变化', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)
for i, (s, sp) in enumerate(zip(sizes, speedups)):
    axes[1].annotate('%.1fx' % sp, (s, sp), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10)

plt.suptitle('%s 性能测试' % gpu_name, fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\gpu_speed_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("图表2已保存: gpu_speed_comparison.png")

# ============================================================
# 3. GPU 信息面板
# ============================================================
mem_allocated = torch.cuda.memory_allocated() / 1024**2
mem_reserved = torch.cuda.memory_reserved() / 1024**2
mem_total = 12288  # 12GB

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

info_text = """
GPU 信息面板
═══════════════════════════════════════
设备名称:    %s
CUDA 版本:   %s
PyTorch 版本: %s
═══════════════════════════════════════
显存总量:    12288 MB (12 GB)
已分配显存:  %.1f MB
已预留显存:  %.1f MB
可用显存:    %.1f MB
═══════════════════════════════════════
最大加速比:  %.1f 倍 (5000x5000矩阵)
GPU 耗时:    %.3f 秒
CPU 耗时:    %.3f 秒
═══════════════════════════════════════
""" % (gpu_name, torch.version.cuda, torch.__version__,
       mem_allocated, mem_reserved, mem_total - mem_reserved,
       speedups[-1], gpu_times[-1], cpu_times[-1])

ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

# 显存占用条
ax2 = fig.add_axes([0.55, 0.3, 0.4, 0.15])
used_pct = mem_reserved / mem_total * 100
ax2.barh([0], [used_pct], color='#1f77b4', height=0.5)
ax2.barh([0], [100-used_pct], left=[used_pct], color='#e0e0e0', height=0.5)
ax2.set_xlim(0, 100)
ax2.set_yticks([])
ax2.set_xlabel('显存占用百分比 (%)', fontsize=11)
ax2.set_title('显存占用: %.1f MB / 12288 MB (%.1f%%)' % (mem_reserved, used_pct), fontsize=12)
for spine in ax2.spines.values():
    spine.set_visible(False)

plt.savefig(r'J:\phd\tensorflow_demo\gpu_info_panel.png', dpi=150, bbox_inches='tight')
plt.close()
print("图表3已保存: gpu_info_panel.png")

print("\n全部图表生成完成！")
