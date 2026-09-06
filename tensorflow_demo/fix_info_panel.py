"""重新生成 GPU 信息面板（修复中文字体）"""
import torch
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

gpu_name = torch.cuda.get_device_name(0)
mem_allocated = torch.cuda.memory_allocated() / 1024**2
mem_reserved = torch.cuda.memory_reserved() / 1024**2
mem_total = 12288

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axis('off')

# 用表格形式展示信息
info = [
    ["GPU 名称", gpu_name],
    ["CUDA 版本", torch.version.cuda],
    ["PyTorch 版本", torch.__version__],
    ["显存总量", "12288 MB (12 GB)"],
    ["已分配显存", "%.1f MB" % mem_allocated],
    ["已预留显存", "%.1f MB" % mem_reserved],
    ["可用显存", "%.1f MB" % (mem_total - mem_reserved)],
    ["最大加速比", "8.8 倍 (4000x4000)"],
    ["GPU 5000x5000 耗时", "0.059 秒"],
    ["CPU 5000x5000 耗时", "0.489 秒"],
]

# 绘制标题
ax.text(0.5, 0.95, "GPU 信息面板", transform=ax.transAxes,
        fontsize=18, fontweight='bold', ha='center', va='top')

# 绘制分隔线
ax.plot([0.05, 0.95], [0.88, 0.88], transform=ax.transAxes, color='gray', linewidth=1)

# 绘制信息行
y_start = 0.82
row_height = 0.075
for i, (key, val) in enumerate(info):
    y = y_start - i * row_height
    ax.text(0.08, y, key, transform=ax.transAxes, fontsize=12, va='center', color='#555555')
    ax.text(0.45, y, val, transform=ax.transAxes, fontsize=12, va='center', fontweight='bold', color='#1a1a1a')

# 显存占用条
ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.06])
used_pct = mem_reserved / mem_total * 100
ax2.barh([0], [used_pct], color='#1f77b4', height=0.6)
ax2.barh([0], [100 - used_pct], left=[used_pct], color='#e0e0e0', height=0.6)
ax2.set_xlim(0, 100)
ax2.set_yticks([])
ax2.set_xlabel('显存占用百分比 (%)', fontsize=11)
ax2.set_title('显存占用: %.1f MB / 12288 MB (%.1f%%)' % (mem_reserved, used_pct), fontsize=12)
for spine in ax2.spines.values():
    spine.set_visible(False)

plt.savefig(r'J:\phd\tensorflow_demo\gpu_info_panel.png', dpi=150, bbox_inches='tight')
plt.close()
print("信息面板已重新生成")
