"""绘制 U-Net 结构示意图"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色定义
enc_color = '#4A90D9'    # 编码器 - 蓝色
dec_color = '#50B86A'    # 解码器 - 绿色
bot_color = '#E8A838'    # 瓶颈 - 橙色
skip_color = '#999999'   # 跳跃连接 - 灰色

def draw_block(x, y, w, h, color, label, sublabel=''):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center',
                fontsize=8, color='white')

def arrow(x1, y1, x2, y2, color='black', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ===== 标题 =====
ax.text(8, 9.5, 'U-Net 网络结构示意图', ha='center', fontsize=18, fontweight='bold')
ax.text(8, 9.0, '编码器（下采样） → 瓶颈 → 解码器（上采样），灰色虚线为跳跃连接',
        ha='center', fontsize=11, color='#555555')

# ===== 输入 =====
draw_block(0.3, 4.2, 1.2, 1.6, '#333333', '输入', '28×28×1')
arrow(1.5, 5.0, 2.2, 5.0)

# ===== 编码器（左侧，从上到下）=====
# enc1: 28×28×32
draw_block(2.2, 4.0, 1.8, 2.0, enc_color, 'Conv 32', '28×28')
# enc2: 14×14×64
draw_block(4.2, 3.0, 1.8, 2.0, enc_color, 'Conv 64', '14×14')
# enc3: 7×7×128
draw_block(6.2, 2.0, 1.8, 2.0, enc_color, 'Conv 128', '7×7')

# 编码器之间的下采样箭头（向下）
arrow(3.1, 4.0, 3.1, 3.5, color=enc_color, style='->', lw=2)
ax.text(3.3, 3.7, '池化\n↓2', fontsize=8, color=enc_color)
arrow(5.1, 3.0, 5.1, 2.5, color=enc_color, style='->', lw=2)
ax.text(5.3, 2.7, '池化\n↓2', fontsize=8, color=enc_color)

# 编码器到瓶颈
arrow(7.1, 2.0, 7.6, 1.5, color=enc_color, style='->', lw=2)

# ===== 瓶颈（最底部）=====
draw_block(7.6, 0.5, 1.8, 2.0, bot_color, 'Conv 256', '7×7 瓶颈')
ax.text(8.5, 0.2, '（特征最压缩的地方）', ha='center', fontsize=8, color='#888888')

# 瓶颈到解码器
arrow(9.4, 1.5, 9.9, 2.0, color=dec_color, style='->', lw=2)

# ===== 解码器（右侧，从下到上）=====
# dec3: 7×7×128
draw_block(9.9, 2.0, 1.8, 2.0, dec_color, 'Conv 128', '7×7')
# dec2: 14×14×64
draw_block(11.9, 3.0, 1.8, 2.0, dec_color, 'Conv 64', '14×14')
# dec1: 28×28×32
draw_block(13.9, 4.0, 1.8, 2.0, dec_color, 'Conv 32', '28×28')

# 解码器之间的上采样箭头（向上）
arrow(10.8, 4.0, 10.8, 4.5, color=dec_color, style='->', lw=2)
ax.text(11.0, 4.2, '上采样\n↑2', fontsize=8, color=dec_color)
arrow(12.8, 5.0, 12.8, 5.5, color=dec_color, style='->', lw=2)
ax.text(13.0, 5.2, '上采样\n↑2', fontsize=8, color=dec_color)

# ===== 跳跃连接（灰色虚线，横向）=====
# enc1 -> dec1 (28×28)
ax.annotate('', xy=(13.9, 5.0), xytext=(4.0, 5.0),
            arrowprops=dict(arrowstyle='->', color=skip_color, lw=2, linestyle='dashed'))
ax.text(8.9, 5.3, '跳跃连接：拼接 32+32=64 通道', ha='center', fontsize=9, color=skip_color, fontweight='bold')

# enc2 -> dec2 (14×14)
ax.annotate('', xy=(11.9, 4.0), xytext=(6.0, 4.0),
            arrowprops=dict(arrowstyle='->', color=skip_color, lw=2, linestyle='dashed'))
ax.text(8.9, 4.2, '跳跃连接：拼接 64+64=128 通道', ha='center', fontsize=9, color=skip_color)

# ===== 输出 =====
arrow(15.7, 5.0, 16.2, 5.0)
draw_block(16.2, 4.2, 1.2, 1.6, '#333333', '输出', '28×28×1')
ax.text(16.8, 3.8, '去噪后图片', ha='center', fontsize=9, color='#333333')

# ===== 底部说明 =====
ax.text(0.3, 0.3, '关键设计：', fontsize=11, fontweight='bold', color='#333333')
ax.text(0.3, -0.1, '① 编码器逐步缩小图片、增加通道数（提取高级特征）', fontsize=9, color='#555555')
ax.text(0.3, -0.5, '② 解码器逐步放大图片、减少通道数（恢复空间细节）', fontsize=9, color='#555555')
ax.text(0.3, -0.9, '③ 跳跃连接把编码器的原始细节直接传给解码器，防止信息丢失', fontsize=9, color='#555555')
ax.text(0.3, -1.3, '④ 整体形状像字母 "U"，所以叫 U-Net', fontsize=9, color='#555555')

plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\unet_architecture.png', dpi=150, bbox_inches='tight')
print("已保存: J:\\phd\\tensorflow_demo\\unet_architecture.png")
