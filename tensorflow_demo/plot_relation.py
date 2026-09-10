"""绘制 Transformer / TensorFlow / PyTorch 关系示意图"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

def draw_box(x, y, w, h, color, title, subtitle='', title_size=14, sub_size=10):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.25, title, ha='center', va='center',
            fontsize=title_size, fontweight='bold', color='white')
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.35, subtitle, ha='center', va='center',
                fontsize=sub_size, color='white')

def arrow(x1, y1, x2, y2, text='', color='black', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.15, text, ha='center', fontsize=10, color=color, fontweight='bold')

# ===== 标题 =====
ax.text(7, 8.5, 'Transformer / TensorFlow / PyTorch 三者关系',
        ha='center', fontsize=18, fontweight='bold')
ax.text(7, 8.0, '一个是"设计图纸"，两个是"施工工具"',
        ha='center', fontsize=12, color='#555555')

# ===== 第一层：Transformer（架构）=====
draw_box(4.5, 6.2, 5, 1.2, '#E74C3C', 'Transformer',
         '模型架构（设计图纸）· 2017年 Google 提出', title_size=16, sub_size=10)

# 标注 Transformer 内部
ax.text(7, 5.9, '自注意力 + FFN + 残差 + LayerNorm',
        ha='center', fontsize=10, color='#C0392B', style='italic')

# ===== 第二层：两个框架 =====
# TensorFlow
draw_box(0.8, 3.5, 5, 1.5, '#4A90D9', 'TensorFlow',
         '深度学习框架（施工工具）· 2015年 Google 开源', title_size=15, sub_size=9)

# PyTorch
draw_box(8.2, 3.5, 5, 1.5, '#EE6C2A', 'PyTorch',
         '深度学习框架（施工工具）· 2017年 Meta 开源', title_size=15, sub_size=9)

# 箭头：Transformer → 两个框架
arrow(5.8, 6.2, 3.3, 5.0, '用 TF 实现', color='#4A90D9')
arrow(8.2, 6.2, 10.7, 5.0, '用 PyTorch 实现', color='#EE6C2A')

# ===== 第三层：各自的 API 实现 =====
# TensorFlow 侧
tf_box_y = 1.2
draw_box(0.3, tf_box_y, 2.4, 1.5, '#7FB3E8', 'tf.keras.layers',
         'MultiHeadAttention\nDense / LayerNorm\nTransformerEncoder', title_size=10, sub_size=8)

draw_box(2.9, tf_box_y, 2.4, 1.5, '#7FB3E8', 'tf.keras',
         'Model / Sequential\nfit / compile\n自定义训练循环', title_size=10, sub_size=8)

# 箭头 TF → API
arrow(2.2, 3.5, 1.5, 2.7, color='#4A90D9', lw=1.5)
arrow(3.8, 3.5, 4.1, 2.7, color='#4A90D9', lw=1.5)

# PyTorch 侧
draw_box(8.0, tf_box_y, 2.4, 1.5, '#F5A96E', 'torch.nn',
         'MultiheadAttention\nLinear / LayerNorm\nTransformerEncoder', title_size=10, sub_size=8)

draw_box(10.6, tf_box_y, 2.4, 1.5, '#F5A96E', 'torch',
         'Module / Sequential\nbackward / step\nDataLoader', title_size=10, sub_size=8)

# 箭头 PyTorch → API
arrow(9.8, 3.5, 9.2, 2.7, color='#EE6C2A', lw=1.5)
arrow(11.4, 3.5, 11.8, 2.7, color='#EE6C2A', lw=1.5)

# ===== 底部说明 =====
ax.text(7, 0.5, '关键结论：Transformer 是"做什么"，TensorFlow/PyTorch 是"用什么做"。',
        ha='center', fontsize=12, fontweight='bold', color='#333333')
ax.text(7, 0.15, '同一个 Transformer 架构，可以用 TensorFlow 写，也可以用 PyTorch 写，两者独立、互不依赖。',
        ha='center', fontsize=10, color='#666666')

plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\tf_torch_transformer_relation.png', dpi=150, bbox_inches='tight')
print("已保存: J:\\phd\\tensorflow_demo\\tf_torch_transformer_relation.png")
