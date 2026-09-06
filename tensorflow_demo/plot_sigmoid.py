"""生成 sigmoid 函数图像"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

z = np.linspace(-6, 6, 500)
y = 1 / (1 + np.exp(-z))

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(z, y, 'b-', lw=3, label='sigmoid(z) = 1/(1+e^(-z))')

# 标注关键点
points = [(-1, 0.269, 'z=-1 → 0.269'),
          (0, 0.5, 'z=0 → 0.500'),
          (1, 0.731, 'z=1 → 0.731')]
for pz, py, label in points:
    ax.plot(pz, py, 'ro', ms=8)
    ax.annotate(label, xy=(pz, py), xytext=(pz+0.3, py-0.08),
                fontsize=11, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

# 渐近线
ax.axhline(y=1, color='gray', ls='--', alpha=0.5)
ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
ax.axvline(x=0, color='gray', ls=':', alpha=0.5)

ax.text(5.2, 0.95, 'y→1', fontsize=12, color='gray')
ax.text(5.2, 0.03, 'y→0', fontsize=12, color='gray')

ax.set_xlabel('z（输入，任意实数）', fontsize=13)
ax.set_ylabel('sigmoid(z)（输出，0~1之间）', fontsize=13)
ax.set_title('Sigmoid 函数（S 型激活函数）', fontsize=16, fontweight='bold')
ax.set_xlim(-6, 6)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=12, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\sigmoid_curve.png', dpi=150, bbox_inches='tight')
print("已保存: J:\\phd\\tensorflow_demo\\sigmoid_curve.png")
