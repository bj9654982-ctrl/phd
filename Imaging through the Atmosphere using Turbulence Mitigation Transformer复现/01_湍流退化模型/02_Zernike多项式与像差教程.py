"""
知识点2：Zernike多项式与像差基础
================================

【为什么要学这个？】
TMT论文说"湍流退化 = 倾斜 + 模糊"，这个分解的数学基础就是Zernike多项式。
理解Zernike多项式，你才能真正理解为什么TMT的两阶段设计是合理的。

【Zernike多项式是什么？】
Zernike多项式是定义在单位圆上的一组正交多项式，是光学中描述波前像差的标准工具。
- 正交性：不同模式之间互不相关，可以独立分析
- 完备性：任何波前都可以用Zernike多项式的线性组合表示
- 物理意义：每一阶都对应一种具体的像差类型

【本教程你将学会】
1. Zernike多项式的基本概念和性质
2. 前10阶Zernike模式分别对应什么像差
3. 为什么前两阶（活塞、倾斜）是"几何变形"
4. 为什么高阶像差会导致"模糊"
5. 这和TMT两阶段设计的关系
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import factorial
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\01_湍流退化模型'


def zernike_radial(n, m, rho):
    """
    计算Zernike多项式的径向部分
    
    Zernike多项式的一般形式：
        Z_n^m(ρ, θ) = R_n^m(ρ) * cos(mθ)   (m ≥ 0)
        Z_n^m(ρ, θ) = R_n^{|m|}(ρ) * sin(|m|θ)  (m < 0)
    
    其中径向部分：
        R_n^m(ρ) = Σ_{k=0}^{(n-m)/2} (-1)^k * (n-k)! / [k! * ((n+m)/2 - k)! * ((n-m)/2 - k)!] * ρ^{n-2k}
    
    参数:
        n: 径向阶数 (非负整数)
        m: 角向频率 (整数，|m| ≤ n，n-|m|为偶数)
        rho: 径向坐标 (0 ≤ ρ ≤ 1)
    
    返回:
        R: 径向部分的值
    """
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        numerator = factorial(n - k)
        denominator = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        R += ((-1) ** k) * (numerator / denominator) * (rho ** (n - 2 * k))
    return R


def zernike(n, m, rho, theta):
    """
    计算Zernike多项式的值
    
    参数:
        n: 径向阶数
        m: 角向频率
        rho, theta: 极坐标
    
    返回:
        Z: Zernike多项式的值
    """
    if m >= 0:
        return zernike_radial(n, m, rho) * np.cos(m * theta)
    else:
        return zernike_radial(n, -m, rho) * np.sin(-m * theta)


def generate_zernike_basis(size=256, n_max=6):
    """
    生成前几阶Zernike多项式的基函数
    
    参数:
        size: 图像尺寸
        n_max: 最大径向阶数
    
    返回:
        basis: Zernike基函数列表 [(n, m, name, Z)]
    """
    # 生成网格
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # 转换为极坐标
    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    
    # 单位圆外设为NaN
    mask = rho > 1
    
    # Zernike模式列表 (n, m, 名称)
    # 按Noll索引排序
    modes = [
        (0, 0, "活塞 (Piston)"),
        (1, -1, "倾斜Y (Tilt Y)"),
        (1, 1, "倾斜X (Tilt X)"),
        (2, -2, "散光 (Astigmatism)"),
        (2, 0, "离焦 (Defocus)"),
        (2, 2, "散光 (Astigmatism)"),
        (3, -3, "三叶草 (Trefoil)"),
        (3, -1, "彗差 (Coma)"),
        (3, 1, "彗差 (Coma)"),
        (3, 3, "三叶草 (Trefoil)"),
        (4, -4, "四叶草 (Tetrafoil)"),
        (4, -2, "二次散光 (Secondary Astigmatism)"),
        (4, 0, "球差 (Spherical Aberration)"),
        (4, 2, "二次散光 (Secondary Astigmatism)"),
        (4, 4, "四叶草 (Tetrafoil)"),
    ]
    
    basis = []
    for n, m, name in modes:
        if n > n_max:
            continue
        Z = zernike(n, m, rho, theta)
        Z[mask] = np.nan  # 单位圆外设为NaN
        basis.append((n, m, name, Z))
    
    return basis


def demo_zernike_modes():
    """
    演示1：前15阶Zernike模式可视化
    展示每一阶Zernike多项式长什么样
    """
    print("=" * 60)
    print("演示1：前15阶Zernike模式可视化")
    print("=" * 60)
    
    basis = generate_zernike_basis(size=256, n_max=4)
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (n, m, name, Z) in enumerate(basis):
        if i >= 15:
            break
        
        # 归一化到0-1用于显示
        Z_display = Z.copy()
        valid = ~np.isnan(Z_display)
        if valid.any():
            Z_min = Z_display[valid].min()
            Z_max = Z_display[valid].max()
            if Z_max > Z_min:
                Z_display[valid] = (Z_display[valid] - Z_min) / (Z_max - Z_min)
        
        im = axes[i].imshow(Z_display, cmap='RdBu_r', vmin=0, vmax=1)
        axes[i].set_title(f'Z_{n}^{m}\n{name}', fontsize=10)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(basis), 15):
        axes[i].axis('off')
    
    plt.suptitle('Zernike多项式前15阶（Noll排序）', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_Zernike模式可视化.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 05_Zernike模式可视化.png")
    plt.close()


def demo_tilt_vs_higher_order():
    """
    演示2：倾斜像差 vs 高阶像差的效果对比
    
    【核心概念】
    - 低阶像差（活塞、倾斜）：主要导致图像的整体位移/变形
    - 高阶像差（离焦、散光、彗差、球差等）：主要导致图像模糊
    
    这就是TMT两阶段设计的物理基础！
    """
    print("\n" + "=" * 60)
    print("演示2：倾斜像差 vs 高阶像差的效果对比")
    print("=" * 60)
    
    # 创建测试图像（点光源，看PSF）
    size = 256
    img = np.zeros((size, size), dtype=np.float32)
    img[size//2, size//2] = 1.0  # 点光源
    
    # 生成网格
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    
    def wavefront_to_psf(phase, wavelength=0.5, pixel_size=1.0):
        """
        波前 → 点扩散函数(PSF)
        
        物理原理：
        PSF = |Fourier(exp(j * 2π * phase / wavelength))|²
        
        相位畸变会导致点扩散函数展宽，即图像模糊。
        """
        # 复振幅
        aperture = (rho <= 1).astype(np.float32)
        complex_amplitude = aperture * np.exp(1j * 2 * np.pi * phase)
        
        # 傅里叶变换得到PSF
        psf = np.abs(np.fft.fftshift(np.fft.fft2(complex_amplitude)))**2
        
        # 归一化
        psf = psf / psf.max()
        
        return psf
    
    # 1. 只有倾斜（X倾斜 + Y倾斜）
    phase_tilt = 2 * zernike(1, 1, rho, theta) + 1.5 * zernike(1, -1, rho, theta)
    psf_tilt = wavefront_to_psf(phase_tilt)
    
    # 2. 只有离焦
    phase_defocus = 1.0 * zernike(2, 0, rho, theta)
    psf_defocus = wavefront_to_psf(phase_defocus)
    
    # 3. 只有散光
    phase_astigmatism = 1.0 * zernike(2, 2, rho, theta)
    psf_astigmatism = wavefront_to_psf(phase_astigmatism)
    
    # 4. 只有彗差
    phase_coma = 1.0 * zernike(3, 1, rho, theta)
    psf_coma = wavefront_to_psf(phase_coma)
    
    # 5. 只有球差
    phase_spherical = 1.0 * zernike(4, 0, rho, theta)
    psf_spherical = wavefront_to_psf(phase_spherical)
    
    # 6. 组合像差（模拟真实湍流）
    phase_combo = (0.5 * zernike(1, 1, rho, theta) + 
                   0.3 * zernike(2, 0, rho, theta) + 
                   0.4 * zernike(2, 2, rho, theta) +
                   0.3 * zernike(3, 1, rho, theta) +
                   0.2 * zernike(4, 0, rho, theta))
    psf_combo = wavefront_to_psf(phase_combo)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 显示中心区域
    crop = 64
    center = size // 2
    
    axes[0, 0].imshow(psf_tilt[center-crop:center+crop, center-crop:center+crop], cmap='hot')
    axes[0, 0].set_title('倾斜 (Tilt)\n→ 整体位移，不模糊')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(psf_defocus[center-crop:center+crop, center-crop:center+crop], cmap='hot')
    axes[0, 1].set_title('离焦 (Defocus)\n→ 均匀模糊')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(psf_astigmatism[center-crop:center+crop, center-crop:center+crop], cmap='hot')
    axes[0, 2].set_title('散光 (Astigmatism)\n→ 方向模糊')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(psf_coma[center-crop:center+crop, center-crop:center+crop], cmap='hot')
    axes[1, 0].set_title('彗差 (Coma)\n→ 拖尾模糊')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(psf_spherical[center-crop:center+crop, center-crop:center+crop], cmap='hot')
    axes[1, 1].set_title('球差 (Spherical)\n→ 中心模糊')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(psf_combo[center-crop:center+crop, center-crop:center+crop], cmap='hot')
    axes[1, 2].set_title('组合像差\n→ 真实湍流模糊')
    axes[1, 2].axis('off')
    
    plt.suptitle('不同像差对应的点扩散函数(PSF)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_像差与PSF.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 06_像差与PSF.png")
    plt.close()


def demo_why_two_stage_physical():
    """
    演示3：从Zernike角度理解TMT的两阶段设计
    
    【TMT的物理基础】
    Zernike分解告诉我们：
    - 第0阶：活塞（常数项，不影响图像）
    - 第1阶：X倾斜、Y倾斜 → 像素位移（几何变形）
    - 第2阶及以上：离焦、散光、彗差、球差... → 图像模糊
    
    因此，湍流退化可以分解为：
    1. 倾斜（Tilt）：低阶Zernike，导致像素位移
    2. 模糊（Blur）：高阶Zernike，导致像素混合
    
    TMT的两阶段设计正是基于这个物理分解！
    """
    print("\n" + "=" * 60)
    print("演示3：从Zernike角度理解TMT两阶段设计")
    print("=" * 60)
    
    print("""
【Zernike多项式的阶数与像差类型】

阶数 n=0: 活塞 (Piston)
    → 常数相位，不影响图像强度

阶数 n=1: 倾斜 (Tilt)
    → X方向倾斜、Y方向倾斜
    → 效果：图像整体位移（像素偏移）
    → 特点：几何变形，信息没有丢失
    → 难度：⭐⭐ 容易处理

阶数 n=2: 离焦 + 散光
    → 离焦：均匀模糊
    → 散光：方向性模糊
    → 效果：图像开始模糊
    → 难度：⭐⭐⭐ 中等

阶数 n=3: 彗差 + 三叶草
    → 彗差：彗星状拖尾
    → 效果：更复杂的模糊
    → 难度：⭐⭐⭐⭐ 较难

阶数 n≥4: 球差及更高阶
    → 球差、二次散光、四叶草等
    → 效果：更精细的模糊结构
    → 难度：⭐⭐⭐⭐⭐ 最难

【TMT两阶段设计的物理依据】

阶段1：去倾斜 (Tilt-Removal)
    → 处理 n=1 的低阶像差
    → 目标：把像素移回正确位置
    → 方法：3D U-Net（轻量级网络）
    → 原因：倾斜相对简单，小网络就够了

阶段2：去模糊 (Blur-Removal)
    → 处理 n≥2 的高阶像差
    → 目标：恢复被混合的像素
    → 方法：Transformer（复杂网络）
    → 原因：模糊更难，需要强大的特征提取能力

这就是为什么TMT的两阶段设计这么有效！
它不是拍脑袋想出来的，而是有坚实的物理基础。
    """)


def demo_turbulence_zernike_spectrum():
    """
    演示4：湍流的Zernike系数分布
    
    【Kolmogorov湍流模型】
    大气湍流的Zernike系数方差有特定的分布：
    - 低阶系数方差大（倾斜占主导）
    - 高阶系数方差逐渐减小
    
    这就是为什么：
    1. 倾斜是湍流退化的主要成分
    2. 先去倾斜可以去掉大部分退化
    3. 剩下的模糊虽然难，但占比小一些
    """
    print("\n" + "=" * 60)
    print("演示4：湍流的Zernike系数分布")
    print("=" * 60)
    
    # Kolmogorov湍流的Zernike方差（近似）
    # 参考：Noll's formula
    # 方差随阶数增加而减小
    n_modes = 15
    modes = list(range(n_modes))
    
    # 简化的Kolmogorov谱模型
    # 方差 ∝ (n+1)^(-11/3) （近似）
    variances = []
    for i in range(n_modes):
        # Noll索引对应的径向阶数n
        if i == 0:
            n = 0
        elif i <= 2:
            n = 1
        elif i <= 5:
            n = 2
        elif i <= 9:
            n = 3
        else:
            n = 4
        # Kolmogorov谱近似
        var = (n + 1) ** (-11/3)
        variances.append(var)
    
    variances = np.array(variances)
    variances = variances / variances.sum()  # 归一化
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2196F3'] + ['#4CAF50'] * 2 + ['#FF9800'] * 3 + ['#F44336'] * 4 + ['#9C27B0'] * 5
    
    bars = ax.bar(range(n_modes), variances, color=colors[:n_modes])
    
    # 添加标签
    mode_names = ['活塞', 'X倾斜', 'Y倾斜', '散光', '离焦', '散光', 
                  '三叶草', '彗差', '彗差', '三叶草',
                  '四叶草', '二次散光', '球差', '二次散光', '四叶草']
    
    ax.set_xticks(range(n_modes))
    ax.set_xticklabels(mode_names[:n_modes], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('方差占比', fontsize=12)
    ax.set_title('Kolmogorov湍流的Zernike系数方差分布', fontsize=14)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='n=0 (活塞)'),
        Patch(facecolor='#4CAF50', label='n=1 (倾斜) → Tilt'),
        Patch(facecolor='#FF9800', label='n=2 (离焦+散光)'),
        Patch(facecolor='#F44336', label='n=3 (彗差+三叶草) → Blur'),
        Patch(facecolor='#9C27B0', label='n=4 (球差等)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 标注Tilt和Blur的分界
    ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2)
    ax.text(1.2, max(variances)*0.9, 'Tilt\n(倾斜)', ha='center', fontsize=12, color='red', fontweight='bold')
    ax.text(8, max(variances)*0.9, 'Blur\n(模糊)', ha='center', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_湍流Zernike谱分布.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 07_湍流Zernike谱分布.png")
    plt.close()
    
    # 计算倾斜占比
    tilt_variance = variances[1:3].sum()  # 第1阶和第2阶（X和Y倾斜）
    total_variance = variances.sum()
    print(f"倾斜(Tilt)占总像差方差的比例: {tilt_variance/total_variance*100:.1f}%")
    print(f"高阶模糊(Blur)占总像差方差的比例: {(total_variance-tilt_variance)/total_variance*100:.1f}%")
    print("→ 倾斜占了很大一部分！先去倾斜可以解决大部分问题")


def main():
    print("=" * 60)
    print("知识点2：Zernike多项式与像差基础")
    print("=" * 60)
    print()
    print("【核心概念】")
    print("  Zernike多项式 = 光学中描述波前像差的标准工具")
    print("  每一阶Zernike对应一种具体的像差类型")
    print()
    print("  低阶（n≤1）：活塞、倾斜 → 几何变形（像素位移）")
    print("  高阶（n≥2）：离焦、散光、彗差、球差... → 图像模糊")
    print()
    print("  → 这就是TMT两阶段设计的物理基础！")
    print()
    
    # 运行所有演示
    demo_zernike_modes()
    demo_tilt_vs_higher_order()
    demo_why_two_stage_physical()
    demo_turbulence_zernike_spectrum()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. Zernike多项式是描述光学像差的标准工具")
    print("2. 低阶Zernike（倾斜）导致像素位移，高阶导致模糊")
    print("3. 湍流退化 = 倾斜 + 模糊，有坚实的物理基础")
    print("4. TMT两阶段设计 = 先去倾斜，后去模糊")
    print("5. 倾斜占湍流退化的很大一部分，先处理很划算")
    print()
    print("【下一步】")
    print("  → 下一个知识点：3D U-Net与去倾斜模块")


if __name__ == '__main__':
    main()
