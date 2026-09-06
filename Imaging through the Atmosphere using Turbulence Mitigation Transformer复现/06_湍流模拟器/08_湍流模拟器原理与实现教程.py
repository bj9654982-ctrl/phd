"""
知识点8：湍流模拟器原理与实现
================================

【为什么要学这个？】
深度学习需要大量数据，但真实的湍流图像数据很难获取。
所以我们需要湍流模拟器，用清晰图像生成湍流图像，用来训练网络。
TMT对湍流模拟器做了三个重要改进，让生成的数据更真实、更灵活。

【本教程你将学会】
1. 为什么需要湍流模拟器？（数据从哪来？）
2. 传统湍流模拟器的问题
3. TMT改进1：Fourier-based采样器（密集场模拟）
4. TMT改进2：时间相关性（自回归模型）
5. TMT改进3：灵活核尺寸（从9×9到33×33）
6. 用代码生成湍流退化序列
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.special import jv  # 贝塞尔函数
import cv2
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\06_湍流模拟器'
os.makedirs(output_dir, exist_ok=True)


def demo_why_need_simulator():
    """
    演示1：为什么需要湍流模拟器？
    """
    print("=" * 60)
    print("演示1：为什么需要湍流模拟器？")
    print("=" * 60)
    
    print("""
【深度学习的痛点：数据从哪来？】

深度学习需要大量成对的数据：
  - 输入：湍流图像
  - 标签：清晰图像（GT）

但真实的湍流数据很难获取：
1. 很难在真实场景中同时拍到清晰图和湍流图
2. 真实数据的湍流强度不可控
3. 真实数据数量少，不够训练
4. 真实数据没有GT，没法监督学习

【解决方案：湍流模拟器】

用清晰图像 + 湍流模型 → 生成湍流图像
  → 这样就有无限多的成对数据了！
  → 湍流强度可以精确控制
  → GT是已知的，可以监督训练

【这是图像复原领域的常规操作】
- 去模糊：用清晰图 + 模糊核 → 生成模糊图
- 去噪：用清晰图 + 噪声 → 生成噪声图
- 去雨：用清晰图 + 雨层 → 生成雨图
- 去湍流：用清晰图 + 湍流模型 → 生成湍流图

【但问题是：模拟器够不够真实？】
- 如果模拟器和真实湍流差太远
- 那在模拟数据上训的网络，在真实数据上效果就差
- 这就是"域差距"（domain gap）问题

所以，一个好的湍流模拟器非常重要！
TMT的贡献之一就是改进了湍流模拟器，让它更真实、更灵活。
    """)


def generate_phase_screen(size, r0, L0=100, l0=0.01):
    """
    生成Kolmogorov湍流相位屏
    
    用Fourier方法生成符合Kolmogorov谱的相位屏
    
    参数:
        size: 相位屏尺寸
        r0: Fried参数（相干长度），越小湍流越强
        L0: 外尺度
        l0: 内尺度
    
    返回:
        phase: 相位屏，形状 (size, size)
    """
    # 频率网格
    fx = np.fft.fftfreq(size)
    fy = np.fft.fftfreq(size)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX**2 + FY**2)
    f[0, 0] = 1e-10  # 避免除以0
    
    # von Karman湍流谱（修正的Kolmogorov谱）
    # φ(f) = 0.023 r0^(-5/3) f^(-11/3) exp(-f²l0²) / (1 + f²L0²)^(1/6)
    # 简化版：用Kolmogorov谱
    spectrum = 0.023 * r0**(-5/3) * f**(-11/3)
    spectrum[0, 0] = 0  # 直流分量为0
    
    # 生成随机相位
    random_phase = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    
    # 频域采样
    phase_fft = np.sqrt(spectrum) * random_phase * size
    
    # 逆傅里叶变换到时域
    phase = np.real(np.fft.ifft2(phase_fft))
    
    return phase


def demo_fourier_sampler():
    """
    演示2：Fourier-based采样器
    
    TMT改进1：用Fourier方法生成密集的相位场
    """
    print("\n" + "=" * 60)
    print("演示2：Fourier-based采样器（改进1）")
    print("=" * 60)
    
    print("""
【传统方法的问题】

传统的湍流模拟器：
  - 先在稀疏网格上生成相位值
  - 然后用插值得到密集的相位场
  
问题：
  - 插值会引入误差
  - 特别是在高频部分，插值不准
  - 生成的PSF不够真实

【TMT的改进：Fourier-based采样器】

直接在频域采样，然后逆FFT得到密集的相位场：
  1. 生成频率网格
  2. 按湍流谱（比如Kolmogorov谱）采样
  3. 加随机相位
  4. 逆FFT得到空间域的相位屏

优点：
  - 直接生成密集场，不需要插值
  - 频谱更准确，PSF更真实
  - 计算效率高（FFT很快）

【为什么叫"采样器"？】
因为湍流是随机过程，每次生成都不一样
我们需要从湍流的概率分布中"采样"一个实现
Fourier方法就是一种采样方法。
    """)
    
    # 生成不同强度的相位屏
    print("【相位屏可视化】")
    
    size = 256
    r0_values = [0.5, 0.2, 0.1]  # r0越小，湍流越强
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, r0 in enumerate(r0_values):
        phase = generate_phase_screen(size, r0)
        im = axes[i].imshow(phase, cmap='RdBu_r', vmin=-np.max(np.abs(phase)), vmax=np.max(np.abs(phase)))
        axes[i].set_title(f'r0 = {r0}（湍流{"弱" if r0 > 0.3 else "中" if r0 > 0.15 else "强"}）', fontsize=14)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Kolmogorov湍流相位屏（Fourier采样）', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_相位屏可视化.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 09_相位屏可视化.png")
    plt.close()


def phase_to_psf(phase, wavelength=0.5e-6, D=0.1, pixel_size=1e-6):
    """
    相位屏转PSF（点扩散函数）
    
    PSF = |FFT(exp(j*phase))|²
    
    参数:
        phase: 相位屏（单位：弧度）
        wavelength: 波长（米）
        D: 孔径直径（米）
        pixel_size: 像素大小（米）
    
    返回:
        psf: 点扩散函数
    """
    size = phase.shape[0]
    
    # 孔径函数（圆形孔径）
    y, x = np.mgrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x**2 + y**2)
    aperture = (r < size//4).astype(float)  # 简化：用1/4大小的孔径
    
    # 复振幅
    complex_amplitude = aperture * np.exp(1j * phase)
    
    # FFT得到PSF
    psf = np.abs(np.fft.fftshift(np.fft.fft2(complex_amplitude)))**2
    
    # 归一化
    psf = psf / psf.sum()
    
    return psf


def demo_psf_from_phase():
    """
    演示3：相位屏转PSF
    
    展示不同湍流强度下的PSF
    """
    print("\n" + "=" * 60)
    print("演示3：相位屏转PSF")
    print("=" * 60)
    
    print("""
【相位和PSF的关系】

相位屏 → 复振幅 → FFT → PSF

  PSF = |FFT(exp(j·phase))|²

物理意义：
  - 相位屏代表波前的畸变
  - PSF是点光源经过畸变后的像
  - 湍流越强，PSF越弥散，图像越模糊

【不同湍流强度的PSF】
  - 弱湍流：PSF比较集中，接近衍射极限
  - 中湍流：PSF开始弥散
  - 强湍流：PSF很弥散，图像严重模糊
    """)
    
    size = 256
    r0_values = [0.5, 0.2, 0.1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, r0 in enumerate(r0_values):
        phase = generate_phase_screen(size, r0)
        psf = phase_to_psf(phase)
        
        # 裁剪PSF中心区域
        crop = 64
        psf_crop = psf[size//2-crop:size//2+crop, size//2-crop:size//2+crop]
        
        im = axes[i].imshow(psf_crop, cmap='hot')
        axes[i].set_title(f'r0 = {r0}', fontsize=14)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('不同湍流强度的PSF（点扩散函数）', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_不同湍流强度PSF.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 10_不同湍流强度PSF.png")
    plt.close()


def demo_temporal_correlation():
    """
    演示4：时间相关性（改进2）
    
    TMT改进2：自回归模型生成时间相关的湍流序列
    """
    print("\n" + "=" * 60)
    print("演示4：时间相关性（改进2）")
    print("=" * 60)
    
    print("""
【传统方法的问题】

传统的湍流模拟器：
  - 每一帧都是独立生成的
  - 帧与帧之间没有相关性
  
但真实的湍流：
  - 是随时间平滑变化的
  - 相邻帧的湍流很相似
  - 有时间相关性

如果训练数据没有时间相关性：
  - 网络学不到利用时间相关性复原
  - 在真实数据上效果就差

【TMT的改进：自回归模型】

用自回归（AR）模型生成时间相关的序列：

  w_t = α · w_{t-1} + √(1-α²) · z_t

  w_t: 第t帧的相位屏（或其系数）
  α: 相关系数（0~1之间）
  z_t: 第t帧的高斯白噪声

特点：
  - α越大，时间相关性越强，变化越慢
  - α越小，时间相关性越弱，变化越快
  - 每帧的方差保持不变（因为有√(1-α²)）

【为什么在随机种子上做，而不是在最终样本上做？】
  - 保证每帧的统计特性一致
  - 如果直接在相位屏上做，方差会变
  - 在种子上做，可以保持方差不变

【α的取值】
  - α = 0.9：强相关（变化很慢）
  - α = 0.5：中等相关
  - α = 0.2：弱相关（变化很快）
  
TMT论文中：α从0.2到0.95随机采样
    """)
    
    # 演示时间相关性
    print("【时间相关性演示】")
    
    num_frames = 20
    alpha_values = [0.95, 0.7, 0.3]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    for i, alpha in enumerate(alpha_values):
        # 生成自回归序列
        values = np.zeros(num_frames)
        values[0] = np.random.randn()
        
        for t in range(1, num_frames):
            z = np.random.randn()
            values[t] = alpha * values[t-1] + np.sqrt(1 - alpha**2) * z
        
        axes[i].plot(values, 'o-', linewidth=2, markersize=6)
        axes[i].set_title(f'α = {alpha}（{"强" if alpha > 0.8 else "中" if alpha > 0.5 else "弱"}相关）', fontsize=14)
        axes[i].set_xlabel('帧', fontsize=12)
        axes[i].set_ylabel('相位值', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linewidth=0.5)
    
    plt.suptitle('自回归模型生成的时间序列', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_时间相关性演示.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 11_时间相关性演示.png")
    plt.close()


def generate_turbulent_sequence(image, num_frames=10, r0=0.2, alpha=0.8, kernel_size=33):
    """
    生成湍流退化序列（简化版）
    
    参数:
        image: 清晰图像 (H, W)
        num_frames: 帧数
        r0: Fried参数（湍流强度）
        alpha: 时间相关系数
        kernel_size: PSF核大小
    
    返回:
        sequence: 湍流序列 (T, H, W)
    """
    H, W = image.shape
    
    # 生成基函数（Zernike或其他）
    # 简化版：直接生成相位屏序列
    
    # 初始化
    sequence = np.zeros((num_frames, H, W))
    
    # 第一帧
    phase = generate_phase_screen(max(H, W), r0)
    psf = phase_to_psf(phase)
    
    # 裁剪PSF到指定大小
    center = psf.shape[0] // 2
    half = kernel_size // 2
    psf_crop = psf[center-half:center+half+1, center-half:center+half+1]
    psf_crop = psf_crop / psf_crop.sum()
    
    # 模糊图像
    sequence[0] = cv2.filter2D(image, -1, psf_crop)
    
    # 生成倾斜（像素位移）
    # 简化版：用高斯随机场生成位移
    np.random.seed(42)
    
    # 生成位移场的种子
    dx_seed = np.random.randn(H, W)
    dy_seed = np.random.randn(H, W)
    
    for t in range(1, num_frames):
        # 自回归更新种子
        dx_seed = alpha * dx_seed + np.sqrt(1 - alpha**2) * np.random.randn(H, W)
        dy_seed = alpha * dy_seed + np.sqrt(1 - alpha**2) * np.random.randn(H, W)
        
        # 生成位移场（平滑一下，让位移空间相关）
        dx = gaussian_filter(dx_seed * 2, sigma=5)  # 最大位移约2像素
        dy = gaussian_filter(dy_seed * 2, sigma=5)
        
        # 应用倾斜（像素位移）
        y, x = np.mgrid[0:H, 0:W]
        coords = np.array([y + dy, x + dx])
        warped = map_coordinates(image, coords, order=1, mode='reflect')
        
        # 应用模糊
        # 简化：每帧用稍微不同的PSF
        phase_t = generate_phase_screen(max(H, W), r0 * (1 + 0.1 * np.random.randn()))
        psf_t = phase_to_psf(phase_t)
        psf_t_crop = psf_t[center-half:center+half+1, center-half:center+half+1]
        psf_t_crop = psf_t_crop / psf_t_crop.sum()
        
        sequence[t] = cv2.filter2D(warped, -1, psf_t_crop)
    
    return sequence


def demo_flexible_kernel_size():
    """
    演示5：灵活核尺寸（改进3）
    
    TMT改进3：支持从9×9到33×33的灵活核尺寸
    """
    print("\n" + "=" * 60)
    print("演示5：灵活核尺寸（改进3）")
    print("=" * 60)
    
    print("""
【传统方法的问题】

传统的湍流模拟器：
  - PSF核大小是固定的（比如33×33）
  - 不能根据场景调整

但实际情况：
  - 静态场景、窄视场：PSF大，需要大核
  - 动态场景、宽视场：PSF小，用小核就行
  - 固定大核浪费计算，固定小核不够准确

【TMT的改进：灵活核尺寸】

支持从9×9到33×33的任意核尺寸：
  - 静态场景：用大核（33×33），更准确
  - 动态场景：用小核（9×9），更高效
  - 可以根据需要灵活选择

【怎么实现的？】
  - 重新生成更高分辨率的基函数
  - 避免混叠（aliasing）问题
  - 不同核尺寸对应不同的采样率

【为什么这很重要？】
1. 计算效率：小核比大核快很多
2. 数据多样性：不同核尺寸的训练数据，网络更鲁棒
3. 场景适配：不同场景用不同核，更真实

TMT论文中：
  - 静态场景：固定33×33
  - 动态场景：9~33随机采样
    """)
    
    # 演示不同核尺寸的PSF
    print("【不同核尺寸的PSF对比】")
    
    size = 256
    r0 = 0.2
    kernel_sizes = [9, 17, 33]
    
    phase = generate_phase_screen(size, r0)
    psf = phase_to_psf(phase)
    center = psf.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, ksize in enumerate(kernel_sizes):
        half = ksize // 2
        psf_crop = psf[center-half:center+half+1, center-half:center+half+1]
        psf_crop = psf_crop / psf_crop.sum()
        
        im = axes[i].imshow(psf_crop, cmap='hot')
        axes[i].set_title(f'{ksize}×{ksize} 核', fontsize=14)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('不同核尺寸的PSF', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '12_不同核尺寸PSF.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 12_不同核尺寸PSF.png")
    plt.close()


def demo_turbulent_sequence():
    """
    演示6：生成完整的湍流序列
    
    把前面的所有改进合起来，生成一个完整的湍流序列
    """
    print("\n" + "=" * 60)
    print("演示6：生成完整的湍流序列")
    print("=" * 60)
    
    # 生成一个测试图像（棋盘格，方便看效果）
    size = 128
    image = np.zeros((size, size))
    for i in range(0, size, 16):
        for j in range(0, size, 16):
            if (i//16 + j//16) % 2 == 0:
                image[i:i+16, j:j+16] = 1.0
    
    # 生成湍流序列
    print("生成湍流序列中...")
    sequence = generate_turbulent_sequence(image, num_frames=8, r0=0.15, alpha=0.8, kernel_size=21)
    print("生成完成！")
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for t in range(8):
        axes[t].imshow(sequence[t], cmap='gray', vmin=0, vmax=1)
        axes[t].set_title(f'第 {t+1} 帧', fontsize=12)
        axes[t].axis('off')
    
    plt.suptitle('湍流退化序列（8帧）', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '13_湍流序列演示.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 13_湍流序列演示.png")
    plt.close()


def demo_three_improvements_summary():
    """
    演示7：TMT三个改进总结
    """
    print("\n" + "=" * 60)
    print("演示7：TMT湍流模拟器的三个改进总结")
    print("=" * 60)
    
    print("""
【TMT对湍流模拟器的三个改进】

┌─────────────────┬──────────────────────┬──────────────────────┐
│     改进点       │     传统方法问题      │     TMT改进          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 1. Fourier采样器 │ 稀疏网格+插值，不准  │ 频域直接采样，准确   │
│ 2. 时间相关性   │ 帧间独立，不真实      │ 自回归模型，有相关性 │
│ 3. 灵活核尺寸   │ 固定33×33，不灵活    │ 9~33任意尺寸，灵活   │
└─────────────────┴──────────────────────┴──────────────────────┘

【为什么这些改进重要？】

1. 数据更真实 → 网络训练效果更好
2. 数据更多样 → 网络泛化能力更强
3. 计算更高效 → 数据生成更快

【和其他方法的对比】

TMT不是第一个做湍流模拟器的，
但它的改进让模拟器更实用、更灵活、更真实。

这也是TMT效果好的原因之一：
  - 好的训练数据 → 好的模型
  - 垃圾进，垃圾出（Garbage in, garbage out）

【对你的启发】

做深度学习研究，不要只盯着网络结构
数据也非常重要！
  - 数据质量
  - 数据多样性
  - 数据增强
  - 数据生成方法

很多时候，改进数据比改进网络效果还明显。
    """)


def main():
    print("=" * 60)
    print("知识点8：湍流模拟器原理与实现")
    print("=" * 60)
    print()
    print("【核心思想】")
    print("  湍流模拟器 = 用清晰图生成湍流图，用来训练网络")
    print()
    print("  TMT的三个改进：")
    print("    1. Fourier采样器：直接生成密集相位场，更准确")
    print("    2. 时间相关性：自回归模型，更真实")
    print("    3. 灵活核尺寸：9~33任意大小，更灵活")
    print()
    
    # 运行所有演示
    demo_why_need_simulator()
    demo_fourier_sampler()
    demo_psf_from_phase()
    demo_temporal_correlation()
    demo_flexible_kernel_size()
    demo_turbulent_sequence()
    demo_three_improvements_summary()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. 为什么需要湍流模拟器？→ 真实数据太少，需要合成数据")
    print("2. Fourier采样器 → 直接频域采样，密集场，不用插值")
    print("3. 时间相关性 → 自回归模型 w_t = αw_{t-1} + √(1-α²)z_t")
    print("4. 灵活核尺寸 → 9~33都支持，适应不同场景")
    print("5. 好的数据很重要，不一定比改进网络效果差")
    print()
    print("【下一步】")
    print("  → 下一个知识点：TMT数据集构建")
    print("  → （静态场景、动态场景、参数设置）")


if __name__ == '__main__':
    main()
