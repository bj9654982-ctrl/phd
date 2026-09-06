"""
知识点1：大气湍流退化模型
========================

【为什么要学这个？】
TMT论文的核心思想是"湍流退化 = 倾斜 + 模糊"，并据此设计了两阶段网络。
理解这个物理模型，是理解整篇论文的基础。

【核心公式】
湍流退化模型：I = B ∘ T (J)
    I: 退化图像 (Corrupted image)
    J: 清晰图像 (Clean image)
    T: 倾斜算子 (Tilt operator) - 像素位移
    B: 模糊算子 (Blur operator) - 空间变化模糊
    ∘: 复合运算 (先倾斜，后模糊)

【本教程你将学会】
1. 大气湍流的物理本质（折射率随机波动）
2. 倾斜(Tilt)是什么：像素的随机位移
3. 模糊(Blur)是什么：空间变化的模糊核
4. 为什么是"先倾斜，后模糊"（Tilt-then-Blur）
5. 幸运成像原理：多帧平均为什么能去湍流
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\01_湍流退化模型'


def generate_tilt_field(H, W, strength=5.0, scale=20):
    """
    生成倾斜场（Tilt Field）
    
    【物理原理】
    大气湍流导致折射率随机波动，使得光线发生偏折。
    从图像角度看，就是每个像素点发生了随机位移（倾斜）。
    
    倾斜场不是完全随机的噪声，而是具有空间相关性的平滑随机场。
    因为大气湍流是大尺度的流体运动，相邻位置的折射率变化是相关的。
    
    参数:
        H, W: 图像高度和宽度
        strength: 倾斜强度（像素）
        scale: 空间相关性尺度（越大越平滑）
    
    返回:
        dx, dy: x和y方向的位移场
    """
    # 生成低频随机噪声（模拟空间相关性）
    # 方法：生成小尺寸噪声，然后放大到原尺寸
    small_H = max(1, H // scale)
    small_W = max(1, W // scale)
    
    # 生成随机噪声
    noise_x = np.random.randn(small_H, small_W)
    noise_y = np.random.randn(small_H, small_W)
    
    # 放大到原尺寸（双线性插值，实现空间平滑）
    dx = cv2.resize(noise_x, (W, H), interpolation=cv2.INTER_LINEAR) * strength
    dy = cv2.resize(noise_y, (W, H), interpolation=cv2.INTER_LINEAR) * strength
    
    return dx, dy


def apply_tilt(image, dx, dy):
    """
    施加倾斜（像素位移）
    
    【数学原理】
    倾斜后的图像 I_tilt(x, y) = J(x + dx(x,y), y + dy(x,y))
    
    也就是说，每个像素 (x,y) 的值，来自原始图像中偏移后的位置。
    
    参数:
        image: 原始清晰图像
        dx, dy: 位移场
    
    返回:
        tilted: 倾斜后的图像
    """
    H, W = image.shape[:2]
    
    # 生成网格坐标
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)
    
    # 加上位移
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    
    # 重映射（双线性插值）
    if len(image.shape) == 2:
        tilted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        tilted = np.zeros_like(image)
        for c in range(image.shape[2]):
            tilted[:, :, c] = cv2.remap(image[:, :, c], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return tilted


def generate_spatially_varying_blur(H, W, kernel_size=15, strength=1.0):
    """
    生成空间变化的模糊场
    
    【物理原理】
    大气湍流不仅导致像素位移（倾斜），还会导致局部模糊。
    这是因为高阶像差（high-order aberrations）使得点扩散函数(PSF)变宽。
    
    不同位置的模糊程度不同（空间变化），因为不同位置的湍流强度不同。
    
    参数:
        H, W: 图像尺寸
        kernel_size: 模糊核大小
        strength: 模糊强度
    
    返回:
        blur_map: 每个位置的模糊强度图
    """
    # 生成平滑的模糊强度图
    small_H = max(1, H // 30)
    small_W = max(1, W // 30)
    noise = np.random.rand(small_H, small_W)
    blur_map = cv2.resize(noise, (W, H), interpolation=cv2.INTER_LINEAR)
    blur_map = blur_map * strength + 0.5  # 最小0.5，最大1.5
    
    return blur_map


def apply_spatially_varying_blur(image, blur_map, max_kernel=21):
    """
    施加空间变化模糊
    
    【注意】
    真正的空间变化模糊应该每个位置用不同的核，计算很慢。
    这里我们用近似方法：对不同区域用不同程度的高斯模糊。
    
    参数:
        image: 输入图像
        blur_map: 模糊强度图
        max_kernel: 最大模糊核大小
    
    返回:
        blurred: 模糊后的图像
    """
    H, W = image.shape[:2]
    
    # 简单近似：用平均模糊强度做全局高斯模糊
    # （真正的空间变化模糊更复杂，这里为了演示先简化）
    avg_blur = blur_map.mean()
    kernel_size = int(avg_blur * max_kernel)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)
    
    sigma = kernel_size / 6.0
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return blurred


def demo_basic_tilt():
    """
    演示1：基础倾斜效果
    展示不同强度的倾斜是什么样的
    """
    print("=" * 60)
    print("演示1：大气倾斜（像素位移）效果")
    print("=" * 60)
    
    # 创建测试图像（网格图案，便于观察位移）
    img = np.ones((256, 256), dtype=np.float32) * 255
    # 画网格
    for i in range(0, 256, 32):
        img[i, :] = 0
        img[:, i] = 0
    # 画几个圆点
    cv2.circle(img, (64, 64), 10, 0, -1)
    cv2.circle(img, (192, 192), 10, 0, -1)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('清晰图像')
    axes[0].axis('off')
    
    for i, strength in enumerate([2, 5, 10]):
        dx, dy = generate_tilt_field(256, 256, strength=strength, scale=20)
        tilted = apply_tilt(img, dx, dy)
        axes[i+1].imshow(tilted, cmap='gray')
        axes[i+1].set_title(f'倾斜强度={strength}像素')
        axes[i+1].axis('off')
    
    plt.suptitle('大气倾斜效果：像素的随机位移', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_基础倾斜效果.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 01_基础倾斜效果.png")
    plt.close()


def demo_tilt_field_visualization():
    """
    演示2：倾斜场可视化
    看看位移场长什么样
    """
    print("\n" + "=" * 60)
    print("演示2：倾斜场（位移场）可视化")
    print("=" * 60)
    
    dx, dy = generate_tilt_field(256, 256, strength=5, scale=20)
    
    # 计算位移大小
    magnitude = np.sqrt(dx**2 + dy**2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(dx, cmap='RdBu_r')
    axes[0].set_title('X方向位移 (dx)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(dy, cmap='RdBu_r')
    axes[1].set_title('Y方向位移 (dy)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    im3 = axes[2].imshow(magnitude, cmap='hot')
    axes[2].set_title('位移大小 (magnitude)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle('倾斜场：每个像素的随机位移向量', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_倾斜场可视化.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 02_倾斜场可视化.png")
    plt.close()


def demo_tilt_then_blur():
    """
    演示3：先倾斜后模糊（Tilt-then-Blur）
    论文的核心模型：湍流 = 倾斜 + 模糊，且顺序是先倾斜后模糊
    """
    print("\n" + "=" * 60)
    print("演示3：先倾斜后模糊（Tilt-then-Blur）")
    print("=" * 60)
    
    # 创建测试图像（文字，便于观察效果）
    img = np.ones((256, 256), dtype=np.uint8) * 255
    cv2.putText(img, 'TMT', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
    img = img.astype(np.float32)
    
    # 生成倾斜
    dx, dy = generate_tilt_field(256, 256, strength=5, scale=20)
    
    # 生成模糊
    blur_map = generate_spatially_varying_blur(256, 256, strength=1.0)
    
    # 三种情况对比
    # 1. 只有倾斜
    only_tilt = apply_tilt(img, dx, dy)
    
    # 2. 只有模糊
    only_blur = apply_spatially_varying_blur(img, blur_map)
    
    # 3. 先倾斜后模糊（论文模型）
    tilt_then_blur = apply_spatially_varying_blur(apply_tilt(img, dx, dy), blur_map)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('清晰图像')
    axes[0].axis('off')
    
    axes[1].imshow(only_tilt, cmap='gray')
    axes[1].set_title('只有倾斜')
    axes[1].axis('off')
    
    axes[2].imshow(only_blur, cmap='gray')
    axes[2].set_title('只有模糊')
    axes[2].axis('off')
    
    axes[3].imshow(tilt_then_blur, cmap='gray')
    axes[3].set_title('先倾斜后模糊 (完整湍流)')
    axes[3].axis('off')
    
    plt.suptitle('湍流退化 = 倾斜 + 模糊（先倾斜后模糊）', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_先倾斜后模糊.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 03_先倾斜后模糊.png")
    plt.close()


def demo_lucky_imaging():
    """
    演示4：幸运成像原理
    为什么多帧图像可以用来去湍流？
    
    【幸运成像(Lucky Imaging)原理】
    1. 湍流是随机变化的，每一帧的倾斜都不一样
    2. 有些帧的倾斜比较小（"幸运帧"），图像比较清晰
    3. 把多帧图像对齐后取平均，可以抵消随机倾斜
    4. 这就是多帧湍流复原的基础思想
    
    TMT论文用深度学习来实现这个过程，比传统方法更强大。
    """
    print("\n" + "=" * 60)
    print("演示4：幸运成像原理（多帧平均）")
    print("=" * 60)
    
    # 创建测试图像
    img = np.ones((256, 256), dtype=np.uint8) * 255
    cv2.putText(img, 'TEST', (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 0, 4)
    img = img.astype(np.float32)
    
    # 生成100帧湍流图像
    num_frames = 100
    frames = []
    for i in range(num_frames):
        dx, dy = generate_tilt_field(256, 256, strength=5, scale=20)
        tilted = apply_tilt(img, dx, dy)
        frames.append(tilted)
    
    frames = np.array(frames)
    
    # 不同帧数的平均
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('清晰图像 (Ground Truth)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(frames[0], cmap='gray')
    axes[0, 1].set_title('单帧湍流图像')
    axes[0, 1].axis('off')
    
    for i, n in enumerate([5, 20, 50, 100]):
        avg = frames[:n].mean(axis=0)
        row = (i + 2) // 3
        col = (i + 2) % 3
        axes[row, col].imshow(avg, cmap='gray')
        axes[row, col].set_title(f'{n}帧平均')
        axes[row, col].axis('off')
    
    plt.suptitle('幸运成像：多帧平均可以抵消随机倾斜', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_幸运成像原理.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 04_幸运成像原理.png")
    plt.close()
    
    # 计算PSNR
    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    print(f"单帧PSNR: {psnr(img, frames[0]):.2f} dB")
    print(f"5帧平均PSNR: {psnr(img, frames[:5].mean(axis=0)):.2f} dB")
    print(f"20帧平均PSNR: {psnr(img, frames[:20].mean(axis=0)):.2f} dB")
    print(f"50帧平均PSNR: {psnr(img, frames[:50].mean(axis=0)):.2f} dB")
    print(f"100帧平均PSNR: {psnr(img, frames[:100].mean(axis=0)):.2f} dB")
    print("→ 帧数越多，平均后越清晰！这就是多帧复原的基础")


def demo_why_two_stage():
    """
    演示5：为什么要两阶段（先去倾斜，后去模糊）？
    
    【TMT的核心洞察】
    倾斜和模糊是两种不同性质的退化：
    - 倾斜：几何变形，时间上随机，平均可以抵消
    - 模糊：像素混合，需要去卷积，更难恢复
    
    分开处理的好处：
    1. 倾斜比较容易处理，先去掉可以降低后续难度
    2. 模糊处理可以专注于真正的模糊问题
    3. 两阶段设计符合物理模型，泛化性更好
    """
    print("\n" + "=" * 60)
    print("演示5：为什么TMT用两阶段设计？")
    print("=" * 60)
    
    print("""
【倾斜 vs 模糊 的本质区别】

1. 倾斜 (Tilt):
   - 本质：像素的位移（几何变形）
   - 原因：低阶Zernike像差（整体偏折）
   - 特点：时间上随机波动，多帧平均可以抵消
   - 难度：⭐⭐ 相对容易

2. 模糊 (Blur):
   - 本质：像素的混合（卷积）
   - 原因：高阶Zernike像差（PSF展宽）
   - 特点：空间变化，每帧都有，无法通过平均完全消除
   - 难度：⭐⭐⭐⭐ 更难恢复

【为什么两阶段更好？】
1. 符合物理模型：湍流本身就是倾斜+模糊的组合
2. 降低难度：先去掉容易的倾斜，剩下的模糊问题更纯粹
3. 提高泛化：物理驱动的设计，在真实数据上表现更好
4. 节省内存：两阶段可以串行处理，内存占用更低

这就是TMT论文的核心思想！
    """)


def main():
    print("=" * 60)
    print("知识点1：大气湍流退化模型")
    print("=" * 60)
    print()
    print("【核心公式】")
    print("  I = B ∘ T (J)")
    print("  I: 退化图像, J: 清晰图像")
    print("  T: 倾斜算子(像素位移), B: 模糊算子(空间变化模糊)")
    print("  ∘: 复合运算（先倾斜，后模糊）")
    print()
    
    # 运行所有演示
    demo_basic_tilt()
    demo_tilt_field_visualization()
    demo_tilt_then_blur()
    demo_lucky_imaging()
    demo_why_two_stage()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. 大气湍流 = 倾斜 + 模糊（先倾斜后模糊）")
    print("2. 倾斜是像素的随机位移，具有空间相关性")
    print("3. 模糊是空间变化的卷积，由高阶像差导致")
    print("4. 幸运成像：多帧平均可以抵消随机倾斜")
    print("5. TMT两阶段设计：先去倾斜，后去模糊，符合物理模型")
    print()
    print("【下一步】")
    print("  → 下一个知识点：Zernike多项式与像差基础")


if __name__ == '__main__':
    main()
