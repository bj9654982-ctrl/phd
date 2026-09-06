"""
知识点1：图像退化模型与运动模糊
================================

【为什么要学这个？】
论文第3章研究的是"工业振动环境下的红外图像去模糊"。
在写去模糊网络之前，我们必须先理解：图像为什么会模糊？模糊的数学本质是什么？

【核心公式】
图像退化模型：B = I ⊗ k + n
    B: 模糊图像 (Blurred image)
    I: 清晰图像 (Sharp image)
    k: 模糊核 (Blur kernel / Point Spread Function, PSF)
    ⊗: 卷积运算 (Convolution)
    n: 噪声 (Noise)

【本教程你将学会】
1. 理解图像退化的数学原理
2. 手动生成运动模糊核
3. 用卷积生成模糊图像
4. 理解为什么去模糊是"病态问题"
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，保存图片不弹窗
import matplotlib.pyplot as plt
from scipy import signal
import os

# 设置中文字体（Windows环境）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_motion_blur_kernel(length=15, angle=0):
    """
    生成运动模糊核
    
    【原理】
    相机和物体之间的相对运动会导致运动模糊。
    最简单的运动模糊模型是：匀速直线运动。
    
    想象一下：相机曝光时间内，物体移动了一段距离。
    每个像素点的最终值，等于移动路径上所有像素的平均值。
    这就是为什么模糊图像看起来像是"拖影"。
    
    参数:
        length: 模糊长度（像素），运动越剧烈，长度越大
        angle: 运动方向（角度），0度表示水平运动
    
    返回:
        kernel: 模糊核（归一化的二维数组）
    """
    # 创建一个正方形的核
    kernel_size = length * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    
    # 计算运动方向的终点坐标
    center = length
    rad = np.deg2rad(angle)
    x_end = int(center + length * np.cos(rad))
    y_end = int(center - length * np.sin(rad))  # y轴向下为正，所以取负
    
    # 在起点和终点之间画一条直线（运动轨迹）
    cv2.line(kernel, (center, center), (x_end, y_end), 1, thickness=1)
    
    # 归一化：核的所有元素之和为1
    # 为什么要归一化？因为模糊不应该改变图像的整体亮度
    kernel = kernel / kernel.sum()
    
    return kernel


def apply_motion_blur(image, kernel):
    """
    对图像施加运动模糊
    
    【数学原理】
    模糊图像 = 清晰图像 卷积 模糊核
    
    卷积的物理意义：每个输出像素，是输入图像中一个区域的加权平均。
    权重由模糊核决定。
    
    参数:
        image: 清晰图像 (H, W) 或 (H, W, C)
        kernel: 模糊核
    
    返回:
        blurred: 模糊图像
    """
    if len(image.shape) == 2:
        # 灰度图
        blurred = cv2.filter2D(image, -1, kernel)
    else:
        # 彩色图，对每个通道分别卷积
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
    
    return blurred


def demo_basic_blur():
    """
    演示1：基础运动模糊
    展示不同模糊长度和角度的效果
    """
    print("=" * 60)
    print("演示1：基础运动模糊效果")
    print("=" * 60)
    
    # 创建一张简单的测试图像（白色背景上的黑色方块）
    img = np.ones((200, 200), dtype=np.float32) * 255
    img[50:150, 50:150] = 0  # 黑色方块
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 第一行：原图 + 不同模糊长度
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('清晰图像 (Sharp)')
    axes[0, 0].axis('off')
    
    for i, length in enumerate([5, 15, 30]):
        kernel = generate_motion_blur_kernel(length=length, angle=0)
        blurred = apply_motion_blur(img, kernel)
        axes[0, i+1].imshow(blurred, cmap='gray')
        axes[0, i+1].set_title(f'水平模糊, 长度={length}')
        axes[0, i+1].axis('off')
    
    # 第二行：不同模糊角度
    for i, angle in enumerate([0, 30, 60, 90]):
        kernel = generate_motion_blur_kernel(length=15, angle=angle)
        blurred = apply_motion_blur(img, kernel)
        axes[1, i].imshow(blurred, cmap='gray')
        axes[1, i].set_title(f'角度={angle}度, 长度=15')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = r'J:\phd\图像复原博士学位论文\复现项目\模块1_图像去模糊\01_图像退化模型'
    plt.savefig(os.path.join(output_dir, '01_基础运动模糊.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存到: {output_dir}\\01_基础运动模糊.png")
    plt.close()


def demo_kernel_visualization():
    """
    演示2：模糊核可视化
    看看模糊核长什么样
    """
    print("\n" + "=" * 60)
    print("演示2：模糊核（PSF）可视化")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, (length, angle) in enumerate([(10, 0), (20, 0), (15, 45)]):
        kernel = generate_motion_blur_kernel(length=length, angle=angle)
        axes[i].imshow(kernel, cmap='hot')
        axes[i].set_title(f'模糊核 (长度={length}, 角度={angle}°)')
        axes[i].axis('off')
        print(f"模糊核尺寸: {kernel.shape}, 元素和: {kernel.sum():.4f}")
    
    plt.tight_layout()
    output_dir = r'J:\phd\图像复原博士学位论文\复现项目\模块1_图像去模糊\01_图像退化模型'
    plt.savefig(os.path.join(output_dir, '02_模糊核可视化.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存到: {output_dir}\\02_模糊核可视化.png")
    plt.close()


def demo_why_deblur_is_hard():
    """
    演示3：为什么去模糊是"病态问题"？
    
    【关键概念：病态问题 (Ill-posed Problem)】
    
    模糊过程：清晰图像 → 模糊图像（信息丢失，不可逆）
    
    为什么难？
    1. 信息丢失：模糊相当于"平均"，高频细节被抹掉了
    2. 多解性：很多不同的清晰图像，模糊后可能看起来一样
    3. 噪声放大：直接逆卷积会把噪声放得很大
    
    这就是为什么我们需要深度学习！
    深度学习的作用：学习"先验知识"，从无数可能的解中选出最合理的那个。
    """
    print("\n" + "=" * 60)
    print("演示3：为什么去模糊是病态问题？")
    print("=" * 60)
    
    # 创建两张不同的清晰图像
    img1 = np.ones((100, 100), dtype=np.float32) * 128
    img1[40:60, 40:60] = 255  # 亮方块
    
    img2 = np.ones((100, 100), dtype=np.float32) * 128
    img2[35:65, 35:65] = 200  # 更大但更暗的方块
    
    # 施加相同的模糊
    kernel = generate_motion_blur_kernel(length=10, angle=0)
    blurred1 = apply_motion_blur(img1, kernel)
    blurred2 = apply_motion_blur(img2, kernel)
    
    # 计算两张模糊图像的差异
    diff = np.abs(blurred1 - blurred2)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"两张清晰图像差异: 最大={np.abs(img1-img2).max():.1f}, 平均={np.abs(img1-img2).mean():.1f}")
    print(f"两张模糊图像差异: 最大={max_diff:.1f}, 平均={mean_diff:.1f}")
    print(f"→ 模糊后差异变小了！说明模糊丢失了区分信息")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(img1, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('清晰图像A')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('清晰图像B')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(img1-img2), cmap='hot')
    axes[0, 2].set_title('清晰图像差异')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(blurred1, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('模糊图像A')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(blurred2, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('模糊图像B')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title(f'模糊图像差异\n(最大={max_diff:.1f})')
    axes[1, 2].axis('off')
    
    plt.suptitle('病态问题演示：不同清晰图 → 模糊后几乎一样', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_dir = r'J:\phd\图像复原博士学位论文\复现项目\模块1_图像去模糊\01_图像退化模型'
    plt.savefig(os.path.join(output_dir, '03_病态问题演示.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存到: {output_dir}\\03_病态问题演示.png")
    plt.close()


def demo_real_image_blur():
    """
    演示4：真实图像模糊效果
    用一张实际的图片看看模糊效果
    """
    print("\n" + "=" * 60)
    print("演示4：真实图像模糊效果")
    print("=" * 60)
    
    # 创建一张合成的"类红外图像"（模拟铸坯红外图）
    # 红外图像特点：对比度较低、热梯度、边缘模糊
    img = np.zeros((256, 256), dtype=np.float32)
    
    # 模拟铸坯背景（渐变热分布）
    for i in range(256):
        img[i, :] = 100 + 50 * i / 256  # 从上到下逐渐变亮
    
    # 添加一些"氧化铁皮"区域（更亮或更暗的斑块）
    cv2.circle(img, (80, 80), 30, 200, -1)
    cv2.circle(img, (180, 120), 25, 50, -1)
    cv2.rectangle(img, (100, 160), (160, 200), 180, -1)
    
    # 添加一些"裂纹"（细长暗线）
    cv2.line(img, (50, 50), (100, 100), 30, 2)
    cv2.line(img, (150, 180), (200, 220), 20, 2)
    
    # 添加噪声
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img + noise, 0, 255)
    
    # 施加不同程度的模糊
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    axes[0].imshow(img, cmap='inferno')
    axes[0].set_title('模拟红外图 (清晰)')
    axes[0].axis('off')
    
    for i, length in enumerate([5, 15, 30]):
        kernel = generate_motion_blur_kernel(length=length, angle=0)
        blurred = apply_motion_blur(img, kernel)
        axes[i+1].imshow(blurred, cmap='inferno')
        axes[i+1].set_title(f'运动模糊 (长度={length})')
        axes[i+1].axis('off')
    
    plt.suptitle('模拟铸坯红外图像的运动模糊效果', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_dir = r'J:\phd\图像复原博士学位论文\复现项目\模块1_图像去模糊\01_图像退化模型'
    plt.savefig(os.path.join(output_dir, '04_红外图像模糊.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存到: {output_dir}\\04_红外图像模糊.png")
    plt.close()


def main():
    print("=" * 60)
    print("知识点1：图像退化模型与运动模糊")
    print("=" * 60)
    print()
    print("【核心公式】")
    print("  B = I ⊗ k + n")
    print("  B: 模糊图像, I: 清晰图像, k: 模糊核, ⊗: 卷积, n: 噪声")
    print()
    
    # 运行所有演示
    demo_basic_blur()
    demo_kernel_visualization()
    demo_why_deblur_is_hard()
    demo_real_image_blur()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. 模糊 = 清晰图像 卷积 模糊核")
    print("2. 运动模糊的模糊核是一条直线（匀速运动）")
    print("3. 模糊长度越大，图像越模糊")
    print("4. 去模糊是病态问题：信息丢失，多解性")
    print("5. 深度学习的作用：学习先验，从多解中选最优")
    print()
    print("【下一步】")
    print("  → 下一个知识点：U-Net网络结构（编码器-解码器）")


if __name__ == '__main__':
    main()
