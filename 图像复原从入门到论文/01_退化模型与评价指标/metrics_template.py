"""
第一阶段：图像质量评价指标
功能：PSNR 和 SSIM（SSIM默认使用scikit-image标准实现，确保论文结果可复现）
修复：S2-03 手写SSIM与标准实现有偏差，改用skimage标准实现
修复：S1-06 评价指标返回原始浮点值，不在函数内四舍五入
"""

import numpy as np
import cv2

# 尝试导入skimage的标准SSIM实现
try:
    from skimage.metrics import structural_similarity as skimage_ssim
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def compute_mse(img1, img2):
    """
    计算均方误差 MSE

    参数:
        img1, img2: 输入图像 (H x W 或 H x W x C)，uint8或float

    返回:
        mse: 均方误差（标量）
    """
    # 转成float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # TODO 1: 计算MSE
    # 公式：MSE = (1/N) * sum((img1 - img2)^2)
    # 思路：先算逐像素差，再平方，最后取所有元素的均值
    diff = img1 - img2
    mse = np.mean(diff ** 2)
    return mse


def compute_psnr(img1, img2, max_val=255.0):
    """
    计算峰值信噪比 PSNR

    参数:
        img1, img2: 输入图像，uint8
        max_val: 像素最大值（8位图=255，归一化图=1.0）

    返回:
        psnr: 峰值信噪比（dB）
    """
    mse = compute_mse(img1, img2)

    # TODO 2: 计算PSNR
    # 公式：PSNR = 10 * log10(max_val^2 / MSE)
    # 思路：MSE=0时两张图完全相同，PSNR无穷大，返回100.0作为上限
    if mse == 0:
        return 100.0
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return psnr


def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """
    计算结构相似性 SSIM（单通道）

    默认使用scikit-image标准实现，确保论文结果可复现、与文献可比。
    若skimage不可用，回退到手写实现（仅用于教学，不建议用于论文）。

    参数:
        img1, img2: 单通道灰度图，uint8
        window_size: 局部窗口大小（奇数），仅手写实现使用
        sigma: 高斯窗口的标准差，仅手写实现使用

    返回:
        ssim: 结构相似性（0~1，标准SSIM可为负）
    """
    if _HAS_SKIMAGE:
        # 使用skimage标准实现：data_range根据图像dtype自动推断
        # win_size=7是skimage默认值，与大多数文献一致
        return skimage_ssim(img1, img2, data_range=255, win_size=7)

    # 回退：手写实现（仅用于教学，不建议用于论文）
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    coords = np.arange(window_size) - window_size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = np.outer(g, g)
    window = window / window.sum()

    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1 ** 2
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2 ** 2
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1 * mu2

    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map.mean()


def compute_ssim_color(img1, img2):
    """
    计算彩色图像的SSIM（在Y通道上计算）

    参数:
        img1, img2: BGR彩色图，uint8

    返回:
        ssim: 结构相似性
    """
    # 转YCbCr，在Y（亮度）通道上计算SSIM
    img1_y = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return compute_ssim(img1_y, img2_y)


def evaluate_restoration(restored, ground_truth):
    """
    综合评价复原效果

    返回:
        dict: 包含 PSNR, SSIM, MSE
    """
    if len(restored.shape) == 3:
        # 彩色图
        psnr = compute_psnr(restored, ground_truth)
        ssim = compute_ssim_color(restored, ground_truth)
    else:
        # 灰度图
        psnr = compute_psnr(restored, ground_truth)
        ssim = compute_ssim(restored, ground_truth)

    mse = compute_mse(restored, ground_truth)

    # 返回原始浮点值，不在函数内四舍五入（修复S1-06：调参时不应使用四舍五入后的值）
    # 显示时再用:.2f或:.4f格式化
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'MSE': mse
    }


if __name__ == "__main__":
    # 批量计算所有清晰图对应的模糊图的评价指标
    import os

    clean_dir = "../data/clean"
    blur_dir = "../data/synthetic_blur"

    # 收集所有清晰图
    clean_files = []
    if os.path.isdir(clean_dir):
        for f in sorted(os.listdir(clean_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                clean_files.append(f)

    if not clean_files:
        print(f"未在 {clean_dir} 找到图片")
        exit()

    blur_types = ['motion', 'defocus', 'gaussian', 'motion_noisy']
    blur_names = ['运动模糊', '散焦模糊', '高斯模糊', '运动+噪声']

    # 遍历每张清晰图
    for clean_file in clean_files:
        img_name = os.path.splitext(clean_file)[0]
        clean_path = os.path.join(clean_dir, clean_file)
        clean = cv2.imread(clean_path)
        if clean is None:
            continue

        print(f"\n{'='*60}")
        print(f"清晰图: {clean_file} ({clean.shape[1]}x{clean.shape[0]})")
        print(f"{'='*60}")
        print(f"{'模糊类型':<12} {'PSNR(dB)':<12} {'SSIM':<10} {'MSE':<10}")
        print("-" * 50)

        for blur_type, blur_name in zip(blur_types, blur_names):
            blur_path = os.path.join(blur_dir, f"{img_name}_{blur_type}.jpg")
            blurred = cv2.imread(blur_path)
            if blurred is None:
                print(f"{blur_name:<12} {'(未生成)':<12}")
                continue

            metrics = evaluate_restoration(blurred, clean)
            print(f"{blur_name:<12} {metrics['PSNR']:<12} {metrics['SSIM']:<10} {metrics['MSE']:<10}")

    print("\n" + "="*60)
    print("思考题：")
    print("1. 不同图片上，哪种模糊的PSNR最高？是否一致？")
    print("2. 加噪声后PSNR下降了多少？不同图一样吗？")
    print("3. PSNR和SSIM的排序一致吗？为什么？")
    print("4. 哪张图的模糊最严重（PSNR最低）？和图像内容有关吗？")
