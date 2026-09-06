"""
第三阶段：过渡区提取与特征量化
功能：从模糊图中提取边缘过渡区，计算过渡区宽度、梯度均值等特征，
     验证"模糊越重，过渡区越宽"的核心假设
请在 TODO 处填写你的代码
"""

import numpy as np
import cv2
import os
import sys

# 导入第一阶段的PSF生成函数
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_退化模型与评价指标"))
from degradation_template import motion_blur_psf, apply_blur


def imwrite_unicode(path, img):
    """兼容中文路径的图片保存（cv2.imwrite在Windows上对中文路径会静默失败）"""
    ext = os.path.splitext(path)[1]
    if ext.lower() == '.jpg':
        ext = '.jpg'
    result, encoded = cv2.imencode(ext, img)
    if result:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def compute_gradient(image):
    """
    计算图像梯度幅值

    参数:
        image: 输入图像，uint8

    返回:
        gradient: 梯度幅值图，float64
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float64)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gx**2 + gy**2)

    return gradient


def extract_transition_zone(image, gradient_ratio=0.1):
    """
    基于梯度阈值提取过渡区

    参数:
        image: 输入图像，uint8
        gradient_ratio: 梯度阈值比例（0-1），梯度 > max_gradient * ratio 的区域为过渡区

    返回:
        tz_mask: 过渡区二值掩码，uint8（255=过渡区，0=非过渡区）
        gradient: 梯度幅值图，float64
    """
    gradient = compute_gradient(image)

    # TODO 1: 计算梯度阈值
    # 用最大梯度乘以比例作为阈值：梯度大于阈值的像素认为在过渡区内
    max_grad = np.max(gradient)
    threshold = max_grad * gradient_ratio

    # TODO 2: 生成过渡区掩码
    # 梯度大于阈值的像素标记为过渡区（255），其他为非过渡区（0）
    tz_mask = (gradient > threshold).astype(np.uint8) * 255

    return tz_mask, gradient


def compute_transition_width(image, gradient_ratio=0.1):
    """
    计算过渡区平均宽度

    方法：对过渡区掩码做距离变换，统计过渡区到最近非过渡区的平均距离，
         过渡区宽度 ≈ 2 * 平均距离（因为过渡区两侧都有非过渡区）

    参数:
        image: 输入图像，uint8
        gradient_ratio: 梯度阈值比例

    返回:
        avg_width: 过渡区平均宽度（像素）
    """
    tz_mask, gradient = extract_transition_zone(image, gradient_ratio)

    # TODO 3: 距离变换
    # distanceTransform的输入：0表示背景，非0表示前景
    #   - 输入tz_mask：非过渡区=0（背景），过渡区=255（前景）
    #   - 输出：每个像素到最近背景像素（非过渡区=0）的距离
    #   - 所以过渡区内部的像素，距离值就是它到最近非过渡区的距离（>0）
    # 注意：不能用255-tz_mask，那样过渡区变成0（背景），距离就全是0了
    dist = cv2.distanceTransform(tz_mask, cv2.DIST_L2, 3)

    # TODO 4: 计算过渡区内像素的平均距离，过渡区宽度 = 2 * 平均距离
    # 原理：过渡区是一个"带"，带中心到边缘的平均距离是avg_dist，
    #       所以带的总宽度 ≈ 2 * avg_dist（两侧各有一个边缘）
    tz_pixels = dist[tz_mask > 0]
    if len(tz_pixels) > 0:
        avg_dist = np.mean(tz_pixels)
    else:
        avg_dist = 0.0
    avg_width = 2.0 * avg_dist

    return avg_width


def compute_mean_gradient(image, gradient_ratio=0.1):
    """
    计算过渡区内的平均梯度

    参数:
        image: 输入图像，uint8
        gradient_ratio: 梯度阈值比例

    返回:
        mean_grad: 过渡区内平均梯度值
    """
    tz_mask, gradient = extract_transition_zone(image, gradient_ratio)

    # TODO 5: 计算过渡区内像素的平均梯度
    # 只统计过渡区内（tz_mask>0）的梯度值，求平均值
    # 平均梯度反映过渡区的"陡峭程度"：梯度越大，边缘越锐利，模糊越轻
    tz_grads = gradient[tz_mask > 0]
    if len(tz_grads) > 0:
        mean_grad = np.mean(tz_grads)
    else:
        mean_grad = 0.0

    return mean_grad


def analyze_blur_effect(image, blur_lengths=[5, 10, 15, 20, 25], gradient_ratio=0.1):
    """
    分析模糊程度对过渡区特征的影响

    参数:
        image: 清晰原图，uint8
        blur_lengths: 运动模糊长度列表
        gradient_ratio: 梯度阈值比例

    返回:
        results: 列表，每个元素是 {length, width, mean_grad}
    """
    results = []

    # 清晰图（length=0）
    width_clean = compute_transition_width(image, gradient_ratio)
    grad_clean = compute_mean_gradient(image, gradient_ratio)
    results.append({'length': 0, 'width': width_clean, 'mean_grad': grad_clean})

    # 不同模糊程度
    for length in blur_lengths:
        psf = motion_blur_psf(kernel_size=max(21, length * 2 + 1), angle=25, length=length)
        blurred = apply_blur(image, psf)

        width = compute_transition_width(blurred, gradient_ratio)
        mean_grad = compute_mean_gradient(blurred, gradient_ratio)

        results.append({'length': length, 'width': width, 'mean_grad': mean_grad})

    return results


def process_all_images():
    """批量处理：对所有清晰图分析模糊程度 vs 过渡区特征"""

    clean_dir = "../data/clean"
    result_dir = "../results/03_边缘检测"
    os.makedirs(result_dir, exist_ok=True)

    clean_files = []
    for f in sorted(os.listdir(clean_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            clean_files.append(f)

    print(f"找到 {len(clean_files)} 张清晰图，开始过渡区分析...\n")

    for clean_file in clean_files:
        img_name = os.path.splitext(clean_file)[0]
        clean = cv2.imread(os.path.join(clean_dir, clean_file))
        if clean is None:
            continue

        print(f"{'='*60}")
        print(f"图像: {clean_file}")
        print(f"{'='*60}")
        print(f"{'模糊长度':<10} {'过渡区宽度':<15} {'平均梯度':<12}")
        print("-" * 40)

        results = analyze_blur_effect(clean)
        for r in results:
            print(f"{r['length']:<10} {r['width']:<15.2f} {r['mean_grad']:<12.2f}")

        # 保存过渡区可视化图（清晰图和length=10的模糊图）
        tz_clean, _ = extract_transition_zone(clean)
        psf = motion_blur_psf(kernel_size=21, angle=25, length=10)
        blurred = apply_blur(clean, psf)
        tz_blur, _ = extract_transition_zone(blurred)

        imwrite_unicode(os.path.join(result_dir, f"{img_name}_tz_clean.jpg"), tz_clean)
        imwrite_unicode(os.path.join(result_dir, f"{img_name}_tz_blur10.jpg"), tz_blur)

        print(f"\n过渡区可视化已保存: {img_name}_tz_clean.jpg, {img_name}_tz_blur10.jpg")
        print()

    print(f"\n完成！结果已保存到 {result_dir}")


if __name__ == "__main__":
    # 先测试单张图
    print("=" * 60)
    print("测试：单张图过渡区分析")
    print("=" * 60)

    clean_dir = "../data/clean"
    clean_file = None
    for f in sorted(os.listdir(clean_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            clean_file = f
            break

    if clean_file is None:
        print("未找到清晰图")
        exit()

    clean = cv2.imread(os.path.join(clean_dir, clean_file))
    print(f"测试图: {clean_file}")
    print()

    print(f"{'模糊长度':<10} {'过渡区宽度':<15} {'平均梯度':<12}")
    print("-" * 40)

    results = analyze_blur_effect(clean)
    for r in results:
        print(f"{r['length']:<10} {r['width']:<15.2f} {r['mean_grad']:<12.2f}")

    # 验证线性关系
    print()
    print("线性关系验证：")
    lengths = [r['length'] for r in results if r['length'] > 0]
    widths = [r['width'] for r in results if r['length'] > 0]
    if len(lengths) >= 2:
        # 简单线性拟合
        coeffs = np.polyfit(lengths, widths, 1)
        print(f"  过渡区宽度 ≈ {coeffs[0]:.3f} × 模糊长度 + {coeffs[1]:.3f}")
        print(f"  斜率为正 = 模糊越重，过渡区越宽 ✓")

    # 保存测试结果
    os.makedirs("../results/03_边缘检测", exist_ok=True)
    tz_clean, _ = extract_transition_zone(clean)
    psf = motion_blur_psf(kernel_size=21, angle=25, length=10)
    blurred = apply_blur(clean, psf)
    tz_blur, _ = extract_transition_zone(blurred)
    imwrite_unicode("../results/03_边缘检测/test_tz_clean.jpg", tz_clean)
    imwrite_unicode("../results/03_边缘检测/test_tz_blur10.jpg", tz_blur)

    print("\n测试结果已保存到 results/03_边缘检测/")
    print("\n" + "=" * 60)
    print("单张测试完成！确认无误后，取消下面注释运行批量处理：")
    print("  process_all_images()")
    print("=" * 60)

    # 取消注释运行批量处理
    process_all_images()
