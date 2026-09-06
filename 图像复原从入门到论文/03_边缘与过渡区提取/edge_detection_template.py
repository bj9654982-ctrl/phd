"""
第三阶段：边缘检测
功能：实现Sobel、Canny、Laplacian三种边缘检测，对比清晰图和模糊图的边缘差异
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


def sobel_edge(image, ksize=3):
    """
    Sobel边缘检测

    参数:
        image: 输入图像，uint8（灰度或彩色）
        ksize: Sobel核大小，必须是奇数，默认3

    返回:
        edges: 边缘强度图，uint8（0-255，值越大边缘越强）
    """
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 转float，避免溢出
    gray_f = gray.astype(np.float64)

    # TODO 1: 计算水平方向梯度 Gx
    # cv2.Sobel的参数：(图像, 输出深度, x方向导数阶数, y方向导数阶数, 核大小)
    # dx=1, dy=0 表示只对x方向求一阶导数，检测垂直边缘
    gx = cv2.Sobel(gray_f, cv2.CV_64F, dx=1, dy=0, ksize=ksize)

    # TODO 2: 计算垂直方向梯度 Gy
    # dx=0, dy=1 表示只对y方向求一阶导数，检测水平边缘
    gy = cv2.Sobel(gray_f, cv2.CV_64F, dx=0, dy=1, ksize=ksize)

    # TODO 3: 计算梯度幅值 G = sqrt(Gx^2 + Gy^2)
    # 梯度幅值表示灰度变化的剧烈程度，值越大越可能是边缘
    magnitude = np.sqrt(gx**2 + gy**2)

    # 归一化到0-255：除以最大值，乘以255
    # 加1e-8防止除以零（全黑图时最大值为0）
    mag_max = np.max(magnitude)
    if mag_max > 1e-8:
        edges = (magnitude / mag_max * 255).astype(np.uint8)
    else:
        edges = np.zeros_like(magnitude, dtype=np.uint8)

    return edges


def canny_edge(image, low_threshold=50, high_threshold=150):
    """
    Canny边缘检测（调用OpenCV实现）

    参数:
        image: 输入图像，uint8
        low_threshold: 低阈值
        high_threshold: 高阈值（通常是低阈值的2-3倍）

    返回:
        edges: 二值边缘图，uint8（0或255）
    """
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Canny边缘检测（OpenCV已实现，直接调用）
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    return edges


def laplacian_edge(image, ksize=3):
    """
    Laplacian边缘检测（二阶导数，找零交叉点）

    参数:
        image: 输入图像，uint8
        ksize: Laplacian核大小

    返回:
        zero_crossing: 零交叉点图，uint8（零交叉点为255，其他为0）
    """
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float64)

    # TODO 4: 计算Laplacian（二阶导数）
    # Laplacian是二阶导数，在边缘中心处二阶导数过零（零交叉点）
    # 平坦区域Laplacian≈0，边缘一侧为正、另一侧为负
    laplacian = cv2.Laplacian(gray_f, cv2.CV_64F, ksize=ksize)

    # TODO 5: 找零交叉点（Zero-Crossing）
    # 原理：Laplacian在边缘中心处从正变负（或负变正），
    #       如果一个像素和它的邻居符号相反（乘积<0），就是零交叉点。
    # 方法：检查4个方向（右、下、右下、左下）的邻居，任一方向符号相反即标记。

    h, w = laplacian.shape
    zero_crossing = np.zeros((h, w), dtype=np.float64)

    # 方向1：和右边邻居比较（水平方向）
    sign_change_h = laplacian[:, :-1] * laplacian[:, 1:] < 0
    zero_crossing[:, :-1] = np.maximum(zero_crossing[:, :-1], sign_change_h)
    zero_crossing[:, 1:] = np.maximum(zero_crossing[:, 1:], sign_change_h)

    # 方向2：和下边邻居比较（垂直方向）
    sign_change_v = laplacian[:-1, :] * laplacian[1:, :] < 0
    zero_crossing[:-1, :] = np.maximum(zero_crossing[:-1, :], sign_change_v)
    zero_crossing[1:, :] = np.maximum(zero_crossing[1:, :], sign_change_v)

    # 方向3：和右下邻居比较（对角线）
    sign_change_d1 = laplacian[:-1, :-1] * laplacian[1:, 1:] < 0
    zero_crossing[:-1, :-1] = np.maximum(zero_crossing[:-1, :-1], sign_change_d1)
    zero_crossing[1:, 1:] = np.maximum(zero_crossing[1:, 1:], sign_change_d1)

    # 方向4：和左下邻居比较（反对角线）
    sign_change_d2 = laplacian[:-1, 1:] * laplacian[1:, :-1] < 0
    zero_crossing[:-1, 1:] = np.maximum(zero_crossing[:-1, 1:], sign_change_d2)
    zero_crossing[1:, :-1] = np.maximum(zero_crossing[1:, :-1], sign_change_d2)

    # 转成0/255的二值图
    zero_crossing = (zero_crossing > 0).astype(np.uint8) * 255

    return zero_crossing


def process_all_images():
    """批量处理：对清晰图和不同程度的模糊图做边缘检测，对比效果"""

    clean_dir = "../data/clean"
    result_dir = "../results/03_边缘检测"
    os.makedirs(result_dir, exist_ok=True)

    # 收集清晰图
    clean_files = []
    for f in sorted(os.listdir(clean_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            clean_files.append(f)

    # 模糊程度：运动长度5/10/15
    blur_levels = [5, 10, 15]

    print(f"找到 {len(clean_files)} 张清晰图，开始边缘检测对比...\n")

    for clean_file in clean_files:
        img_name = os.path.splitext(clean_file)[0]
        clean = cv2.imread(os.path.join(clean_dir, clean_file))
        if clean is None:
            continue

        print(f"{'='*60}")
        print(f"图像: {clean_file}")
        print(f"{'='*60}")

        # 清晰图的边缘检测
        sobel_clean = sobel_edge(clean)
        canny_clean = canny_edge(clean)
        lap_clean = laplacian_edge(clean)

        # 保存清晰图边缘
        imwrite_unicode(os.path.join(result_dir, f"{img_name}_clean_sobel.jpg"), sobel_clean)
        imwrite_unicode(os.path.join(result_dir, f"{img_name}_clean_canny.jpg"), canny_clean)
        imwrite_unicode(os.path.join(result_dir, f"{img_name}_clean_laplacian.jpg"), lap_clean)

        print(f"  清晰图: Sobel/Canny/Laplacian 已保存")

        # 不同模糊程度的边缘检测
        for length in blur_levels:
            psf = motion_blur_psf(kernel_size=21, angle=25, length=length)
            blurred = apply_blur(clean, psf)

            sobel_blur = sobel_edge(blurred)
            canny_blur = canny_edge(blurred)
            lap_blur = laplacian_edge(blurred)

            # 保存模糊图边缘
            imwrite_unicode(os.path.join(result_dir, f"{img_name}_blur{length}_sobel.jpg"), sobel_blur)
            imwrite_unicode(os.path.join(result_dir, f"{img_name}_blur{length}_canny.jpg"), canny_blur)
            imwrite_unicode(os.path.join(result_dir, f"{img_name}_blur{length}_laplacian.jpg"), lap_blur)

            print(f"  模糊(length={length}): Sobel/Canny/Laplacian 已保存")

        print()

    print(f"\n完成！边缘检测结果已保存到 {result_dir}")


if __name__ == "__main__":
    # 先测试单张图，确认代码正确
    print("=" * 60)
    print("测试：单张图边缘检测")
    print("=" * 60)

    # 找一张清晰图
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

    # 清晰图边缘检测
    print("Sobel边缘检测中...")
    sobel_clean = sobel_edge(clean)
    print("Canny边缘检测中...")
    canny_clean = canny_edge(clean)
    print("Laplacian边缘检测中...")
    lap_clean = laplacian_edge(clean)

    # 模糊图边缘检测
    psf = motion_blur_psf(kernel_size=21, angle=25, length=10)
    blurred = apply_blur(clean, psf)
    print("模糊图Sobel边缘检测中...")
    sobel_blur = sobel_edge(blurred)
    print("模糊图Canny边缘检测中...")
    canny_blur = canny_edge(blurred)
    print("模糊图Laplacian边缘检测中...")
    lap_blur = laplacian_edge(blurred)

    # 保存测试结果
    os.makedirs("../results/03_边缘检测", exist_ok=True)
    imwrite_unicode("../results/03_边缘检测/test_clean_sobel.jpg", sobel_clean)
    imwrite_unicode("../results/03_边缘检测/test_clean_canny.jpg", canny_clean)
    imwrite_unicode("../results/03_边缘检测/test_clean_laplacian.jpg", lap_clean)
    imwrite_unicode("../results/03_边缘检测/test_blur_sobel.jpg", sobel_blur)
    imwrite_unicode("../results/03_边缘检测/test_blur_canny.jpg", canny_blur)
    imwrite_unicode("../results/03_边缘检测/test_blur_laplacian.jpg", lap_blur)

    print("\n测试结果已保存到 results/03_边缘检测/")
    print("\n" + "=" * 60)
    print("单张测试完成！确认无误后，取消下面注释运行批量处理：")
    print("  process_all_images()")
    print("=" * 60)

    # 取消注释运行批量处理
    process_all_images()
