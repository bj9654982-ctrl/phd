"""
第一阶段：图像退化模型
功能：生成各种PSF，用卷积合成模糊图
请在 TODO 处填写你的代码
"""

import numpy as np
import cv2


def motion_blur_psf(kernel_size=15, angle=0, length=10):
    """
    生成运动模糊PSF（线段形状）

    长度定义：线段两端点之间的几何距离（像素）。
    实现方法：8倍超采样分辨率画线段，再用8x8平均池化下采样，
    提供亚像素精度，所有角度一致，质心居中。

    修复历史：
    - v1: range(-length//2, length//2+1)，Python向下取整导致奇数长度多1个采样点
    - v2: int(round(x)) + cv2.line，"亚像素"声明不实，质心偏移0.5像素
    - v3: cv2.line shift参数，行为不稳定（整数端点时线段消失）
    - v4（当前）: 8倍超采样 + 平均池化下采样，稳定且准确

    参数:
        kernel_size: PSF的尺寸（奇数）
        angle: 运动方向角度（度），0=水平，90=垂直
        length: 运动长度（像素，几何端点距离，必须>0）

    返回:
        psf: kernel_size x kernel_size 的numpy数组，float32，元素和为1
    """
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    # 核支撑校验：线段最大投影长度为length（当angle=0或90时）
    # kernel_size必须足够容纳整个线段，否则会静默裁剪导致真值错误
    # 留2像素边距以容纳抗锯齿
    if length > kernel_size - 3:
        raise ValueError(
            f"kernel_size={kernel_size} too small for length={length}. "
            f"Need kernel_size >= length + 3 = {int(np.ceil(length)) + 3}. "
            f"Current would silently truncate the PSF."
        )

    supersample = 8  # 8倍超采样
    big_size = kernel_size * supersample
    big_psf = np.zeros((big_size, big_size), dtype=np.float32)
    big_center = big_size // 2

    rad = np.deg2rad(angle)
    half = length * supersample / 2.0

    # 在超采样分辨率上画线段（整数坐标，端点距离 = length * supersample）
    x1 = int(round(big_center - half * np.cos(rad)))
    y1 = int(round(big_center - half * np.sin(rad)))
    x2 = int(round(big_center + half * np.cos(rad)))
    y2 = int(round(big_center + half * np.sin(rad)))

    # LINE_8（无抗锯齿）在超采样分辨率上画，下采样后自然产生抗锯齿效果
    cv2.line(big_psf, (x1, y1), (x2, y2), 1.0, 1, lineType=cv2.LINE_8)

    # 8x8平均池化下采样回原始分辨率
    # reshape为 (kernel_size, 8, kernel_size, 8)，对(1,3)轴求平均
    psf = big_psf.reshape(kernel_size, supersample, kernel_size, supersample).mean(axis=(1, 3))

    # 归一化：所有元素之和为1（能量守恒）
    total = psf.sum()
    if total <= 0:
        # 极端情况（length极小导致线段完全落在一个像素内），退化为点扩散
        center = kernel_size // 2
        psf[center, center] = 1.0
    else:
        psf = psf / total

    return psf.astype(np.float32)


def defocus_blur_psf(kernel_size=15, radius=5):
    """
    生成散焦模糊PSF（均匀圆盘形状）

    参数:
        kernel_size: PSF的尺寸（奇数）
        radius: 圆盘半径（像素）

    返回:
        psf: kernel_size x kernel_size 的numpy数组，元素和为1
    """
    psf = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # TODO 2: 在psf上画一个半径为radius的圆盘
    # 思路：散焦模糊是镜头没对上焦，点光源变成均匀光斑（圆盘）
    # 用numpy向量化计算，比for循环快
    y_coords, x_coords = np.mgrid[0:kernel_size, 0:kernel_size]
    dist = np.sqrt((x_coords - center) ** 2 + (y_coords - center) ** 2)
    psf[dist <= radius] = 1.0


    psf = psf / psf.sum()
    return psf


def gaussian_blur_psf(kernel_size=15, sigma=3.0):
    """
    生成高斯模糊PSF

    参数:
        kernel_size: PSF的尺寸（奇数）
        sigma: 高斯标准差

    返回:
        psf: kernel_size x kernel_size 的numpy数组，元素和为1
    """
    psf = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # TODO 3: 生成二维高斯分布
    # 思路：高斯模糊是最常见的平滑，PSF是二维高斯分布（中心亮，四周指数衰减）
    y_coords, x_coords = np.mgrid[0:kernel_size, 0:kernel_size]
    # 二维高斯公式：G(x,y) = exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
    psf = np.exp(-((x_coords - center) ** 2 + (y_coords - center) ** 2) / (2 * sigma ** 2))


    psf = psf / psf.sum()
    return psf


def apply_blur(image, psf):
    """
    对图像施加模糊（卷积）

    参数:
        image: 清晰图像 (H x W 或 H x W x C)，uint8
        psf: 模糊核，元素和为1

    返回:
        blurred: 模糊图像，uint8
    """
    # 使用OpenCV的filter2D做卷积
    # borderType=cv2.BORDER_REFLECT 用于减少边界伪影
    blurred = cv2.filter2D(image, -1, psf, borderType=cv2.BORDER_REFLECT)
    return blurred


def add_gaussian_noise(image, sigma=0.01, seed=None):
    """
    给图像加高斯噪声

    参数:
        image: 输入图像，uint8
        sigma: 噪声标准差（相对于[0,1]范围）。例如8-bit DN=10对应sigma=10/255≈0.0392
        seed: 随机种子。可以是：
              - int: 每次调用创建新的Generator（可复现但每次相同）
              - numpy.random.Generator: 复用已有Generator，连续调用产生独立噪声
              - None: 使用全局随机状态

    返回:
        noisy: 加噪图像，uint8
    """
    # 归一化到[0,1]
    img_float = image.astype(np.float32) / 255.0
    # 根据seed类型选择随机数生成器
    if isinstance(seed, np.random.Generator):
        rng = seed
    elif seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random
    noise = rng.normal(0, sigma, img_float.shape).astype(np.float32)
    noisy = img_float + noise
    # 裁剪到[0,1]，防止像素值越界
    noisy = np.clip(noisy, 0.0, 1.0)

    noisy = (noisy * 255).astype(np.uint8)
    return noisy


def save_psf_visualization(psf, save_path):
    """
    把PSF可视化保存为图片（放大后便于观察）
    """
    # 归一化到0-255
    psf_vis = (psf / psf.max() * 255).astype(np.uint8)
    # 放大10倍
    psf_vis = cv2.resize(psf_vis, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(save_path, psf_vis)
    print(f"PSF可视化已保存: {save_path}")


def process_one_image(img_path, clean_dir):
    """处理单张图片：施加4种模糊，保存结果"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  跳过（无法读取）: {img_path}")
        return

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"  处理: {img_name} ({img.shape[1]}x{img.shape[0]})")

    # 1. 运动模糊
    psf_motion = motion_blur_psf(kernel_size=21, angle=25, length=10)
    blurred_motion = apply_blur(img, psf_motion)
    cv2.imwrite(f"../data/synthetic_blur/{img_name}_motion.jpg", blurred_motion)

    # 2. 散焦模糊
    psf_defocus = defocus_blur_psf(kernel_size=21, radius=5)
    blurred_defocus = apply_blur(img, psf_defocus)
    cv2.imwrite(f"../data/synthetic_blur/{img_name}_defocus.jpg", blurred_defocus)

    # 3. 高斯模糊
    psf_gaussian = gaussian_blur_psf(kernel_size=21, sigma=3.0)
    blurred_gaussian = apply_blur(img, psf_gaussian)
    cv2.imwrite(f"../data/synthetic_blur/{img_name}_gaussian.jpg", blurred_gaussian)

    # 4. 运动模糊+噪声
    blurred_noisy = add_gaussian_noise(blurred_motion, sigma=0.02)
    cv2.imwrite(f"../data/synthetic_blur/{img_name}_motion_noisy.jpg", blurred_noisy)


if __name__ == "__main__":
    # 批量处理 data/clean/ 目录下所有清晰图片
    import os
    clean_dir = "../data/clean"

    # 收集所有图片
    image_files = []
    if os.path.isdir(clean_dir):
        for f in sorted(os.listdir(clean_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(clean_dir, f))

    if not image_files:
        print(f"未在 {clean_dir} 找到图片")
        print("请先放清晰图（jpg/png）到 data/clean/ 目录下")
        exit()

    print(f"找到 {len(image_files)} 张清晰图，开始批量处理...\n")

    # PSF只需要生成一次可视化（所有图用相同PSF参数）
    psf_motion = motion_blur_psf(kernel_size=21, angle=25, length=10)
    psf_defocus = defocus_blur_psf(kernel_size=21, radius=5)
    psf_gaussian = gaussian_blur_psf(kernel_size=21, sigma=3.0)
    save_psf_visualization(psf_motion, "../results/psf_motion.png")
    save_psf_visualization(psf_defocus, "../results/psf_defocus.png")
    save_psf_visualization(psf_gaussian, "../results/psf_gaussian.png")

    # 批量处理每张图
    for img_path in image_files:
        process_one_image(img_path, clean_dir)

    print(f"\n完成！共处理 {len(image_files)} 张图，模糊图已保存到 data/synthetic_blur/")
    print("PSF可视化已保存到 results/")
