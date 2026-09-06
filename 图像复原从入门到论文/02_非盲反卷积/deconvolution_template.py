"""
第二阶段：非盲反卷积
功能：已知PSF时，用维纳滤波、RL反卷积、TV正则化复原模糊图
请在 TODO 处填写你的代码
"""

import numpy as np
import cv2
import os
import sys

# 导入第一阶段的评价指标
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_退化模型与评价指标"))
from metrics_template import evaluate_restoration

# 导入第一阶段的PSF生成函数
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_退化模型与评价指标"))
from degradation_template import motion_blur_psf, defocus_blur_psf, gaussian_blur_psf


def imwrite_unicode(path, img):
    """
    兼容中文路径的图片保存函数
    cv2.imwrite在Windows上对中文路径支持不好（会静默失败），
    所以用cv2.imencode + 文件写入的方式替代。
    """
    ext = os.path.splitext(path)[1]
    if ext.lower() == '.jpg':
        ext = '.jpg'
    result, encoded = cv2.imencode(ext, img)
    if result:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def wiener_filter(blurred, psf, K=0.02):
    """
    维纳滤波反卷积

    参数:
        blurred: 模糊图像 (H x W 或 H x W x C)，uint8
        psf: 模糊核，元素和为1
        K: 噪声信号比（正则化参数），0.001~0.05

    返回:
        restored: 复原图像，uint8
    """
    # 转float
    img = blurred.astype(np.float64) / 255.0

    # 处理彩色图：逐通道处理
    if len(img.shape) == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_wiener_single_channel(img[:, :, c], psf, K))
        restored = np.stack(channels, axis=2)
    else:
        restored = _wiener_single_channel(img, psf, K)

    # 裁剪到[0,1]，转uint8
    restored = np.clip(restored, 0, 1)
    return (restored * 255).astype(np.uint8)


def _wiener_single_channel(img, psf, K):
    """
    单通道维纳滤波（核心实现）

    参数:
        img: 单通道模糊图，float [0,1]
        psf: 模糊核
        K: 正则化参数

    返回:
        restored: 复原图，float [0,1]
    """
    h, w = img.shape
    kh, kw = psf.shape

    # 关键修复1：cv2.filter2D做的是相关(correlation)，不是卷积(convolution)
    # 对于非对称PSF（如运动模糊），相关等价于用翻转后的PSF做卷积
    # 维纳滤波公式假设的是卷积模型，所以这里要把PSF翻转，匹配cv2.filter2D的相关运算
    psf = psf[::-1, ::-1]

    # 关键修复2：反射边界扩展，消除FFT周期性假设导致的边界振铃条纹
    # FFT默认图像无限周期重复，左右/上下边界直接拼接会产生不连续
    # 用反射方式扩展图像边缘，让边界连续，复原后再裁掉扩展区域
    pad = max(kh, kw)  # 扩展量取PSF大小，足够消除边界影响
    img_padded = np.pad(img, pad_width=pad, mode='reflect')
    ph, pw = img_padded.shape

    # TODO 1: 把PSF填充到和扩展后图像一样大（中心对齐，周围补零）
    # 思路：FFT默认原点在左上角，所以PSF中心要先移到左上角（ifftshift）
    psf_padded = np.zeros((ph, pw), dtype=np.float64)
    # 计算PSF在大图中的起始位置
    start_h = ph // 2 - kh // 2
    start_w = pw // 2 - kw // 2
    # 把PSF放到中心
    psf_padded[start_h:start_h + kh, start_w:start_w + kw] = psf
    # 关键：把PSF中心从图像中心移到左上角，匹配FFT的原点约定
    psf_padded = np.fft.ifftshift(psf_padded)

    # TODO 2: 对图像和PSF做FFT
    # 卷积定理：空域卷积 = 频域乘法，所以转到频域计算更快
    img_fft = np.fft.fft2(img_padded)
    psf_fft = np.fft.fft2(psf_padded)

    # TODO 3: 维纳滤波公式
    # 公式：F_hat = (H* / (|H|^2 + K)) * G
    # 其中 H*是复共轭，|H|^2是功率谱，K是噪声信号比（正则化参数）
    H_conj = np.conj(psf_fft)           # 复共轭
    H_abs_sq = np.abs(psf_fft) ** 2     # 功率谱 |H|^2
    Wiener = H_conj / (H_abs_sq + K)     # 维纳滤波器
    restored_fft = Wiener * img_fft       # 频域复原

    # TODO 4: 逆FFT回到空域，取实部，裁掉边界扩展区域
    # 由于之前用了ifftshift把PSF中心移到左上角，这里逆FFT后不需要再shift
    restored_padded = np.fft.ifft2(restored_fft)
    restored_padded = np.real(restored_padded)  # 取实部，去掉浮点误差产生的虚部
    # 裁掉四周扩展的pad像素，恢复原始尺寸
    restored = restored_padded[pad:pad + h, pad:pad + w]

    return restored


def richardson_lucy(blurred, psf, iterations=30):
    """
    Richardson-Lucy 反卷积

    参数:
        blurred: 模糊图像，uint8
        psf: 模糊核，元素和为1
        iterations: 迭代次数

    返回:
        restored: 复原图像，uint8
    """
    img = blurred.astype(np.float64) / 255.0

    if len(img.shape) == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_rl_single_channel(img[:, :, c], psf, iterations))
        restored = np.stack(channels, axis=2)
    else:
        restored = _rl_single_channel(img, psf, iterations)

    restored = np.clip(restored, 0, 1)
    return (restored * 255).astype(np.uint8)


def _rl_single_channel(img, psf, iterations):
    """
    单通道RL反卷积（核心实现）

    参数:
        img: 单通道模糊图，float [0,1]
        psf: 模糊核
        iterations: 迭代次数

    返回:
        restored: 复原图，float [0,1]
    """
    # 翻转PSF（用于相关运算）
    psf_flip = psf[::-1, ::-1]

    # TODO 5: 初始化估计值
    # RL反卷积是迭代方法，需要一个初始估计。用模糊图本身作为初始值最常见，
    # 因为它已经包含了图像的大致结构，迭代收敛更快。
    estimate = img.copy()

    for i in range(iterations):
        # TODO 6: RL迭代核心公式
        # RL反卷积基于最大似然估计，假设噪声服从泊松分布
        # 迭代公式：estimate = estimate * [(img / (estimate * psf)) * psf_flip]
        # 分为4步：

        # 第1步：用当前估计和PSF卷积，得到"预测的模糊图"
        est_conv = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)

        # 第2步：计算相对误差 = 真实模糊图 / 预测模糊图
        # 加1e-8防止除以零
        relative = img / (est_conv + 1e-8)

        # 第3步：相对误差和翻转后的PSF做卷积（相关运算）
        # 翻转PSF是因为RL迭代中需要做互相关而不是卷积
        correction = cv2.filter2D(relative, -1, psf_flip, borderType=cv2.BORDER_REFLECT)

        # 第4步：用修正量更新估计（乘法形式，天然保证非负）
        estimate = estimate * correction

    return estimate


def tv_deconvolution(blurred, psf, weight=0.002):
    """
    总变分（TV）正则化反卷积
    策略：先用我们自己的RL反卷积恢复细节，再用TV去噪抑制振铃
    （scikit-image的RL用卷积模型，和我们cv2.filter2D的相关运算不匹配，所以不用它的RL）

    参数:
        blurred: 模糊图像，uint8
        psf: 模糊核
        weight: TV正则化权重，0.001~0.01，越大越平滑

    返回:
        restored: 复原图像，uint8
    """
    # 第一步：用我们自己的RL反卷积（已验证和模糊生成方式自洽，效果好）
    rl_result = richardson_lucy(blurred, psf, iterations=30)

    try:
        from skimage.restoration import denoise_tv_chambolle
        # 第二步：用scikit-image的专业TV去噪抑制振铃
        img_f = rl_result.astype(np.float64) / 255.0
        if len(img_f.shape) == 3:
            restored = np.zeros_like(img_f)
            for c in range(img_f.shape[2]):
                restored[:, :, c] = denoise_tv_chambolle(img_f[:, :, c], weight=weight)
        else:
            restored = denoise_tv_chambolle(img_f, weight=weight)
        restored = np.clip(restored, 0, 1)
        return (restored * 255).astype(np.uint8)
    except ImportError:
        print("scikit-image未安装，直接返回RL结果（无TV后处理）")
        return rl_result


def _tv_simple(blurred, psf, weight=0.1, iterations=100):
    """
    简化版TV反卷积（梯度下降），不需要scikit-image
    注意：这只是教学用的简化实现，效果不如专业优化算法
    """
    img = blurred.astype(np.float64) / 255.0

    if len(img.shape) == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_tv_simple_single(img[:, :, c], psf, weight, iterations))
        restored = np.stack(channels, axis=2)
    else:
        restored = _tv_simple_single(img, psf, weight, iterations)

    restored = np.clip(restored, 0, 1)
    return (restored * 255).astype(np.uint8)


def _tv_simple_single(img, psf, weight, iterations):
    """单通道简化TV"""
    estimate = img.copy()
    lr = 0.01  # 学习率

    for i in range(iterations):
        # 数据保真项梯度：(estimate * psf - img) * psf_flip
        est_conv = cv2.filter2D(estimate, -1, psf)
        data_grad = cv2.filter2D(est_conv - img, -1, psf[::-1, ::-1])

        # TV正则项梯度（简化：用梯度的L1近似）
        grad_x = np.diff(estimate, axis=1, append=estimate[:, -1:])
        grad_y = np.diff(estimate, axis=0, append=estimate[-1:, :])
        tv_grad = np.zeros_like(estimate)
        tv_grad[:, :-1] -= np.sign(grad_x[:, :-1])
        tv_grad[:, 1:] += np.sign(grad_x[:, :-1])
        tv_grad[:-1, :] -= np.sign(grad_y[:-1, :])
        tv_grad[1:, :] += np.sign(grad_y[:-1, :])

        # 梯度下降更新
        estimate = estimate - lr * (data_grad + weight * tv_grad)
        estimate = np.clip(estimate, 0, 1)

    return estimate


def process_all_images():
    """批量处理：对第一阶段生成的所有模糊图做复原，计算评价指标"""

    clean_dir = "../data/clean"
    blur_dir = "../data/synthetic_blur"
    result_dir = "../results/02_非盲反卷积"
    os.makedirs(result_dir, exist_ok=True)

    # 收集清晰图
    clean_files = []
    for f in sorted(os.listdir(clean_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            clean_files.append(f)

    # 模糊类型和对应的PSF参数
    blur_configs = [
        ('motion', motion_blur_psf(kernel_size=21, angle=25, length=10)),
        ('defocus', defocus_blur_psf(kernel_size=21, radius=5)),
        ('gaussian', gaussian_blur_psf(kernel_size=21, sigma=3.0)),
    ]

    methods = ['wiener', 'rl', 'tv']

    print(f"找到 {len(clean_files)} 张清晰图，开始批量复原...\n")

    all_results = []

    for clean_file in clean_files:
        img_name = os.path.splitext(clean_file)[0]
        clean_path = os.path.join(clean_dir, clean_file)
        clean = cv2.imread(clean_path)
        if clean is None:
            continue

        print(f"{'='*70}")
        print(f"清晰图: {clean_file}")
        print(f"{'='*70}")
        print(f"{'模糊类型':<10} {'方法':<8} {'PSNR(dB)':<12} {'SSIM':<10} {'MSE':<10}")
        print("-" * 60)

        for blur_type, psf in blur_configs:
            blur_path = os.path.join(blur_dir, f"{img_name}_{blur_type}.jpg")
            blurred = cv2.imread(blur_path)
            if blurred is None:
                continue

            for method in methods:
                # 复原
                if method == 'wiener':
                    restored = wiener_filter(blurred, psf, K=0.01)
                elif method == 'rl':
                    restored = richardson_lucy(blurred, psf, iterations=30)
                elif method == 'tv':
                    restored = tv_deconvolution(blurred, psf, weight=0.002)

                # 计算评价指标
                metrics = evaluate_restoration(restored, clean)
                print(f"{blur_type:<10} {method:<8} {metrics['PSNR']:<12} {metrics['SSIM']:<10} {metrics['MSE']:<10}")

                # 保存结果
                save_path = os.path.join(result_dir, f"{img_name}_{blur_type}_{method}.jpg")
                imwrite_unicode(save_path, restored)

                all_results.append({
                    'image': img_name,
                    'blur': blur_type,
                    'method': method,
                    **metrics
                })

        print()

    print(f"\n完成！复原结果已保存到 {result_dir}")
    return all_results


if __name__ == "__main__":
    # 先测试单张图，确认代码正确
    print("=" * 60)
    print("测试：单张图复原对比")
    print("=" * 60)

    # 找一张清晰图
    clean_dir = "../data/clean"
    clean_file = None
    for f in sorted(os.listdir(clean_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            clean_file = f
            break

    if clean_file is None:
        print("未找到清晰图，请先运行第一阶段代码")
        exit()

    clean = cv2.imread(os.path.join(clean_dir, clean_file))
    psf = motion_blur_psf(kernel_size=21, angle=25, length=10)
    blurred = cv2.imread(f"../data/synthetic_blur/{os.path.splitext(clean_file)[0]}_motion.jpg")

    if blurred is None:
        print("未找到模糊图，请先运行第一阶段代码")
        exit()

    print(f"测试图: {clean_file}")
    print(f"模糊类型: 运动模糊 (angle=25, length=10)")
    print()

    # 模糊图的基线指标
    baseline = evaluate_restoration(blurred, clean)
    print(f"{'模糊图(基线)':<15} PSNR={baseline['PSNR']:<8} SSIM={baseline['SSIM']:<8}")
    print()

    # 三种方法复原
    print("维纳滤波复原中...")
    restored_wiener = wiener_filter(blurred, psf, K=0.01)
    m_wiener = evaluate_restoration(restored_wiener, clean)
    print(f"{'维纳滤波':<15} PSNR={m_wiener['PSNR']:<8} SSIM={m_wiener['SSIM']:<8}")

    print("RL反卷积复原中...")
    restored_rl = richardson_lucy(blurred, psf, iterations=30)
    m_rl = evaluate_restoration(restored_rl, clean)
    print(f"{'RL反卷积':<15} PSNR={m_rl['PSNR']:<8} SSIM={m_rl['SSIM']:<8}")

    print("TV正则化复原中...")
    restored_tv = tv_deconvolution(blurred, psf, weight=0.002)
    m_tv = evaluate_restoration(restored_tv, clean)
    print(f"{'TV正则化':<15} PSNR={m_tv['PSNR']:<8} SSIM={m_tv['SSIM']:<8}")

    # 保存测试结果
    os.makedirs("../results/02_非盲反卷积", exist_ok=True)
    imwrite_unicode("../results/02_非盲反卷积/test_blurred.jpg", blurred)
    imwrite_unicode("../results/02_非盲反卷积/test_wiener.jpg", restored_wiener)
    imwrite_unicode("../results/02_非盲反卷积/test_rl.jpg", restored_rl)
    imwrite_unicode("../results/02_非盲反卷积/test_tv.jpg", restored_tv)

    print("\n测试结果已保存到 results/02_非盲反卷积/")
    print("\n" + "=" * 60)
    print("单张测试完成！确认无误后，取消下面注释运行批量处理：")
    print("  process_all_images()")
    print("=" * 60)

    # 取消注释运行批量处理
    # process_all_images()
