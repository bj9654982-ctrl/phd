"""
第五阶段：完整盲复原流程
功能：端到端实现盲图像复原——输入模糊图（PSF未知），输出复原图像
     流程：模糊图 → 过渡区提取 → PSF估计 → 反卷积 → 复原图
请在 TODO 处填写你的代码
"""

import numpy as np
import cv2
import os
import sys

# 导入第一阶段的PSF生成、模糊、评价指标
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_退化模型与评价指标"))
from degradation_template import motion_blur_psf, apply_blur
from metrics_template import evaluate_restoration

# 导入第三阶段的过渡区提取
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "03_边缘与过渡区提取"))
from transition_zone_template import compute_mean_gradient

# 导入第二阶段的反卷积
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_非盲反卷积"))
from deconvolution_template import wiener_filter, richardson_lucy


def imwrite_unicode(path, img):
    """兼容中文路径的图片保存"""
    ext = os.path.splitext(path)[1]
    if ext.lower() == '.jpg':
        ext = '.jpg'
    result, encoded = cv2.imencode(ext, img)
    if result:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def compute_self_normalized_grad(blurred_image, gradient_ratio=0.05):
    """
    计算自归一化梯度：过渡区平均梯度 / 过渡区最大梯度

    优点：只依赖模糊图自身，不需要估计清晰图基线，避免了基线估计误差。
    物理意义：过渡区的平均陡峭程度相对于最锐利边缘的比例。
             模糊越重，平均梯度下降越多，最大梯度下降较少，比值越小。

    参数:
        blurred_image: 模糊图
        gradient_ratio: 过渡区梯度阈值比例

    返回:
        normalized_grad: 自归一化梯度（0~1之间）
    """
    from transition_zone_template import extract_transition_zone

    # 提取过渡区掩码和梯度图
    tz_mask, gradient = extract_transition_zone(blurred_image, gradient_ratio)

    # 取过渡区内的梯度值
    tz_grads = gradient[tz_mask > 0]

    if len(tz_grads) == 0:
        return 0.0

    mean_grad = np.mean(tz_grads)
    max_grad = np.max(tz_grads)

    if max_grad > 0:
        normalized_grad = mean_grad / max_grad
    else:
        normalized_grad = 0.0

    return normalized_grad


def build_reference_table_selfnorm(clean_images, lengths=range(1, 26), angle=25, gradient_ratio=0.05):
    """
    建立自归一化梯度查找表：用"过渡区平均梯度/过渡区最大梯度"建立映射

    自归一化梯度只依赖模糊图自身，不需要清晰图，避免了基线估计误差。

    参数:
        clean_images: 清晰图列表（用于生成不同长度的模糊图）
        lengths: 运动长度列表
        angle: 运动角度
        gradient_ratio: 梯度阈值比例

    返回:
        table: 字典，key=运动长度，value=自归一化梯度（多张图平均）
    """
    print(f"建立自归一化梯度查找表（{len(clean_images)}张图平均，运动长度{min(lengths)}~{max(lengths)}）...")

    # 收集每张图在每个长度下的自归一化梯度
    sn_grad_collect = {L: [] for L in lengths}

    for img in clean_images:
        for length in lengths:
            kernel_size = max(21, length * 2 + 1)
            psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=length)
            blurred = apply_blur(img, psf)

            # 自归一化梯度 = 过渡区平均梯度 / 过渡区最大梯度
            sn_grad = compute_self_normalized_grad(blurred, gradient_ratio)
            sn_grad_collect[length].append(sn_grad)

    # 对每个长度，计算多张图的平均自归一化梯度
    table = {}
    for length in lengths:
        table[length] = np.mean(sn_grad_collect[length])
        if length % 5 == 0:
            print(f"  length={length:2d}: 自归一化梯度={table[length]:.4f}")

    print(f"自归一化梯度查找表建立完成，共 {len(table)} 个条目")
    return table


def blind_deblur_selfnorm(blurred_image, lookup_table, angle=25, gradient_ratio=0.05, method='wiener'):
    """
    用自归一化梯度的完整盲复原函数

    流程：
        1. 计算模糊图的自归一化梯度（平均/最大）
        2. 用自归一化梯度查查找表，估计运动长度
        3. 生成PSF，反卷积

    参数:
        blurred_image: 输入模糊图
        lookup_table: 自归一化梯度查找表
        angle: 运动角度
        gradient_ratio: 梯度阈值比例
        method: 反卷积方法

    返回:
        restored: 复原图像
        estimated_length: 估计的运动长度
        sn_grad: 自归一化梯度值
    """
    # 1. 计算自归一化梯度
    sn_grad = compute_self_normalized_grad(blurred_image, gradient_ratio)

    # 2. 用自归一化梯度查查找表，估计运动长度
    estimated_length = estimate_length_from_grad(sn_grad, lookup_table)

    # 3. 生成PSF，反卷积（8倍超采样PSF支持浮点长度）
    kernel_size = max(21, int(np.ceil(estimated_length)) * 2 + 1)
    psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=estimated_length)

    if method == 'wiener':
        restored = wiener_filter(blurred_image, psf, K=0.02)
    elif method == 'rl':
        restored = richardson_lucy(blurred_image, psf, iterations=30)
    else:
        restored = wiener_filter(blurred_image, psf, K=0.02)

    return restored, estimated_length, sn_grad


def build_reference_table(clean_images, lengths=range(1, 26), angle=25, gradient_ratio=0.05):
    """
    建立通用查找表：用多张清晰图的平均特征建立"平均梯度 → 运动长度"映射

    参数:
        clean_images: 清晰图列表（多张图，用于平均）
        lengths: 运动长度列表
        angle: 运动角度（固定）
        gradient_ratio: 梯度阈值比例

    返回:
        table: 字典，key=运动长度，value=平均梯度（多张图平均）
    """
    print(f"建立通用查找表（{len(clean_images)}张图平均，运动长度{min(lengths)}~{max(lengths)}）...")

    # 收集每张图在每个长度下的平均梯度
    grad_collect = {L: [] for L in lengths}

    for img in clean_images:
        for length in lengths:
            kernel_size = max(21, length * 2 + 1)
            psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=length)
            blurred = apply_blur(img, psf)
            mean_grad = compute_mean_gradient(blurred, gradient_ratio)
            grad_collect[length].append(mean_grad)

    # 对每个长度，计算多张图的平均梯度
    table = {}
    for length in lengths:
        table[length] = np.mean(grad_collect[length])
        if length % 5 == 0:
            print(f"  length={length:2d}: 平均梯度={table[length]:.2f}")

    print(f"通用查找表建立完成，共 {len(table)} 个条目")
    return table


def estimate_length_from_grad(measured_grad, lookup_table, verbose=False):
    """
    从测量的平均梯度估计运动长度（线性插值）

    注意：此方法假定平均梯度随长度单调递减。若序列非单调，线性插值可能失效，
    会自动回退到最近邻估计。

    参数:
        measured_grad: 模糊图的平均梯度
        lookup_table: 查找表（key=长度，value=平均梯度）
        verbose: 是否打印警告

    返回:
        estimated_length: 估计的运动长度（float）
    """
    # 按长度排序
    sorted_lengths = sorted(lookup_table.keys())
    sorted_grads = [lookup_table[L] for L in sorted_lengths]

    # 单调性检查
    is_mono = all(sorted_grads[i] >= sorted_grads[i + 1] for i in range(len(sorted_grads) - 1))
    if not is_mono and verbose:
        violations = sum(1 for i in range(len(sorted_grads) - 1) if sorted_grads[i] < sorted_grads[i + 1])
        print(f"  [警告] 平均梯度序列非单调递减，违反{violations}处")

    # 处理边界情况
    if measured_grad >= sorted_grads[0]:
        return float(sorted_lengths[0])
    elif measured_grad <= sorted_grads[-1]:
        return float(sorted_lengths[-1])

    # 线性插值（要求降序区间）
    for i in range(len(sorted_grads) - 1):
        if sorted_grads[i + 1] <= measured_grad <= sorted_grads[i] and sorted_grads[i] > sorted_grads[i + 1]:
            g0, g1 = sorted_grads[i], sorted_grads[i + 1]
            L0, L1 = sorted_lengths[i], sorted_lengths[i + 1]
            ratio = (g0 - measured_grad) / (g0 - g1)
            return L0 + ratio * (L1 - L0)

    # 未找到有效区间（非单调），回退到最近邻
    diffs = [abs(g - measured_grad) for g in sorted_grads]
    nearest_idx = int(np.argmin(diffs))
    if verbose:
        print(f"  [回退] 未找到有效插值区间，使用最近邻: L={sorted_lengths[nearest_idx]}")
    return float(sorted_lengths[nearest_idx])


def blind_deblur(blurred_image, lookup_table, angle=25, gradient_ratio=0.05, method='wiener'):
    """
    完整盲复原函数：输入模糊图，输出复原图像

    流程：
        1. 提取过渡区，测量平均梯度
        2. 从查找表估计运动长度
        3. 生成估计的PSF
        4. 用估计的PSF做反卷积

    参数:
        blurred_image: 输入模糊图
        lookup_table: 查找表
        angle: 运动角度（固定，后续可扩展）
        gradient_ratio: 梯度阈值比例
        method: 反卷积方法，'wiener'或'rl'

    返回:
        restored: 复原图像
        estimated_length: 估计的运动长度
    """
    # TODO 1: 提取过渡区，测量平均梯度
    # 过渡区内的平均梯度反映了模糊程度：梯度越小，模糊越重
    measured_grad = compute_mean_gradient(blurred_image, gradient_ratio)

    # TODO 2: 从平均梯度估计运动长度
    # 在查找表中找平均梯度最接近的条目，用线性插值得到亚像素精度的估计
    estimated_length = estimate_length_from_grad(measured_grad, lookup_table)

    # TODO 3: 生成估计的PSF
    # 用估计的运动长度生成运动模糊PSF
    # 8倍超采样PSF支持亚像素浮点长度，不再用int截断
    # kernel_size要足够大以容纳整个运动线段，所以用 ceil(length)*2+1
    kernel_size = max(21, int(np.ceil(estimated_length)) * 2 + 1)
    psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=estimated_length)

    # TODO 4: 用估计的PSF做反卷积
    # 维纳滤波：速度快，对PSF误差鲁棒，适合盲复原
    # RL反卷积：迭代方法，效果可能更好但对PSF误差敏感、速度慢
    if method == 'wiener':
        restored = wiener_filter(blurred_image, psf, K=0.02)
    elif method == 'rl':
        restored = richardson_lucy(blurred_image, psf, iterations=30)
    else:
        restored = wiener_filter(blurred_image, psf, K=0.02)

    return restored, estimated_length


def compare_blind_vs_nonblind(blurred_image, clean_image, true_length, lookup_table, angle=25):
    """
    对比盲复原和非盲复原的效果

    参数:
        blurred_image: 模糊图
        clean_image: 清晰原图（用于计算PSNR）
        true_length: 真实运动长度（用于非盲复原）
        lookup_table: 查找表
        angle: 运动角度

    返回:
        results: 字典，包含盲复原和非盲复原的PSNR/SSIM
    """
    # 盲复原
    restored_blind, est_length = blind_deblur(blurred_image, lookup_table, angle=angle)

    # 非盲复原（用真实PSF）
    true_kernel_size = max(21, true_length * 2 + 1)
    true_psf = motion_blur_psf(kernel_size=true_kernel_size, angle=angle, length=true_length)
    restored_nonblind = wiener_filter(blurred_image, true_psf, K=0.02)

    # 计算评价指标
    metrics_blind = evaluate_restoration(restored_blind, clean_image)
    metrics_nonblind = evaluate_restoration(restored_nonblind, clean_image)

    results = {
        'estimated_length': est_length,
        'true_length': true_length,
        'length_error': abs(est_length - true_length),
        'psnr_blind': metrics_blind['PSNR'],
        'psnr_nonblind': metrics_nonblind['PSNR'],
        'psnr_drop': metrics_nonblind['PSNR'] - metrics_blind['PSNR'],
        'ssim_blind': metrics_blind['SSIM'],
        'ssim_nonblind': metrics_nonblind['SSIM'],
        'restored_blind': restored_blind,
        'restored_nonblind': restored_nonblind,
    }

    return results


def process_all_images():
    """批量处理：对比绝对梯度 vs 相对梯度的盲复原效果"""

    clean_dir = "../data/clean"
    blur_dir = "../data/synthetic_blur"
    result_dir = "../results/05_完整盲复原"
    os.makedirs(result_dir, exist_ok=True)

    # 读取所有清晰图
    clean_images = []
    clean_names = []
    for f in sorted(os.listdir(clean_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img = cv2.imread(os.path.join(clean_dir, f))
            if img is not None:
                clean_images.append(img)
                clean_names.append(os.path.splitext(f)[0])

    print(f"找到 {len(clean_images)} 张清晰图\n")

    # 建立两种查找表
    print("=" * 60)
    table_abs = build_reference_table(clean_images, lengths=range(1, 26))
    print()
    table_sn = build_reference_table_selfnorm(clean_images, lengths=range(1, 26))

    # 处理所有运动模糊图
    print(f"\n{'='*100}")
    print("绝对梯度 vs 自归一化梯度 盲复原对比")
    print(f"{'='*100}")
    print(f"{'图像':<15} {'真实长度':<8} {'绝对估计':<10} {'绝对误差':<10} {'绝对PSNR降':<12} "
          f"{'自归一估计':<10} {'自归一误差':<10} {'自归一PSNR降':<12}")
    print("-" * 100)

    for blur_file in sorted(os.listdir(blur_dir)):
        if not blur_file.lower().endswith('.jpg'):
            continue
        if '_motion' not in blur_file or '_noisy' in blur_file:
            continue

        img_name = blur_file.replace('_motion.jpg', '')
        if img_name not in clean_names:
            continue

        blurred = cv2.imread(os.path.join(blur_dir, blur_file))
        clean_idx = clean_names.index(img_name)
        clean = clean_images[clean_idx]

        true_length = 10

        # 绝对梯度盲复原
        restored_abs, est_abs = blind_deblur(blurred, table_abs, angle=25)
        metrics_abs = evaluate_restoration(restored_abs, clean)

        # 自归一化梯度盲复原
        restored_sn, est_sn, sn_val = blind_deblur_selfnorm(blurred, table_sn, angle=25)
        metrics_sn = evaluate_restoration(restored_sn, clean)

        # 非盲复原（上限）
        true_psf = motion_blur_psf(kernel_size=21, angle=25, length=true_length)
        restored_true = wiener_filter(blurred, true_psf, K=0.02)
        metrics_true = evaluate_restoration(restored_true, clean)

        psnr_drop_abs = metrics_true['PSNR'] - metrics_abs['PSNR']
        psnr_drop_sn = metrics_true['PSNR'] - metrics_sn['PSNR']

        print(f"{img_name:<15} {true_length:<8} {est_abs:<10.1f} {abs(est_abs-true_length):<10.1f} "
              f"{psnr_drop_abs:<12.2f} {est_sn:<10.1f} {abs(est_sn-true_length):<10.1f} "
              f"{psnr_drop_sn:<12.2f}")

        # 保存对比图（模糊 | 绝对梯度复原 | 自归一化复原 | 非盲复原）
        comparison = np.hstack([blurred, restored_abs, restored_sn, restored_true])
        imwrite_unicode(os.path.join(result_dir, f"{img_name}_abs_vs_selfnorm.jpg"), comparison)

    print(f"\n完成！结果已保存到 {result_dir}")
    print("对比图从左到右：模糊图 | 绝对梯度复原 | 自归一化梯度复原 | 非盲复原（真实PSF）")


if __name__ == "__main__":
    # 先测试单张图
    print("=" * 60)
    print("测试：单张图完整盲复原")
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

    # 生成一张测试模糊图（运动长度=10）
    true_length = 10
    psf = motion_blur_psf(kernel_size=21, angle=25, length=true_length)
    blurred = apply_blur(clean, psf)
    print(f"生成模糊图：运动长度={true_length}, 角度=25°")

    # 建立查找表（用这张图自己）
    table = build_reference_table([clean], lengths=range(1, 26))

    # 盲复原
    restored, est_length = blind_deblur(blurred, table, angle=25)
    print(f"\n估计的运动长度: {est_length:.1f}（真实: {true_length}）")
    print(f"长度误差: {abs(est_length - true_length):.1f}")

    # 非盲复原（对比）
    restored_nonblind = wiener_filter(blurred, psf, K=0.02)

    # 计算PSNR
    metrics_blind = evaluate_restoration(restored, clean)
    metrics_nonblind = evaluate_restoration(restored_nonblind, clean)
    print(f"盲复原 PSNR: {metrics_blind['PSNR']:.2f} dB")
    print(f"非盲复原 PSNR: {metrics_nonblind['PSNR']:.2f} dB")
    print(f"PSNR 下降: {metrics_nonblind['PSNR'] - metrics_blind['PSNR']:.2f} dB")

    # 保存测试结果
    result_dir = "../results/05_完整盲复原"
    os.makedirs(result_dir, exist_ok=True)
    comparison = np.hstack([blurred, restored, restored_nonblind])
    imwrite_unicode(os.path.join(result_dir, "test_comparison.jpg"), comparison)
    print(f"\n对比图已保存: {result_dir}/test_comparison.jpg")
    print("从左到右：模糊图 | 盲复原 | 非盲复原（真实PSF）")

    print("\n" + "=" * 60)
    print("单张测试完成！接下来运行批量处理：")
    print("=" * 60)

    # 运行批量处理
    process_all_images()
