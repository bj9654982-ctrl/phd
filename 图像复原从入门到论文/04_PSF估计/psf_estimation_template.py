"""
第四阶段：PSF 估计
功能：从模糊图的过渡区特征反推PSF参数（运动模糊长度），验证估计精度
请在 TODO 处填写你的代码
"""

import numpy as np
import cv2
import os
import sys

# 导入第一阶段的PSF生成和模糊函数
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_退化模型与评价指标"))
from degradation_template import motion_blur_psf, apply_blur

# 导入第三阶段的过渡区提取函数
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "03_边缘与过渡区提取"))
from transition_zone_template import compute_transition_width, compute_mean_gradient, extract_transition_zone

# 导入第二阶段的评价指标和维纳滤波
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_退化模型与评价指标"))
from metrics_template import evaluate_restoration
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_非盲反卷积"))
from deconvolution_template import wiener_filter


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


def build_lookup_table(clean_image, lengths=range(1, 31), angle=25, gradient_ratio=0.05):
    """
    建立"过渡区特征 → 运动长度"的查找表

    参数:
        clean_image: 清晰原图
        lengths: 运动长度列表（默认1到30）
        angle: 运动角度（固定25度，简化问题）
        gradient_ratio: 过渡区梯度阈值比例

    返回:
        table: 字典，key=运动长度，value={'width':过渡区宽度, 'mean_grad':平均梯度}
    """
    table = {}

    # 先测量清晰图的基线（length=0）
    base_width = compute_transition_width(clean_image, gradient_ratio)
    base_grad = compute_mean_gradient(clean_image, gradient_ratio)
    table[0] = {'width': base_width, 'mean_grad': base_grad}

    print(f"清晰图基线: 宽度={base_width:.2f}, 平均梯度={base_grad:.2f}")
    print(f"建立查找表（运动长度 {min(lengths)}~{max(lengths)}）...")

    for length in lengths:
        # 生成指定长度的运动模糊
        kernel_size = max(21, length * 2 + 1)
        psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=length)
        blurred = apply_blur(clean_image, psf)

        # 测量过渡区特征
        width = compute_transition_width(blurred, gradient_ratio)
        mean_grad = compute_mean_gradient(blurred, gradient_ratio)

        table[length] = {'width': width, 'mean_grad': mean_grad}

        if length % 5 == 0:
            print(f"  length={length:2d}: 宽度={width:.2f}, 平均梯度={mean_grad:.2f}")

    print(f"查找表建立完成，共 {len(table)} 个条目")
    return table


def check_monotonic(values, increasing=True):
    """
    检查序列是否单调

    参数:
        values: 数值列表
        increasing: True=检查单调递增，False=检查单调递减

    返回:
        is_monotonic: bool
        violations: 违反单调性的位置数量
    """
    if len(values) < 2:
        return True, 0
    violations = 0
    for i in range(len(values) - 1):
        if increasing:
            if values[i + 1] < values[i]:
                violations += 1
        else:
            if values[i + 1] > values[i]:
                violations += 1
    return violations == 0, violations


def estimate_length_by_width(blurred_image, lookup_table, gradient_ratio=0.05, verbose=False):
    """
    用过渡区宽度估计运动长度

    注意：此方法假定宽度随长度单调递增。若序列非单调，线性插值可能失效，
    会自动回退到最近邻估计并给出警告。

    参数:
        blurred_image: 模糊图
        lookup_table: 查找表
        gradient_ratio: 梯度阈值比例
        verbose: 是否打印单调性警告

    返回:
        estimated_length: 估计的运动长度（float）
    """
    # 测量模糊图的过渡区宽度
    measured_width = compute_transition_width(blurred_image, gradient_ratio)

    # 提取排序后的长度和宽度（跳过length=0的基线）
    sorted_lengths = sorted([L for L in lookup_table.keys() if L > 0])
    sorted_widths = [lookup_table[L]['width'] for L in sorted_lengths]

    # 单调性检查
    is_mono, violations = check_monotonic(sorted_widths, increasing=True)
    if not is_mono and verbose:
        print(f"  [警告] 宽度序列非单调递增，违反{violations}处，插值可能不准确")

    # 初始化默认值（最近邻），防止未赋值
    estimated_length = float(sorted_lengths[0])

    # 处理边界情况：measured_width超出查找表范围
    if measured_width <= sorted_widths[0]:
        estimated_length = float(sorted_lengths[0])
    elif measured_width >= sorted_widths[-1]:
        estimated_length = float(sorted_lengths[-1])
    else:
        # 找到measured_width介于哪两个宽度之间（要求升序区间）
        found = False
        for i in range(len(sorted_widths) - 1):
            if sorted_widths[i] <= measured_width <= sorted_widths[i + 1] and sorted_widths[i] < sorted_widths[i + 1]:
                # 线性插值：在宽度i和i+1之间，按比例计算长度
                w0, w1 = sorted_widths[i], sorted_widths[i + 1]
                L0, L1 = sorted_lengths[i], sorted_lengths[i + 1]
                ratio = (measured_width - w0) / (w1 - w0)
                estimated_length = L0 + ratio * (L1 - L0)
                found = True
                break

        # 如果没找到有效区间（非单调序列），回退到最近邻
        if not found:
            diffs = [abs(w - measured_width) for w in sorted_widths]
            nearest_idx = np.argmin(diffs)
            estimated_length = float(sorted_lengths[nearest_idx])
            if verbose:
                print(f"  [回退] 未找到有效插值区间，使用最近邻: L={estimated_length:.1f}")

    return estimated_length


def estimate_length_by_grad(blurred_image, lookup_table, gradient_ratio=0.05, verbose=False):
    """
    用平均梯度估计运动长度

    注意：此方法假定平均梯度随长度单调递减。若序列非单调，线性插值可能失效，
    会自动回退到最近邻估计并给出警告。

    参数:
        blurred_image: 模糊图
        lookup_table: 查找表
        gradient_ratio: 梯度阈值比例
        verbose: 是否打印单调性警告

    返回:
        estimated_length: 估计的运动长度（float）
    """
    # 测量模糊图的平均梯度
    measured_grad = compute_mean_gradient(blurred_image, gradient_ratio)

    # 提取排序后的长度和平均梯度（跳过length=0的基线）
    sorted_lengths = sorted([L for L in lookup_table.keys() if L > 0])
    sorted_grads = [lookup_table[L]['mean_grad'] for L in sorted_lengths]

    # 单调性检查（梯度应随长度递减）
    is_mono, violations = check_monotonic(sorted_grads, increasing=False)
    if not is_mono and verbose:
        print(f"  [警告] 平均梯度序列非单调递减，违反{violations}处，插值可能不准确")

    # 初始化默认值（最近邻），防止未赋值
    estimated_length = float(sorted_lengths[0])

    # 处理边界情况
    # 注意：梯度是递减的，所以measured_grad >= sorted_grads[0] 对应最小长度
    if measured_grad >= sorted_grads[0]:
        estimated_length = float(sorted_lengths[0])
    elif measured_grad <= sorted_grads[-1]:
        estimated_length = float(sorted_lengths[-1])
    else:
        # 找到measured_grad介于哪两个梯度之间（要求降序区间）
        found = False
        for i in range(len(sorted_grads) - 1):
            # sorted_grads[i] > sorted_grads[i+1]（递减）
            if sorted_grads[i + 1] <= measured_grad <= sorted_grads[i] and sorted_grads[i] > sorted_grads[i + 1]:
                g0, g1 = sorted_grads[i], sorted_grads[i + 1]
                L0, L1 = sorted_lengths[i], sorted_lengths[i + 1]
                ratio = (g0 - measured_grad) / (g0 - g1)
                estimated_length = L0 + ratio * (L1 - L0)
                found = True
                break

        # 如果没找到有效区间（非单调序列），回退到最近邻
        if not found:
            diffs = [abs(g - measured_grad) for g in sorted_grads]
            nearest_idx = np.argmin(diffs)
            estimated_length = float(sorted_lengths[nearest_idx])
            if verbose:
                print(f"  [回退] 未找到有效插值区间，使用最近邻: L={estimated_length:.1f}")

    return estimated_length


def estimate_length_by_linear_fit(blurred_image, lookup_table, gradient_ratio=0.05):
    """
    用线性拟合公式估计运动长度（更优雅的方法）

    思路：假设宽度和长度是线性关系 W = a*L + b，
         用查找表的数据拟合出a和b，然后 L = (W - b) / a

    参数:
        blurred_image: 模糊图
        lookup_table: 查找表
        gradient_ratio: 梯度阈值比例

    返回:
        estimated_length: 估计的运动长度
    """
    # 从查找表中提取长度和宽度数据（跳过length=0的基线）
    lengths = []
    widths = []
    for L, feat in lookup_table.items():
        if L > 0:
            lengths.append(L)
            widths.append(feat['width'])

    lengths = np.array(lengths)
    widths = np.array(widths)

    # TODO 3: 线性拟合 W = a*L + b
    # np.polyfit(x, y, 1) 对 y = a*x + b 做最小二乘拟合，返回 [a, b]
    # 这里 x=lengths（运动长度），y=widths（过渡区宽度）
    # a = 斜率（每增加1像素长度，宽度增加多少）
    # b = 截距（长度为0时的宽度，即清晰图基线宽度）
    coeffs = np.polyfit(lengths, widths, 1)
    a = coeffs[0]  # 斜率
    b = coeffs[1]  # 截距

    # 测量模糊图的宽度
    measured_width = compute_transition_width(blurred_image, gradient_ratio)

    # TODO 4: 用拟合公式反推长度 L = (W - b) / a
    # 从 W = a*L + b 解出 L：
    #   a*L = W - b
    #   L = (W - b) / a
    # 注意：a不能为0（否则宽度不随长度变化，无法估计）
    if abs(a) > 1e-10:
        estimated_length = (measured_width - b) / a
    else:
        estimated_length = 0.0

    # 限制估计长度在合理范围内
    estimated_length = max(0.0, min(float(max(lengths)), estimated_length))

    return estimated_length


def evaluate_estimation(clean_image, true_length, estimated_length, angle=25):
    """
    评价PSF估计精度：用估计的PSF做反卷积，看复原效果

    参数:
        clean_image: 清晰原图
        true_length: 真实运动长度
        estimated_length: 估计的运动长度
        angle: 运动角度

    返回:
        results: 字典，包含参数误差和复原PSNR
    """
    # 参数误差
    abs_error = abs(estimated_length - true_length)
    rel_error = abs_error / true_length * 100 if true_length > 0 else 0

    # 用真实PSF生成模糊图
    kernel_size = max(21, true_length * 2 + 1)
    true_psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=true_length)
    blurred = apply_blur(clean_image, true_psf)

    # 用估计的PSF做维纳滤波复原（8倍超采样PSF支持浮点长度）
    est_kernel_size = max(21, int(np.ceil(estimated_length)) * 2 + 1)
    est_psf = motion_blur_psf(kernel_size=est_kernel_size, angle=angle, length=estimated_length)

    # 简单的维纳滤波复原（用第二阶段的方法）
    restored = wiener_filter(blurred, est_psf, K=0.02)

    # 计算复原PSNR
    metrics = evaluate_restoration(restored, clean_image)

    # 用真实PSF复原的PSNR（上限）
    restored_true = wiener_filter(blurred, true_psf, K=0.02)
    metrics_true = evaluate_restoration(restored_true, clean_image)

    results = {
        'true_length': true_length,
        'estimated_length': estimated_length,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'psnr_estimated': metrics['PSNR'],
        'psnr_true': metrics_true['PSNR'],
        'psnr_drop': metrics_true['PSNR'] - metrics['PSNR'],
    }

    return results


def process_all_images():
    """批量处理：对所有清晰图测试不同运动长度的估计精度，保存CSV结果"""

    clean_dir = "../data/clean"
    result_dir = "../results/04_PSF估计"
    os.makedirs(result_dir, exist_ok=True)

    clean_files = []
    for f in sorted(os.listdir(clean_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            clean_files.append(f)

    # 测试的运动长度
    test_lengths = [3, 5, 8, 10, 15, 20]

    print(f"找到 {len(clean_files)} 张清晰图，开始PSF估计精度测试...\n")

    # 收集所有结果用于CSV
    all_results = []

    for clean_file in clean_files:
        img_name = os.path.splitext(clean_file)[0]
        clean = cv2.imread(os.path.join(clean_dir, clean_file))
        if clean is None:
            continue

        print(f"{'='*70}")
        print(f"图像: {clean_file}")
        print(f"{'='*70}")

        # 建立查找表
        table = build_lookup_table(clean, lengths=range(1, 26))

        print(f"\n{'真实长度':<10} {'估计(宽度法)':<12} {'估计(梯度法)':<12} {'宽度误差':<10} {'梯度误差':<10}")
        print("-" * 60)

        for true_L in test_lengths:
            # 生成模糊图
            kernel_size = max(21, true_L * 2 + 1)
            psf = motion_blur_psf(kernel_size=kernel_size, angle=25, length=true_L)
            blurred = apply_blur(clean, psf)

            # 用宽度法和梯度法分别估计长度（打开verbose以记录非单调警告）
            est_L_width = estimate_length_by_width(blurred, table, verbose=True)
            est_L_grad = estimate_length_by_grad(blurred, table, verbose=True)

            err_width = abs(est_L_width - true_L)
            err_grad = abs(est_L_grad - true_L)

            print(f"{true_L:<10} {est_L_width:<12.2f} {est_L_grad:<12.2f} {err_width:<10.2f} {err_grad:<10.2f}")

            all_results.append({
                'image': img_name,
                'true_length': true_L,
                'est_width': round(est_L_width, 4),
                'est_grad': round(est_L_grad, 4),
                'err_width': round(err_width, 4),
                'err_grad': round(err_grad, 4),
            })

        print()

    # 保存CSV结果（明确标注为同图泄漏式教学演示，不能作为论文证据）
    import csv
    csv_path = os.path.join(result_dir, "same_image_leaky_demo_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'true_length', 'est_width', 'est_grad', 'err_width', 'err_grad'])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n完成！结果已保存到 {result_dir}")
    print(f"CSV文件: {csv_path}（共{len(all_results)}条记录）")
    print("注意: 此CSV为同图同长度节点循环验证数据，不能作为论文证据")


def leave_one_image_out(success_threshold_db=0.5):
    """
    留一场景诊断性交叉验证（N4-01修正措辞）

    每折留出1张图作为测试集，用剩余图建立通用查找表。
    测试长度不进入查找表训练节点（包括非整数长度）。

    改进（第四轮复审）:
    - N4-06: 成功定义加阈值(默认+0.5dB)，报告连续增益分布
    - S1-05: 增加真值PSF复原上界，分离核估计损失与反卷积损失
    - N4-03: CSV增加SSIM、K来源、角度、真值上界等字段
    - 报告均值、中位数、场景级统计

    注意: 此函数使用全局梯度特征+通用查找表，预期在2/3图像上失败。
    失败是方法本身的局限，不是bug。
    """
    import csv

    clean_dir = "../data/clean"
    result_dir = "../results/04_PSF估计"
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

    n_images = len(clean_images)
    if n_images < 2:
        print("需要至少2张图才能做留一图验证")
        return

    print(f"{'='*70}")
    print(f"留一场景诊断性交叉验证 (Scene-wise Leave-One-Out)")
    print(f"共 {n_images} 个场景，{n_images} 折；场景级样本量={n_images}")
    print(f"成功阈值: PSNR增益 > +{success_threshold_db} dB")
    print(f"{'='*70}")

    # 训练长度节点（稀疏，测试长度不进入此集合）
    train_lengths = [2, 6, 10, 14, 18, 22]
    # 测试长度（未进入训练表，包括非整数）
    test_lengths = [4.0, 7.5, 9.0, 12.5, 16.0, 20.0]
    angle = 25  # 已知角度（S1-03: 角度估计尚未实现）
    gradient_ratio = 0.05
    K_wiener = 0.02  # 固定K（S1-05: 嵌套交叉验证待实现）

    all_results = []
    total_tests = 0
    failed_any = 0  # 增益 <= 0
    failed_threshold = 0  # 增益 <= success_threshold_db

    for test_idx in range(n_images):
        test_name = clean_names[test_idx]
        test_image = clean_images[test_idx]
        train_images = [clean_images[i] for i in range(n_images) if i != test_idx]
        train_names = [clean_names[i] for i in range(n_images) if i != test_idx]

        print(f"\n--- 折 {test_idx+1}/{n_images}: 测试图={test_name}, 训练图={train_names} ---")

        # 用训练图建立通用查找表（平均梯度）
        print(f"  建立通用查找表（训练长度节点: {train_lengths}）...")
        grad_collect = {L: [] for L in train_lengths}
        for img in train_images:
            for L in train_lengths:
                kernel_size = max(21, int(np.ceil(L)) * 2 + 1)
                psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=L)
                blurred = apply_blur(img, psf)
                mean_grad = compute_mean_gradient(blurred, gradient_ratio)
                grad_collect[L].append(mean_grad)

        lookup_table = {L: {'width': 0.0, 'mean_grad': np.mean(grad_collect[L])} for L in train_lengths}

        # 对测试图的每个测试长度进行估计和复原
        for true_L in test_lengths:
            kernel_size = max(21, int(np.ceil(true_L)) * 2 + 1)
            psf_true = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=true_L)
            blurred = apply_blur(test_image, psf_true)

            # 估计长度（梯度法）
            est_L = estimate_length_by_grad(blurred, lookup_table, verbose=False)

            # 用估计的PSF复原
            est_kernel_size = max(21, int(np.ceil(est_L)) * 2 + 1)
            psf_est = motion_blur_psf(kernel_size=est_kernel_size, angle=angle, length=est_L)
            restored = wiener_filter(blurred, psf_est, K=K_wiener)

            # S1-05: 真值PSF复原上界（分离核估计损失与反卷积损失）
            restored_oracle = wiener_filter(blurred, psf_true, K=K_wiener)

            # 计算指标
            metrics_input = evaluate_restoration(blurred, test_image)
            metrics_restored = evaluate_restoration(restored, test_image)
            metrics_oracle = evaluate_restoration(restored_oracle, test_image)

            input_psnr = float(metrics_input['PSNR'])
            input_ssim = float(metrics_input['SSIM'])
            restored_psnr = float(metrics_restored['PSNR'])
            restored_ssim = float(metrics_restored['SSIM'])
            oracle_psnr = float(metrics_oracle['PSNR'])
            oracle_ssim = float(metrics_oracle['SSIM'])

            psnr_gain = restored_psnr - input_psnr
            oracle_gain = oracle_psnr - input_psnr
            estimation_loss = oracle_psnr - restored_psnr  # 核估计造成的损失

            success_any = psnr_gain > 0
            success_threshold = psnr_gain > success_threshold_db

            length_error = abs(est_L - true_L)
            total_tests += 1
            if not success_any:
                failed_any += 1
            if not success_threshold:
                failed_threshold += 1

            result = {
                'fold': test_idx + 1,
                'test_image': test_name,
                'train_images': '+'.join(train_names),
                'true_length': true_L,
                'true_angle': angle,
                'angle_mode': 'oracle_known (not estimated)',
                'estimated_length': round(est_L, 4),
                'length_error': round(length_error, 4),
                'input_psnr': round(input_psnr, 4),
                'input_ssim': round(input_ssim, 4),
                'restored_psnr': round(restored_psnr, 4),
                'restored_ssim': round(restored_ssim, 4),
                'psnr_gain': round(psnr_gain, 4),
                'ssim_gain': round(restored_ssim - input_ssim, 4),
                'matched_psf_psnr': round(oracle_psnr, 4),
                'matched_psf_ssim': round(oracle_ssim, 4),
                'matched_psf_gain': round(oracle_gain, 4),
                'kernel_mismatch_loss_db': round(estimation_loss, 4),
                'K_wiener': K_wiener,
                'K_source': 'fixed_global (not nested CV)',
                'gradient_ratio': gradient_ratio,
                'success_any_gain': success_any,
                f'success_gain_gt_{success_threshold_db}db': success_threshold,
            }
            all_results.append(result)

            status = "PASS" if success_threshold else ("MARGINAL" if success_any else "FAIL")
            print(f"  L={true_L:5.1f}: 估计={est_L:6.2f}, 误差={length_error:5.2f}, "
                  f"增益={psnr_gain:+6.2f}dB, 上界={oracle_gain:+6.2f}dB, "
                  f"估计损失={estimation_loss:5.2f}dB [{status}]")

    # 保存CSV（原子写入：先写临时文件，回读验证后再替换）
    csv_path = os.path.join(result_dir, "leave_one_image_out_results.csv")
    tmp_path = csv_path + ".tmp"
    fieldnames = list(all_results[0].keys())
    with open(tmp_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    # 回读验证
    with open(tmp_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) != len(all_results):
            raise RuntimeError(f"CSV回读验证失败：写入{len(all_results)}行，回读{len(rows)}行")
        for i, (written, read) in enumerate(zip(all_results, rows)):
            for key in fieldnames:
                if str(written[key]) != read[key]:
                    raise RuntimeError(f"CSV回读验证失败：第{i}行字段{key}不一致")
    # 原子替换
    os.replace(tmp_path, csv_path)

    # 汇总统计（使用与result字典一致的键名）
    gains = np.array([r['psnr_gain'] for r in all_results])
    oracle_gains = np.array([r['matched_psf_gain'] for r in all_results])
    est_losses = np.array([r['kernel_mismatch_loss_db'] for r in all_results])
    length_errors = np.array([r['length_error'] for r in all_results])

    # 场景级统计
    scene_stats = {}
    for name in clean_names:
        scene_gains = [r['psnr_gain'] for r in all_results if r['test_image'] == name]
        scene_stats[name] = {
            'mean_gain': np.mean(scene_gains),
            'median_gain': np.median(scene_gains),
            'n_pass': sum(1 for g in scene_gains if g > success_threshold_db),
            'n_total': len(scene_gains),
        }

    print(f"\n{'='*70}")
    print(f"留一场景诊断性交叉验证汇总")
    print(f"{'='*70}")
    print(f"退化条件总数: {total_tests}（场景级样本量={n_images}，不独立）")
    print(f"失败数（增益<=0）: {failed_any}")
    print(f"失败数（增益<=+{success_threshold_db}dB）: {failed_threshold}")
    print(f"通过率（增益>+{success_threshold_db}dB）: {(total_tests-failed_threshold)/total_tests*100:.1f}%")
    print(f"\nPSNR增益分布:")
    print(f"  均值:   {np.mean(gains):+.2f} dB")
    print(f"  中位数: {np.median(gains):+.2f} dB")
    print(f"  标准差: {np.std(gains):.2f} dB")
    print(f"  最小:   {np.min(gains):+.2f} dB")
    print(f"  最大:   {np.max(gains):+.2f} dB")
    print(f"\nMatched-PSF Wiener参考（同一K、同一复原器下的核失配损失分离）:")
    print(f"  matched-PSF平均增益: {np.mean(oracle_gains):+.2f} dB")
    print(f"  核失配平均损失: {np.mean(est_losses):.2f} dB")
    print(f"  长度估计平均误差: {np.mean(length_errors):.2f} px")
    print(f"\n分场景统计:")
    for name, stats in scene_stats.items():
        print(f"  {name}: 均值增益={stats['mean_gain']:+.2f}dB, "
              f"中位数={stats['median_gain']:+.2f}dB, "
              f"通过={stats['n_pass']}/{stats['n_total']}")
    print(f"\nCSV文件: {csv_path}")
    print(f"\n注意: 此结果为留一场景诊断性交叉验证（测试图不参与建表，测试长度不进入训练节点）")
    print(f"场景级样本量仅为{n_images}，{total_tests}个退化条件不独立，不宜称为最终独立验证")
    print(f"核估计平均损失={np.mean(est_losses):.2f}dB说明：即使反卷积器完美，PSF估计错误仍导致显著质量下降")
    print(f"如果失败率高，说明全局梯度特征不能跨图辨识PSF，需要改用ESF/LSF方法。")


if __name__ == "__main__":
    # 先测试单张图
    print("=" * 60)
    print("测试：单张图PSF估计")
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

    # 建立查找表
    table = build_lookup_table(clean, lengths=range(1, 26))

    # 测试几个运动长度
    test_lengths = [5, 10, 15, 20]
    print(f"\n{'真实长度':<10} {'估计(宽度法)':<12} {'估计(梯度法)':<12} {'估计(线性拟合)':<14}")
    print("-" * 50)

    for true_L in test_lengths:
        kernel_size = max(21, true_L * 2 + 1)
        psf = motion_blur_psf(kernel_size=kernel_size, angle=25, length=true_L)
        blurred = apply_blur(clean, psf)

        est_width = estimate_length_by_width(blurred, table)
        est_grad = estimate_length_by_grad(blurred, table)
        est_fit = estimate_length_by_linear_fit(blurred, table)

        print(f"{true_L:<10} {est_width:<12.1f} {est_grad:<12.1f} {est_fit:<14.1f}")

    print("\n" + "=" * 60)
    print("单张测试完成！接下来运行批量处理（含CSV保存）：")
    print("=" * 60)

    # 运行批量处理（同图循环验证，仅作教学演示）
    process_all_images()

    # 运行留一场景诊断性交叉验证（S1-01修复，N4-01修正措辞）
    print("\n" + "=" * 60)
    print("接下来运行留一场景诊断性交叉验证:")
    print("=" * 60)
    leave_one_image_out()
