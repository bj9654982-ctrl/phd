"""
基于边缘扩散函数(ESF)和线扩散函数(LSF)的运动模糊参数估计

第五轮复审修复（S1-01~S1-09）:
- S1-01: 投影公式统一使用边缘法线角，w = L * |cos(motion_angle - normal_angle)|
- S1-02: 按法线角分箱，逐边缘保留(normal_angle, width, uncertainty)，禁止跨方向混合
- S1-04: Canny阈值自适应（梯度百分位），无边缘返回结构化失败
- S1-05: 边缘质量筛选（直线性、方向方差、两侧平台稳定性、对比度）
- S1-06: 真正亚像素对齐（scipy.ndimage.shift，不round成整数）
- S1-07: 每条剖面拟合两侧平台，归一化到0-1，暗到亮统一
- S1-08: LSF宽度用FWHM亚像素交点，不用固定0.1阈值硬切
- S1-09: 引入基线宽度w0，不直接令L=LSF宽度
- N5-04: 理想阶跃解析测试矩阵

物理原理:
- 理想阶跃边缘经二维PSF模糊后，沿边缘法线的剖面为ESF
- ESF的导数为LSF，等于PSF沿边缘切线方向的积分(Radon投影)
- 对于线性运动模糊PSF，LSF近似为矩形函数
- 投影宽度 w_i = L * |cos(theta_motion - theta_normal_i)|
- 多方向边缘联合拟合: w_i = w0 + L * |cos(theta_motion - theta_normal_i)|
  其中w0是静态成像基线宽度（像素孔径、镜头像差等）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_退化模型与评价指标'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_非盲反卷积'))
import numpy as np
import cv2
from scipy import ndimage
from degradation_template import motion_blur_psf, apply_blur
from deconvolution_template import wiener_filter
from metrics_template import evaluate_restoration


def detect_edges_adaptive(gray, low_percentile=20, high_percentile=80):
    """
    自适应Canny边缘检测（S1-04修复）

    基于图像梯度幅值的百分位设置阈值，避免固定阈值在强模糊时丢失边缘。

    参数:
        gray: 灰度图，uint8或float32
        low_percentile: 低阈值百分位
        high_percentile: 高阈值百分位

    返回:
        edges: 二值边缘掩码
        low_thresh: 使用的低阈值
        high_thresh: 使用的高阈值
    """
    if gray.dtype == np.uint8:
        gray_float = gray.astype(np.float32) / 255.0
    else:
        gray_float = gray.astype(np.float32)

    # 计算梯度幅值
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 用非零梯度的百分位设置阈值
    nonzero_grads = grad_mag[grad_mag > 1e-6]
    if len(nonzero_grads) == 0:
        return np.zeros_like(gray, dtype=np.uint8), 0, 0

    # 对于梯度大部分为0的图像（如理想阶跃），用非零梯度的统计量
    low_thresh = np.percentile(nonzero_grads, low_percentile)
    high_thresh = np.percentile(nonzero_grads, high_percentile)

    # 确保高阈值 > 低阈值，且不超过梯度最大值
    max_grad = np.max(nonzero_grads)
    if high_thresh <= low_thresh:
        high_thresh = min(low_thresh * 1.5, max_grad * 0.9)
    if high_thresh > max_grad * 0.9:
        high_thresh = max_grad * 0.9
    if low_thresh >= high_thresh:
        low_thresh = high_thresh * 0.4

    # 确保高阈值 > 低阈值
    if high_thresh <= low_thresh:
        high_thresh = low_thresh * 2.0

    # 转换回8-bit比例用于Canny
    low_8bit = float(np.clip(low_thresh * 255, 1, 255))
    high_8bit = float(np.clip(high_thresh * 255, 2, 255))

    edges = cv2.Canny(gray, low_8bit, high_8bit)
    return edges, low_thresh, high_thresh


def estimate_edge_orientation(gray, edge_mask):
    """估计每个边缘点的法线方向（S1-01修复：统一使用法线角）"""
    if gray.dtype == np.uint8:
        gray_float = gray.astype(np.float32) / 255.0
    else:
        gray_float = gray.astype(np.float32)

    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 法线方向 = 梯度方向
    normal_angle = np.arctan2(grad_y, grad_x)  # [-pi, pi]

    return normal_angle, grad_mag


def normalize_profile(profile):
    """
    对比度归一化和平台校正（S1-07修复）

    拟合两侧平台，统一为暗到亮方向，归一化到0-1。

    参数:
        profile: 一维灰度剖面

    返回:
        normalized: 归一化后的剖面
        valid: 是否有效（两侧平台稳定、对比度足够）
        contrast: 边缘对比度
    """
    n = len(profile)
    edge_region = n // 4  # 边缘两侧各取1/4作为平台

    # 左侧平台（暗侧）和右侧平台（亮侧）
    left_platform = profile[:edge_region]
    right_platform = profile[-edge_region:]

    left_mean = np.mean(left_platform)
    right_mean = np.mean(right_platform)
    left_std = np.std(left_platform)
    right_std = np.std(right_platform)

    # 对比度
    contrast = abs(right_mean - left_mean)

    # 有效性检查：平台稳定（标准差小）、对比度足够
    platform_stability = (left_std < 0.05) and (right_std < 0.05)
    sufficient_contrast = contrast > 0.05

    if not (platform_stability and sufficient_contrast):
        return profile, False, contrast

    # 统一为暗到亮方向（如果左亮右暗，翻转）
    if left_mean > right_mean:
        profile = profile[::-1]
        left_mean, right_mean = right_mean, left_mean

    # 归一化到0-1
    normalized = (profile - left_mean) / (right_mean - left_mean + 1e-10)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized, True, contrast


def sample_normal_profile_subpixel(gray, y, x, normal_angle, profile_length=41):
    """
    沿法线方向亚像素采样边缘剖面（双线性插值）

    返回:
        profile: 一维剖面
        valid: 是否有效
    """
    half = profile_length // 2
    dx = np.cos(normal_angle)
    dy = np.sin(normal_angle)

    h, w = gray.shape
    profile = np.zeros(profile_length, dtype=np.float32)
    valid = True

    for i in range(profile_length):
        offset = i - half
        sy = y + offset * dy
        sx = x + offset * dx

        y0 = int(np.floor(sy))
        x0 = int(np.floor(sx))
        y1 = y0 + 1
        x1 = x0 + 1

        if y0 < 0 or y1 >= h or x0 < 0 or x1 >= w:
            valid = False
            break

        wy = sy - y0
        wx = sx - x0

        val = (gray[y0, x0] * (1 - wy) * (1 - wx) +
               gray[y0, x1] * (1 - wy) * wx +
               gray[y1, x0] * wy * (1 - wx) +
               gray[y1, x1] * wy * wx)
        profile[i] = val

    return profile, valid


def align_profile_subpixel(profile, target_center=None):
    """
    真正的亚像素ESF对齐（S1-06修复：用scipy.ndimage.shift，不round成整数）

    找到ESF的最大梯度位置（亚像素），然后用分数位移对齐到目标中心。

    参数:
        profile: 输入ESF剖面
        target_center: 目标中心位置（默认剖面中心）

    返回:
        aligned: 亚像素对齐后的剖面
        edge_center: 检测到的边缘中心（亚像素）
    """
    n = len(profile)
    if target_center is None:
        target_center = n / 2.0

    # 计算梯度
    grad = np.gradient(profile)

    # 找到最大绝对梯度位置
    max_idx = np.argmax(np.abs(grad))

    # 亚像素精修：抛物线拟合
    if 0 < max_idx < len(grad) - 1:
        a = grad[max_idx - 1]
        b = grad[max_idx]
        c = grad[max_idx + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-10:
            subpixel_offset = 0.5 * (a - c) / denom
        else:
            subpixel_offset = 0.0
    else:
        subpixel_offset = 0.0

    edge_center = max_idx + subpixel_offset

    # 计算需要的位移（保留小数，S1-06修复）
    shift = target_center - edge_center

    # 用scipy.ndimage.shift做亚像素位移（样条插值）
    aligned = ndimage.shift(profile.reshape(1, -1), [0, shift], order=3).flatten()

    return aligned, edge_center


def compute_lsf_regularized(esf, sigma=1.0):
    """
    从ESF计算LSF（S1-08修复：明确为高斯平滑正则化，不称Tikhonov）

    参数:
        esf: 边缘扩散函数
        sigma: 高斯平滑标准差（正则化强度）

    返回:
        lsf: 线扩散函数
    """
    if esf is None or len(esf) < 3:
        return None

    # 中心差分
    lsf = np.zeros_like(esf)
    lsf[1:-1] = (esf[2:] - esf[:-2]) / 2.0
    lsf[0] = esf[1] - esf[0]
    lsf[-1] = esf[-1] - esf[-2]

    # 高斯平滑正则化
    if sigma > 0:
        kernel_size = int(np.ceil(sigma * 3)) * 2 + 1
        lsf = cv2.GaussianBlur(lsf.reshape(1, -1), (kernel_size, 1), sigma).flatten()

    # 取绝对值并归一化
    lsf_abs = np.abs(lsf)
    total = np.sum(lsf_abs)
    if total > 0:
        lsf_normalized = lsf_abs / total
    else:
        lsf_normalized = lsf_abs

    return lsf_normalized


def estimate_fwhm_subpixel(lsf):
    """
    从LSF估计半高全宽（FWHM），亚像素交点（S1-08修复）

    参数:
        lsf: 线扩散函数（已取绝对值并归一化）

    返回:
        fwhm: 半高全宽（亚像素精度）
        center: LSF中心（加权平均）
        valid: 是否有效
    """
    if lsf is None or len(lsf) < 3:
        return None, None, False

    max_val = np.max(lsf)
    if max_val < 1e-10:
        return None, None, False

    half_max = max_val / 2.0
    above_half = lsf > half_max

    if not np.any(above_half):
        return None, None, False

    # 找到最大连续区域
    indices = np.where(above_half)[0]
    # 简单取所有超过半高的范围（后续可改进为最大连续区域）
    left_idx = indices[0]
    right_idx = indices[-1]

    # 亚像素左交点：线性插值
    if left_idx > 0:
        y0, y1 = lsf[left_idx - 1], lsf[left_idx]
        if y1 != y0:
            left_sub = left_idx - 1 + (half_max - y0) / (y1 - y0)
        else:
            left_sub = left_idx
    else:
        left_sub = left_idx

    # 亚像素右交点
    if right_idx < len(lsf) - 1:
        y0, y1 = lsf[right_idx], lsf[right_idx + 1]
        if y1 != y0:
            right_sub = right_idx + (half_max - y0) / (y1 - y0)
        else:
            right_sub = right_idx
    else:
        right_sub = right_idx

    fwhm = right_sub - left_sub

    # LSF中心（加权平均）
    weights = lsf
    if np.sum(weights) > 0:
        center = np.sum(np.arange(len(lsf)) * weights) / np.sum(weights)
    else:
        center = (left_sub + right_sub) / 2.0

    return fwhm, center, True


def estimate_esf_lsf_per_edge(gray, edge_mask, normal_angle, grad_mag,
                                 profile_length=41, min_edge_length=15):
    """
    逐边缘ESF/LSF估计（S1-02修复：不跨方向混合，每条边缘独立估计）

    参数:
        gray: 灰度图，float32 [0,1]
        edge_mask: 二值边缘掩码
        normal_angle: 每个像素的法线角
        grad_mag: 梯度幅值
        profile_length: 剖面长度
        min_edge_length: 最小边缘段长度

    返回:
        edge_results: 列表，每个元素是字典:
            - normal_angle: 边缘法线角（中位数）
            - fwhm: LSF半高全宽
            - contrast: 边缘对比度
            - n_profiles: 使用的剖面数
            - straightness: 直线性评分
            - valid: 是否有效
    """
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edge_mask.astype(np.uint8), connectivity=8
    )

    edge_results = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_edge_length:
            continue

        ys, xs = np.where(labels == i)
        if len(ys) < min_edge_length:
            continue

        # 边缘法线角（中位数，鲁棒）
        normals = normal_angle[ys, xs]
        median_normal = np.median(normals)
        # 方向方差（无向角度，用圆方差）
        angle_var = 1 - np.abs(np.mean(np.exp(1j * normals * 2))) / max(np.abs(np.median(np.exp(1j * normals * 2))), 1e-10)

        # 直线性：用PCA主成分分析（S1-6-04修复：不依赖像素顺序）
        # 第一主成分方差/总方差，越接近1越直
        if len(ys) >= 3:
            coords = np.column_stack([xs, ys]).astype(np.float64)
            coords_centered = coords - np.mean(coords, axis=0)
            # PCA：协方差矩阵特征值分解
            cov = np.cov(coords_centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
            total_var = np.sum(eigenvalues)
            if total_var > 1e-10:
                straightness = eigenvalues[0] / total_var
            else:
                straightness = 1.0
        else:
            straightness = 1.0

        # 质量筛选（S1-05）：方向方差小、直线性高
        if angle_var > 0.5 or straightness < 0.5:
            continue

        # 采样剖面（边缘段上均匀采样）
        profiles = []
        profile_contrasts = []  # S1-6-04修复：保存每条剖面的contrast
        step = max(1, len(ys) // 10)
        for j in range(0, len(ys), step):
            y, x = ys[j], xs[j]
            na = normal_angle[y, x]
            profile, valid = sample_normal_profile_subpixel(
                gray, y, x, na, profile_length
            )
            if valid:
                # 对比度归一化和平台校正（S1-07）
                norm_profile, prof_valid, contrast = normalize_profile(profile)
                if prof_valid:
                    profiles.append(norm_profile)
                    profile_contrasts.append(contrast)  # 保存contrast

        if len(profiles) < 3:
            continue

        # 段级contrast：所有剖面contrast的中位数（S1-6-04修复）
        segment_contrast = float(np.median(profile_contrasts)) if profile_contrasts else 0.0

        # 亚像素对齐（S1-06）
        aligned_profiles = []
        for p in profiles:
            aligned, _ = align_profile_subpixel(p)
            aligned_profiles.append(aligned)

        # 稳健聚合（中位数）
        esf = np.median(aligned_profiles, axis=0)

        # 计算LSF
        lsf = compute_lsf_regularized(esf, sigma=1.0)

        # 估计FWHM
        fwhm, center, fwhm_valid = estimate_fwhm_subpixel(lsf)

        if not fwhm_valid:
            continue

        edge_results.append({
            'normal_angle': float(median_normal),
            'fwhm': float(fwhm),
            'contrast': segment_contrast,  # S1-6-04修复：所有剖面contrast的中位数
            'n_profiles': len(profiles),
            'straightness': float(straightness),
            'angle_var': float(angle_var),
            'valid': True,
            'esf': esf,
            'lsf': lsf,
        })

    return edge_results


def fit_motion_with_known_angle(edge_results, motion_angle_deg,
                                  condition_threshold=1e6,
                                  max_relative_residual=0.5):
    """
    已知运动方向时，固定θ，只拟合L和w0（S1-6-02修复）

    模型: fwhm_i = w0 + L * |cos(theta_motion - theta_normal_i)|
    其中theta_motion已知，只拟合L和w0两个参数。

    参数:
        edge_results: 逐边缘估计结果列表
        motion_angle_deg: 已知运动方向（度）
        condition_threshold: 条件数拒识阈值
        max_relative_residual: 最大允许相对残差

    返回:
        result: 字典，与joint_fit_motion_params格式兼容
    """
    if len(edge_results) < 2:
        return {
            'L': None, 'theta_motion': float(motion_angle_deg), 'w0': None,
            'residual': None, 'n_edges': len(edge_results),
            'condition': None, 'rank': None,
            'n_independent_directions': 0,
            'valid': False,
            'status': 'insufficient_edges (need >=2 for known angle)',
        }

    normals = np.array([e['normal_angle'] for e in edge_results])
    fwhms = np.array([e['fwhm'] for e in edge_results])
    raw_weights = np.array([max(e.get('contrast', 0), 1e-6) *
                             max(e.get('straightness', 0), 1e-6)
                             for e in edge_results])
    weights = raw_weights / np.sum(raw_weights)

    theta_m = np.deg2rad(motion_angle_deg)
    proj = np.abs(np.cos(theta_m - normals))

    # 加权最小二乘: fwhm = w0 + L * proj（固定θ）
    A = np.column_stack([np.ones_like(proj), proj])
    sqrtW = np.sqrt(weights)
    A_weighted = A * sqrtW[:, np.newaxis]
    fwhms_weighted = fwhms * sqrtW

    try:
        params, residuals, rank, s = np.linalg.lstsq(A_weighted, fwhms_weighted, rcond=None)
        w0, L = params
        pred = w0 + L * proj
        residual = np.sum(weights * (fwhms - pred)**2)
        condition = np.linalg.cond(A_weighted)
    except np.linalg.LinAlgError:
        return {
            'L': None, 'theta_motion': float(motion_angle_deg), 'w0': None,
            'residual': None, 'n_edges': len(edge_results),
            'condition': None, 'rank': None,
            'valid': False,
            'status': 'linalg_error',
        }

    # 门禁：参数有效性
    if not (L > 0 and w0 > 0 and np.isfinite(condition)):
        return {
            'L': float(L) if np.isfinite(L) else None,
            'theta_motion': float(motion_angle_deg),
            'w0': float(w0) if np.isfinite(w0) else None,
            'residual': float(residual),
            'n_edges': len(edge_results),
            'condition': float(condition) if np.isfinite(condition) else None,
            'rank': int(rank) if rank is not None else None,
            'valid': False,
            'status': f'invalid_params (L={L:.3f}, w0={w0:.3f}, condition={condition})',
        }

    # 门禁：条件数
    if condition > condition_threshold:
        return {
            'L': float(L), 'theta_motion': float(motion_angle_deg),
            'w0': float(w0),
            'residual': float(residual),
            'n_edges': len(edge_results),
            'condition': float(condition),
            'rank': int(rank) if rank is not None else None,
            'valid': False,
            'status': f'condition_too_large ({condition:.2e} > {condition_threshold:.2e})',
        }

    # 门禁：相对残差
    mean_fwhm = np.average(fwhms, weights=weights)
    relative_residual = np.sqrt(residual) / max(mean_fwhm, 1e-6)
    if relative_residual > max_relative_residual:
        return {
            'L': float(L), 'theta_motion': float(motion_angle_deg),
            'w0': float(w0),
            'residual': float(residual),
            'relative_residual': float(relative_residual),
            'n_edges': len(edge_results),
            'condition': float(condition),
            'rank': int(rank) if rank is not None else None,
            'valid': False,
            'status': f'excessive_residual (relative={relative_residual:.3f} > {max_relative_residual})',
        }

    return {
        'L': float(L),
        'theta_motion': float(motion_angle_deg),
        'w0': float(w0),
        'residual': float(residual),
        'relative_residual': float(relative_residual),
        'n_edges': len(edge_results),
        'condition': float(condition),
        'rank': int(rank) if rank is not None else None,
        'n_independent_directions': len(np.unique(np.round(np.rad2deg(normals) / 15))),
        'valid': True,
        'status': 'ok_known_angle',
        'predictions': pred,
        'measurements': fwhms,
        'normals': normals,
        'weights': weights,
    }


def joint_fit_motion_params(edge_results, w0_init=1.0,
                             condition_threshold=1e6,
                             min_direction_diversity_deg=15.0,
                             max_relative_residual=0.5):
    """
    多方向边缘联合拟合运动参数（第六轮修复：加权最小二乘+病态拒识门禁）

    模型: fwhm_i = w0 + L * |cos(theta_motion - theta_normal_i)|
    其中w0是静态基线宽度，L是运动长度，theta_motion是运动方向

    门禁（S1-6-01修复）:
        - 独立方向数量 >= 2（方向差 > min_direction_diversity_deg）
        - 设计矩阵秩 >= 2
        - 条件数有限且 < condition_threshold
        - 加权相对残差 < max_relative_residual
        - L > 0 且 w0 > 0

    参数:
        edge_results: 逐边缘估计结果列表
        w0_init: 基线宽度初始值（未使用，保留接口兼容）
        condition_threshold: 条件数拒识阈值
        min_direction_diversity_deg: 最小方向多样性（度）
        max_relative_residual: 最大允许相对残差

    返回:
        result: 字典，包含L, theta_motion, w0, residual, n_edges, condition, valid, status
    """
    if len(edge_results) < 3:
        return {
            'L': None, 'theta_motion': None, 'w0': None,
            'residual': None, 'n_edges': len(edge_results),
            'condition': None, 'rank': None,
            'n_independent_directions': 0,
            'valid': False,
            'status': 'insufficient_edges (need >=3)',
        }

    # 收集数据
    normals = np.array([e['normal_angle'] for e in edge_results])
    fwhms = np.array([e['fwhm'] for e in edge_results])
    # 质量权重：对比度 * 直线性（S1-6-04修复：实际使用）
    raw_weights = np.array([max(e.get('contrast', 0), 1e-6) *
                             max(e.get('straightness', 0), 1e-6)
                             for e in edge_results])
    weights = raw_weights / np.sum(raw_weights)

    # === 门禁1：方向多样性检查 ===
    # 计算独立方向数量（方向差 > min_direction_diversity_deg 的方向组数）
    unique_normals = []
    for n in normals:
        # 无向角度：0-180度
        n_mod = n % 180
        is_new = True
        for un in unique_normals:
            diff = abs(n_mod - un)
            diff = min(diff, 180 - diff)  # 无向角度差
            if diff < min_direction_diversity_deg:
                is_new = False
                break
        if is_new:
            unique_normals.append(n_mod)

    n_independent_dirs = len(unique_normals)
    if n_independent_dirs < 2:
        return {
            'L': None, 'theta_motion': None, 'w0': None,
            'residual': None, 'n_edges': len(edge_results),
            'condition': None, 'rank': None,
            'n_independent_directions': n_independent_dirs,
            'valid': False,
            'status': f'insufficient_direction_diversity (only {n_independent_dirs} independent directions, need >=2)',
        }

    # 网格搜索运动方向（0-180度，无向）
    best_residual = np.inf
    best_params = None
    best_condition = np.inf
    best_rank = None

    for theta_deg in np.linspace(0, 180, 181):
        theta = np.deg2rad(theta_deg)
        # 投影因子: |cos(theta_motion - theta_normal)|
        proj = np.abs(np.cos(theta - normals))

        # 加权最小二乘: fwhm = w0 + L * proj
        # 设计矩阵 [1, proj]，使用 sqrt(W) 加权（S1-6-04修复）
        A = np.column_stack([np.ones_like(proj), proj])
        sqrtW = np.sqrt(weights)
        A_weighted = A * sqrtW[:, np.newaxis]
        fwhms_weighted = fwhms * sqrtW

        try:
            params, residuals, rank, s = np.linalg.lstsq(A_weighted, fwhms_weighted, rcond=None)
            w0, L = params
            # 加权残差
            pred = w0 + L * proj
            residual = np.sum(weights * (fwhms - pred)**2)
            condition = np.linalg.cond(A_weighted)

            if (residual < best_residual and L > 0 and w0 > 0
                    and np.isfinite(condition) and condition < condition_threshold):
                best_residual = residual
                best_params = (L, theta, w0, pred)
                best_condition = condition
                best_rank = rank
        except np.linalg.LinAlgError:
            continue

    if best_params is None:
        return {
            'L': None, 'theta_motion': None, 'w0': None,
            'residual': None, 'n_edges': len(edge_results),
            'condition': best_condition if np.isfinite(best_condition) else None,
            'rank': best_rank,
            'n_independent_directions': n_independent_dirs,
            'valid': False,
            'status': 'fit_failed (no valid solution with L>0, w0>0, finite condition)',
        }

    L, theta_motion, w0, pred = best_params

    # === 门禁2：相对残差检查 ===
    mean_fwhm = np.average(fwhms, weights=weights)
    relative_residual = np.sqrt(best_residual) / max(mean_fwhm, 1e-6)
    if relative_residual > max_relative_residual:
        return {
            'L': float(L), 'theta_motion': float(np.rad2deg(theta_motion)),
            'w0': float(w0),
            'residual': float(best_residual),
            'relative_residual': float(relative_residual),
            'n_edges': len(edge_results),
            'condition': float(best_condition),
            'rank': int(best_rank) if best_rank is not None else None,
            'n_independent_directions': n_independent_dirs,
            'valid': False,
            'status': f'excessive_residual (relative={relative_residual:.3f} > {max_relative_residual})',
        }

    return {
        'L': float(L),
        'theta_motion': float(np.rad2deg(theta_motion)),
        'w0': float(w0),
        'residual': float(best_residual),
        'relative_residual': float(relative_residual),
        'n_edges': len(edge_results),
        'condition': float(best_condition),
        'rank': int(best_rank) if best_rank is not None else None,
        'n_independent_directions': n_independent_dirs,
        'valid': True,
        'status': 'ok',
        'predictions': pred,
        'measurements': fwhms,
        'normals': normals,
        'weights': weights,
    }


def esf_lsf_pipeline(image, profile_length=41, min_edge_length=15,
                      motion_angle_known=None, verbose=False):
    """
    ESF/LSF完整流水线（第五轮修复版）

    参数:
        image: 输入模糊图像，uint8
        profile_length: 边缘剖面采样长度
        min_edge_length: 最小边缘段长度
        motion_angle_known: 如果已知运动方向，传入角度（度），则只做长度估计
        verbose: 是否打印详细信息

    返回:
        result: 字典
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_float = gray.astype(np.float32) / 255.0

    # 1. 自适应Canny边缘检测（S1-04）
    edges, low_thresh, high_thresh = detect_edges_adaptive(gray)

    n_edges_total = np.sum(edges > 0)
    if n_edges_total == 0:
        return {
            'valid': False,
            'status': 'no_edges_detected',
            'n_edges_total': 0,
            'message': '自适应Canny未检测到边缘，可能模糊过强或图像内容不足',
        }

    # 2. 边缘方向估计（S1-01：统一使用法线角）
    normal_angle, grad_mag = estimate_edge_orientation(gray_float, edges)

    # 3. 逐边缘ESF/LSF估计（S1-02：不跨方向混合）
    edge_results = estimate_esf_lsf_per_edge(
        gray_float, edges, normal_angle, grad_mag,
        profile_length=profile_length, min_edge_length=min_edge_length
    )

    if len(edge_results) < 3:
        # 单方向或边缘不足时（S1-6-02修复）
        if len(edge_results) >= 1:
            fwhms = [e['fwhm'] for e in edge_results]
            median_fwhm = float(np.median(fwhms))
            median_normal = float(np.median([e['normal_angle'] for e in edge_results]))

            if motion_angle_known is not None:
                # 已知运动方向：从投影宽度反推长度 L = w_proj / |cos(theta_motion - theta_normal)|
                theta_m = np.deg2rad(motion_angle_known)
                proj_factor = abs(np.cos(theta_m - median_normal))
                if proj_factor > 0.1:  # 投影因子太小则不可辨识
                    estimated_L = median_fwhm / proj_factor
                    return {
                        'valid': True,
                        'status': 'single_direction_known_angle',
                        'estimated_length': float(estimated_L),
                        'estimated_angle': float(motion_angle_known),
                        'projection_width': median_fwhm,
                        'projection_factor': float(proj_factor),
                        'baseline_width_w0': None,
                        'fit_residual': None,
                        'fit_condition': None,
                        'n_edges_total': int(n_edges_total),
                        'n_valid_edges': len(edge_results),
                        'edge_results': edge_results,
                        'fit_result': None,
                        'canny_low': float(low_thresh),
                        'canny_high': float(high_thresh),
                        'median_fwhm': median_fwhm,
                        'median_normal_angle': median_normal,
                        'message': f'单方向已知角度估计（角度={motion_angle_known}°，投影因子={proj_factor:.3f}），L≈{estimated_L:.2f}px',
                    }
                else:
                    return {
                        'valid': False,
                        'status': 'projection_factor_too_small',
                        'estimated_length': None,
                        'estimated_angle': float(motion_angle_known),
                        'projection_width': median_fwhm,
                        'projection_factor': float(proj_factor),
                        'n_edges_total': int(n_edges_total),
                        'n_valid_edges': len(edge_results),
                        'message': f'投影因子过小（{proj_factor:.3f}<0.1），运动方向与边缘法线近似垂直，长度不可辨识',
                    }
            else:
                # 未知角度：单方向只能给出投影宽度，不能反推真实长度（S1-6-02修复）
                return {
                    'valid': False,
                    'status': 'single_direction_projection_only',
                    'estimated_length': None,  # 单方向未知角度不能估计真实长度
                    'estimated_angle': None,
                    'projection_width': median_fwhm,  # 只返回投影宽度
                    'baseline_width_w0': None,
                    'fit_residual': None,
                    'fit_condition': None,
                    'n_edges_total': int(n_edges_total),
                    'n_valid_edges': len(edge_results),
                    'edge_results': edge_results,
                    'fit_result': None,
                    'canny_low': float(low_thresh),
                    'canny_high': float(high_thresh),
                    'median_fwhm': median_fwhm,
                    'median_normal_angle': median_normal,
                    'message': f'单方向仅投影宽度（{len(edge_results)}个边缘，法线角={np.rad2deg(median_normal):.1f}°），FWHM={median_fwhm:.2f}px，未知运动方向不能反推真实长度',
                }
        return {
            'valid': False,
            'status': 'insufficient_valid_edges',
            'n_edges_total': int(n_edges_total),
            'n_valid_edges': len(edge_results),
            'message': f'有效边缘不足（{len(edge_results)}/1），无法估计',
        }

    # 4. 多方向联合拟合（S1-6-02：已知角度模式）
    if motion_angle_known is not None:
        # 已知运动方向：固定θ，只拟合L和w0
        fit_result = fit_motion_with_known_angle(edge_results, motion_angle_known)
    else:
        # 未知角度：全参数联合拟合
        fit_result = joint_fit_motion_params(edge_results)

    if verbose:
        print(f"检测到边缘像素: {n_edges_total}")
        print(f"有效边缘段: {len(edge_results)}")
        if fit_result['valid']:
            print(f"估计运动长度: {fit_result['L']:.2f} px")
            print(f"估计运动方向: {fit_result['theta_motion']:.1f}°")
            print(f"基线宽度w0: {fit_result['w0']:.2f} px")
            print(f"加权残差: {fit_result['residual']:.4f}")
            print(f"设计矩阵条件数: {fit_result['condition']:.2f}")

    return {
        'valid': fit_result['valid'],
        'status': fit_result['status'] if 'status' in fit_result else ('ok' if fit_result['valid'] else 'unknown_error'),
        'estimated_length': fit_result['L'],
        'estimated_angle': fit_result['theta_motion'],
        'baseline_width_w0': fit_result['w0'],
        'fit_residual': fit_result['residual'],
        'fit_condition': fit_result['condition'],
        'n_edges_total': int(n_edges_total),
        'n_valid_edges': len(edge_results),
        'edge_results': edge_results,
        'fit_result': fit_result,
        'canny_low': float(low_thresh),
        'canny_high': float(high_thresh),
    }


def test_ideal_step():
    """
    理想阶跃解析测试（N5-04修复）

    构造单条理想阶跃边缘，验证投影宽度随L*|cos(delta_theta)|单调变化。
    """
    print("=" * 70)
    print("理想阶跃解析测试")
    print("=" * 70)

    # 构造理想阶跃图像（左暗右亮）
    size = 256
    img = np.zeros((size, size), dtype=np.float32)
    img[:, size//2:] = 1.0
    img_uint8 = (img * 255).astype(np.uint8)

    # 测试不同运动长度和方向
    test_lengths = [2, 5, 10, 15, 20, 25]
    # 边缘法线方向 = 0°（水平，因为阶跃是竖直的）
    edge_normal = 0.0

    print(f"\n边缘法线方向: {edge_normal}°（水平）")
    print(f"{'运动长度':<10} {'运动方向':<10} {'投影因子':<10} {'真值投影':<10} {'估计FWHM':<10} {'误差':<10}")
    print("-" * 65)

    results = []
    for L in test_lengths:
        for motion_deg in [0, 30, 45, 60, 90]:
            motion_rad = np.deg2rad(motion_deg)
            # 投影因子 = |cos(motion - normal)|
            proj_factor = abs(np.cos(motion_rad - np.deg2rad(edge_normal)))
            true_projection = L * proj_factor

            if true_projection < 0.5:
                continue  # 投影太小，跳过

            kernel_size = max(21, int(np.ceil(L)) * 2 + 1)
            psf = motion_blur_psf(kernel_size=kernel_size, angle=motion_deg, length=L)
            blurred = apply_blur(img_uint8, psf)

            # 用ESF/LSF流水线估计
            result = esf_lsf_pipeline(blurred, profile_length=41, min_edge_length=10)

            if result['valid']:
                if result['status'] == 'single_direction_estimate':
                    # 单方向：用median_fwhm / 投影因子估计运动长度
                    est_proj = result['median_fwhm']
                    est_L = est_proj / proj_factor if proj_factor > 0.1 else None
                else:
                    # 联合拟合
                    est_L = result['estimated_length']
                    est_proj = est_L * proj_factor

                if est_L is not None:
                    error = abs(est_L - L)
                    print(f"{L:<10} {motion_deg:<10} {proj_factor:<10.3f} {true_projection:<10.2f} "
                          f"{est_L:<10.2f} {error:<10.2f} [{result['status']}]")
                    results.append({
                        'true_L': L, 'motion_deg': motion_deg,
                        'proj_factor': proj_factor, 'true_proj': true_projection,
                        'est_L': est_L, 'error': error,
                        'status': result['status'],
                    })
                else:
                    print(f"{L:<10} {motion_deg:<10} {proj_factor:<10.3f} {true_projection:<10.2f} "
                          f"{'N/A':<10} {'N/A':<10} [proj_too_small]")
            else:
                print(f"{L:<10} {motion_deg:<10} {proj_factor:<10.3f} {true_projection:<10.2f} "
                      f"{'FAIL':<10} {result['status']}")

    if results:
        errors = [r['error'] for r in results]
        print(f"\n汇总: {len(results)}个测试，平均误差={np.mean(errors):.2f}px, "
              f"中位数误差={np.median(errors):.2f}px, 最大误差={np.max(errors):.2f}px")

    return results


def test_on_synthetic():
    """在合成模糊图上测试ESF/LSF流水线"""
    clean_dir = "../data/clean"
    result_dir = "../results/04_PSF估计"
    os.makedirs(result_dir, exist_ok=True)

    clean_files = [f for f in sorted(os.listdir(clean_dir))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not clean_files:
        print("未找到清晰图")
        return

    print(f"测试图: {clean_files[0]}")

    clean = cv2.imread(os.path.join(clean_dir, clean_files[0]))
    test_lengths = [5, 10, 15, 20]
    angle = 25

    print(f"\n{'真实长度':<10} {'估计长度':<12} {'估计方向':<12} {'基线w0':<10} {'有效边缘':<10} {'状态':<15}")
    print("-" * 75)

    for true_L in test_lengths:
        kernel_size = max(21, true_L * 2 + 1)
        psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=true_L)
        blurred = apply_blur(clean, psf)

        result = esf_lsf_pipeline(blurred, profile_length=41, min_edge_length=15, verbose=False)

        if result['valid']:
            print(f"{true_L:<10} {result['estimated_length']:<12.2f} "
                  f"{result['estimated_angle']:<12.1f}° {result['baseline_width_w0']:<10.2f} "
                  f"{result['n_valid_edges']:<10} {'OK':<15}")
        else:
            print(f"{true_L:<10} {'N/A':<12} {'N/A':<12} {'N/A':<10} "
                  f"{result.get('n_valid_edges', 0):<10} {result['status']:<15}")


if __name__ == "__main__":
    # 先运行理想阶跃解析测试
    test_ideal_step()

    print("\n" + "=" * 70)
    print("接下来在自然图像上测试:")
    print("=" * 70)
    test_on_synthetic()
