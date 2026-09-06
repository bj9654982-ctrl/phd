"""
精细K值调优：两阶段搜索（粗搜 + 细搜）
对所有测试图分别找最优K，输出K-PSNR曲线和平均最优K
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_退化模型与评价指标'))
from deconvolution_template import wiener_filter, motion_blur_psf
from metrics_template import evaluate_restoration
import cv2
import numpy as np

clean_dir = "../data/clean"
blur_dir = "../data/synthetic_blur"

# 收集所有清晰图
clean_files = []
for f in sorted(os.listdir(clean_dir)):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        clean_files.append(f)

print("=" * 70)
print("精细K值调优（运动模糊，两阶段搜索）")
print("=" * 70)
print()

all_best_ks = []
all_results = {}

for clean_file in clean_files:
    img_name = os.path.splitext(clean_file)[0]
    clean = cv2.imread(os.path.join(clean_dir, clean_file))
    psf = motion_blur_psf(kernel_size=21, angle=25, length=10)
    blurred = cv2.imread(os.path.join(blur_dir, f"{img_name}_motion.jpg"))
    if clean is None or blurred is None:
        continue

    print(f"【{clean_file}】")
    print(f"  模糊图基线 PSNR = {evaluate_restoration(blurred, clean)['PSNR']:.2f} dB")

    # ========== 第一阶段：粗搜 ==========
    # 范围 0.0001 ~ 0.1，对数均匀取30个点
    coarse_Ks = np.logspace(-4, -1, 30)  # 0.0001 到 0.1，对数间隔
    coarse_psnrs = []
    for K in coarse_Ks:
        restored = wiener_filter(blurred, psf, K=K)
        m = evaluate_restoration(restored, clean)
        coarse_psnrs.append(m['PSNR'])

    coarse_best_idx = np.argmax(coarse_psnrs)
    coarse_best_K = coarse_Ks[coarse_best_idx]
    coarse_best_psnr = coarse_psnrs[coarse_best_idx]

    print(f"  粗搜最优: K={coarse_best_K:.6f}, PSNR={coarse_best_psnr:.4f} dB")

    # ========== 第二阶段：细搜 ==========
    # 在粗搜最优值附近，取 ±50% 范围，步长 0.0005
    fine_min = max(coarse_best_K * 0.5, 0.0001)
    fine_max = coarse_best_K * 1.5
    fine_Ks = np.arange(fine_min, fine_max, 0.0005)
    fine_psnrs = []
    for K in fine_Ks:
        restored = wiener_filter(blurred, psf, K=K)
        m = evaluate_restoration(restored, clean)
        fine_psnrs.append(m['PSNR'])

    fine_best_idx = np.argmax(fine_psnrs)
    fine_best_K = fine_Ks[fine_best_idx]
    fine_best_psnr = fine_psnrs[fine_best_idx]

    print(f"  细搜最优: K={fine_best_K:.6f}, PSNR={fine_best_psnr:.4f} dB")
    print(f"  细搜范围: {fine_min:.6f} ~ {fine_max:.6f}，共{len(fine_Ks)}个点，步长0.0005")
    print()

    all_best_ks.append(fine_best_K)
    all_results[img_name] = {
        'best_K': fine_best_K,
        'best_PSNR': fine_best_psnr,
        'fine_Ks': fine_Ks,
        'fine_psnrs': fine_psnrs,
    }

# ========== 汇总 ==========
print("=" * 70)
print("汇总：各图最优K值")
print("=" * 70)
print(f"{'图像':<15} {'最优K':<12} {'最优PSNR(dB)':<15}")
print("-" * 45)
for img_name, res in all_results.items():
    print(f"{img_name:<15} {res['best_K']:<12.6f} {res['best_PSNR']:<15.4f}")

avg_K = np.mean(all_best_ks)
median_K = np.median(all_best_ks)
print("-" * 45)
print(f"{'平均最优K':<15} {avg_K:<12.6f}")
print(f"{'中位最优K':<15} {median_K:<12.6f}")
print()

# 输出K-PSNR曲线数据（用于画图）
print("=" * 70)
print("K-PSNR曲线数据（building图，细搜阶段）")
print("=" * 70)
if 'building' in all_results:
    res = all_results['building']
    print(f"{'K':<12} {'PSNR(dB)':<12}")
    print("-" * 25)
    for K, psnr in zip(res['fine_Ks'], res['fine_psnrs']):
        marker = "  ← 最优" if abs(K - res['best_K']) < 1e-6 else ""
        print(f"{K:<12.6f} {psnr:<12.4f}{marker}")

print()
print(f"结论：对运动模糊(angle=25,length=10)，")
print(f"  平均最优K = {avg_K:.6f}")
print(f"  中位最优K = {median_K:.6f}")
print(f"  建议取 K = {round(avg_K, 4)} 或 {round(median_K, 4)}")
