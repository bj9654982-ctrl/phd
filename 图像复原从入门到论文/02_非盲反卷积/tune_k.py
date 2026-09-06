import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_退化模型与评价指标'))
from deconvolution_template import wiener_filter, motion_blur_psf
from metrics_template import evaluate_restoration
import cv2

clean = cv2.imread('../data/clean/building.jpg')
psf = motion_blur_psf(kernel_size=21, angle=25, length=10)
blurred = cv2.imread('../data/synthetic_blur/building_motion.jpg')

baseline = evaluate_restoration(blurred, clean)
print('模糊图基线: PSNR=%.2f, SSIM=%.4f' % (baseline['PSNR'], baseline['SSIM']))
print()
print('%-12s %-12s %-10s' % ('K值', 'PSNR(dB)', 'SSIM'))
print('-' * 35)

best_k = None
best_psnr = 0
for K in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
    restored = wiener_filter(blurred, psf, K=K)
    m = evaluate_restoration(restored, clean)
    print('%-12s %-12s %-10s' % (K, m['PSNR'], m['SSIM']))
    if m['PSNR'] > best_psnr:
        best_psnr = m['PSNR']
        best_k = K

print()
print('最优K值: %s, 最优PSNR: %.2f dB' % (best_k, best_psnr))
