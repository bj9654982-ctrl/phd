import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_退化模型与评价指标'))
from deconvolution_template import tv_deconvolution, motion_blur_psf
from metrics_template import evaluate_restoration
import cv2

clean = cv2.imread('../data/clean/building.jpg')
psf = motion_blur_psf(kernel_size=21, angle=25, length=10)
blurred = cv2.imread('../data/synthetic_blur/building_motion.jpg')

baseline = evaluate_restoration(blurred, clean)
print('模糊图基线: PSNR=%.2f, SSIM=%.4f' % (baseline['PSNR'], baseline['SSIM']))
print()
print('%-12s %-12s %-10s' % ('weight', 'PSNR(dB)', 'SSIM'))
print('-' * 35)

best_w = None
best_psnr = 0
for w in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
    restored = tv_deconvolution(blurred, psf, weight=w)
    m = evaluate_restoration(restored, clean)
    print('%-12s %-12s %-10s' % (w, m['PSNR'], m['SSIM']))
    if m['PSNR'] > best_psnr:
        best_psnr = m['PSNR']
        best_w = w

print()
print('最优weight: %s, 最优PSNR: %.2f dB' % (best_w, best_psnr))
