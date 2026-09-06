import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_退化模型与评价指标'))
from deconvolution_template import richardson_lucy, motion_blur_psf
from metrics_template import evaluate_restoration
import cv2
import numpy as np
from skimage.restoration import richardson_lucy as sk_rl, denoise_tv_chambolle

clean = cv2.imread('../data/clean/building.jpg')
psf = motion_blur_psf(kernel_size=21, angle=25, length=10)
blurred = cv2.imread('../data/synthetic_blur/building_motion.jpg')

baseline = evaluate_restoration(blurred, clean)
print('模糊图基线: PSNR=%.2f, SSIM=%.4f' % (baseline['PSNR'], baseline['SSIM']))
print()

# 测试1：我们自己的RL
print('--- 我们自己的RL（30次迭代）---')
our_rl = richardson_lucy(blurred, psf, iterations=30)
m = evaluate_restoration(our_rl, clean)
print('PSNR=%.2f, SSIM=%.4f' % (m['PSNR'], m['SSIM']))
print()

# 测试2：scikit-image的RL
print('--- scikit-image的RL（20次迭代）---')
img_f = blurred.astype(np.float64) / 255.0
sk_rl_result = np.zeros_like(img_f)
for c in range(3):
    sk_rl_result[:, :, c] = sk_rl(img_f[:, :, c], psf, num_iter=20)
sk_rl_u8 = np.clip(sk_rl_result, 0, 1)
sk_rl_u8 = (sk_rl_u8 * 255).astype(np.uint8)
m = evaluate_restoration(sk_rl_u8, clean)
print('PSNR=%.2f, SSIM=%.4f' % (m['PSNR'], m['SSIM']))
print()

# 测试3：我们的RL + scikit-image的TV后处理
print('--- 我们的RL + TV后处理（weight=0.02）---')
rl_f = our_rl.astype(np.float64) / 255.0
tv_result = np.zeros_like(rl_f)
for c in range(3):
    tv_result[:, :, c] = denoise_tv_chambolle(rl_f[:, :, c], weight=0.02)
tv_u8 = np.clip(tv_result, 0, 1)
tv_u8 = (tv_u8 * 255).astype(np.uint8)
m = evaluate_restoration(tv_u8, clean)
print('PSNR=%.2f, SSIM=%.4f' % (m['PSNR'], m['SSIM']))
print()

# 测试4：不同weight的TV后处理
print('--- 我们的RL + TV后处理，不同weight ---')
print('%-12s %-12s %-10s' % ('weight', 'PSNR(dB)', 'SSIM'))
print('-' * 35)
for w in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    tv_result = np.zeros_like(rl_f)
    for c in range(3):
        tv_result[:, :, c] = denoise_tv_chambolle(rl_f[:, :, c], weight=w)
    tv_u8 = np.clip(tv_result, 0, 1)
    tv_u8 = (tv_u8 * 255).astype(np.uint8)
    m = evaluate_restoration(tv_u8, clean)
    print('%-12s %-12s %-10s' % (w, m['PSNR'], m['SSIM']))
