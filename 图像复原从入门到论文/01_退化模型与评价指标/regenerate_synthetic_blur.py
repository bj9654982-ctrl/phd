"""
用修复后的8倍超采样PSF重新生成所有合成模糊图
运行: python regenerate_synthetic_blur.py

修复历史:
- N-01: 噪声sigma单位错误（10.0→10/255）
- N3-01: 每张图复用同一噪声场 → 入口创建一次RNG连续使用
- N3-06: 保存失败时误报整体成功 → 累积失败列表，最终抛异常
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from degradation_template import motion_blur_psf, defocus_blur_psf, gaussian_blur_psf, apply_blur, add_gaussian_noise


def imwrite_unicode(path, img):
    """兼容中文路径的图片保存"""
    ext = os.path.splitext(path)[1]
    result, encoded = cv2.imencode(ext, img)
    if result:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def main():
    clean_dir = "../data/clean"
    blur_dir = "../data/synthetic_blur"
    os.makedirs(blur_dir, exist_ok=True)

    # 模糊参数
    motion_params = {'kernel_size': 21, 'angle': 25, 'length': 10}
    defocus_params = {'kernel_size': 21, 'radius': 5}
    gaussian_params = {'kernel_size': 21, 'sigma': 2.0}
    # 噪声标准差：8-bit DN=10 对应 [0,1] 范围的 sigma=10/255≈0.0392
    noise_sigma_dn = 10.0
    noise_sigma = noise_sigma_dn / 255.0
    master_seed = 42
    # N3-01修复: 入口创建一次RNG，连续使用，确保每张图噪声场独立
    rng = np.random.default_rng(master_seed)

    # 生成PSF
    print("生成PSF...")
    psf_motion = motion_blur_psf(**motion_params)
    psf_defocus = defocus_blur_psf(**defocus_params)
    psf_gaussian = gaussian_blur_psf(**gaussian_params)

    print(f"  运动PSF: sum={psf_motion.sum():.6f}, shape={psf_motion.shape}")
    print(f"  散焦PSF: sum={psf_defocus.sum():.6f}, shape={psf_defocus.shape}")
    print(f"  高斯PSF: sum={psf_gaussian.sum():.6f}, shape={psf_gaussian.shape}")

    # 处理每张清晰图
    clean_files = [f for f in sorted(os.listdir(clean_dir))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"\n找到 {len(clean_files)} 张清晰图，开始重新生成合成模糊图...\n")

    # N3-06修复: 累积失败列表，最终抛异常
    failed_files = []
    success_count = 0

    for clean_file in clean_files:
        img_name = os.path.splitext(clean_file)[0]
        clean_path = os.path.join(clean_dir, clean_file)
        clean = cv2.imread(clean_path)
        if clean is None:
            print(f"  [跳过] 无法读取 {clean_file}")
            failed_files.append(f"{clean_file} (读取失败)")
            continue

        print(f"处理: {clean_file} ({clean.shape[1]}x{clean.shape[0]})")

        # 运动模糊
        blurred_motion = apply_blur(clean, psf_motion)
        if not imwrite_unicode(os.path.join(blur_dir, f"{img_name}_motion.jpg"), blurred_motion):
            print(f"  [错误] 保存失败: {img_name}_motion.jpg")
            failed_files.append(f"{img_name}_motion.jpg")
            continue
        print(f"  OK {img_name}_motion.jpg")
        success_count += 1

        # 散焦模糊
        blurred_defocus = apply_blur(clean, psf_defocus)
        if not imwrite_unicode(os.path.join(blur_dir, f"{img_name}_defocus.jpg"), blurred_defocus):
            print(f"  [错误] 保存失败: {img_name}_defocus.jpg")
            failed_files.append(f"{img_name}_defocus.jpg")
            continue
        print(f"  OK {img_name}_defocus.jpg")
        success_count += 1

        # 高斯模糊
        blurred_gaussian = apply_blur(clean, psf_gaussian)
        if not imwrite_unicode(os.path.join(blur_dir, f"{img_name}_gaussian.jpg"), blurred_gaussian):
            print(f"  [错误] 保存失败: {img_name}_gaussian.jpg")
            failed_files.append(f"{img_name}_gaussian.jpg")
            continue
        print(f"  OK {img_name}_gaussian.jpg")
        success_count += 1

        # 运动模糊 + 高斯噪声（传入rng对象，连续使用）
        blurred_motion_noisy = add_gaussian_noise(blurred_motion, sigma=noise_sigma, seed=rng)
        if not imwrite_unicode(os.path.join(blur_dir, f"{img_name}_motion_noisy.jpg"), blurred_motion_noisy):
            print(f"  [错误] 保存失败: {img_name}_motion_noisy.jpg")
            failed_files.append(f"{img_name}_motion_noisy.jpg")
            continue
        print(f"  OK {img_name}_motion_noisy.jpg (noise sigma={noise_sigma_dn} DN, master_seed={master_seed})")
        success_count += 1

    # N3-06修复: 有失败时抛异常，不允许误报整体成功
    if failed_files:
        print(f"\n[ERROR] 生成完成但有 {len(failed_files)} 个文件失败:")
        for f in failed_files:
            print(f"  - {f}")
        print(f"成功: {success_count} 个文件")
        raise RuntimeError(f"合成模糊图生成失败: {len(failed_files)} 个文件未成功保存")

    print(f"\n完成！所有 {success_count} 张合成模糊图已重新生成到 {blur_dir}")
    print(f"参数: 运动 angle={motion_params['angle']}°, length={motion_params['length']}; "
          f"散焦 radius={defocus_params['radius']}; 高斯 sigma={gaussian_params['sigma']}; "
          f"噪声 sigma={noise_sigma:.4f} [0,1]")


if __name__ == "__main__":
    main()
