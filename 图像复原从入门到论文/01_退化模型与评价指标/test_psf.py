"""
PSF 单元测试
验证运动模糊PSF的长度、质心、能量、角度等性质
运行: python test_psf.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from degradation_template import motion_blur_psf, defocus_blur_psf, gaussian_blur_psf


def test_motion_psf_energy():
    """测试：PSF元素和为1（能量守恒）"""
    for length in [1, 2, 3, 5, 8, 10, 15, 20, 25]:
        for angle in [0, 25, 45, 90, 135]:
            psf = motion_blur_psf(kernel_size=31, angle=angle, length=length)
            total = psf.sum()
            assert abs(total - 1.0) < 1e-5, f"length={length}, angle={angle}: sum={total}"
    print("[OK] 能量守恒测试通过（所有PSF和为1）")


def test_motion_psf_centroid():
    """测试：PSF质心居中"""
    kernel_size = 31
    center = kernel_size // 2
    y_coords, x_coords = np.mgrid[0:kernel_size, 0:kernel_size]

    for length in [3, 5, 10, 15, 20]:
        for angle in [0, 25, 45, 90]:
            psf = motion_blur_psf(kernel_size=kernel_size, angle=angle, length=length)
            cx = (psf * x_coords).sum()
            cy = (psf * y_coords).sum()
            assert abs(cx - center) < 0.5, f"length={length}, angle={angle}: cx={cx:.2f}, expected {center}"
            assert abs(cy - center) < 0.5, f"length={length}, angle={angle}: cy={cy:.2f}, expected {center}"
    print("[OK] 质心居中测试通过")


def test_motion_psf_length_horizontal():
    """测试：0°水平运动模糊的非零支撑长度约等于请求长度"""
    kernel_size = 51
    center = kernel_size // 2

    for length in [3, 5, 8, 10, 15, 20, 25]:
        psf = motion_blur_psf(kernel_size=kernel_size, angle=0, length=length)
        # 取中心行，统计非零像素范围
        row = psf[center, :]
        nonzero = np.where(row > 1e-6)[0]
        if len(nonzero) > 0:
            span = nonzero[-1] - nonzero[0] + 1
            # 水平运动的支撑像素数应约等于length（抗锯齿可能多1-2个像素）
            assert abs(span - length) <= 2, f"length={length}: span={span}, expected ~{length}"
    print("[OK] 水平运动长度测试通过（非零支撑约等于请求长度）")


def test_motion_psf_angle_direction():
    """测试：运动方向与角度一致"""
    kernel_size = 31

    # 0°应该是水平的（x方向延伸范围 > y方向延伸范围）
    psf_0 = motion_blur_psf(kernel_size=kernel_size, angle=0, length=15)
    nonzero_0 = np.where(psf_0 > 1e-6)
    x_range_0 = nonzero_0[1].max() - nonzero_0[1].min()  # 列方向范围
    y_range_0 = nonzero_0[0].max() - nonzero_0[0].min()  # 行方向范围
    assert x_range_0 > y_range_0, f"0°应该是水平运动: x_range={x_range_0}, y_range={y_range_0}"

    # 90°应该是垂直的（y方向延伸范围 > x方向延伸范围）
    psf_90 = motion_blur_psf(kernel_size=kernel_size, angle=90, length=15)
    nonzero_90 = np.where(psf_90 > 1e-6)
    x_range_90 = nonzero_90[1].max() - nonzero_90[1].min()
    y_range_90 = nonzero_90[0].max() - nonzero_90[0].min()
    assert y_range_90 > x_range_90, f"90°应该是垂直运动: x_range={x_range_90}, y_range={y_range_90}"

    print("[OK] 运动方向测试通过（0°水平，90°垂直）")


def test_defocus_psf():
    """测试：散焦模糊PSF"""
    psf = defocus_blur_psf(kernel_size=31, radius=10)
    assert abs(psf.sum() - 1.0) < 1e-5, f"散焦PSF和={psf.sum()}"
    # 散焦应该是对称的
    assert np.allclose(psf, psf.T), "散焦PSF应该对称"
    print("[OK] 散焦模糊PSF测试通过")


def test_gaussian_psf():
    """测试：高斯模糊PSF"""
    psf = gaussian_blur_psf(kernel_size=31, sigma=3.0)
    assert abs(psf.sum() - 1.0) < 1e-5, f"高斯PSF和={psf.sum()}"
    assert np.allclose(psf, psf.T), "高斯PSF应该对称"
    print("[OK] 高斯模糊PSF测试通过")


def test_invalid_inputs():
    """测试：非法输入应该抛出异常"""
    try:
        motion_blur_psf(kernel_size=31, angle=0, length=0)
        assert False, "length=0应该抛出异常"
    except ValueError:
        pass

    try:
        motion_blur_psf(kernel_size=30, angle=0, length=10)  # 偶数kernel_size
        assert False, "偶数kernel_size应该抛出异常"
    except ValueError:
        pass

    print("[OK] 非法输入测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("PSF 单元测试")
    print("=" * 60)

    test_motion_psf_energy()
    test_motion_psf_centroid()
    test_motion_psf_length_horizontal()
    test_motion_psf_angle_direction()
    test_defocus_psf()
    test_gaussian_psf()
    test_invalid_inputs()

    print("\n" + "=" * 60)
    print("所有测试通过！[OK]")
    print("=" * 60)
