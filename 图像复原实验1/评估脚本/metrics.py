"""
图像复原评估工具：PSNR / SSIM / LPIPS
PSNR 和 SSIM 纯 PyTorch 实现，无需额外依赖；
LPIPS 需安装 lpips 库（可选）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def _to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """RGB -> YCbCr 的 Y 通道（与 MATLAB 评估一致，GOPRO 标准做法）。
    img: (B, 3, H, W) in [0, 1]
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    return y / 255.0


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 0,
    test_y_channel: bool = True,
) -> float:
    """计算 PSNR (dB)。
    pred, target: (B, 3, H, W) in [0, 1]
    """
    if crop_border > 0:
        pred = pred[:, :, crop_border:-crop_border, crop_border:-crop_border]
        target = target[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        pred = _to_y_channel(pred)
        target = _to_y_channel(target)

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    # 避免除零
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()


def _gaussian_kernel(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    """生成 1D 高斯核。"""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    """创建 2D 高斯窗口，用于 SSIM。"""
    kernel_1d = _gaussian_kernel(window_size, 1.5).unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 0,
    test_y_channel: bool = True,
    window_size: int = 11,
) -> float:
    """计算 SSIM。
    pred, target: (B, 3, H, W) in [0, 1]
    """
    if crop_border > 0:
        pred = pred[:, :, crop_border:-crop_border, crop_border:-crop_border]
        target = target[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        pred = _to_y_channel(pred)
        target = _to_y_channel(target)

    channel = pred.shape[1]
    window = _create_window(window_size, channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def calculate_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model: Optional[torch.nn.Module] = None,
) -> float:
    """计算 LPIPS（需要安装 lpips 库）。
    pred, target: (B, 3, H, W) in [0, 1]，会自动归一化到 [-1, 1]。
    """
    if lpips_model is None:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(pred.device)
        except ImportError:
            return float("nan")

    pred_norm = pred * 2.0 - 1.0
    target_norm = target * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = lpips_model(pred_norm, target_norm)
    return lpips_val.mean().item()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_lpips: bool = False,
) -> dict:
    """在整个测试集上评估模型，返回平均 PSNR/SSIM/LPIPS。"""
    model.eval()
    psnr_list = []
    ssim_list = []
    lpips_list = []

    lpips_model = None
    if compute_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device)
        except ImportError:
            pass

    for batch in dataloader:
        if len(batch) == 3:
            blur, sharp, _ = batch
        else:
            blur, sharp = batch

        blur = blur.to(device)
        sharp = sharp.to(device)

        # 全图推理（可能需要 pad 到 8 的倍数）
        _, _, h, w = blur.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            blur_pad = F.pad(blur, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            blur_pad = blur

        with torch.amp.autocast("cuda"):
            restored = model(blur_pad)

        restored = restored[:, :, :h, :w].clamp(0, 1)

        psnr_list.append(calculate_psnr(restored, sharp))
        ssim_list.append(calculate_ssim(restored, sharp))
        if lpips_model is not None:
            lpips_list.append(calculate_lpips(restored, sharp, lpips_model))

    results = {
        "PSNR": sum(psnr_list) / len(psnr_list),
        "SSIM": sum(ssim_list) / len(ssim_list),
        "num_images": len(psnr_list),
    }
    if lpips_list:
        results["LPIPS"] = sum(lpips_list) / len(lpips_list)

    return results
