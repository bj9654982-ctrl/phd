"""
Transformer 去模糊直观示例（PyTorch版）
========================================
用一个极简的窗口自注意力 Transformer 网络，对合成的运动模糊图像做训练，
可视化：清晰GT → 模糊输入 → 训练中间过程 → 最终输出 → 注意力图

使用方式：
  # 使用内置合成图像（默认）
  python transformer_deblur_demo.py

  # 使用用户自定义图片
  python transformer_deblur_demo.py --image "路径/到/你的图片.jpg"

  # 自定义参数
  python transformer_deblur_demo.py --image "test.jpg" --size 128 --steps 1000 --lr 0.001

依赖：torch, numpy, matplotlib, Pillow
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# 必须在 import matplotlib 之前过滤字体警告
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib
matplotlib.use("Agg")
# 必须在 import pyplot 之前设置字体和减号参数
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 关键：用普通连字符代替数学减号U+2212
matplotlib.rcParams["axes.formatter.use_mathtext"] = False
matplotlib.rcParams["mathtext.default"] = "regular"
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import sys

# ============================================================
# 配置区域：直接在这里修改参数，无需命令行参数
# ============================================================
IMAGE_PATH = "building.jpg"                        # 输入图片路径（改成你自己的图片路径）
OUTPUT_PATH = "transformer_deblur_result.png"    # 输出结果图路径
BLURRED_OUTPUT_PATH = "blurred_input.png"         # 生成的模糊图像保存路径
RESTORED_OUTPUT_PATH = "restored_result.png"      # 复原后的图像保存路径
MODEL_SAVE_PATH = "deblur_model.pt"               # 训练好的模型权重保存路径
LOAD_MODEL = False                                  # 是否加载已保存的模型（True则跳过训练，直接用已有模型推理）
TARGET_PSNR = None                                  # 目标PSNR，训练中达到此值则提前停止（设为None则不启用早停，跑满全部步数）
IMAGE_SIZE = 160                                   # 图像处理尺寸（建议64-256，越大越慢）
MOTION_LENGTH = 15                                  # 运动模糊长度（像素，越大模糊越严重）
MOTION_ANGLE = 28.0                                 # 运动模糊角度（度）
TRAIN_STEPS = 5000                                 # 训练步数（越多恢复越充分但越慢）
LEARNING_RATE = 5e-4                               # 学习率
DIM = 64                                            # 特征维度（越大模型容量越大）
NUM_BLOCKS = 8                                      # Transformer Block数量
WINDOW_SIZE = 8                                     # 自注意力窗口大小（必须能整除 IMAGE_SIZE）
NUM_HEADS = 8                                       # 注意力头数（必须能整除 DIM）
USE_GRADIENT_LOSS = True                            # 是否启用梯度损失（鼓励边缘清晰）
USE_SSIM_LOSS = True                                # 是否启用SSIM损失（鼓励结构相似）
USE_FREQ_LOSS = True                                # 是否启用频域损失（FFT，恢复高频细节）
USE_TTA = True                                      # 是否启用测试时增强（翻转推理取平均）
USE_SE_ATTENTION = True                             # 是否启用通道注意力SE模块（增强特征表达）
USE_UNET = True                                      # 是否使用三尺度U-Net+转置自注意力（Restormer风格，效果更好）
# U-Net网络参数
UNET_EMBED_DIM = 32                                 # U-Net初始通道数（24/32/48）
UNET_DEPTHS = (2, 2, 4)                             # 三个阶段的Transformer Block数量
UNET_HEADS = (2, 4, 8)                              # 三个阶段的注意力头数
UNET_EXPANSION = 2.0                                # FFN扩展倍率
# 训练数据增强
RANDOM_BLUR = False                                  # 是否随机生成多种模糊（单图训练建议False，多图训练建议True）
NOISE_STD = 0.01                                     # 随机噪声标准差
NUM_PATCHES_PER_EPOCH = 50                          # 每个epoch随机裁剪的patch数量（模拟多图训练）
# ============================================================

# ============================================================
# 1. 图像加载与生成
# ============================================================
def load_user_image(image_path, target_size=128):
    """
    加载用户自定义图片，resize到目标尺寸（保持比例，居中裁剪）
    返回: (H, W, 3) float32 数组，值域 [0, 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法打开图片文件: {image_path}\n错误: {e}")

    w, h = img.size
    print(f"  原始图片尺寸: {w} x {h}")

    # 居中裁剪为正方形
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))

    # resize到目标尺寸
    img = img.resize((target_size, target_size), Image.LANCZOS)
    print(f"  处理后尺寸: {target_size} x {target_size}（居中裁剪+缩放）")

    return np.array(img).astype(np.float32) / 255.0


def make_sharp_image(size=128):
    """生成一张有清晰边缘和文字的测试图像"""
    img = Image.new("RGB", (size, size), (20, 20, 30))
    draw = ImageDraw.Draw(img)

    # 画几个几何图形（有清晰边缘）
    draw.rectangle([15, 15, 55, 55], fill=(220, 80, 60))
    draw.ellipse([70, 20, 115, 65], fill=(60, 180, 100))
    draw.polygon([(20, 80), (60, 70), (50, 115), (10, 100)], fill=(70, 120, 220))
    draw.rectangle([75, 80, 118, 118], outline=(240, 200, 60), width=3)

    # 画几条细线（运动模糊后最难恢复的）
    for i in range(5):
        y = 95 + i * 5
        draw.line([(75, y), (118, y)], fill=(255, 255, 255), width=1)

    # 加文字
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 60), "DEBLUR", fill=(255, 255, 255), font=font)

    return np.array(img).astype(np.float32) / 255.0


def motion_blur(image, motion_length=15, motion_angle=28.0):
    """对图像施加运动模糊"""
    kernel = np.zeros((motion_length, motion_length), dtype=np.float32)
    kernel[motion_length // 2, :] = 1.0 / motion_length

    # 旋转核（expand=True确保不被裁剪，之后中心裁剪回严格尺寸）
    kernel_pil = Image.fromarray((kernel * 255).astype(np.uint8))
    kernel_pil = kernel_pil.rotate(motion_angle, expand=True)
    kernel = np.array(kernel_pil).astype(np.float32)

    # 严格控制核尺寸：中心裁剪回 motion_length × motion_length
    k_h, k_w = kernel.shape
    if k_h > motion_length or k_w > motion_length:
        start_h = (k_h - motion_length) // 2
        start_w = (k_w - motion_length) // 2
        kernel = kernel[start_h:start_h + motion_length, start_w:start_w + motion_length]

    # 归一化（裁剪后sum可能变化，重新归一化保证亮度不变）
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel = kernel / kernel_sum

    # 对每个通道做卷积（用scipy风格的简单实现）
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    blurred = np.zeros_like(image)
    for c in range(3):
        padded = np.pad(image[:, :, c], ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        from numpy.fft import fft2, ifft2
        F_padded = fft2(padded)
        F_kernel = fft2(kernel, s=padded.shape)
        blurred[:, :, c] = np.real(ifft2(F_padded * F_kernel))[pad_h:pad_h+image.shape[0], pad_w:pad_w+image.shape[1]]
    return np.clip(blurred, 0, 1)


# ============================================================
# 2. 极简 Transformer 去模糊网络
#    结构：Patch Embedding → N个 Transformer块 → 重建
#    每个Transformer块 = 窗口自注意力(W-MSA) + 前馈网络(FFN)
# ============================================================
class WindowAttention(nn.Module):
    """窗口自注意力：把特征图分成不重叠窗口，每个窗口内做自注意力"""
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        ws = self.window_size

        # 确保能被窗口大小整除
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        # (B,C,H,W) → (B,H,W,C)
        x = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x)

        # 分成窗口：(B, H/ws, W/ws, ws, ws, C) → (B*num_windows, ws*ws, C)
        x_norm = x_norm.reshape(B, Hp // ws, ws, Wp // ws, ws, C)
        x_norm = x_norm.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)

        # QKV
        qkv = self.qkv(x_norm).reshape(-1, ws * ws, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nw, heads, N, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*nw, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn_weights = attn  # 保存用于可视化

        # 加权求和
        out = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        out = self.proj(out)

        # 还原窗口结构
        out = out.reshape(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        out = out.permute(0, 3, 1, 2)  # (B,C,H,W)

        # 裁剪掉padding
        out = out[:, :, :H, :W]

        return out, attn_weights


class TransformerBlock(nn.Module):
    """Transformer块：窗口自注意力 + FFN + SE通道注意力，带残差连接"""
    def __init__(self, dim, window_size=8, num_heads=4, mlp_ratio=2.0, use_se=True):
        super().__init__()
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # SE通道注意力（可选）
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(dim, reduction=4)

    def forward(self, x):
        # 自注意力 + 残差
        attn_out, attn_weights = self.attn(x)
        x = x + attn_out

        # FFN + 残差
        B, C, H, W = x.shape
        x_t = x.permute(0, 2, 3, 1)  # (B,H,W,C)
        x_t = x_t + self.ffn(self.norm2(x_t))
        x = x_t.permute(0, 3, 1, 2)

        # SE通道注意力
        if self.use_se:
            x = self.se(x)

        return x, attn_weights


class SimpleTransformerDeblur(nn.Module):
    """
    极简 Transformer 去模糊网络
    输入：模糊图像 (B,3,H,W)
    输出：残差图像 (B,3,H,W)，最终恢复 = 输入 + 残差
    """
    def __init__(self, in_ch=3, dim=32, num_blocks=4, window_size=8, num_heads=4, use_se=True):
        super().__init__()
        # Patch Embedding：用卷积把图像映射到特征空间
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)

        # 多个 Transformer 块
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, window_size, num_heads, use_se=use_se)
            for _ in range(num_blocks)
        ])

        # 加深重建头：3层卷积 + GELU，把特征映射回3通道残差
        self.recon = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, in_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        residual = x
        feat = self.patch_embed(x)

        attn_maps = []
        for block in self.blocks:
            feat, attn_w = block(feat)
            attn_maps.append(attn_w)

        out = self.recon(feat)
        return residual + out, attn_maps


# ============================================================
# 3. 计算 PSNR
# ============================================================
def compute_psnr(img1, img2):
    """计算PSNR，输入为numpy数组 [0,1]"""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def gradient_loss(pred, target):
    """
    梯度损失：鼓励预测图像的边缘/梯度与目标一致，提升清晰度
    输入: (B, C, H, W) tensor
    """
    # 水平方向梯度
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    # 垂直方向梯度
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def ssim_loss(pred, target, window_size=11, reduction='mean'):
    """
    SSIM损失（结构相似性）：鼓励预测图像在结构上与目标一致
    输入: (B, C, H, W) tensor，值域 [0, 1]
    返回: 1 - SSIM（损失，越小越好）
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 创建高斯窗口
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(pred.shape[1], 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=pred.shape[1])
    mu2 = F.conv2d(target, window, padding=pad, groups=target.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=target.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=pred.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if reduction == 'mean':
        return 1 - ssim_map.mean()
    return 1 - ssim_map.mean(dim=[1, 2, 3])


class SEBlock(nn.Module):
    """
    通道注意力SE模块：自适应调整每个通道的权重，增强重要特征、抑制无用特征
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 8)),
            nn.GELU(),
            nn.Linear(max(dim // reduction, 8), dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        y = x.mean(dim=[2, 3])  # 全局平均池化 (B, C)
        y = self.fc(y)  # (B, C)
        return x * y.unsqueeze(-1).unsqueeze(-1)  # 通道加权


# ============================================================
# Restormer风格模块：转置自注意力 + 门控FFN + 三尺度U-Net
# ============================================================
class LayerNorm2d(nn.Module):
    """在通道维度做LayerNorm，保持 (B,C,H,W) 格式"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class TransposedSelfAttention(nn.Module):
    """
    转置自注意力（Restormer核心创新）：在通道维度做注意力，而非空间维度
    复杂度 O(C²) 而非 O((HW)²)，可以全局建模，不需要窗口限制
    """
    def __init__(self, channels, heads):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, padding=1,
            groups=channels * 3, bias=False
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        C_per_head = C // self.heads

        def split_heads(t):
            return t.reshape(B, self.heads, C_per_head, H * W)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # 通道维度注意力：(B, heads, C_per_head, C_per_head)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, C_per_head, H*W)
        out = out.reshape(B, C, H, W)
        return self.project(out)


class GatedFeedForward(nn.Module):
    """门控前馈网络：1x1卷积 + 3x3深度可分离卷积 + 门控机制（GLU风格）"""
    def __init__(self, channels, expansion=2.0):
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1,
            groups=hidden * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        first, gate = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * gate)


class RestormerBlock(nn.Module):
    """Restormer风格Transformer Block：转置自注意力 + 门控FFN，带残差和LayerNorm"""
    def __init__(self, channels, heads, expansion=2.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = TransposedSelfAttention(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedFeedForward(channels, expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def make_stage(channels, depth, heads, expansion):
    """创建一个阶段的多个RestormerBlock"""
    return nn.Sequential(*(RestormerBlock(channels, heads, expansion) for _ in range(depth)))


class DeblurTransformerUNet(nn.Module):
    """
    三尺度U-Net Transformer去模糊网络（迷你版Restormer）
    编码器：dim → 2dim → 4dim，逐步下采样
    解码器：4dim → 2dim → dim，逐步上采样，带跳跃连接
    输出：残差学习，最终恢复 = 输入 + 残差
    """
    def __init__(self, embed_dim=32, depths=(2, 2, 4), heads=(2, 4, 8), expansion=2.0):
        super().__init__()
        depths = tuple(depths)
        heads = tuple(heads)
        dim1, dim2, dim3 = embed_dim, embed_dim * 2, embed_dim * 4
        self.model_config = {"embed_dim": embed_dim, "depths": depths, "heads": heads, "expansion": expansion}

        # 编码器
        self.input_proj = nn.Conv2d(3, dim1, kernel_size=3, padding=1, bias=False)
        self.encoder1 = make_stage(dim1, depths[0], heads[0], expansion)
        self.down1 = nn.Conv2d(dim1, dim2, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder2 = make_stage(dim2, depths[1], heads[1], expansion)
        self.down2 = nn.Conv2d(dim2, dim3, kernel_size=3, stride=2, padding=1, bias=False)
        self.bottleneck = make_stage(dim3, depths[2], heads[2], expansion)

        # 解码器
        self.up2 = nn.ConvTranspose2d(dim3, dim2, kernel_size=2, stride=2, bias=False)
        self.reduce2 = nn.Conv2d(dim2 * 2, dim2, kernel_size=1, bias=False)
        self.decoder2 = make_stage(dim2, depths[1], heads[1], expansion)
        self.up1 = nn.ConvTranspose2d(dim2, dim1, kernel_size=2, stride=2, bias=False)
        self.reduce1 = nn.Conv2d(dim1 * 2, dim1, kernel_size=1, bias=False)
        self.decoder1 = make_stage(dim1, depths[0], heads[0], expansion)
        self.output_proj = nn.Conv2d(dim1, 3, kernel_size=3, padding=1, bias=False)

    @staticmethod
    def _pad_to_multiple(x, multiple=4):
        """把输入pad到4的倍数，保证下采样/上采样尺寸匹配"""
        H, W = x.shape[-2:]
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h or pad_w:
            mode = "reflect" if H > pad_h and W > pad_w else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x, H, W

    def forward(self, x):
        x, orig_H, orig_W = self._pad_to_multiple(x)
        residual = x

        # 编码器
        enc1 = self.encoder1(self.input_proj(x))
        enc2 = self.encoder2(self.down1(enc1))
        latent = self.bottleneck(self.down2(enc2))

        # 解码器 + 跳跃连接
        dec2 = self.up2(latent)
        dec2 = self.decoder2(self.reduce2(torch.cat((dec2, enc2), dim=1)))
        dec1 = self.up1(dec2)
        dec1 = self.decoder1(self.reduce1(torch.cat((dec1, enc1), dim=1)))

        # 残差学习（注意：训练时不clamp，避免梯度消失；评估/保存时再clamp）
        restored = self.output_proj(dec1) + residual
        restored = restored[..., :orig_H, :orig_W]
        return restored


# ============================================================
# 随机模糊生成：抗锯齿运动模糊核 + 高斯模糊核
# ============================================================
def gaussian_kernel(size, sigma):
    """生成高斯模糊核"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    k1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    k1d /= k1d.sum()
    return torch.outer(k1d, k1d)


def motion_kernel_antialiased(size, angle_degrees):
    """
    生成抗锯齿运动模糊核（超采样+双线性插值）
    比简单PIL旋转更精确、更平滑
    """
    import math
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    angle = math.radians(angle_degrees)
    samples = max(16, size * 8)  # 超采样
    for offset in np.linspace(-center, center, samples):
        x = center + offset * math.cos(angle)
        y = center + offset * math.sin(angle)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        dx, dy = x - x0, y - y0
        for yy, wy in ((y0, 1.0 - dy), (y0 + 1, dy)):
            for xx, wx in ((x0, 1.0 - dx), (x0 + 1, dx)):
                if 0 <= xx < size and 0 <= yy < size:
                    kernel[yy, xx] += wx * wy
    kernel /= max(float(kernel.sum()), 1e-8)
    return torch.from_numpy(kernel)


def apply_kernel_tensor(image_tensor, kernel):
    """对 (C,H,W) tensor 应用卷积核"""
    C = image_tensor.shape[0]
    pad = kernel.shape[-1] // 2
    padded = F.pad(image_tensor.unsqueeze(0), (pad,) * 4, mode="reflect")
    weight = kernel.to(image_tensor).expand(C, 1, -1, -1)
    return F.conv2d(padded, weight, groups=C).squeeze(0)


def random_blur_tensor(sharp_tensor, noise_std=0.01):
    """
    对清晰图 (C,H,W) tensor 随机施加模糊：
    50%概率高斯模糊，50%概率运动模糊，参数全部随机，再加随机噪声
    """
    if torch.rand(1).item() < 0.5:
        size = int(torch.randint(3, 12, (1,)).item())
        if size % 2 == 0:
            size += 1
        sigma = float(torch.empty(1).uniform_(0.6, max(0.7, size / 3.0)).item())
        kernel = gaussian_kernel(size, sigma)
    else:
        size = int(torch.randint(5, 16, (1,)).item())
        if size % 2 == 0:
            size += 1
        angle = float(torch.empty(1).uniform_(0.0, 180.0).item())
        kernel = motion_kernel_antialiased(size, angle)
    blurred = apply_kernel_tensor(sharp_tensor, kernel)
    if noise_std > 0:
        blurred = blurred + torch.randn_like(blurred) * float(torch.empty(1).uniform_(0.0, noise_std).item())
    return blurred.clamp(0.0, 1.0)


# ============================================================
# 频域损失
# ============================================================
def frequency_loss(pred, target):
    """FFT频域损失：约束高频成分，恢复边缘和纹理细节"""
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return torch.mean(torch.abs(pred_fft - target_fft))


def random_augment(blur_t, sharp_t):
    """
    随机数据增强：对模糊图和清晰图做相同的随机翻转/旋转
    输入: (B, C, H, W) tensor
    """
    # 随机水平翻转
    if torch.rand(1).item() > 0.5:
        blur_t = torch.flip(blur_t, dims=[3])
        sharp_t = torch.flip(sharp_t, dims=[3])
    # 随机垂直翻转
    if torch.rand(1).item() > 0.5:
        blur_t = torch.flip(blur_t, dims=[2])
        sharp_t = torch.flip(sharp_t, dims=[2])
    # 随机90度旋转（0/1/2/3次）
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        blur_t = torch.rot90(blur_t, k=k, dims=[2, 3])
        sharp_t = torch.rot90(sharp_t, k=k, dims=[2, 3])
    return blur_t, sharp_t


def tta_inference(model, x):
    """
    测试时增强（TTA）：8个视角推理后取平均
    兼容单输出网络（U-Net）和双输出网络（旧网络带attn）
    """
    model.eval()

    def _out(model_out):
        return model_out[0] if isinstance(model_out, tuple) else model_out

    with torch.no_grad():
        results = []

        # 视角1：原图
        results.append(_out(model(x)))

        # 视角2：水平翻转
        out = _out(model(torch.flip(x, dims=[3])))
        results.append(torch.flip(out, dims=[3]))

        # 视角3：垂直翻转
        out = _out(model(torch.flip(x, dims=[2])))
        results.append(torch.flip(out, dims=[2]))

        # 视角4：水平+垂直翻转
        out = _out(model(torch.flip(x, dims=[2, 3])))
        results.append(torch.flip(out, dims=[2, 3]))

        # 视角5：旋转90度
        x_rot = torch.rot90(x, k=1, dims=[2, 3])
        out = _out(model(x_rot))
        results.append(torch.rot90(out, k=-1, dims=[2, 3]))

        # 视角6：旋转90度 + 水平翻转
        x_rot_hflip = torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3])
        out = _out(model(x_rot_hflip))
        results.append(torch.rot90(torch.flip(out, dims=[3]), k=-1, dims=[2, 3]))

        # 视角7：旋转90度 + 垂直翻转
        x_rot_vflip = torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
        out = _out(model(x_rot_vflip))
        results.append(torch.rot90(torch.flip(out, dims=[2]), k=-1, dims=[2, 3]))

        # 视角8：旋转90度 + 水平+垂直翻转
        x_rot_hvflip = torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2, 3])
        out = _out(model(x_rot_hvflip))
        results.append(torch.rot90(torch.flip(out, dims=[2, 3]), k=-1, dims=[2, 3]))

        # 取平均
        return torch.stack(results, dim=0).mean(dim=0)


# ============================================================
# 4. 训练 + 可视化
# ============================================================
def train_and_visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载/生成数据 ---
    print("\n[1/4] 加载图像并生成运动模糊...")
    # 判断是否使用用户图片（文件存在才用，否则回退到合成图）
    is_user_image = args.image is not None and os.path.exists(args.image)

    if is_user_image:
        print(f"  使用用户图片: {args.image}")
        try:
            sharp = load_user_image(args.image, target_size=args.size)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n错误: {e}")
            sys.exit(1)
        image_name = os.path.basename(args.image)
    else:
        if args.image is not None and not os.path.exists(args.image):
            print(f"  提示: 指定的图片 '{args.image}' 不存在，自动回退到内置合成图像")
        print(f"  使用内置合成图像（尺寸: {args.size}）")
        sharp = make_sharp_image(size=args.size)
        image_name = "合成测试图"

    blurred = motion_blur(sharp, motion_length=args.motion_length, motion_angle=args.motion_angle)
    print(f"  运动模糊长度: {args.motion_length}px, 模糊角度: {args.motion_angle}度")
    print(f"  模糊图像 PSNR: {compute_psnr(blurred, sharp):.2f} dB")

    # 保存模糊图像到当前路径
    blurred_pil = Image.fromarray((blurred * 255).astype(np.uint8))
    blurred_pil.save(BLURRED_OUTPUT_PATH)
    print(f"  模糊图像已保存: {BLURRED_OUTPUT_PATH}")

    # 转tensor (1,3,H,W)
    sharp_t = torch.from_numpy(sharp).permute(2, 0, 1).unsqueeze(0).to(device)
    blurred_t = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).to(device)

    # --- 创建模型 ---
    if USE_UNET:
        print(f"\n[2/4] 创建三尺度U-Net Transformer模型（Restormer风格，embed_dim={UNET_EMBED_DIM}, depths={UNET_DEPTHS}, heads={UNET_HEADS}）...")
        model = DeblurTransformerUNet(
            embed_dim=UNET_EMBED_DIM, depths=UNET_DEPTHS,
            heads=UNET_HEADS, expansion=UNET_EXPANSION
        ).to(device)
    else:
        print(f"\n[2/4] 创建直筒型Transformer模型（{args.blocks}个Block, 窗口{args.window}×{args.window}）...")
        model = SimpleTransformerDeblur(
            in_ch=3, dim=args.dim, num_blocks=args.blocks,
            window_size=args.window, num_heads=args.heads,
            use_se=USE_SE_ATTENTION
        ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # AdamW优化器（带weight_decay，比Adam泛化更好）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # --- 训练或加载已有模型 ---
    num_steps = args.steps
    save_steps = sorted(set([1, 10, 50, 100, 200, 500, num_steps]))
    save_steps = [s for s in save_steps if s <= num_steps]
    history = {}
    losses = []

    if LOAD_MODEL and os.path.exists(MODEL_SAVE_PATH):
        # 加载已保存的模型，跳过训练
        print(f"\n[3/4] 加载已保存的模型权重: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            if USE_TTA:
                out = tta_inference(model, blurred_t)
            else:
                model_out = model(blurred_t)
                out = model_out[0] if isinstance(model_out, tuple) else model_out
            # U-Net没有注意力图，用零矩阵占位
            attn_maps = [torch.zeros(1, 1, 64, 64, device=device)]
            history[num_steps] = {
                "image": out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy(),
                "attn": attn_maps,
            }
        losses = [0.0]
        actual_steps = 0
        early_stopped = False
        print(f"  模型加载完成，跳过训练")
    else:
        # 正常训练
        print(f"\n[3/4] 开始训练 {args.steps} 步...")
        if TARGET_PSNR is not None:
            print(f"  早停目标: PSNR > {TARGET_PSNR} dB")
        model.train()
        early_stopped = False
        actual_steps = num_steps
        for step in range(1, num_steps + 1):
            optimizer.zero_grad()

            # 随机模糊：每步对清晰图随机生成不同的模糊（高斯+运动，参数随机）
            if RANDOM_BLUR:
                with torch.no_grad():
                    # 从清晰图随机裁剪patch（模拟多图训练）
                    _, _, H, W = sharp_t.shape
                    patch_size = min(H, W, max(64, args.size // 2))
                    if H > patch_size and W > patch_size:
                        top = torch.randint(0, H - patch_size + 1, (1,)).item()
                        left = torch.randint(0, W - patch_size + 1, (1,)).item()
                        sharp_patch = sharp_t[:, :, top:top+patch_size, left:left+patch_size]
                    else:
                        sharp_patch = sharp_t
                    # 随机生成模糊
                    blurred_rand = random_blur_tensor(sharp_patch[0], NOISE_STD).unsqueeze(0)
                    aug_blur, aug_sharp = random_augment(blurred_rand, sharp_patch)
            else:
                # 固定模糊：用预先生成的blurred_t
                aug_blur, aug_sharp = random_augment(blurred_t, sharp_t)

            # 前向传播（U-Net只返回输出，旧网络返回(output, attn)）
            model_out = model(aug_blur)
            output = model_out[0] if isinstance(model_out, tuple) else model_out

            # 复合损失：L1 + 0.1*MSE + 0.1*梯度损失 + 0.2*SSIM损失 + 0.05*频域损失
            loss = F.l1_loss(output, aug_sharp) + 0.1 * F.mse_loss(output, aug_sharp)
            if USE_GRADIENT_LOSS:
                loss = loss + 0.1 * gradient_loss(output, aug_sharp)
            if USE_SSIM_LOSS:
                loss = loss + 0.2 * ssim_loss(output, aug_sharp)
            if USE_FREQ_LOSS:
                loss = loss + 0.05 * frequency_loss(output, aug_sharp)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            if step in save_steps:
                with torch.no_grad():
                    model.eval()
                    model_out = model(blurred_t)
                    out = model_out[0] if isinstance(model_out, tuple) else model_out
                    # U-Net没有注意力图，用零矩阵占位
                    if isinstance(model_out, tuple):
                        attn_maps = model_out[1]
                    else:
                        attn_maps = [torch.zeros(1, 1, 64, 64, device=device)]
                    current_img = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    current_psnr = compute_psnr(current_img, sharp)
                    history[step] = {
                        "image": current_img,
                        "attn": attn_maps,
                    }
                    model.train()
                if step % max(1, num_steps // 10) == 0 or step in [1, 10, 50, 100]:
                    print(f"  Step {step:4d}/{num_steps} | Loss: {loss.item():.6f} | PSNR: {current_psnr:.2f} dB")

                # 早停检查：达到目标PSNR则提前停止
                if TARGET_PSNR is not None and current_psnr > TARGET_PSNR:
                    print(f"\n  ★ 达到目标PSNR {TARGET_PSNR} dB！当前PSNR: {current_psnr:.2f} dB，提前停止训练")
                    # 把当前结果存为最终步数的key，方便后续代码统一访问
                    history[num_steps] = history[step]
                    actual_steps = step
                    early_stopped = True
                    break

        # 训练完成后保存模型权重
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        if early_stopped:
            print(f"  模型权重已保存（早停于第{actual_steps}步）: {MODEL_SAVE_PATH}")
        else:
            print(f"\n  模型权重已保存: {MODEL_SAVE_PATH}")
            if TARGET_PSNR is not None and num_steps in history:
                final_psnr = compute_psnr(history[num_steps]['image'], sharp)
                print(f"  注意: 训练{num_steps}步后PSNR为{final_psnr:.2f} dB，未达到目标{TARGET_PSNR} dB")

    # 用最终模型（训练好的或加载的）生成复原结果并保存
    with torch.no_grad():
        model.eval()
        if USE_TTA:
            # 测试时增强：8个视角推理后取平均
            final_out = tta_inference(model, blurred_t)
        else:
            model_out = model(blurred_t)
            final_out = model_out[0] if isinstance(model_out, tuple) else model_out
        final_restored = final_out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored_pil = Image.fromarray((final_restored * 255).astype(np.uint8))
    restored_pil.save(RESTORED_OUTPUT_PATH)
    print(f"\n  复原图像已保存: {RESTORED_OUTPUT_PATH}")

    # --- 可视化 ---
    print("\n[4/4] 生成可视化结果...")
    # 再次强制设置字体和减号参数（防止被matplotlib内部重置）
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.formatter.use_mathtext"] = False
    plt.rcParams["mathtext.default"] = "regular"
    # 临时重定向stderr，抑制matplotlib渲染时的字体缺失警告（U+2212减号等）
    _original_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.15)

    # 第1行：清晰GT、模糊输入、训练过程
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(sharp)
    title = "① 用户原图 (GT)" if is_user_image else "① 清晰图像 (GT)"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(blurred)
    ax.set_title(f"② 运动模糊输入\nPSNR={compute_psnr(blurred, sharp):.2f}dB", fontsize=12, fontweight="bold")
    ax.axis("off")

    # 展示训练过程（选4个关键步数）
    display_steps = [s for s in [1, 10, 50, num_steps] if s in history]
    if len(display_steps) < 4:
        display_steps = list(history.keys())[:4]
    for idx, s in enumerate(display_steps[:4]):
        ax = fig.add_subplot(gs[0, 2 + idx])
        ax.imshow(history[s]["image"])
        if idx < len(display_steps) - 1:
            title = f"③ 训练第{s}步\nPSNR={compute_psnr(history[s]['image'], sharp):.2f}dB"
        else:
            title = f"④ 最终输出 (第{s}步)\nPSNR={compute_psnr(history[s]['image'], sharp):.2f}dB"
        ax.set_title(title, fontsize=10, fontweight="bold" if idx == len(display_steps) - 1 else "normal")
        ax.axis("off")

    # 第2行：注意力图可视化（取最后一个block的第1个head）
    ax = fig.add_subplot(gs[1, 0])
    ax.text(0.5, 0.5, "Transformer\n自注意力图\n(最后一层Block,\n第1个Head,\n中心像素\n关注哪些位置)",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    ax.set_title("⑤ 注意力图说明", fontsize=12, fontweight="bold")
    ax.axis("off")

    # 选4个不同位置的像素，看它们的注意力分布
    last_attn = history[num_steps]["attn"][-1]  # (B*nw, heads, N, N)
    attn_matrix = last_attn[0, 0].cpu().numpy()  # (ws*ws, ws*ws)
    ws = args.window
    N = attn_matrix.shape[0]
    actual_ws = int(np.sqrt(N))

    center_pixels = [0, N//3, 2*N//3, N-1]
    labels = ["左上", "中上", "中下", "右下"]
    for idx, pix in enumerate(center_pixels):
        ax = fig.add_subplot(gs[1, 1 + idx])
        attn_map = attn_matrix[pix].reshape(actual_ws, actual_ws)
        im = ax.imshow(attn_map, cmap="hot", interpolation="nearest")
        r, c = pix // actual_ws, pix % actual_ws
        ax.plot(c, r, "c*", markersize=15, markeredgecolor="white")
        ax.set_title(f"⑥ {labels[idx]}像素({r},{c})\n的注意力分布", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 第3行：损失曲线 + 局部放大对比
    ax = fig.add_subplot(gs[2, 0:2])
    ax.plot(losses, color="#2196F3", linewidth=1.5)
    ax.set_xlabel("训练步数", fontsize=11)
    ax.set_ylabel("Loss (L1 + 0.1*MSE)", fontsize=11)
    ax.set_title("⑦ 训练损失曲线（损失下降 = 图像逐渐清晰）", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 局部放大：中心区域对比
    H, W = sharp.shape[:2]
    crop_h1, crop_h2 = H//4, 3*H//4
    crop_w1, crop_w2 = W//4, 3*W//4

    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(sharp[crop_h1:crop_h2, crop_w1:crop_w2])
    ax.set_title("⑧ 原图中心区域放大", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 3])
    ax.imshow(blurred[crop_h1:crop_h2, crop_w1:crop_w2])
    ax.set_title("⑨ 模糊中心区域", fontsize=10)
    ax.axis("off")

    final_img = history[num_steps]["image"]
    ax = fig.add_subplot(gs[2, 4])
    ax.imshow(final_img[crop_h1:crop_h2, crop_w1:crop_w2])
    ax.set_title("⑩ Transformer恢复后", fontsize=10)
    ax.axis("off")

    # 总结说明
    ax = fig.add_subplot(gs[2, 5])
    final_psnr = compute_psnr(final_img, sharp)
    blurred_psnr = compute_psnr(blurred, sharp)
    improvement = final_psnr - blurred_psnr
    summary_text = (
        f"关键观察：\n\n"
        f"1. 训练{actual_steps}步后，\n   PSNR从{blurred_psnr:.2f}提升到\n   {final_psnr:.2f} dB (+{improvement:.2f})\n\n"
        f"2. 注意力图显示：\n   每个像素不只看\n   局部3×3，而是\n   参考整个{actual_ws}×{actual_ws}窗口\n\n"
        f"3. 全局上下文帮助\n   恢复被模糊破坏的\n   结构一致性"
    )
    if is_user_image:
        summary_text += f"\n\n图片: {image_name}"
    ax.text(0.05, 0.95, summary_text,
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", alpha=0.8))
    ax.set_title("11. 总结", fontsize=12, fontweight="bold")
    ax.axis("off")

    title_suffix = f"（用户图片: {image_name}）" if is_user_image else ""
    plt.suptitle(f"Transformer 去模糊直观示例：从运动模糊到清晰恢复的全过程 {title_suffix}",
                 fontsize=16, fontweight="bold", y=0.98)

    # 确定输出路径（优先用命令行参数，否则用配置区域的 OUTPUT_PATH）
    if args.output is not None:
        output_path = args.output
    else:
        output_path = OUTPUT_PATH

    # 保存图片时忽略所有字体警告（matplotlib渲染时可能触发U+2212减号警告）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # 恢复stderr
    sys.stderr.close()
    sys.stderr = _original_stderr
    print(f"\n  可视化结果已保存: {output_path}")

    # 计算最终PSNR（直接用已保存的复原结果，避免重复推理）
    final_psnr = compute_psnr(final_restored, sharp)
    blurred_psnr = compute_psnr(blurred, sharp)

    print(f"\n{'='*60}")
    print(f"最终结果：")
    print(f"  模糊输入 PSNR:   {blurred_psnr:.2f} dB")
    print(f"  恢复输出 PSNR:   {final_psnr:.2f} dB")
    print(f"  PSNR 提升:       +{final_psnr - blurred_psnr:.2f} dB")
    print(f"  训练步数:        {actual_steps}")
    if early_stopped:
        print(f"  早停:            是（达到目标PSNR {TARGET_PSNR} dB）")
    print(f"  模型参数量:      {total_params:,}")
    if is_user_image:
        print(f"  使用图片:        {args.image}")
    print(f"{'='*60}")
    print(f"\n结果图已保存到: {output_path}")

    return output_path


# ============================================================
# 5. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Transformer 去模糊直观示例（PyTorch版，含训练过程）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python transformer_deblur_demo.py                                # 使用内置合成图
  python transformer_deblur_demo.py --image test.jpg               # 使用用户图片
  python transformer_deblur_demo.py --image test.jpg --steps 1000 --size 128
  python transformer_deblur_demo.py --image test.jpg --blocks 6 --window 16 --dim 64
        """
    )
    parser.add_argument("--image", type=str, default=IMAGE_PATH,
                        help=f"用户自定义图片路径（默认使用配置区域的 IMAGE_PATH = {IMAGE_PATH}，文件不存在时自动回退到合成图）")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE,
                        help=f"图像处理尺寸（默认 {IMAGE_SIZE}，建议64-256，越大越慢）")
    parser.add_argument("--motion-length", type=int, default=MOTION_LENGTH,
                        help=f"运动模糊长度（默认 {MOTION_LENGTH} 像素，越大模糊越严重）")
    parser.add_argument("--motion-angle", type=float, default=MOTION_ANGLE,
                        help=f"运动模糊角度（默认 {MOTION_ANGLE} 度）")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS,
                        help=f"训练步数（默认 {TRAIN_STEPS}，越多恢复越充分但越慢）")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"学习率（默认 {LEARNING_RATE}）")
    parser.add_argument("--dim", type=int, default=DIM,
                        help=f"特征维度（默认 {DIM}，越大模型容量越大）")
    parser.add_argument("--blocks", type=int, default=NUM_BLOCKS,
                        help=f"Transformer Block数量（默认 {NUM_BLOCKS}）")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE,
                        help=f"自注意力窗口大小（默认 {WINDOW_SIZE}，必须能整除 size）")
    parser.add_argument("--heads", type=int, default=NUM_HEADS,
                        help=f"注意力头数（默认 {NUM_HEADS}，必须能整除 dim）")
    parser.add_argument("--output", type=str, default=None,
                        help=f"输出图片路径（默认使用配置区域的 OUTPUT_PATH = {OUTPUT_PATH}）")

    args = parser.parse_args()

    # 参数校验
    if args.size % args.window != 0:
        print(f"警告: image size ({args.size}) 不能被 window size ({args.window}) 整除")
        print(f"  程序会自动padding，但建议设置 size 为 window 的整数倍")
        print()

    if args.dim % args.heads != 0:
        print(f"错误: dim ({args.dim}) 必须能被 heads ({args.heads}) 整除")
        sys.exit(1)

    print("=" * 60)
    print("Transformer 去模糊直观示例（PyTorch版，含训练过程）")
    print("=" * 60)

    try:
        train_and_visualize(args)
    except KeyboardInterrupt:
        print("\n\n用户中断训练。")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
