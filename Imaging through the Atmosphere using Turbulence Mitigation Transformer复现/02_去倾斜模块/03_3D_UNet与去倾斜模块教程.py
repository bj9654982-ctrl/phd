"""
知识点3：3D U-Net与去倾斜模块
================================

【为什么要学这个？】
TMT的第一阶段是"去倾斜模块"，用的是一个轻量级的3D U-Net。
理解3D U-Net，你才能理解TMT是如何从多帧湍流图像中去掉倾斜的。

【本教程你将学会】
1. 什么是3D卷积？和2D卷积有什么区别？
2. 为什么用3D卷积处理多帧图像？
3. 深度可分离3D卷积是什么？为什么能节省内存？
4. U-Net结构是什么？编码器-解码器+跳跃连接
5. 多尺度监督怎么实现？
6. 用PyTorch实现一个简化版的Tilt-Removal Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\02_去倾斜模块'
os.makedirs(output_dir, exist_ok=True)


def demo_conv2d_vs_conv3d():
    """
    演示1：2D卷积 vs 3D卷积
    
    【2D卷积】
    - 输入形状: (B, C, H, W)
    - 卷积核形状: (out_C, in_C, kH, kW)
    - 只在空间维度（H, W）上滑动
    
    【3D卷积】
    - 输入形状: (B, C, T, H, W)
    - 卷积核形状: (out_C, in_C, kT, kH, kW)
    - 在时间+空间维度（T, H, W）上滑动
    
    为什么用3D卷积处理多帧图像？
    → 因为3D卷积可以同时在时间和空间上提取特征，
      自然地利用多帧之间的相关性来对齐图像。
    """
    print("=" * 60)
    print("演示1：2D卷积 vs 3D卷积")
    print("=" * 60)
    
    # 2D卷积示例
    print("\n【2D卷积】")
    conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    x2d = torch.randn(1, 3, 64, 64)  # (B, C, H, W)
    y2d = conv2d(x2d)
    print(f"  输入形状: {x2d.shape}")
    print(f"  输出形状: {y2d.shape}")
    print(f"  参数量: {sum(p.numel() for p in conv2d.parameters()):,}")
    
    # 3D卷积示例
    print("\n【3D卷积】")
    conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    x3d = torch.randn(1, 3, 10, 64, 64)  # (B, C, T, H, W)
    y3d = conv3d(x3d)
    print(f"  输入形状: {x3d.shape}")
    print(f"  输出形状: {y3d.shape}")
    print(f"  参数量: {sum(p.numel() for p in conv3d.parameters()):,}")
    
    print("\n【对比】")
    print(f"  3D卷积参数量 / 2D卷积参数量 = {sum(p.numel() for p in conv3d.parameters()) / sum(p.numel() for p in conv2d.parameters()):.1f} 倍")
    print("  → 3D卷积参数量更大，因为多了一个时间维度")
    print("  → 但它能同时处理多帧，提取时空特征")


def demo_depthwise_separable_conv3d():
    """
    演示2：深度可分离3D卷积
    
    【普通3D卷积的问题】
    参数量和计算量太大！
    比如：输入C=64，输出C=64，kernel=3×3×3
    参数量 = 64 × 64 × 3 × 3 × 3 = 110,592
    
    【深度可分离3D卷积】
    把一个3D卷积拆成两步：
    1. 深度卷积 (Depthwise Conv)：每个通道单独卷积
       - 卷积核形状: (C, 1, kT, kH, kW)
    2. 逐点卷积 (Pointwise Conv)：用1×1×1卷积混合通道
       - 卷积核形状: (out_C, in_C, 1, 1, 1)
    
    参数量 = C × kT × kH × kW + C × out_C
    同样的例子：64×27 + 64×64 = 1,728 + 4,096 = 5,824
    减少了约 19 倍！
    
    这就是TMT用"depth-wise 3D UNet"的原因——节省内存和计算。
    """
    print("\n" + "=" * 60)
    print("演示2：深度可分离3D卷积")
    print("=" * 60)
    
    C_in = 64
    C_out = 64
    kernel_size = 3
    
    # 普通3D卷积
    conv3d = nn.Conv3d(C_in, C_out, kernel_size, padding=kernel_size//2)
    params_normal = sum(p.numel() for p in conv3d.parameters())
    
    # 深度可分离3D卷积
    depthwise = nn.Conv3d(C_in, C_in, kernel_size, padding=kernel_size//2, groups=C_in)
    pointwise = nn.Conv3d(C_in, C_out, kernel_size=1)
    params_ds = sum(p.numel() for p in depthwise.parameters()) + sum(p.numel() for p in pointwise.parameters())
    
    print(f"\n【普通3D卷积】")
    print(f"  参数量: {params_normal:,}")
    
    print(f"\n【深度可分离3D卷积】")
    print(f"  深度卷积参数量: {sum(p.numel() for p in depthwise.parameters()):,}")
    print(f"  逐点卷积参数量: {sum(p.numel() for p in pointwise.parameters()):,}")
    print(f"  总参数量: {params_ds:,}")
    
    print(f"\n【对比】")
    print(f"  参数量减少: {(1 - params_ds/params_normal)*100:.1f}%")
    print(f"  压缩比: {params_normal/params_ds:.1f} 倍")
    print("\n  → 这就是为什么TMT用深度可分离3D卷积！")
    print("  → 大幅节省内存和计算，性能损失很小")


class DepthwiseSeparableConv3d(nn.Module):
    """深度可分离3D卷积模块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 深度卷积：每个通道单独卷积
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels
        )
        # 逐点卷积：1×1×1卷积混合通道
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # 激活函数
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        return x


class ConvBlock3d(nn.Module):
    """3D卷积块：两个深度可分离3D卷积 + 残差连接"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv3d(channels, channels)
        self.conv2 = DepthwiseSeparableConv3d(channels, channels)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual  # 残差连接


class TiltRemovalUNet(nn.Module):
    """
    简化版TMT去倾斜模块 (Tilt-Removal Module)
    
    【结构】
    - 3尺度U-Net：编码器2层 + 瓶颈层 + 解码器2层
    - 跳跃连接：编码器特征直接传到解码器对应层
    - 多尺度输出：每个尺度都输出去倾斜结果
    
    通道数变化（base_channels=16为例）：
    输入: 1
    编码器层1: 16  ←→  解码器层1: 32 (16+16)
    编码器层2: 32  ←→  解码器层2: 64 (32+32)
    瓶颈层: 64
    """
    
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        
        C = base_channels
        
        # ========== 编码器 ==========
        # 初始卷积
        self.input_conv = DepthwiseSeparableConv3d(in_channels, C)
        
        # 编码器第1层
        self.enc1 = ConvBlock3d(C)
        self.down1 = nn.Conv3d(C, C*2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # 编码器第2层
        self.enc2 = ConvBlock3d(C*2)
        self.down2 = nn.Conv3d(C*2, C*4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # 瓶颈层
        self.bottleneck = ConvBlock3d(C*4)
        
        # ========== 解码器 ==========
        # 解码器第2层（对应编码器第2层）
        self.up2 = nn.ConvTranspose3d(C*4, C*2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ConvBlock3d(C*4)  # C*2 + C*2 = C*4
        self.out2 = nn.Conv3d(C*4, in_channels, kernel_size=1)
        
        # 解码器第1层（对应编码器第1层）
        self.up1 = nn.ConvTranspose3d(C*4, C*2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock3d(C*3)  # C*2 + C = C*3
        self.out1 = nn.Conv3d(C*3, in_channels, kernel_size=1)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (B, C, T, H, W)
        
        返回:
            outputs: 多尺度输出列表 [细尺度, 粗尺度]
        """
        # 编码器
        x = self.input_conv(x)  # C
        
        e1 = self.enc1(x)      # C
        x = self.down1(e1)     # C*2
        
        e2 = self.enc2(x)      # C*2
        x = self.down2(e2)     # C*4
        
        # 瓶颈层
        x = self.bottleneck(x)  # C*4
        
        # 解码器第2层
        x = self.up2(x)         # C*2
        x = torch.cat([x, e2], dim=1)  # C*4
        x = self.dec2(x)        # C*4
        out2 = self.out2(x)     # in_channels
        
        # 解码器第1层
        x = self.up1(x)         # C*2
        x = torch.cat([x, e1], dim=1)  # C*3
        x = self.dec1(x)        # C*3
        out1 = self.out1(x)     # in_channels
        
        return [out1, out2]  # 从细到粗


def demo_unet_structure():
    """
    演示3：U-Net结构可视化
    
    展示U-Net的编码器-解码器结构和跳跃连接
    """
    print("\n" + "=" * 60)
    print("演示3：U-Net结构与多尺度输出")
    print("=" * 60)
    
    # 创建一个简化的U-Net
    model = TiltRemovalUNet(in_channels=1, base_channels=16)
    
    # 测试前向传播
    x = torch.randn(1, 1, 8, 64, 64)  # (B, C, T, H, W)
    print(f"\n输入形状: {x.shape}")
    print(f"  Batch: 1")
    print(f"  通道: 1 (灰度图)")
    print(f"  帧数: 8")
    print(f"  分辨率: 64×64")
    
    outputs = model(x)
    
    print(f"\n多尺度输出:")
    for i, out in enumerate(outputs):
        print(f"  尺度{i+1}: {out.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"  → 这是一个非常轻量的网络！")
    print(f"  → TMT的去倾斜模块就是这么设计的，小网络足够处理倾斜问题")


def demo_multi_scale_supervision():
    """
    演示4：多尺度监督
    
    【什么是多尺度监督？】
    不仅在最终输出上计算损失，在中间每个尺度的输出上也计算损失。
    
    【为什么有用？】
    1. 梯度更容易回传，训练更稳定
    2. 每个尺度都学到对应尺度的特征
    3. 低分辨率先学大的倾斜，高分辨率学精细的倾斜
    4. 类似"课程学习"，从易到难
    
    【TMT的多尺度损失】
    L_tilt = Σ(γ_ℓ · L_char(Ĵ_ℓ, J̃_ℓ))
    
    其中：
    - γ_ℓ 是每个尺度的权重（低分辨率权重更大）
    - L_char 是Charbonnier损失（比L1更平滑）
    - Ĵ_ℓ 是第ℓ尺度的预测
    - J̃_ℓ 是第ℓ尺度的GT（下采样后的清晰图像）
    """
    print("\n" + "=" * 60)
    print("演示4：多尺度监督")
    print("=" * 60)
    
    print("""
【多尺度监督的思想】

想象你要教一个人画画：
1. 先让他画大轮廓（低分辨率）
2. 再让他画细节（高分辨率）
3. 每一步都给他反馈（损失函数）

这比直接让他画完整的画（只在最高分辨率监督）更容易学会！

【TMT的多尺度损失公式】

  L_tilt = Σ(γ_ℓ · L_char(Ĵ_ℓ, J̃_ℓ))

  γ_ℓ: 第ℓ尺度的权重（低分辨率权重更大）
  L_char: Charbonnier损失
  Ĵ_ℓ: 第ℓ尺度的预测去倾斜图像
  J̃_ℓ: 第ℓ尺度的清晰图像（GT下采样）

【为什么低分辨率权重更大？】
1. 低分辨率的倾斜更"大"、更明显，更容易学
2. 先学好大的位移，再学精细的位移
3. 符合"从粗到细"的优化策略
    """)
    
    # 模拟多尺度损失计算
    print("【示例：3尺度的损失计算】")
    
    # 假设的权重（TMT论文中的设置）
    gamma = [0.6, 0.3, 0.1]  # 从粗到细
    print(f"\n  尺度权重 γ = {gamma}")
    print(f"  权重和 = {sum(gamma)}（归一化的）")
    print(f"  → 低分辨率（粗尺度）权重最大")
    print(f"  → 高分辨率（细尺度）权重最小")


def demo_why_3d_unet_for_tilt():
    """
    演示5：为什么用3D U-Net来去倾斜？
    
    【倾斜的特点】
    1. 倾斜是像素位移，是几何变换
    2. 倾斜在时间上随机变化，每一帧都不一样
    3. 多帧之间有相关性，倾斜是平滑变化的
    
    【为什么3D U-Net适合？】
    1. 3D卷积可以同时看多帧，利用时间相关性
    2. U-Net的多尺度结构可以处理不同大小的倾斜
    3. 端到端学习，不需要显式计算光流
    
    【和光流法的对比】
    - 光流法：两帧之间计算光流，然后对齐
      → 只能两帧两帧地处理，不能利用更多帧
      → 光流计算本身可能有误差
    - 3D U-Net：一次处理多帧，直接输出对齐后的图像
      → 可以利用更多帧的信息
      → 端到端训练，更鲁棒
    """
    print("\n" + "=" * 60)
    print("演示5：为什么用3D U-Net来去倾斜？")
    print("=" * 60)
    
    print("""
【倾斜的特点】
1. 倾斜 = 像素的随机位移（几何变形）
2. 每一帧的倾斜都不一样（时间上随机）
3. 相邻帧的倾斜是相关的（平滑变化）

【为什么3D U-Net适合处理倾斜？】

1. ✅ 同时处理多帧
   - 3D卷积的时间维度 = 多帧
   - 可以同时看10帧、20帧，利用更多信息
   - 光流法只能两帧两帧地处理

2. ✅ 多尺度结构
   - 大的倾斜在低分辨率更容易检测
   - 小的倾斜在高分辨率更容易修正
   - U-Net的多尺度正好匹配这个特点

3. ✅ 端到端学习
   - 不需要显式计算光流
   - 直接输出去倾斜后的图像
   - 训练更简单，效果更好

4. ✅ 轻量级
   - 倾斜相对简单，不需要复杂网络
   - 用深度可分离3D卷积，参数量很小
   - TMT的去倾斜模块只有几百万参数

【和其他方法的对比】

| 方法 | 帧数 | 参数量 | 效果 |
|------|------|--------|------|
| 光流法 | 2帧 | - | 一般，误差会累积 |
| 2D CNN | 单帧 | 小 | 差，没有时间信息 |
| 3D U-Net (TMT) | 多帧(12+) | 中等 | 好，利用时间相关性 |

→ 这就是TMT选择3D U-Net的原因！
    """)


def main():
    print("=" * 60)
    print("知识点3：3D U-Net与去倾斜模块")
    print("=" * 60)
    print()
    print("【核心思想】")
    print("  TMT第一阶段：用轻量级3D U-Net去掉多帧图像中的倾斜")
    print()
    print("  为什么用3D卷积？→ 同时处理多帧，利用时间相关性")
    print("  为什么用U-Net？→ 多尺度结构，从粗到细去倾斜")
    print("  为什么用深度可分离？→ 大幅减少参数量，节省内存")
    print("  为什么多尺度监督？→ 训练更稳定，效果更好")
    print()
    
    # 运行所有演示
    demo_conv2d_vs_conv3d()
    demo_depthwise_separable_conv3d()
    demo_unet_structure()
    demo_multi_scale_supervision()
    demo_why_3d_unet_for_tilt()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. 3D卷积 = 2D卷积 + 时间维度，可以同时处理多帧")
    print("2. 深度可分离卷积 = 深度卷积 + 逐点卷积，大幅减少参数")
    print("3. U-Net = 编码器 + 解码器 + 跳跃连接，多尺度特征")
    print("4. 多尺度监督 = 在每个尺度都计算损失，训练更稳定")
    print("5. TMT去倾斜模块 = 轻量级3D U-Net，专门处理倾斜问题")
    print()
    print("【下一步】")
    print("  → 下一个知识点：多尺度损失函数（Charbonnier损失）")


if __name__ == '__main__':
    main()
