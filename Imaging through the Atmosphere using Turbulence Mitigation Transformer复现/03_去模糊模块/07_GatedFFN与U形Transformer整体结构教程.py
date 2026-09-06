"""
知识点7：Gated FFN与U形Transformer整体结构
================================================

【为什么要学这个？】
前面我们学了TCJA注意力机制，但它只是一个模块。
现在我们要把它放到完整的网络里，看看TMT的去模糊模块整体长什么样。
同时还要学习Gated FFN，这是Transformer里另一个重要组件。

【本教程你将学会】
1. Gated FFN是什么？和普通FFN有什么区别？
2. 为什么要用Gated FFN？有什么好处？
3. TMT的Transformer Block完整结构
4. U形Transformer的整体架构（编码器-解码器+跳跃连接）
5. 多尺度输入怎么处理？
6. 用PyTorch实现简化版的TMT去模糊模块
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

output_dir = r'J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\03_去模糊模块'
os.makedirs(output_dir, exist_ok=True)


def demo_gated_ffn():
    """
    演示1：Gated FFN vs 普通FFN
    
    对比两种FFN的结构和区别
    """
    print("=" * 60)
    print("演示1：Gated FFN vs 普通FFN")
    print("=" * 60)
    
    print("""
【普通FFN】

标准Transformer里的FFN：
  FFN(x) = Linear2(GELU(Linear1(x)))
  
结构：
  x → Linear1 → GELU → Linear2 → 输出
  
特点：
  - 两个线性层，中间一个激活函数
  - 第一层升维（通常是4倍），第二层降维回来
  - 每个token独立处理

【Gated FFN】

带门控的FFN：
  GatedFFN(x) = Linear2(GELU(Linear1(x)) ⊙ Linear3(x))
  
结构：
  x → Linear1 → GELU → ┐
                        ⊙ → Linear2 → 输出
  x → Linear3 ──────────┘
  
  （⊙是逐元素相乘）

特点：
  - 有三个线性层
  - 一路做特征变换，一路做门控
  - 门控控制哪些特征通过
  - 表达能力更强

【为什么叫"门控"？】
门控（Gate）的意思是"开关"：
  - 门控值大 → 特征通过得多
  - 门控值小 → 特征被抑制
  
就像水龙头一样，控制水流的大小。

【为什么Gated FFN更好？】
1. 表达能力更强：门控机制可以动态选择特征
2. 训练更稳定：门控可以缓解梯度消失
3. 效果更好：在很多任务上都比普通FFN好

【代价】
- 参数量多了一点（多了一个线性层）
- 计算量也多了一点
- 但通常收益大于成本

【TMT的选择】
TMT用的是Gated FFN，不是普通FFN。
这也是TMT效果好的一个小细节。
    """)


class GatedFFN(nn.Module):
    """Gated FFN实现"""
    
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4  # 通常是4倍
        
        # 普通路径
        self.fc1 = nn.Linear(dim, hidden_dim)
        # 门控路径
        self.fc_gate = nn.Linear(dim, hidden_dim)
        # 输出层
        self.fc2 = nn.Linear(hidden_dim, dim)
        
        self.act = nn.GELU()
    
    def forward(self, x):
        """
        参数:
            x: (..., dim)
        返回:
            out: (..., dim)
        """
        # 普通路径：特征变换
        hidden = self.act(self.fc1(x))
        # 门控路径：产生门控信号
        gate = self.fc_gate(x)
        # 逐元素相乘（门控）
        hidden = hidden * gate
        # 输出投影
        out = self.fc2(hidden)
        return out


class TMTBlock(nn.Module):
    """
    TMT的Transformer Block
    
    结构：
    - LayerNorm → TCJA → 残差连接
    - LayerNorm → Gated FFN → 残差连接
    
    注意：这里用简化版的TCJA，重点是展示整体结构
    """
    
    def __init__(self, dim, num_frames, num_heads=4):
        super().__init__()
        
        # 第一部分：TCJA注意力（简化版，用时间注意力代替）
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # 第二部分：Gated FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedFFN(dim)
    
    def forward(self, x):
        """
        参数:
            x: (B, T, H, W, C)
        返回:
            out: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # ========== TCJA注意力（简化版：时间注意力） ==========
        # 把空间维度展平
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        
        # LayerNorm + 注意力
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        # 残差连接
        x_flat = x_flat + attn_out
        
        # ========== Gated FFN ==========
        x_norm = self.norm2(x_flat)
        ffn_out = self.ffn(x_norm)
        
        # 残差连接
        x_flat = x_flat + ffn_out
        
        # 恢复形状
        x = x_flat.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)
        
        return x


def demo_transformer_block():
    """
    演示2：TMT Transformer Block完整结构
    """
    print("\n" + "=" * 60)
    print("演示2：TMT Transformer Block完整结构")
    print("=" * 60)
    
    print("""
【完整结构】

一个TMT Transformer Block：

  ┌─────────────────────────────────┐
  │            输入 x               │
  └───────────────┬─────────────────┘
                  │
          ┌───────▼───────┐
          │   LayerNorm   │  ← Pre-Norm
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │     TCJA      │  ← 时间-通道联合注意力
          └───────┬───────┘
                  │
                  └─────────┐
                            │ 残差连接
          ┌─────────────────▼─────────────────┐
          │           x + TCJA(x)             │
          └─────────────────┬─────────────────┘
                            │
                    ┌───────▼───────┐
                    │   LayerNorm   │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │   Gated FFN   │  ← 门控前馈网络
                    └───────┬───────┘
                            │
                            └─────────┐
                                      │ 残差连接
          ┌───────────────────────────▼───────────────────────────┐
          │                    x + FFN(x)                         │
          └───────────────────────────────────────────────────────┘

【两个子层】
1. TCJA注意力层：捕捉时间-通道依赖
2. Gated FFN层：特征变换和非线性

【两个关键设计】
1. Pre-Norm：先归一化再子层（现在的标准做法）
2. 残差连接：解决梯度消失，深层网络能训练

【和标准Transformer的区别】
- 标准：自注意力 + 普通FFN
- TMT：TCJA注意力 + Gated FFN
- 都是在标准基础上的改进
    """)
    
    # 测试一下
    block = TMTBlock(dim=64, num_frames=8, num_heads=4)
    x = torch.randn(1, 8, 32, 32, 64)
    out = block(x)
    
    print(f"\n【测试】")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in block.parameters()):,}")


class TMT_U_Net(nn.Module):
    """
    简化版TMT去模糊模块（U形Transformer）
    
    整体结构：
    - 编码器：多个下采样阶段，每个阶段有几个TMT Block
    - 瓶颈层：最深层的TMT Block
    - 解码器：多个上采样阶段，每个阶段有几个TMT Block
    - 跳跃连接：编码器特征直接传到解码器对应层
    
    注意：这是简化版，重点是展示整体结构
    """
    
    def __init__(self, in_channels=1, base_channels=32, num_blocks=2, num_frames=8):
        super().__init__()
        
        C = base_channels
        
        # ========== 输入投影 ==========
        self.input_proj = nn.Conv2d(in_channels, C, kernel_size=3, padding=1)
        # 注意：这里用2D卷积，因为输入是多帧堆叠的，我们先把时间维度当通道
        # 实际TMT的处理更复杂，这里简化了
        
        # ========== 编码器 ==========
        # 编码器第1层
        self.enc1_blocks = nn.ModuleList([
            TMTBlock(C, num_frames) for _ in range(num_blocks)
        ])
        self.down1 = nn.Conv2d(C, C * 2, kernel_size=2, stride=2)
        
        # 编码器第2层
        self.enc2_blocks = nn.ModuleList([
            TMTBlock(C * 2, num_frames) for _ in range(num_blocks)
        ])
        self.down2 = nn.Conv2d(C * 2, C * 4, kernel_size=2, stride=2)
        
        # ========== 瓶颈层 ==========
        self.bottleneck_blocks = nn.ModuleList([
            TMTBlock(C * 4, num_frames) for _ in range(num_blocks)
        ])
        
        # ========== 解码器 ==========
        # 解码器第2层
        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, kernel_size=2, stride=2)
        self.dec2_blocks = nn.ModuleList([
            TMTBlock(C * 4, num_frames) for _ in range(num_blocks)  # C*2 + C*2 = C*4
        ])
        
        # 解码器第1层
        self.up1 = nn.ConvTranspose2d(C * 4, C, kernel_size=2, stride=2)
        self.dec1_blocks = nn.ModuleList([
            TMTBlock(C * 2, num_frames) for _ in range(num_blocks)  # C + C = C*2
        ])
        
        # ========== 输出投影 ==========
        self.output_proj = nn.Conv2d(C, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        参数:
            x: (B, T, C, H, W)  多帧输入
        返回:
            out: (B, T, C, H, W)  去模糊后的多帧输出
        """
        B, T, C, H, W = x.shape
        
        # 把时间维度和batch维度合并，方便用2D卷积
        # 实际TMT的处理更复杂，这里是简化版
        x = x.reshape(B * T, C, H, W)
        
        # 输入投影
        x = self.input_proj(x)
        C_new = x.shape[1]
        
        # 恢复时间维度，变成 (B, T, H, W, C) 给TMT Block用
        x = x.reshape(B, T, C_new, H, W).permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        
        # ========== 编码器 ==========
        # 第1层
        for block in self.enc1_blocks:
            x = block(x)
        enc1 = x  # 保存跳跃连接
        
        # 下采样1（空间下采样）
        # 先转成 (B*T, C, H, W) 再下采样
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C_new, H, W)
        x = self.down1(x)
        C_new = x.shape[1]
        H_new, W_new = x.shape[2], x.shape[3]
        x = x.reshape(B, T, C_new, H_new, W_new).permute(0, 1, 3, 4, 2)
        
        # 第2层
        for block in self.enc2_blocks:
            x = block(x)
        enc2 = x  # 保存跳跃连接
        
        # 下采样2
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C_new, H_new, W_new)
        x = self.down2(x)
        C_new = x.shape[1]
        H_new, W_new = x.shape[2], x.shape[3]
        x = x.reshape(B, T, C_new, H_new, W_new).permute(0, 1, 3, 4, 2)
        
        # ========== 瓶颈层 ==========
        for block in self.bottleneck_blocks:
            x = block(x)
        
        # ========== 解码器 ==========
        # 上采样2
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C_new, H_new, W_new)
        x = self.up2(x)
        C_new = x.shape[1]
        H_new, W_new = x.shape[2], x.shape[3]
        x = x.reshape(B, T, C_new, H_new, W_new).permute(0, 1, 3, 4, 2)
        
        # 跳跃连接
        x = torch.cat([x, enc2], dim=-1)  # 通道维度拼接
        
        # 解码器第2层
        for block in self.dec2_blocks:
            x = block(x)
        
        # 上采样1
        C_new = x.shape[-1]
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C_new, H_new, W_new)
        x = self.up1(x)
        C_new = x.shape[1]
        H_new, W_new = x.shape[2], x.shape[3]
        x = x.reshape(B, T, C_new, H_new, W_new).permute(0, 1, 3, 4, 2)
        
        # 跳跃连接
        x = torch.cat([x, enc1], dim=-1)
        
        # 解码器第1层
        for block in self.dec1_blocks:
            x = block(x)
        
        # ========== 输出投影 ==========
        C_new = x.shape[-1]
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C_new, H_new, W_new)
        x = self.output_proj(x)
        x = x.reshape(B, T, -1, H_new, W_new)
        
        return x


def demo_u_shaped_transformer():
    """
    演示3：U形Transformer整体结构
    """
    print("\n" + "=" * 60)
    print("演示3：U形Transformer整体结构")
    print("=" * 60)
    
    print("""
【整体结构】

TMT的去模糊模块是一个U形Transformer：

  编码器                      解码器
  ───────                     ───────
  输入 ─→ TMT Block ─┐    ┌→ TMT Block ─→ 输出
                    ↓    ↑
              TMT Block ─┐│
                    ↓    ↑│
              TMT Block ─┐│
                    ↓    ↑│
              下采样   上采样
                    ↓    ↑
              TMT Block ─┐│
                    ↓    ↑│
              TMT Block ─┐│
                    ↓    ↑│
              下采样   上采样
                    ↓    ↑
              ┌─瓶颈层─┐  │
              └───────┘──┘

【和U-Net的关系】
- 整体结构和U-Net很像：编码器+解码器+跳跃连接
- 区别是：U-Net用卷积块，TMT用Transformer Block
- 所以叫"U形Transformer"

【为什么用U形结构？】
1. 多尺度特征：不同尺度捕捉不同大小的特征
2. 跳跃连接：保留低层细节（边缘、纹理）
3. 渐进式复原：从粗到细逐步恢复
4. 训练稳定：多尺度监督更容易训练

【每个阶段有什么？】
每个阶段（编码器/解码器的一层）包含：
- 多个TMT Transformer Block
- 一个下采样/上采样层

TMT Block是基本单元，U形结构是整体框架。
    """)
    
    # 测试一下
    print("【简化版测试】")
    
    model = TMT_U_Net(in_channels=1, base_channels=16, num_blocks=1, num_frames=4)
    x = torch.randn(1, 4, 1, 32, 32)  # (B, T, C, H, W)
    
    print(f"\n  输入形状: {x.shape}")
    print(f"  Batch: 1, 帧数: 4, 通道: 1, 分辨率: 32×32")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  总参数量: {total_params:,}")
    print(f"  → 简化版已经有 {total_params/1e6:.2f} M 参数了")
    print(f"  → 完整的TMT有26M参数，主要是通道数更多、层数更多")


def demo_spatial_conv_in_tmt():
    """
    演示4：TMT里的空间卷积
    
    TCJA不在空间维度算注意力，那空间维度怎么处理？
    答案：用卷积！
    """
    print("\n" + "=" * 60)
    print("演示4：TMT里的空间卷积")
    print("=" * 60)
    
    print("""
【回顾：TCJA的设计】

  空间维度 → 用卷积处理
  时间+通道 → 用注意力处理

【空间卷积在哪里？】

1. 输入/输出投影
   - 把输入通道投影到网络通道数
   - 把网络通道投影回输出通道
   - 都是3×3卷积

2. 下采样/上采样
   - 编码器的下采样：2×2卷积，stride=2
   - 解码器的上采样：转置卷积
   - 都是在空间维度做的

3. 跳跃连接后的融合
   - 编码器和解码器的特征拼接后
   - 用卷积来融合通道

4. 每个Transformer Block前后？
   - 有些设计会在注意力前后加卷积
   - TMT的设计是：空间维度纯用卷积连接
   - 注意力只在时间-通道维度

【为什么空间用卷积就够了？】

1. 空间相关性是局部的
   - 一个像素主要和周围的像素相关
   - 卷积的感受野可以覆盖局部区域
   - 深层网络的感受野也会变大

2. 卷积很高效
   - 卷积的计算量是 O(H·W·C²)
   - 比空间注意力的 O(H²·W²·C) 小太多了

3. 卷积有归纳偏置
   - 局部性、平移不变性
   - 这些先验知识对图像任务很有用
   - 可以让网络学得更快、更好

【卷积+注意力的组合】

TMT的设计哲学：
  - 局部的事情交给卷积（高效）
  - 全局的事情交给注意力（强大）
  - 两者结合，既高效又强大

这是现在很多视觉Transformer的设计思路：
  - 用卷积处理局部空间信息
  - 用注意力处理全局依赖关系
  - 取长补短
    """)


def demo_two_stage_design():
    """
    演示5：两阶段设计的整体回顾
    
    把去倾斜和去模糊合起来看
    """
    print("\n" + "=" * 60)
    print("演示5：TMT两阶段设计整体回顾")
    print("=" * 60)
    
    print("""
【完整的TMT流水线】

  输入：多帧湍流图像
    ↓
  ┌─────────────────┐
  │  阶段1：去倾斜  │  ← 3D U-Net（轻量级）
  │  Tilt-Removal  │
  └────────┬────────┘
           ↓
  去倾斜后的多帧图像
           ↓
  ┌─────────────────┐
  │  阶段2：去模糊  │  ← U形Transformer（主力）
  │  Blur-Removal   │
  └────────┬────────┘
           ↓
  输出：复原后的清晰图像

【为什么分两阶段？】

1. 物理上合理
   - 湍流退化 = 倾斜 + 模糊
   - 先去倾斜，再去模糊
   - 符合物理过程

2. 计算上高效
   - 倾斜相对简单，用小网络就行
   - 模糊比较难，用大网络
   - 分开处理，各司其职

3. 训练上稳定
   - 分阶段训练，每个阶段目标明确
   - 先训去倾斜，再训去模糊
   - 比端到端一起训更容易收敛

4. 效果上更好
   - 论文消融实验证明：两阶段比单阶段好 0.02~0.08 dB
   - 分而治之，每个阶段都能做好自己的事

【两阶段的损失函数】

阶段1（去倾斜）：
  L_tilt = Σ γ_ℓ · L_char(Ĵ_ℓ, J̃_ℓ)
  → 多尺度Charbonnier损失

阶段2（去模糊）：
  L_blur = L_char(Î, I)
  → 单尺度Charbonnier损失
  → （也可以加多尺度，但论文里主要是单尺度）

总损失：
  分阶段训练，不是一起训的
  先训阶段1，固定阶段1，再训阶段2
    """)


def main():
    print("=" * 60)
    print("知识点7：Gated FFN与U形Transformer整体结构")
    print("=" * 60)
    print()
    print("【核心思想】")
    print("  TMT去模糊模块 = U形Transformer")
    print("  基本单元 = TMT Block = TCJA + Gated FFN")
    print("  整体框架 = 编码器 + 解码器 + 跳跃连接")
    print()
    print("  空间用卷积，时间+通道用注意力")
    print("  局部+全局结合，既高效又强大")
    print()
    
    # 运行所有演示
    demo_gated_ffn()
    demo_transformer_block()
    demo_u_shaped_transformer()
    demo_spatial_conv_in_tmt()
    demo_two_stage_design()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. Gated FFN = 普通FFN + 门控机制，表达能力更强")
    print("2. TMT Block = TCJA注意力 + Gated FFN + 残差")
    print("3. 整体是U形结构：编码器+解码器+跳跃连接")
    print("4. 空间维度用卷积，时间+通道用注意力")
    print("5. 两阶段设计：先去倾斜，再去模糊")
    print()
    print("【下一步】")
    print("  → 下一个知识点：湍流模拟器原理与实现")
    print("  → （数据生成部分，TMT的另一个重要贡献）")


if __name__ == '__main__':
    main()
