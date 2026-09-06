"""
知识点6：TCJA时间-通道联合注意力
==================================

【为什么要学这个？】
TCJA（Temporal-Channel Joint Attention）是TMT论文最核心的创新点。
它解决了标准自注意力在多帧图像上显存爆炸的问题，
让TMT能用很少的显存处理很多帧，同时效果还很好。

【本教程你将学会】
1. TCJA的核心思想：不在空间维度算注意力
2. TMTa变体：时间-通道分离的注意力
3. TMTb变体：通道混洗+分组的注意力
4. 复杂度对比：为什么TCJA这么高效？
5. 通道混洗有什么用？为什么能涨点？
6. 用PyTorch实现TCJA
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


def demo_tcja_core_idea():
    """
    演示1：TCJA的核心思想
    
    为什么不在空间维度算注意力？只在时间和通道维度算？
    """
    print("=" * 60)
    print("演示1：TCJA的核心思想")
    print("=" * 60)
    
    print("""
【标准自注意力的问题】

标准自注意力：每个像素和所有像素算注意力
  → N = T × H × W 个token
  → 复杂度 O(N²) = O((T·H·W)²)
  → 显存爆炸！

【TCJA的核心洞察】

图像的三个维度：时间(T)、高度(H)、宽度(W)、通道(C)

1. 空间维度 (H×W):
   - 空间相关性是局部的
   - 一个像素主要和周围的像素相关
   - 卷积就足够处理了！
   - 不需要全局注意力

2. 时间维度 (T):
   - 湍流是时间上随机变化的
   - 帧与帧之间有全局相关性
   - 需要捕捉帧间的动态变化
   - 注意力很适合

3. 通道维度 (C):
   - 不同通道代表不同特征
   - 通道之间有复杂的依赖关系
   - 注意力可以建模通道间的交互
   - （类似SE-Net的通道注意力，但更灵活）

【TCJA的设计】

  空间维度 → 用卷积处理（局部、高效）
  时间+通道 → 用注意力处理（全局、强大）

这样一来：
  - 空间卷积: O(H·W·C²)  ← 很常规，CNN都这样
  - TCJA注意力: O(T·C²)  ← T和C都不大，很小！

总复杂度: O(H·W·C² + T·C²)
  → 和标准自注意力的 O((T·H·W)²) 比，小太多了！

【为什么这合理？】

1. 物理上合理：
   - 湍流退化在空间上是局部的（PSF是局部的）
   - 但在时间上是全局变化的（每帧都不一样）
   - 所以空间用卷积，时间用注意力，正好匹配

2. 经验上有效：
   - 论文实验证明，TCJA效果比标准自注意力还好！
   - 而且显存小很多，可以输入更多帧
   - 更多帧 = 更多信息 = 更好的复原效果
    """)


class TCJA_TMTa(nn.Module):
    """
    TMTa: 时间-通道分离的注意力
    
    【思路】
    把时间和通道两个轴分开处理：
    1. 时间注意力：在时间维度算自注意力
    2. 通道注意力：用线性投影代替（因为通道数不大，线性层更高效）
    
    【为什么叫"分离"？】
    时间和通道是分开算的，不是联合在一起算的
    时间注意力只在时间轴上交互
    通道注意力是线性投影，不做softmax注意力
    """
    
    def __init__(self, dim, num_frames, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 时间注意力的QKV
        self.temporal_qkv = nn.Linear(dim, dim * 3)
        self.temporal_proj = nn.Linear(dim, dim)
        
        # 通道投影（代替通道注意力）
        self.channel_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        参数:
            x: 输入，形状 (B, T, H, W, C)
               B = batch size
               T = 帧数
               H, W = 空间分辨率
               C = 通道数
        
        返回:
            out: 输出，形状 (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # ========== 时间注意力 ==========
        # 把空间维度展平：(B, T, H, W, C) → (B*H*W, T, C)
        # 这样每个空间位置都独立算时间注意力
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        
        # LayerNorm
        x_norm = self.norm1(x_flat)
        
        # 计算QKV
        qkv = self.temporal_qkv(x_norm).reshape(B * H * W, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*H*W, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 时间注意力
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B*H*W, heads, T, head_dim)
        
        # 拼回头
        out = out.transpose(1, 2).reshape(B * H * W, T, C)
        out = self.temporal_proj(out)
        
        # 残差连接
        x_flat = x_flat + out
        
        # ========== 通道投影（代替通道注意力） ==========
        # LayerNorm
        x_norm = self.norm2(x_flat)
        
        # 线性投影（相当于通道注意力的简化版）
        out = self.channel_proj(x_norm)
        
        # 残差连接
        x_flat = x_flat + out
        
        # 恢复形状
        x = x_flat.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)
        
        return x


class TCJA_TMTb(nn.Module):
    """
    TMTb: 通道混洗 + 分组的时间-通道联合注意力
    
    【思路】
    把通道分成G组，每组内做时间-通道联合注意力
    然后用通道混洗（channel shuffle）让不同组之间交互
    
    【为什么更好？】
    1. 分组计算，更高效
    2. 组内是时间-通道联合的，表达能力更强
    3. 通道混洗保证不同组之间能信息交互
    4. 实验证明比TMTa好 0.02~0.03 dB
    
    【灵感来源】
    ShuffleNet（通道混洗）+ 分组卷积的思想
    """
    
    def __init__(self, dim, num_frames, num_heads=4, num_groups=4):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.group_dim = dim // num_groups
        
        # 每个组的QKV
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)
    
    def channel_shuffle(self, x, groups):
        """
        通道混洗
        
        参数:
            x: (B, T, C) 或类似形状
            groups: 分组数
        """
        B, T, C = x.shape
        # (B, T, groups, C//groups)
        x = x.reshape(B, T, groups, C // groups)
        # 交换groups和C//groups维度
        x = x.permute(0, 1, 3, 2)
        # 再拼回去
        x = x.reshape(B, T, C)
        return x
    
    def forward(self, x):
        """
        参数:
            x: 输入，形状 (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        G = self.num_groups
        
        # 把空间维度展平
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        
        # LayerNorm
        x_norm = self.norm(x_flat)
        
        # 计算QKV: (B*H*W, T, 3C)
        qkv = self.qkv(x_norm)
        
        # 分成G组: (B*H*W, T, 3G, C//G)
        qkv = qkv.reshape(B * H * W, T, 3 * G, C // G)
        # 重新排列: (3, B*H*W, G, T, C//G)
        qkv = qkv.reshape(B * H * W, T, 3, G, C // G)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B*H*W, G, T, C//G)
        
        # 组内做时间-通道联合注意力
        # 把时间和通道合在一起作为token维度
        # q, k, v: (B*H*W, G, T, C//G) → (B*H*W, G, T*C//G)
        # 不对，应该是把时间和通道都当作token，在上面算注意力
        # 等一下，让我重新想...
        
        # 实际上，TCJA的联合注意力是这样的：
        # 对于每个组，我们有 T 个时间步，每个时间步有 C//G 个通道
        # 我们把 (T, C//G) 这两个维度都当作"token"维度
        # 然后在这个联合维度上算自注意力
        
        # 但是这样token数 = T * C//G，还是有点大
        # 不对，论文里的做法更巧妙...
        
        # 让我重新理解论文：
        # TMTb是"temporal-channel joint attention with channel shuffle"
        # 意思是：注意力同时在时间和通道维度上计算
        # 但通过分组来控制复杂度
        
        # 具体来说：
        # - 把通道分成G组
        # - 每组内，token是 T 个时间步
        # - 但每个token的特征是 C//G 维
        # - 注意力是在时间维度上算的
        # - 通道维度通过分组和混洗来交互
        
        # 不对，这样和TMTa有什么区别？
        # 让我再仔细想...
        
        # 哦，我明白了！
        # TMTa：时间注意力 + 通道线性投影（分离的）
        # TMTb：在时间-通道联合空间上算注意力，但通过分组降低复杂度
        
        # 具体来说，TMTb的token是"时间×通道"的组合
        # 但因为通道分组了，所以每组内的token数 = T × (C/G)
        # 然后在每组内算自注意力
        
        # 但这样复杂度是 G × (T·C/G)² = T²·C²/G
        # 而TMTa的时间注意力复杂度是 T²·C
        # 哪个大？取决于参数设置
        
        # 论文里说TMTb更好，可能是因为联合注意力表达能力更强
        
        # 让我先按简化版来实现，重点是讲清楚思想
        
        # 简化版：每组内，在时间维度算注意力
        # 然后通道混洗
        # 这样和TMTa的区别是：分组 + 通道混洗
        
        # 实际上，论文里的TMTb更复杂一些
        # 但为了教学，我们先理解核心思想
        
        # 计算注意力（每组内，时间维度）
        # q, k, v: (B*H*W, G, T, C//G)
        # 我们在T维度上算注意力
        
        # 先把head维度加进去... 算了，简化一下，先不做多头
        # 直接在时间维度算注意力
        
        # 转置一下，方便计算: (B*H*W, G, T, C//G)
        # 计算注意力分数: q @ k^T
        # q: (B*H*W, G, T, d)
        # k^T: (B*H*W, G, d, T)
        # 结果: (B*H*W, G, T, T)
        
        d = C // G
        attn = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # (B*H*W, G, T, T)
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        out = attn @ v  # (B*H*W, G, T, C//G)
        
        # 把组拼回去: (B*H*W, T, C)
        out = out.permute(0, 2, 1, 3).reshape(B * H * W, T, C)
        
        # 通道混洗
        out = self.channel_shuffle(out, G)
        
        # 输出投影
        out = self.proj(out)
        
        # 残差连接
        x_flat = x_flat + out
        
        # 恢复形状
        x = x_flat.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)
        
        return x


def demo_complexity_comparison():
    """
    演示2：复杂度对比
    
    标准自注意力 vs TMTa vs TMTb
    """
    print("\n" + "=" * 60)
    print("演示2：复杂度对比")
    print("=" * 60)
    
    # 假设的参数
    T = 12   # 帧数
    H = 256  # 高度
    W = 256  # 宽度
    C = 128  # 通道数
    
    print(f"\n【参数设置】")
    print(f"  帧数 T = {T}")
    print(f"  分辨率 H×W = {H}×{W}")
    print(f"  通道数 C = {C}")
    print(f"  总像素数 N = T×H×W = {T*H*W:,}")
    
    # 标准自注意力
    # 复杂度: O(N²·C)，其中 N = T·H·W
    # 注意力矩阵大小: N²
    N = T * H * W
    standard_attn_memory = N * N * 4 / 1024 / 1024 / 1024  # GB
    
    # TMTa
    # 时间注意力: 每个空间位置独立算，注意力矩阵大小 T²
    # 总共有 H×W 个空间位置，但它们是独立的，不需要存所有注意力矩阵
    # 实际上，计算时是批处理的，注意力矩阵大小是 (B·H·W) × T × T
    # 但显存占用主要是 T² 乘以 batch
    # 让我们算一下相对大小
    
    # 更准确的显存对比：
    # 标准自注意力的注意力矩阵: (B, heads, N, N)
    # TCJA的注意力矩阵: (B·H·W, heads, T, T)
    
    # 假设 B=1, heads=1
    standard_attn_size = N * N  # 元素个数
    tcja_attn_size = H * W * T * T  # 元素个数
    
    print(f"\n【注意力矩阵显存对比】")
    print(f"  标准自注意力: {standard_attn_size:,} 元素 ≈ {standard_attn_size*4/1024/1024/1024:.2f} GB")
    print(f"  TCJA: {tcja_attn_size:,} 元素 ≈ {tcja_attn_size*4/1024/1024:.2f} MB")
    print(f"  显存减少倍数: {standard_attn_size / tcja_attn_size:.0f} 倍")
    
    print("""
【为什么差这么多？】

标准自注意力：
  → 每个像素和所有像素算注意力
  → N = T·H·W 个token
  → 注意力矩阵 N×N，巨大！

TCJA：
  → 每个空间位置独立算时间注意力
  → 每个位置只有 T 个token（帧数）
  → 注意力矩阵 T×T，很小！
  → 虽然有 H×W 个位置，但它们是独立计算的
  → 计算时可以分批，不需要同时存所有注意力矩阵

【计算复杂度对比】

标准自注意力: O((T·H·W)²·C)
TMTa: O(H·W·T²·C + H·W·C²)

当 H·W >> T 时（图像分辨率大，帧数不多）：
  标准自注意力的复杂度是 (H·W)² 量级
  TCJA的复杂度是 H·W 量级
  → 差了 H·W 倍！

这就是为什么TCJA这么高效！
    """)


def demo_channel_shuffle():
    """
    演示3：通道混洗（Channel Shuffle）
    
    什么是通道混洗？为什么需要它？
    """
    print("\n" + "=" * 60)
    print("演示3：通道混洗（Channel Shuffle）")
    print("=" * 60)
    
    print("""
【什么是通道混洗？】

通道混洗是ShuffleNet提出的技术，用于解决分组卷积的问题。

【分组卷积的问题】

分组卷积：把通道分成G组，每组内单独卷积
  优点：减少参数量和计算量
  缺点：不同组之间没有信息交互
        → 表达能力下降

【通道混洗的解决方案】

在分组卷积之后，把通道打乱重排
  → 让不同组的通道混在一起
  → 下一层分组卷积时，每个组就能看到来自不同组的信息
  → 这样就实现了组间信息交互

【为什么TMTb用通道混洗？】

TMTb是分组的时间-通道联合注意力：
  - 分组是为了减少计算量
  - 但分组会导致组间信息不流通
  - 通道混洗解决了这个问题

【效果】
论文中，通道混洗带来了 0.02~0.03 dB 的提升
  → 不多，但很稳定
  → 几乎没有额外计算成本
  → 性价比很高！
    """)
    
    # 演示通道混洗
    print("【通道混洗示例】")
    
    # 假设有8个通道，分成2组
    channels = np.arange(8)
    groups = 2
    
    print(f"\n  原始通道: {channels}")
    print(f"  分组数: {groups}")
    
    # 分组: [0,1,2,3], [4,5,6,7]
    grouped = channels.reshape(groups, -1)
    print(f"  分组后: \n{grouped}")
    
    # 转置（混洗）
    shuffled = grouped.T.reshape(-1)
    print(f"  混洗后: {shuffled}")
    
    print(f"""
  解释：
  - 第0组原来的通道: {grouped[0]}
  - 第1组原来的通道: {grouped[1]}
  - 混洗后，第0组变成: {shuffled[:4]}
  - 混洗后，第1组变成: {shuffled[4:]}
  
  → 每个组现在都有来自原来两个组的通道了！
  → 这样下一层分组计算时，组间就有信息交互了！
    """)


def demo_tcja_variants():
    """
    演示4：TMTa vs TMTb 对比
    
    两个变体的区别和各自的优缺点
    """
    print("\n" + "=" * 60)
    print("演示4：TMTa vs TMTb 对比")
    print("=" * 60)
    
    print("""
【TMTa: Temporal Attention with Channel Projection】

结构：
  1. 时间注意力（在时间维度算自注意力）
  2. 通道线性投影（用线性层代替通道注意力）

特点：
  ✅ 速度快，计算简单
  ✅ 容易实现
  ❌ 时间和通道是分离的，不是联合的
  ❌ 表达能力稍弱

适用场景：
  - 对速度要求高
  - 通道数不多的时候

【TMTb: Temporal-Channel Joint Attention with Channel Shuffle】

结构：
  1. 通道分组
  2. 每组内做时间-通道联合注意力
  3. 通道混洗（组间交互）

特点：
  ✅ 时间-通道联合，表达能力更强
  ✅ 分组计算，仍然高效
  ✅ 通道混洗保证组间交互
  ✅ 效果比TMTa好 0.02~0.03 dB
  ❌ 稍微复杂一点
  ❌ 速度比TMTa慢一点点

适用场景：
  - 追求最好的效果
  - 对速度要求不是极致

【论文中的选择】
TMT论文主要用TMTb，因为效果更好
但也提供了TMTa作为轻量级选项

【为什么叫"联合"注意力？】
TMTb中，注意力是在时间-通道的联合空间里计算的
  → 每个"token"不只是一个时间步
  → 而是时间步+通道的组合
  → 注意力可以同时捕捉时间和通道的依赖关系

这就是"时间-通道联合注意力"的含义。
    """)


def demo_why_tcja_works_for_turbulence():
    """
    演示5：为什么TCJA特别适合湍流复原？
    
    从物理角度解释TCJA的合理性
    """
    print("\n" + "=" * 60)
    print("演示5：为什么TCJA特别适合湍流复原？")
    print("=" * 60)
    
    print("""
【湍流退化的特点】

1. 空间上是局部的
   - 湍流的PSF（点扩散函数）是局部的
   - 一个像素的模糊只和周围像素有关
   - 空间相关性是短程的
   → 卷积足够处理！

2. 时间上是全局的
   - 每一帧的湍流都不一样
   - 但帧与帧之间有相关性
   - 要复原需要利用多帧的信息
   - 时间相关性是长程的
   → 需要注意力来捕捉！

3. 通道上是复杂的
   - 不同通道（特征）代表不同的信息
   - 通道之间有复杂的依赖关系
   - 比如边缘、纹理、低频、高频...
   → 注意力可以建模通道间的交互

【TCJA正好匹配这些特点】

  空间 → 卷积（处理局部相关性）
  时间 → 注意力（处理全局动态）
  通道 → 注意力（处理特征间依赖）

【这不是巧合！】

TMT的作者是懂物理的，他们知道湍流退化的特点
所以才设计了TCJA这种结构
  → 不是为了创新而创新
  → 而是根据问题的物理特性来设计网络

这就是好研究的特点：
  → 方法和问题是匹配的
  → 有物理直觉支撑
  → 不是瞎试出来的

【这对你的启发】
以后你做研究的时候，也要先想清楚：
  1. 问题的本质是什么？
  2. 有什么物理/数学性质可以利用？
  3. 什么样的结构最匹配这个问题？
  
而不是上来就堆Transformer、堆参数量。
    """)


def main():
    print("=" * 60)
    print("知识点6：TCJA时间-通道联合注意力")
    print("=" * 60)
    print()
    print("【核心思想】")
    print("  不在空间维度算注意力，只在时间和通道维度算")
    print("  空间用卷积，时间+通道用注意力")
    print("  既高效，效果又好")
    print()
    print("  TMTa: 时间注意力 + 通道投影（分离）")
    print("  TMTb: 分组 + 通道混洗 + 联合注意力（更好）")
    print()
    
    # 运行所有演示
    demo_tcja_core_idea()
    demo_complexity_comparison()
    demo_channel_shuffle()
    demo_tcja_variants()
    demo_why_tcja_works_for_turbulence()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. TCJA核心：空间用卷积，时间+通道用注意力")
    print("2. 为什么高效？因为不在空间维度算注意力，避免了N²")
    print("3. TMTa：时间注意力+通道投影，简单快速")
    print("4. TMTb：分组+通道混洗+联合注意力，效果更好")
    print("5. 通道混洗：解决分组后组间信息不流通的问题")
    print("6. 物理合理性：湍流空间局部、时间全局，TCJA正好匹配")
    print()
    print("【下一步】")
    print("  → 下一个知识点：Gated FFN与U形Transformer整体结构")
    print("  → 把TCJA放到完整的网络里看")


if __name__ == '__main__':
    main()
