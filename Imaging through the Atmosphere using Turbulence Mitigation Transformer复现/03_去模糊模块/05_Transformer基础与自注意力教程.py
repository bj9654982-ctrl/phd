"""
知识点5：Transformer基础与自注意力
====================================

【为什么要学这个？】
TMT的第二阶段（去模糊模块）核心是Transformer结构，
而最创新的TCJA注意力机制，也是在自注意力的基础上改进的。
理解了自注意力，你才能理解TMT的核心创新点。

【本教程你将学会】
1. 什么是自注意力？为什么它这么强？
2. Q、K、V是什么？自注意力的完整计算过程
3. 多头注意力是什么？
4. Transformer Block的完整结构
5. 自注意力的复杂度分析（为什么这么吃显存）
6. 为什么TMT不直接用标准Transformer？（显存问题）
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


def demo_self_attention_intuition():
    """
    演示1：自注意力的直觉理解
    
    用一个简单的例子说明自注意力在做什么
    """
    print("=" * 60)
    print("演示1：自注意力的直觉理解")
    print("=" * 60)
    
    print("""
【自注意力在做什么？】

想象你在读一句话："我爱深度学习"
- 读到"我"的时候，你知道这是主语
- 读到"爱"的时候，你知道这是谓语，和"我"有关
- 读到"深度学习"的时候，你知道这是宾语，和"爱"有关

自注意力就是干这个的：
  → 每个位置都去看其他位置，计算"相关性"
  → 然后把相关位置的信息加权聚合过来

【在图像里呢？】

图像里的每个像素，也可以去看其他像素：
- 边缘附近的像素，应该关注对面的边缘像素
- 纹理区域的像素，应该关注周围相似纹理的像素
- 这样就能捕捉长距离依赖关系

【为什么叫"自"注意力？】
- 因为是自己和自己计算注意力
- 输入是X，输出也是从X来的
- 不像"交叉注意力"是两个不同的序列之间计算
    """)


def demo_qkv_explanation():
    """
    演示2：Q、K、V是什么？
    
    用图书馆找书的比喻来解释
    """
    print("\n" + "=" * 60)
    print("演示2：Q、K、V是什么？")
    print("=" * 60)
    
    print("""
【图书馆找书的比喻】

想象你去图书馆找书：
- 你脑子里想找的书 = Query (Q) → "我想要什么？"
- 每本书的标题/标签 = Key (K) → "这本书是什么？"
- 书的内容 = Value (V) → "这本书里有什么信息？"

找书的过程：
1. 你拿着Q，去和每本书的K对比，算相似度
2. 相似度越高，这本书越相关
3. 然后你把这些书的内容(V)按相似度加权拿过来

【自注意力里的QKV】

输入X，通过三个线性变换得到Q、K、V：
  Q = X · W_Q  →  "我想要什么？"
  K = X · W_K  →  "我有什么？"
  V = X · W_V  →  "我的内容是什么？"

然后：
  Attention(Q, K, V) = softmax(Q·Kᵀ/√d) · V

这个公式就是：
  1. 用Q和K算相似度（点积）
  2. 用softmax归一化成权重
  3. 用权重对V加权求和

【为什么要分成QKV？】
- 如果只有一个变换，表达能力不够
- Q和K可以有不同的变换，更灵活
- V又是另一个变换，负责提取真正要传递的信息
- 三个不同的线性变换，让模型能学到更复杂的注意力模式
    """)


def demo_self_attention_calculation():
    """
    演示3：自注意力的完整计算过程
    
    用一个具体的数值例子，一步步计算自注意力
    """
    print("\n" + "=" * 60)
    print("演示3：自注意力的完整计算过程（数值例子）")
    print("=" * 60)
    
    # 简单例子：3个token，每个token维度是4
    # 比如一句话有3个词，每个词的embedding是4维
    torch.manual_seed(42)
    
    # 输入X: (3, 4)  3个token，每个4维
    X = torch.randn(3, 4)
    print(f"\n【输入X】形状: {X.shape}")
    print(f"  3个token，每个4维")
    print(f"  X = \n{X.numpy().round(3)}")
    
    # 定义Q、K、V的变换矩阵
    # 假设d_k = d_v = 2
    d_k = 2
    W_Q = torch.randn(4, d_k)
    W_K = torch.randn(4, d_k)
    W_V = torch.randn(4, d_k)
    
    print(f"\n【变换矩阵】")
    print(f"  W_Q: {W_Q.shape}, W_K: {W_K.shape}, W_V: {W_V.shape}")
    
    # 计算Q、K、V
    Q = X @ W_Q  # (3, 2)
    K = X @ W_K  # (3, 2)
    V = X @ W_V  # (3, 2)
    
    print(f"\n【计算Q、K、V】")
    print(f"  Q = X · W_Q = \n{Q.numpy().round(3)}")
    print(f"  K = X · W_K = \n{K.numpy().round(3)}")
    print(f"  V = X · W_V = \n{V.numpy().round(3)}")
    
    # 第一步：计算注意力分数（Q·Kᵀ）
    scores = Q @ K.T  # (3, 3)
    print(f"\n【第一步：计算注意力分数】")
    print(f"  scores = Q · Kᵀ = \n{scores.numpy().round(3)}")
    print(f"  形状: {scores.shape} (3个token × 3个token)")
    print(f"  scores[i][j] = 第i个token和第j个token的相似度")
    
    # 第二步：缩放（除以√d_k）
    scores_scaled = scores / np.sqrt(d_k)
    print(f"\n【第二步：缩放（除以√d_k）】")
    print(f"  为什么要缩放？")
    print(f"  → 维度d_k大的时候，点积的值会很大")
    print(f"  → softmax会变得很尖锐，梯度很小")
    print(f"  → 除以√d_k可以让梯度保持在合适的范围")
    print(f"  scaled_scores = scores / √{d_k} = \n{scores_scaled.numpy().round(3)}")
    
    # 第三步：softmax归一化
    attn_weights = F.softmax(scores_scaled, dim=-1)
    print(f"\n【第三步：softmax归一化】")
    print(f"  attn_weights = softmax(scaled_scores) = \n{attn_weights.numpy().round(3)}")
    print(f"  每一行的和 = {attn_weights.sum(dim=1).numpy().round(3)}")
    print(f"  → 每一行都是一个概率分布，代表这个token对其他token的注意力权重")
    
    # 第四步：加权求和V
    output = attn_weights @ V  # (3, 2)
    print(f"\n【第四步：加权求和V】")
    print(f"  output = attn_weights · V = \n{output.numpy().round(3)}")
    print(f"  形状: {output.shape} (3个token，每个2维)")
    
    print(f"\n【完整公式回顾】")
    print(f"  Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V")
    print(f"  ✓ 第一步：Q·Kᵀ → 算相似度")
    print(f"  ✓ 第二步：除以√d_k → 缩放")
    print(f"  ✓ 第三步：softmax → 归一化成权重")
    print(f"  ✓ 第四步：乘V → 加权求和")


class SelfAttention(nn.Module):
    """标准自注意力实现"""
    
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q、K、V的线性变换
        self.qkv = nn.Linear(dim, dim * 3)
        # 输出投影
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        """
        参数:
            x: 输入，形状 (B, N, C)
               B = batch size
               N = token数量（序列长度）
               C = 通道数（维度）
        
        返回:
            out: 输出，形状 (B, N, C)
        """
        B, N, C = x.shape
        
        # 计算Q、K、V
        # (B, N, 3C) → 拆成3个 (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B, heads, N, head_dim)
        
        # 计算注意力
        # q @ k.transpose(-2, -1) → (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        # attn @ v → (B, heads, N, head_dim)
        out = attn @ v
        
        # 把多头拼回去
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        
        return out


def demo_complexity_analysis():
    """
    演示4：自注意力的复杂度分析
    
    分析自注意力的计算量和显存占用
    """
    print("\n" + "=" * 60)
    print("演示4：自注意力的复杂度分析")
    print("=" * 60)
    
    print("""
【自注意力的复杂度】

输入形状：(B, N, C)
  B = batch size
  N = token数量（序列长度）
  C = 通道数（维度）

【计算复杂度】
  QKV变换: O(N·C²)  ← 线性变换
  Q·Kᵀ:    O(N²·C)  ← 点积，这是大头！
  softmax: O(N²)    ← 可以忽略
  attn·V:  O(N²·C)  ← 加权求和，也是大头
  输出投影: O(N·C²) ← 线性变换

  总计算量: O(N²·C + N·C²)

【显存复杂度】
  注意力矩阵: O(N²)  ← 这是最大的问题！
  QKV: O(N·C)
  输出: O(N·C)

  总显存: O(N² + N·C)

【关键问题】
  → 计算量和显存都和 N² 成正比！
  → N越大，越恐怖！
    """)
    
    # 举几个具体的例子
    print("【具体例子】")
    
    examples = [
        ("句子（100个词）", 100, 512),
        ("小图像（32×32=1024像素）", 1024, 256),
        ("中等图像（64×64=4096像素）", 4096, 256),
        ("大图像（256×256=65536像素）", 65536, 256),
    ]
    
    print(f"\n{'场景':<30} {'N':>8} {'C':>6} {'注意力矩阵大小':>16}")
    print("-" * 65)
    
    for name, N, C in examples:
        attn_size = N * N  # 元素个数
        attn_mb = attn_size * 4 / 1024 / 1024  # float32，转MB
        print(f"{name:<30} {N:>8} {C:>6} {attn_mb:>12.2f} MB")
    
    print("""
【看到问题了吗？】
- 句子（100个token）：注意力矩阵只有 0.04 MB，没问题
- 32×32小图：注意力矩阵 4 MB，还可以
- 64×64中图：注意力矩阵 64 MB，开始有点大了
- 256×256大图：注意力矩阵 16 GB！！！

→ 256×256的图，光注意力矩阵就要16GB显存！
→ 这还只是单头、单batch的情况！
→ 实际训练时，batch size=4，多头，还要算梯度，显存直接爆掉！

【这就是为什么图像Transformer这么难】
- ViT（Vision Transformer）把图像切成patch，减少N
- Swin Transformer用窗口注意力，只在窗口内算注意力
- TMT的TCJA也是为了解决这个问题！
    """)


def demo_why_standard_transformer_not_good_for_tmt():
    """
    演示5：为什么TMT不直接用标准Transformer？
    
    结合TMT的任务特点，分析标准Transformer的问题
    """
    print("\n" + "=" * 60)
    print("演示5：为什么TMT不直接用标准Transformer？")
    print("=" * 60)
    
    print("""
【TMT的任务特点】

1. 多帧输入：TMT要处理多帧图像（比如12帧、20帧）
   → 时间维度也有token
   → N = T × H × W
   → 比单帧图像的N大很多！

2. 湍流复原需要捕捉时间相关性
   → 帧与帧之间需要交互
   → 不能只看空间，还要看时间
   → 时间+空间，N就更大了

3. 输入分辨率不能太小
   → 图像复原任务需要高分辨率
   → 不能像ViT那样切成大patch
   → H和W不能太小

【标准自注意力的问题】

假设：12帧输入，每帧256×256
  N = 12 × 256 × 256 = 786,432 个token！

注意力矩阵大小：
  N² = (786,432)² ≈ 6.18 × 10¹¹ 个元素
  ≈ 2.3 TB 显存！！！

→ 这完全不可能！
→ 别说训练了，前向传播都跑不了！

【所以TMT需要一个更高效的注意力机制】

这就是TCJA（Temporal-Channel Joint Attention）的由来！
TCJA的核心思想：
  → 不在空间维度算注意力！
  → 只在时间和通道维度算注意力！
  → 空间维度用卷积来处理！

为什么这样可行？
  1. 空间相关性是局部的 → 卷积足够处理
  2. 时间相关性是全局的 → 需要注意力来捕捉
  3. 通道相关性也是全局的 → 注意力可以建模

这样一来：
  - 空间维度：用卷积，O(H·W·C²)，很高效
  - 时间+通道：用注意力，O(T·C²)，也很小
  - 总复杂度大大降低！

→ 这就是TMT最核心的创新点！
→ 下一个知识点我们详细讲TCJA！
    """)


def demo_transformer_block():
    """
    演示6：Transformer Block的完整结构
    
    自注意力 + FFN + 残差连接 + LayerNorm
    """
    print("\n" + "=" * 60)
    print("演示6：Transformer Block的完整结构")
    print("=" * 60)
    
    print("""
【Transformer Block的结构】

一个标准的Transformer Block长这样：

  ┌─────────────────────────┐
  │         输入 x          │
  └───────────┬─────────────┘
              │
        ┌─────▼─────┐
        │ LayerNorm │  ← 先归一化
        └─────┬─────┘
              │
        ┌─────▼─────┐
        │  自注意力  │  ← 核心
        └─────┬─────┘
              │
              └───────┐
                      │ 残差连接
        ┌─────────────▼─────────────┐
        │         x + attn(x)       │
        └─────────────┬─────────────┘
                      │
                ┌─────▼─────┐
                │ LayerNorm │  ← 再归一化
                └─────┬─────┘
                      │
                ┌─────▼─────┐
                │    FFN    │  ← 前馈网络（两个线性层+激活）
                └─────┬─────┘
                      │
                      └───────┐
                              │ 残差连接
                ┌─────────────▼─────────────┐
                │       x + ffn(x)          │
                └───────────────────────────┘

【两个子层】
1. 自注意力层（Self-Attention）
   → 捕捉token之间的依赖关系
   → 每个token都能看到所有token

2. 前馈网络层（Feed-Forward Network, FFN）
   → 对每个token单独做非线性变换
   → 相当于两个1×1卷积
   → FFN(x) = Linear2(GELU(Linear1(x)))

【两个关键设计】
1. 残差连接（Residual Connection）
   → 输出 = 输入 + 子层(输入)
   → 解决梯度消失问题，让深层网络能训练
   → 和ResNet里的残差连接是一个道理

2. 层归一化（Layer Normalization）
   → 对每个token的特征做归一化
   → 稳定训练，加速收敛
   → 有两种做法：
      - Pre-Norm：先Norm再子层（现在更常用）
      - Post-Norm：先子层再Norm（原始Transformer）

【为什么叫"Transformer"？】
- 因为它完全基于注意力机制，没有RNN、CNN
- 可以"直接变换"（transform）输入到输出
- 不需要像RNN那样一步步处理，并行度很高
    """)


def main():
    print("=" * 60)
    print("知识点5：Transformer基础与自注意力")
    print("=" * 60)
    print()
    print("【核心思想】")
    print("  Transformer = 自注意力 + FFN + 残差 + LayerNorm")
    print()
    print("  自注意力 = 每个token和所有token算相关性，然后加权聚合")
    print("  公式: Attention(Q,K,V) = softmax(QKᵀ/√d) · V")
    print()
    print("  最大问题: 复杂度 O(N²)，N大了显存爆炸")
    print("  TMT的解决方案: TCJA，只在时间+通道维度算注意力")
    print()
    
    # 运行所有演示
    demo_self_attention_intuition()
    demo_qkv_explanation()
    demo_self_attention_calculation()
    demo_complexity_analysis()
    demo_why_standard_transformer_not_good_for_tmt()
    demo_transformer_block()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. 自注意力：QKV → 算相似度 → softmax → 加权求和")
    print("2. 公式：Attention(Q,K,V) = softmax(QKᵀ/√d) · V")
    print("3. 复杂度：O(N²)，N是token数，大了显存爆炸")
    print("4. Transformer Block = 自注意力 + FFN + 残差 + LayerNorm")
    print("5. TMT不能直接用标准Transformer，因为N太大（多帧+高分辨率）")
    print()
    print("【下一步】")
    print("  → 下一个知识点：TCJA时间-通道联合注意力（TMT核心创新！）")
    print("  → 这是论文最有价值的部分，一定要好好理解！")


if __name__ == '__main__':
    main()
