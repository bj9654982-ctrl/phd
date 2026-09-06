"""
知识点4：多尺度损失函数（Charbonnier损失）
==========================================

【为什么要学这个？】
TMT的去倾斜模块和去模糊模块都用了多尺度的Charbonnier损失。
理解这个损失函数，你才能明白TMT训练的细节，以及为什么效果这么好。

【本教程你将学会】
1. L1损失 vs L2损失（MSE）：各自的优缺点
2. Charbonnier损失是什么？为什么它这么受欢迎？
3. 为什么TMT用Charbonnier损失而不是MSE？
4. 多尺度损失怎么具体实现？
5. 用PyTorch实现完整的多尺度Charbonnier损失
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


def l1_loss(pred, target):
    """L1损失（平均绝对误差）"""
    return torch.mean(torch.abs(pred - target))


def l2_loss(pred, target):
    """L2损失（均方误差MSE）"""
    return torch.mean((pred - target) ** 2)


def charbonnier_loss(pred, target, epsilon=1e-3):
    """
    Charbonnier损失
    
    公式：L = mean(sqrt((pred - target)^2 + epsilon^2))
    
    特点：
    - 在0点可导（不像L1损失在0点不可导）
    - 对异常值不敏感（不像L2损失对大误差惩罚很重）
    - 可以看作是L1损失的"平滑版"
    """
    diff = pred - target
    loss = torch.sqrt(diff ** 2 + epsilon ** 2)
    return torch.mean(loss)


def demo_loss_comparison():
    """
    演示1：L1 vs L2 vs Charbonnier 损失对比
    
    可视化三种损失函数的曲线，对比它们的特点
    """
    print("=" * 60)
    print("演示1：L1 vs L2 vs Charbonnier 损失对比")
    print("=" * 60)
    
    # 生成误差值
    x = np.linspace(-2, 2, 400)
    
    # 计算各种损失
    l1 = np.abs(x)
    l2 = x ** 2
    charbonnier = np.sqrt(x ** 2 + 0.1 ** 2)  # epsilon=0.1
    
    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 全范围
    axes[0].plot(x, l1, label='L1损失', linewidth=2)
    axes[0].plot(x, l2, label='L2损失 (MSE)', linewidth=2)
    axes[0].plot(x, charbonnier, label='Charbonnier损失', linewidth=2, linestyle='--')
    axes[0].set_xlabel('误差 (pred - target)', fontsize=12)
    axes[0].set_ylabel('损失值', fontsize=12)
    axes[0].set_title('三种损失函数对比（全范围）', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linewidth=0.5)
    
    # 放大原点附近
    x_zoom = np.linspace(-0.5, 0.5, 400)
    l1_zoom = np.abs(x_zoom)
    l2_zoom = x_zoom ** 2
    charbonnier_zoom = np.sqrt(x_zoom ** 2 + 0.1 ** 2)
    
    axes[1].plot(x_zoom, l1_zoom, label='L1损失', linewidth=2)
    axes[1].plot(x_zoom, l2_zoom, label='L2损失 (MSE)', linewidth=2)
    axes[1].plot(x_zoom, charbonnier_zoom, label='Charbonnier损失', linewidth=2, linestyle='--')
    axes[1].set_xlabel('误差 (pred - target)', fontsize=12)
    axes[1].set_ylabel('损失值', fontsize=12)
    axes[1].set_title('原点附近放大（看可导性）', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_损失函数对比.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 08_损失函数对比.png")
    plt.close()
    
    print("""
【三种损失的特点对比】

L1损失 (MAE):
  ✅ 优点：对异常值不敏感，鲁棒性好
  ❌ 缺点：在0点不可导，训练可能不稳定
  适用：回归问题，对异常值鲁棒要求高

L2损失 (MSE):
  ✅ 优点：处处可导，梯度平滑，训练稳定
  ❌ 缺点：对异常值非常敏感（大误差惩罚很重）
  适用：回归问题，数据比较干净

Charbonnier损失:
  ✅ 优点1：处处可导（平滑版L1）
  ✅ 优点2：对异常值不敏感（类似L1）
  ✅ 优点3：训练稳定，效果好
  ❌ 缺点：有一个超参数epsilon
  适用：图像复原、超分辨率等计算机视觉任务

→ 这就是为什么Charbonnier损失在图像复原领域这么受欢迎！
→ 它结合了L1和L2的优点！
    """)


def demo_why_charbonnier_better():
    """
    演示2：为什么Charbonnier损失更好？
    
    模拟有异常值的情况，看三种损失的表现
    """
    print("\n" + "=" * 60)
    print("演示2：有异常值时，三种损失的表现")
    print("=" * 60)
    
    # 模拟数据：大部分误差很小，但有几个大误差（异常值）
    errors = np.array([0.01, 0.02, 0.015, 0.03, 0.02, 0.025, 0.018, 0.5, 0.6])  # 最后两个是异常值
    
    l1 = np.mean(np.abs(errors))
    l2 = np.mean(errors ** 2)
    charbonnier = np.mean(np.sqrt(errors ** 2 + 0.01 ** 2))
    
    print(f"\n模拟误差: {errors}")
    print(f"  正常误差: {errors[:7]}")
    print(f"  异常值: {errors[7:]}")
    print()
    print(f"L1损失: {l1:.4f}")
    print(f"L2损失: {l2:.4f}")
    print(f"Charbonnier损失: {charbonnier:.4f}")
    
    print("""
【观察】
- L2损失被异常值"带偏"了，损失值很大
- L1损失和Charbonnier损失受异常值影响较小

【为什么图像复原喜欢用Charbonnier？】
1. 图像中经常有 outliers（比如饱和区域、噪声点）
2. MSE对这些outliers太敏感，会让模型去拟合噪声
3. L1鲁棒但在0点不可导，训练不稳定
4. Charbonnier两者兼顾：鲁棒 + 可导

→ 这就是TMT选择Charbonnier损失的原因！
    """)


class MultiScaleCharbonnierLoss(nn.Module):
    """
    多尺度Charbonnier损失
    
    TMT论文中的损失函数：
        L_tilt = Σ(γ_ℓ · L_char(Ĵ_ℓ, J̃_ℓ))
    
    其中：
    - γ_ℓ: 第ℓ尺度的权重
    - L_char: Charbonnier损失
    - Ĵ_ℓ: 第ℓ尺度的预测
    - J̃_ℓ: 第ℓ尺度的GT（需要下采样）
    """
    
    def __init__(self, weights=None, epsilon=1e-3):
        """
        参数:
            weights: 各尺度的权重列表，从细到粗
                     如果为None，默认 [0.6, 0.3, 0.1]（从粗到细）
                     注意：我们的输入是从细到粗，所以要反转一下
            epsilon: Charbonnier损失的平滑参数
        """
        super().__init__()
        self.epsilon = epsilon
        
        # 默认权重（TMT论文设置）
        if weights is None:
            # 从粗到细的权重：粗尺度权重最大
            weights_coarse_to_fine = [0.6, 0.3, 0.1]
            # 我们的输入是从细到粗，所以反转一下
            self.weights = weights_coarse_to_fine[::-1]  # [0.1, 0.3, 0.6]
        else:
            self.weights = weights
    
    def charbonnier(self, pred, target):
        """计算单尺度Charbonnier损失"""
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return torch.mean(loss)
    
    def forward(self, predictions, target):
        """
        计算多尺度损失
        
        参数:
            predictions: 预测列表，从细到粗 [pred1, pred2, ...]
                         pred1是最高分辨率，pred2是下采样后的，以此类推
            target: 最高分辨率的GT（清晰图像）
        
        返回:
            total_loss: 总损失
            loss_list: 各尺度的损失列表
        """
        assert len(predictions) == len(self.weights), \
            f"预测数量({len(predictions)})与权重数量({len(self.weights)})不匹配"
        
        total_loss = 0.0
        loss_list = []
        
        current_target = target
        
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            # 计算当前尺度的损失
            loss = self.charbonnier(pred, current_target)
            loss_list.append(loss.item())
            
            # 加权累加
            total_loss += weight * loss
            
            # 将target下采样，作为下一个更粗尺度的GT
            if i < len(predictions) - 1:
                # 空间维度下采样（平均池化）
                # 注意：3D数据形状是 (B, C, T, H, W)
                # 我们只下采样空间维度H和W，保持时间维度T不变
                if current_target.dim() == 5:  # 3D数据 (B, C, T, H, W)
                    # 用平均池化下采样空间维度
                    current_target = F.avg_pool3d(current_target, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                elif current_target.dim() == 4:  # 2D数据 (B, C, H, W)
                    current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)
        
        return total_loss, loss_list


def demo_multi_scale_loss():
    """
    演示3：多尺度损失的计算过程
    
    模拟一个3尺度的U-Net输出，计算多尺度损失
    """
    print("\n" + "=" * 60)
    print("演示3：多尺度损失计算过程")
    print("=" * 60)
    
    # 创建损失函数
    criterion = MultiScaleCharbonnierLoss()
    
    print(f"\n【损失函数设置】")
    print(f"  尺度权重（从细到粗）: {criterion.weights}")
    print(f"  权重和: {sum(criterion.weights)}")
    print(f"  Charbonnier epsilon: {criterion.epsilon}")
    
    # 模拟预测（3个尺度，从细到粗）
    # 假设：batch=2, 通道=1, 帧数=8
    pred_fine = torch.randn(2, 1, 8, 64, 64)    # 64×64
    pred_mid = torch.randn(2, 1, 8, 32, 32)     # 32×32
    pred_coarse = torch.randn(2, 1, 8, 16, 16)  # 16×16
    
    predictions = [pred_fine, pred_mid, pred_coarse]
    
    # 模拟GT（只有最高分辨率的）
    target = torch.randn(2, 1, 8, 64, 64)
    
    print(f"\n【输入数据】")
    print(f"  预测（从细到粗）:")
    for i, p in enumerate(predictions):
        print(f"    尺度{i+1}: {p.shape}")
    print(f"  GT: {target.shape}")
    
    # 计算损失
    total_loss, loss_list = criterion(predictions, target)
    
    print(f"\n【损失计算结果】")
    for i, (loss, weight) in enumerate(zip(loss_list, criterion.weights)):
        print(f"  尺度{i+1}: 损失 = {loss:.4f}, 权重 = {weight}, 加权 = {loss*weight:.4f}")
    print(f"  总损失: {total_loss.item():.4f}")
    
    print("""
【注意事项】
1. GT只有最高分辨率的，低分辨率GT是自动下采样得到的
2. 下采样用平均池化（不是双线性插值，保持简单）
3. 时间维度T不下采样，只下采样空间维度H和W
4. 粗尺度权重更大，因为：
   - 粗尺度的倾斜更大、更明显，更容易学
   - 先学大的位移，再学小的位移
   - 符合"从粗到细"的优化策略
    """)


def demo_loss_on_real_example():
    """
    演示4：在实际图像上看损失效果
    
    生成一个简单的例子，看不同损失对图像复原的影响
    """
    print("\n" + "=" * 60)
    print("演示4：损失函数对图像复原的影响")
    print("=" * 60)
    
    print("""
【为什么不用MSE？】

MSE（L2损失）的问题：
1. MSE最小化的是"像素级均方误差"
2. 但人眼感知的是"结构相似性"
3. MSE小的图像，人眼看起来不一定好
4. MSE对异常值太敏感，容易产生过平滑的结果

Charbonnier损失的优势：
1. 类似L1，对异常值不敏感
2. 结果更锐利，不会过度平滑
3. 训练更稳定（因为可导）

【TMT的选择】
TMT论文中，去倾斜模块和去模糊模块都用了Charbonnier损失。
这不是随便选的，而是经过实验验证的。

实际上，Charbonnier损失在图像复原领域已经是"标配"了，
比如超分辨率、去模糊、去噪等任务，很多都用它。
    """)


def main():
    print("=" * 60)
    print("知识点4：多尺度损失函数（Charbonnier损失）")
    print("=" * 60)
    print()
    print("【核心思想】")
    print("  TMT用多尺度Charbonnier损失来训练网络")
    print()
    print("  Charbonnier = 平滑版L1损失")
    print("    → 既有L1的鲁棒性，又有L2的可导性")
    print()
    print("  多尺度 = 每个尺度都算损失")
    print("    → 粗尺度权重更大，从粗到细逐步优化")
    print("    → 训练更稳定，效果更好")
    print()
    
    # 运行所有演示
    demo_loss_comparison()
    demo_why_charbonnier_better()
    demo_multi_scale_loss()
    demo_loss_on_real_example()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. L1损失：鲁棒但不可导；L2损失：可导但不鲁棒")
    print("2. Charbonnier损失：两者兼顾，平滑版L1")
    print("3. 图像复原任务常用Charbonnier，效果比MSE好")
    print("4. 多尺度损失 = 每个尺度都算损失，加权求和")
    print("5. 粗尺度权重更大 → 先学大的，再学小的")
    print()
    print("【下一步】")
    print("  → 下一个知识点：Transformer基础与自注意力")
    print("  → （进入TMT第二阶段：去模糊模块的核心）")


if __name__ == '__main__':
    main()
