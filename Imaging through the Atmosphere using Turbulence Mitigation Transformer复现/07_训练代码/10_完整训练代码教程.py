"""
知识点10：TMT完整训练代码（简化版）
======================================

【为什么要学这个？】
前面我们学了TMT的各个组件，现在把它们拼起来，
看看完整的训练流程是怎么跑的。

【本教程你将学会】
1. 完整的训练流程（数据→前向→损失→反向→更新）
2. 分阶段训练怎么实现
3. 学习率调度、模型保存、日志打印
4. 如何在CPU上跑通小实验

【注意】
这是简化版的训练代码，用来学习流程。
真实的TMT训练需要GPU和大量数据。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

# ============================================================
# 第一部分：简化版网络（用我们之前学的）
# ============================================================

class SimpleTiltRemoval(nn.Module):
    """简化版去倾斜模块（2D U-Net代替3D，方便CPU训练）"""
    
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        
        C = base_channels
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1),
            nn.ReLU(),
        )
        self.down1 = nn.Conv2d(C, C*2, 2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(C*2, C*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(C*2, C*2, 3, padding=1),
            nn.ReLU(),
        )
        self.down2 = nn.Conv2d(C*2, C*4, 2, stride=2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(C*4, C*4, 3, padding=1),
            nn.ReLU(),
        )
        
        # 解码器
        self.up2 = nn.ConvTranspose2d(C*4, C*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(C*4, C*2, 3, padding=1),  # C*2 + C*2 = C*4
            nn.ReLU(),
            nn.Conv2d(C*2, C*2, 3, padding=1),
            nn.ReLU(),
        )
        
        self.up1 = nn.ConvTranspose2d(C*2, C, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(C*2, C, 3, padding=1),  # C + C = C*2
            nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1),
            nn.ReLU(),
        )
        
        # 输出
        self.output = nn.Conv2d(C, in_channels, 3, padding=1)
    
    def forward(self, x):
        # x: (B, T, C, H, W) → 把T当batch，用2D卷积（简化）
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        # 编码器
        e1 = self.enc1(x)
        x = self.down1(e1)
        e2 = self.enc2(x)
        x = self.down2(e2)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        # 输出
        x = self.output(x)
        x = x.reshape(B, T, C, H, W)
        
        return x


class SimpleBlurRemoval(nn.Module):
    """简化版去模糊模块（用卷积代替Transformer，方便CPU训练）"""
    
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        
        C = base_channels
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, C, 3, padding=1)
        
        # 几个残差块（代替Transformer Block）
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(C, C, 3, padding=1),
            ) for _ in range(3)
        ])
        
        # 输出投影
        self.output_proj = nn.Conv2d(C, in_channels, 3, padding=1)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        x = self.input_proj(x)
        
        for block in self.res_blocks:
            x = x + block(x)  # 残差连接
        
        x = self.output_proj(x)
        x = x.reshape(B, T, C, H, W)
        
        return x


class CharbonnierLoss(nn.Module):
    """Charbonnier损失"""
    
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff**2 + self.epsilon**2)
        return loss.mean()


# ============================================================
# 第二部分：模拟数据集（用随机数据代替真实数据）
# ============================================================

class DummyTurbulenceDataset(Dataset):
    """
    模拟湍流数据集（用随机数据，方便跑通流程）
    
    真实场景下，这里应该加载真实数据 + 湍流模拟器生成
    """
    
    def __init__(self, num_samples=100, num_frames=4, image_size=64):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 随机生成清晰图像（GT）
        gt = torch.randn(1, self.image_size, self.image_size)
        
        # 模拟湍流图像（GT加噪声，只是演示）
        # 真实场景下应该用湍流模拟器生成
        turbulent = gt + 0.1 * torch.randn(self.num_frames, 1, self.image_size, self.image_size)
        
        return turbulent, gt


# ============================================================
# 第三部分：阶段1训练：去倾斜模块
# ============================================================

def train_tilt_removal():
    """训练去倾斜模块"""
    
    print("=" * 60)
    print("阶段1：训练去倾斜模块")
    print("=" * 60)
    
    # 超参数
    num_epochs = 5
    batch_size = 2
    learning_rate = 2e-4
    num_frames = 4
    image_size = 64
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据
    train_dataset = DummyTurbulenceDataset(num_samples=50, num_frames=num_frames, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型
    model = SimpleTiltRemoval(in_channels=1, base_channels=16).to(device)
    print(f"去倾斜模块参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    criterion = CharbonnierLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 训练循环
    print(f"\n开始训练，共 {num_epochs} 个epoch")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (turbulent, gt) in enumerate(train_loader):
            turbulent = turbulent.to(device)  # (B, T, C, H, W)
            gt = gt.to(device)  # (B, C, H, W)
            
            # 前向传播
            pred = model(turbulent)  # (B, T, C, H, W)
            
            # 计算损失（每帧和GT比）
            gt_expanded = gt.unsqueeze(1).expand_as(pred)  # (B, T, C, H, W)
            loss = criterion(pred, gt_expanded)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 打印日志
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        
        print(f"  Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.6f} "
              f"LR: {lr:.2e} "
              f"Time: {elapsed:.1f}s")
    
    # 保存模型
    save_path = r'J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\07_训练代码\tilt_removal.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 去倾斜模块训练完成！")
    print(f"   模型已保存到: {save_path}")
    
    return model


# ============================================================
# 第四部分：阶段2训练：去模糊模块
# ============================================================

def train_blur_removal(tilt_model):
    """训练去模糊模块（固定去倾斜模块）"""
    
    print("\n" + "=" * 60)
    print("阶段2：训练去模糊模块")
    print("=" * 60)
    
    # 超参数
    num_epochs = 5
    batch_size = 2
    learning_rate = 2e-4
    num_frames = 4
    image_size = 64
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据
    train_dataset = DummyTurbulenceDataset(num_samples=50, num_frames=num_frames, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型
    tilt_model = tilt_model.to(device)
    tilt_model.eval()  # 固定去倾斜模块
    
    blur_model = SimpleBlurRemoval(in_channels=1, base_channels=32).to(device)
    print(f"去模糊模块参数量: {sum(p.numel() for p in blur_model.parameters()):,}")
    
    # 损失函数
    criterion = CharbonnierLoss()
    
    # 优化器（只更新去模糊模块的参数）
    optimizer = optim.Adam(blur_model.parameters(), lr=learning_rate)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 训练循环
    print(f"\n开始训练，共 {num_epochs} 个epoch")
    print("（去倾斜模块参数固定，只更新去模糊模块）")
    
    for epoch in range(num_epochs):
        blur_model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (turbulent, gt) in enumerate(train_loader):
            turbulent = turbulent.to(device)
            gt = gt.to(device)
            
            # 阶段1：去倾斜（固定参数，不计算梯度）
            with torch.no_grad():
                tilt_removed = tilt_model(turbulent)
            
            # 阶段2：去模糊
            pred = blur_model(tilt_removed)  # (B, T, C, H, W)
            
            # 计算损失
            gt_expanded = gt.unsqueeze(1).expand_as(pred)
            loss = criterion(pred, gt_expanded)
            
            # 反向传播（只更新去模糊模块）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 打印日志
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        
        print(f"  Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.6f} "
              f"LR: {lr:.2e} "
              f"Time: {elapsed:.1f}s")
    
    # 保存模型
    save_path = r'J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\07_训练代码\blur_removal.pth'
    torch.save(blur_model.state_dict(), save_path)
    print(f"\n✅ 去模糊模块训练完成！")
    print(f"   模型已保存到: {save_path}")
    
    return blur_model


# ============================================================
# 第五部分：完整的两阶段训练
# ============================================================

def main():
    print("=" * 60)
    print("TMT完整训练流程（简化版）")
    print("=" * 60)
    print()
    print("【说明】")
    print("  这是简化版的训练代码，用来学习流程。")
    print("  - 网络用简化版（卷积代替Transformer，方便CPU跑）")
    print("  - 数据用随机数据（代替真实数据+湍流模拟器）")
    print("  - 训练轮数少（只是演示流程）")
    print()
    print("【真实训练需要】")
    print("  1. GPU（3060 12GB应该够）")
    print("  2. 真实数据集（Place、SVW等）")
    print("  3. 完整的湍流模拟器")
    print("  4. 完整的TMT网络（TCJA等）")
    print("  5. 更多的训练迭代（400K + 600K）")
    print()
    
    # 阶段1：训练去倾斜模块
    tilt_model = train_tilt_removal()
    
    # 阶段2：训练去模糊模块
    blur_model = train_blur_removal(tilt_model)
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. 分阶段训练：先训去倾斜，再训去模糊")
    print("2. 第二阶段固定第一阶段的参数")
    print("3. 学习率用余弦退火，从高到低")
    print("4. 每个epoch打印loss、学习率、时间")
    print("5. 训练完保存模型权重")
    print()
    print("【下一步】")
    print("  如果你想跑真实的TMT训练：")
    print("  1. 下载官方代码和数据集")
    print("  2. 配置GPU环境")
    print("  3. 按官方README的步骤跑")
    print()
    print("  或者，我们可以继续改进这个简化版，")
    print("  把TCJA等真实组件加进去。")


if __name__ == '__main__':
    main()
