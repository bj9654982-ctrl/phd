"""
图像去噪训练 - PyTorch GPU 版本
用 RTX 3060 GPU 训练 U-Net 去噪自编码器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 设备检查
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("设备:", device)
if device.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))
    print("显存: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
print("=" * 60)

# ============================================================
# 2. 加载 MNIST 数据 + 加噪
# ============================================================
print("\n加载数据...")
# 用 tensorflow 加载 MNIST（已经下载过了），转成 PyTorch 张量
import tensorflow as tf
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[:, np.newaxis, :, :]  # (N, 1, 28, 28) PyTorch格式
x_test = x_test[:, np.newaxis, :, :]

noise_factor = 0.3
np.random.seed(42)
x_train_noisy = np.clip(x_train + noise_factor * np.random.normal(0, 1, x_train.shape), 0, 1).astype('float32')
x_test_noisy = np.clip(x_test + noise_factor * np.random.normal(0, 1, x_test.shape), 0, 1).astype('float32')

# 转 PyTorch 张量
train_x = torch.from_numpy(x_train_noisy)
train_y = torch.from_numpy(x_train)
test_x = torch.from_numpy(x_test_noisy)
test_y = torch.from_numpy(x_test)

train_ds = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
test_ds = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

print("训练集:", x_train.shape)
print("测试集:", x_test.shape)
print("噪声强度:", noise_factor)

# ============================================================
# 3. 定义 U-Net 模型
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class DenoiseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        # 瓶颈
        self.bottleneck = DoubleConv(64, 128)
        # 解码器
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        # 输出
        self.out_conv = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)          # (B,32,28,28)
        e2 = self.enc2(self.pool(e1))  # (B,64,14,14)
        # 瓶颈
        b = self.bottleneck(self.pool(e2))  # (B,128,7,7)
        # 解码 + 跳跃连接
        d2 = self.up2(b)                      # (B,64,14,14)
        d2 = torch.cat([d2, e2], dim=1)       # 跳跃连接
        d2 = self.dec2(d2)                    # (B,64,14,14)
        d1 = self.up1(d2)                     # (B,32,28,28)
        d1 = torch.cat([d1, e1], dim=1)       # 跳跃连接
        d1 = self.dec1(d1)                    # (B,32,28,28)
        return self.sigmoid(self.out_conv(d1))

model = DenoiseUNet().to(device)
print("\n模型参数量: {:,}".format(sum(p.numel() for p in model.parameters())))

# ============================================================
# 4. 训练
# ============================================================
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
train_losses = []
val_losses = []

print("\n开始训练 (GPU)...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train = epoch_loss / len(train_loader)
    train_losses.append(avg_train)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            val_loss += criterion(output, batch_y).item()
    avg_val = val_loss / len(test_loader)
    val_losses.append(avg_val)

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print("Epoch %2d/%d | train_loss: %.6f | val_loss: %.6f" %
              (epoch+1, EPOCHS, avg_train, avg_val))

total_time = time.time() - start_time
print("\n训练完成! 总耗时: %.1f 秒 (%.2f 分钟)" % (total_time, total_time/60))
print("平均每个epoch: %.1f 秒" % (total_time / EPOCHS))

# ============================================================
# 5. 评估 PSNR
# ============================================================
model.eval()
with torch.no_grad():
    test_output = model(test_x.to(device)).cpu().numpy()

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 999

p_noisy = np.mean([psnr(x_test[i], x_test_noisy[i]) for i in range(500)])
p_denoise = np.mean([psnr(x_test[i], test_output[i]) for i in range(500)])
print("\n加噪图 PSNR: %.2f dB" % p_noisy)
print("去噪图 PSNR: %.2f dB" % p_denoise)
print("提升: %.2f dB" % (p_denoise - p_noisy))

# ============================================================
# 6. 可视化
# ============================================================
out_dir = r'J:\phd\tensorflow_demo'

# 训练曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, EPOCHS+1), train_losses, 'b-o', ms=4, label='训练损失', lw=2)
ax.plot(range(1, EPOCHS+1), val_losses, 'r-o', ms=4, label='验证损失', lw=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('BCE Loss', fontsize=12)
ax.set_title('PyTorch GPU 训练曲线 (U-Net, %d epoch, %.1fs)' % (EPOCHS, total_time),
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'denoise_training.png'), dpi=150, bbox_inches='tight')
plt.close()

# 去噪对比
n = 10
fig, axes = plt.subplots(3, n, figsize=(18, 6))
for i in range(n):
    axes[0,i].imshow(x_test[i,0], cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(x_test_noisy[i,0], cmap='gray'); axes[1,i].axis('off')
    axes[2,i].imshow(test_output[i,0], cmap='gray'); axes[2,i].axis('off')
axes[0,0].set_ylabel('原图', fontsize=13, rotation=0, labelpad=40)
axes[1,0].set_ylabel('加噪图', fontsize=13, rotation=0, labelpad=40)
axes[2,0].set_ylabel('去噪图', fontsize=13, rotation=0, labelpad=40)
plt.suptitle('PyTorch GPU 去噪效果（加噪 %.1fdB → 去噪 %.1fdB，提升 %.1fdB）' %
             (p_noisy, p_denoise, p_denoise-p_noisy), fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'denoise_result.png'), dpi=150, bbox_inches='tight')
plt.close()

# 加噪对比
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
for i in range(10):
    axes[0,i].imshow(x_test[i,0], cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(x_test_noisy[i,0], cmap='gray'); axes[1,i].axis('off')
axes[0,0].set_ylabel('原图', fontsize=12, rotation=0, labelpad=35)
axes[1,0].set_ylabel('加噪图', fontsize=12, rotation=0, labelpad=35)
plt.suptitle('原图 vs 加噪图（噪声 %.1f）' % noise_factor, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'denoise_input.png'), dpi=150, bbox_inches='tight')
plt.close()

# 残差分析
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
idx = 3
axes[0].imshow(x_test[idx,0], cmap='gray'); axes[0].set_title('原图'); axes[0].axis('off')
axes[1].imshow(x_test_noisy[idx,0], cmap='gray'); axes[1].set_title('加噪图'); axes[1].axis('off')
axes[2].imshow(test_output[idx,0], cmap='gray'); axes[2].set_title('去噪图'); axes[2].axis('off')
err = np.abs(x_test[idx,0] - test_output[idx,0])
im = axes[3].imshow(err, cmap='hot'); axes[3].set_title('残差(越暗越好)'); axes[3].axis('off')
plt.colorbar(im, ax=axes[3], shrink=0.8)
plt.suptitle('单张图片去噪细节与残差', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'denoise_residual.png'), dpi=150, bbox_inches='tight')
plt.close()

# 保存模型
torch.save(model.state_dict(), os.path.join(out_dir, 'denoiser_unet_pytorch.pth'))
print("\n全部完成! 文件保存在:", out_dir)
print("  denoise_result.png    去噪效果对比")
print("  denoise_training.png  训练曲线")
print("  denoise_input.png     原图vs加噪")
print("  denoise_residual.png  残差分析")
print("  denoiser_unet_pytorch.pth  模型权重")
