"""
第6课：卷积神经网络（CNN）—— 图像处理的利器
================================================
MLP 把图片展平成一维向量，丢失了空间结构信息。
CNN 通过卷积操作保留空间结构，参数共享，大幅减少参数量。

核心组件：
- Conv2d：卷积层，提取局部特征（边缘、纹理、形状）
- MaxPool2d：池化层，下采样，扩大感受野
- BatchNorm2d：批归一化，加速训练、稳定收敛

经典网络：LeNet -> AlexNet -> VGG -> ResNet

运行方式：
    conda activate torch-gpu
    python 06_cnn.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("1. 准备 CIFAR-10 数据集")
print("=" * 60)

# CIFAR-10：32x32 彩色图，10个类别（飞机、汽车、鸟、猫、鹿、狗、蛙、马、船、卡车）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 数据增强：随机裁剪
    transforms.RandomHorizontalFlip(),       # 数据增强：随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")
print(f"图片形状: {train_dataset[0][0].shape}")  # [3, 32, 32]
print(f"类别: {classes}")

print("\n" + "=" * 60)
print("2. 定义 CNN 模型")
print("=" * 60)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取部分（卷积 + 池化）
        self.features = nn.Sequential(
            # 第1个卷积块：输入 [3, 32, 32]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # [32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [32, 16, 16]
            nn.Dropout(0.25),

            # 第2个卷积块：输入 [32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # [64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [64, 8, 8]
            nn.Dropout(0.25),
        )

        # 分类部分（全连接）
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),  # 64*8*8 = 4096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)       # 卷积提取特征
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)      # 分类
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
print(f"模型结构:\n{model}")
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"设备: {device}")

print("\n" + "=" * 60)
print("3. 卷积层参数详解")
print("=" * 60)
print("""
Conv2d(in_channels, out_channels, kernel_size, stride, padding)

- in_channels: 输入通道数（RGB图=3，灰度图=1）
- out_channels: 输出通道数（卷积核数量，也是提取的特征数量）
- kernel_size: 卷积核大小（3x3 最常用）
- stride: 步长，默认1（卷积核每次移动几格）
- padding: 填充，默认0（在边缘补0，保持尺寸不变用 padding=kernel_size//2）

输出尺寸公式：
  out_size = (in_size - kernel_size + 2*padding) / stride + 1

例：输入 32x32, kernel=3, padding=1, stride=1
  out = (32 - 3 + 2*1) / 1 + 1 = 32  （尺寸不变）

MaxPool2d(kernel_size=2, stride=2)：宽高各减半
""")

print("\n" + "=" * 60)
print("4. 训练循环")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 学习率调度器：每 10 轮学习率乘以 0.5
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 10
train_losses = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()  # 更新学习率

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)

    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    test_accs.append(test_acc)

    elapsed = time.time() - start_time
    print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
          f"loss={epoch_loss:.4f}  "
          f"train_acc={epoch_acc:.1f}%  "
          f"test_acc={test_acc:.1f}%  "
          f"lr={optimizer.param_groups[0]['lr']:.6f}  "
          f"time={elapsed:.1f}s")

print(f"\n训练完成！最终测试准确率: {test_accs[-1]:.1f}%")

print("\n" + "=" * 60)
print("5. 可视化训练结果和特征图")
print("=" * 60)

fig = plt.figure(figsize=(15, 8))

# 图1：损失曲线
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(train_losses, 'b-o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# 图2：准确率曲线
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(test_accs, 'r-o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Test Accuracy')
ax2.grid(True, alpha=0.3)

# 图3：预测示例
ax3 = fig.add_subplot(2, 3, 3)
model.eval()
images, labels = next(iter(test_loader))
images_gpu = images.to(device)
with torch.no_grad():
    outputs = model(images_gpu)
    _, predicted = outputs.max(1)

for i in range(5):
    ax = fig.add_subplot(2, 5, 5 + i + 1)
    # 反标准化
    img = images[i].permute(1, 2, 0).numpy()
    img = img * (0.2470, 0.2435, 0.2616) + (0.4914, 0.4822, 0.4465)
    img = img.clip(0, 1)
    ax.imshow(img)
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f'Pred:{classes[predicted[i]]}\nTrue:{classes[labels[i]]}', color=color, fontsize=8)
    ax.axis('off')
ax3.set_title('Predictions')
ax3.axis('off')

# 图4-6：卷积特征图可视化
# 取第一张图，通过第一个卷积层看特征
with torch.no_grad():
    first_conv = model.features[0]  # 第一个 Conv2d
    features = first_conv(images_gpu[0:1])

for i in range(6):
    ax = fig.add_subplot(2, 6, 7 + i)
    feat = features[0, i].cpu().numpy()
    ax.imshow(feat, cmap='viridis')
    ax.set_title(f'Filter {i+1}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('06_cnn_result.png', dpi=150)
print("结果图已保存为: 06_cnn_result.png")

print("\n" + "=" * 60)
print("6. CNN 核心概念总结")
print("=" * 60)
print("""
1. 卷积（Convolution）：
   - 用小窗口（卷积核）在图片上滑动，提取局部特征
   - 参数共享：同一个卷积核在整张图上复用，大幅减少参数
   - 浅层提取边缘/纹理，深层提取语义概念

2. 池化（Pooling）：
   - MaxPool：取窗口内最大值，保留最显著特征
   - 作用：下采样（减小尺寸）、扩大感受野、平移不变性

3. 批归一化（BatchNorm）：
   - 对每个 batch 的特征做归一化（均值0，方差1）
   - 作用：加速收敛、允许更大学习率、轻微正则化效果

4. 数据增强（Data Augmentation）：
   - RandomCrop、RandomFlip 等
   - 作用：扩充训练数据多样性，防止过拟合

5. 学习率调度（LR Scheduler）：
   - 训练后期降低学习率，帮助模型收敛到更优解
   - StepLR：每 N 轮乘以 gamma
""")

print("\n" + "=" * 60)
print("第6课完成！CNN 是计算机视觉的基石。")
print("下一课：自定义数据集与 DataLoader")
print("=" * 60)
