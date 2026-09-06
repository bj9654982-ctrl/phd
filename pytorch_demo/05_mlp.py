"""
第5课：多层感知机（MLP）—— 第一个真正的神经网络
====================================================
前面的逻辑回归只有一层线性变换，只能学习线性决策边界。
MLP 在层之间加入非线性激活函数（ReLU），可以学习任意复杂的函数。

网络结构：
    输入层(2) -> 隐藏层(16) -> ReLU -> 隐藏层(8) -> ReLU -> 输出层(1) -> Sigmoid

关键知识点：
- nn.Sequential：快速堆叠层
- ReLU 激活函数：引入非线性
- 训练 MNIST 手写数字数据集（经典入门数据集）

运行方式：
    conda activate torch-gpu
    python 05_mlp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("1. 准备 MNIST 数据集")
print("=" * 60)

# 数据预处理：转 Tensor + 标准化
transform = transforms.Compose([
    transforms.ToTensor(),                    # [0,255] -> [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

# 下载/加载 MNIST（手写数字，28x28 灰度图，10个类别）
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

print(f"训练集: {len(train_dataset)} 张图片")
print(f"测试集: {len(test_dataset)} 张图片")
print(f"图片形状: {train_dataset[0][0].shape}")  # [1, 28, 28]

# DataLoader：分批加载，自动打乱
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"批大小: {batch_size}")
print(f"训练批数: {len(train_loader)}")

print("\n" + "=" * 60)
print("2. 定义 MLP 模型")
print("=" * 60)

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden1=128, hidden2=64, num_classes=10):
        super().__init__()
        # 第1种写法：逐层定义
        self.fc1 = nn.Linear(input_size, hidden1)   # 784 -> 128
        self.fc2 = nn.Linear(hidden1, hidden2)       # 128 -> 64
        self.fc3 = nn.Linear(hidden2, num_classes)   # 64 -> 10
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        # x: [batch, 1, 28, 28] -> 展平成 [batch, 784]
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))   # 线性 + ReLU
        x = self.dropout(x)
        x = self.relu(self.fc2(x))   # 线性 + ReLU
        x = self.dropout(x)
        x = self.fc3(x)               # 输出层，不用激活（CrossEntropyLoss 内部含 softmax）
        return x

# 第2种写法：用 nn.Sequential（更简洁）
class MLPSequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)

# 使用第一种写法（更清晰）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
print(f"模型结构:\n{model}")
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"设备: {device}")

print("\n" + "=" * 60)
print("3. 损失函数与优化器")
print("=" * 60)

# CrossEntropyLoss = LogSoftmax + NLLLoss，适合多分类
criterion = nn.CrossEntropyLoss()

# Adam 优化器：自适应学习率，比 SGD 更稳定，常用默认 lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"损失函数: CrossEntropyLoss")
print(f"优化器: Adam(lr=0.001)")

print("\n" + "=" * 60)
print("4. 训练循环")
print("=" * 60)

num_epochs = 5
train_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    model.train()  # 训练模式（启用 dropout）
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 前向
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向 + 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # 测试集评估
    model.eval()  # 评估模式（关闭 dropout）
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
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"loss={epoch_loss:.4f}  "
          f"train_acc={epoch_acc:.1f}%  "
          f"test_acc={test_acc:.1f}%  "
          f"time={elapsed:.1f}s")

print(f"\n训练完成！最终测试准确率: {test_accs[-1]:.1f}%")

print("\n" + "=" * 60)
print("5. 可视化训练结果")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 图1：损失曲线
axes[0].plot(train_losses, 'b-o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

# 图2：准确率曲线
axes[1].plot(train_accs, 'b-o', label='Train')
axes[1].plot(test_accs, 'r-o', label='Test')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3：预测示例
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    outputs = model(images)
    _, predicted = outputs.max(1)

for i in range(6):
    ax = axes[2] if i == 0 else fig.add_subplot(3, 6, 12 + i + 1)
    img = images[i].cpu().squeeze().numpy()
    ax.imshow(img, cmap='gray')
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f'Pred:{predicted[i].item()}\nTrue:{labels[i].item()}', color=color, fontsize=8)
    ax.axis('off')
axes[2].set_title('Predictions (green=correct)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('05_mlp_result.png', dpi=150)
print("结果图已保存为: 05_mlp_result.png")

print("\n" + "=" * 60)
print("6. 关键概念总结")
print("=" * 60)
print("""
1. 激活函数（ReLU）：引入非线性，让网络能学习复杂模式
   - ReLU(x) = max(0, x)
   - 没有激活函数，多层网络等价于单层线性网络

2. Dropout：训练时随机丢弃部分神经元，防止过拟合
   - 训练时启用，评估时关闭（model.eval()）

3. CrossEntropyLoss：多分类标准损失
   - 内部包含 LogSoftmax，所以模型最后一层不用加 softmax

4. Adam 优化器：自适应学习率，大多数场景默认选择
   - 比 SGD 收敛快，但有时 SGD 泛化更好

5. model.train() / model.eval()：切换训练/评估模式
   - 影响 Dropout、BatchNorm 等层的行为
""")

print("\n" + "=" * 60)
print("第5课完成！MLP 是神经网络的基础。")
print("下一课：卷积神经网络（CNN）—— 图像处理的利器")
print("=" * 60)
