"""
第9课：迁移学习（Transfer Learning）—— 站在巨人的肩膀上
============================================================
在大数据集（如 ImageNet，120万张图）上预训练好的模型，
已经学会了通用的视觉特征（边缘、纹理、形状）。
迁移学习就是把这些特征迁移到你的小数据集任务上。

两种策略：
1. 微调（Fine-tuning）：加载预训练权重，训练所有层（或后面几层）
2. 特征提取（Feature Extraction）：冻结 backbone，只训练最后的分类头

常用预训练模型：ResNet、VGG、EfficientNet、ViT 等

运行方式：
    conda activate torch-gpu
    python 09_transfer_learning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("1. 加载预训练模型")
print("=" * 60)

# 加载在 ImageNet 上预训练的 ResNet18
# weights 参数指定预训练权重（新版 torchvision 用法）
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print(f"ResNet18 结构（部分）:")
print(model)

print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

# ResNet18 原本是 1000 类（ImageNet），我们改成自己的类别数
num_classes = 10  # 假设我们的任务是10分类
model.fc = nn.Linear(model.fc.in_features, num_classes)
print(f"\n替换最后一层为 {num_classes} 类")
print(f"新的 fc 层: {model.fc}")

print("\n" + "=" * 60)
print("2. 策略一：特征提取（冻结 backbone，只训练分类头）")
print("=" * 60)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只让最后一层可训练
for param in model.fc.parameters():
    param.requires_grad = True

# 统计可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"可训练比例: {trainable_params/total_params*100:.2f}%")

# 优化器只传入可训练参数
optimizer_featex = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
print(f"\n特征提取模式：只训练 fc 层，训练快，适合小数据集")

print("\n" + "=" * 60)
print("3. 策略二：微调（训练所有层，用更小学习率）")
print("=" * 60)

# 解冻所有参数
for param in model.parameters():
    param.requires_grad = True

# 微调时通常用更小的学习率，避免破坏预训练特征
# 可以给不同层设置不同学习率（差分学习率）
optimizer_finetune = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 0.00001},  # 最浅层，最小lr
    {'params': model.layer1.parameters(), 'lr': 0.00001},
    {'params': model.layer2.parameters(), 'lr': 0.00005},
    {'params': model.layer3.parameters(), 'lr': 0.0001},
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001},       # 最后层，最大lr
], lr=0.001)

print("微调模式：训练所有层，使用差分学习率")
print("  - 浅层（通用特征）：lr=1e-5")
print("  - 中层：lr=5e-5 ~ 1e-4")
print("  - 分类头：lr=1e-3")
print("适合中等以上数据集，效果通常更好")

print("\n" + "=" * 60)
print("4. 完整迁移学习示例：CIFAR-10 + ResNet18")
print("=" * 60)

# 数据预处理（ResNet 输入需要 224x224，ImageNet 标准化）
transform_train = transforms.Compose([
    transforms.Resize(224),  # CIFAR-10 是 32x32，放大到 224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet 均值方差
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 只用子集加速演示
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 取子集（演示用，实际训练用全部数据）
train_subset = torch.utils.data.Subset(train_dataset, range(2000))
test_subset = torch.utils.data.Subset(test_dataset, range(500))

batch_size = 32
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(f"训练子集: {len(train_subset)}, 测试子集: {len(test_subset)}")

# 重新创建模型，使用特征提取策略
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 冻结 backbone
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

print(f"\n模型设备: {device}")
print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 训练
num_epochs = 3
train_losses = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for images, labels in train_loader:
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

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"loss={epoch_loss:.4f}  "
          f"train_acc={epoch_acc:.1f}%  "
          f"test_acc={test_acc:.1f}%  "
          f"time={time.time()-start:.1f}s")

print(f"\n迁移学习（特征提取）完成！测试准确率: {test_accs[-1]:.1f}%")
print("只用了 2000 张训练图、3 个 epoch，只训练了最后一层！")

print("\n" + "=" * 60)
print("5. 可视化预测结果")
print("=" * 60)

model.eval()
images, labels = next(iter(test_loader))
images_gpu = images.to(device)
with torch.no_grad():
    outputs = model(images_gpu)
    _, predicted = outputs.max(1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    img = images[i].permute(1, 2, 0).numpy()
    # 反标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = img.clip(0, 1)
    ax.imshow(img)
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f'Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}',
                 color=color, fontsize=9)
    ax.axis('off')

plt.suptitle('Transfer Learning Predictions (ResNet18 pretrained on ImageNet)', fontsize=12)
plt.tight_layout()
plt.savefig('09_transfer_learning_result.png', dpi=150)
print("预测结果图已保存为: 09_transfer_learning_result.png")

print("\n" + "=" * 60)
print("6. 迁移学习最佳实践")
print("=" * 60)
print("""
1. 什么时候用迁移学习？
   - 数据集小（< 1万张）：特征提取，只训练分类头
   - 数据集中等（1万-10万张）：微调，训练所有层
   - 数据集大（> 10万张）：可以从头训练，但迁移学习仍有帮助

2. 数据预处理要和预训练时一致：
   - 使用 ImageNet 的均值和方差做 Normalize
   - 输入尺寸要匹配（ResNet 用 224x224）

3. 学习率选择：
   - 特征提取：lr = 0.001 ~ 0.01（只训练分类头，可以大一点）
   - 微调：lr = 0.0001 ~ 0.001（要小，避免破坏预训练特征）
   - 差分学习率：浅层小，深层大

4. 冻结策略：
   - 先冻结 backbone 训练分类头（几个 epoch）
   - 再解冻部分层微调（逐步解冻，从后往前）

5. 常用预训练模型选择：
   - ResNet18/34/50：通用，速度快，效果好
   - EfficientNet：精度高，参数少
   - ViT（Vision Transformer）：大模型，大数据集效果好
""")

print("\n" + "=" * 60)
print("第9课完成！迁移学习是实际项目中最常用的技巧。")
print("下一课：完整项目实战 —— 把所有知识串起来")
print("=" * 60)
