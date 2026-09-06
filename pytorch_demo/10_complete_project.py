"""
第10课：完整项目实战 —— 图像分类全流程
==========================================
把前面所有知识串起来，实现一个完整的图像分类项目：

1. 配置管理（超参数集中管理）
2. 数据准备（自定义 Dataset + DataLoader + 数据增强）
3. 模型构建（支持多种模型选择）
4. 训练循环（含验证、早停、学习率调度、最佳模型保存）
5. 测试评估（混淆矩阵、分类报告）
6. 推理预测（单张图片预测）

这是一个可以直接复用的项目模板。

运行方式：
    conda activate torch-gpu
    python 10_complete_project.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from collections import defaultdict

print("=" * 60)
print("完整项目实战：CIFAR-10 图像分类")
print("=" * 60)

# ============================================================
# 1. 配置管理
# ============================================================
class Config:
    # 数据
    data_root = './data'
    num_classes = 10
    img_size = 32

    # 训练
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-4

    # 模型
    model_name = 'resnet18'  # 'simple_cnn' or 'resnet18'
    pretrained = False
    dropout = 0.25

    # 学习率调度
    scheduler_type = 'cosine'  # 'step' or 'cosine'
    step_size = 5
    gamma = 0.5

    # 早停
    early_stop_patience = 5

    # 路径
    save_dir = './checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机种子
    seed = 42

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

# 设置随机种子，保证可复现
torch.manual_seed(config.seed)
np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

print(f"配置:")
print(f"  模型: {config.model_name}, 预训练: {config.pretrained}")
print(f"  批大小: {config.batch_size}, Epochs: {config.num_epochs}")
print(f"  学习率: {config.learning_rate}, 设备: {config.device}")

# ============================================================
# 2. 数据准备
# ============================================================
print("\n" + "=" * 60)
print("2. 数据准备")
print("=" * 60)

# CIFAR-10 标准化参数
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

train_dataset = CIFAR10(root=config.data_root, train=True, download=True, transform=transform_train)
test_dataset = CIFAR10(root=config.data_root, train=False, download=True, transform=transform_test)

# 划分训练集和验证集（80% 训练，20% 验证）
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(config.seed)
)

train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(f"训练集: {len(train_subset)}, 验证集: {len(val_subset)}, 测试集: {len(test_dataset)}")

# ============================================================
# 3. 模型构建
# ============================================================
print("\n" + "=" * 60)
print("3. 模型构建")
print("=" * 60)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def build_model(config):
    if config.model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=config.num_classes, dropout=config.dropout)
    elif config.model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if config.pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    return model.to(config.device)

model = build_model(config)
print(f"模型: {config.model_name}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 4. 损失函数、优化器、学习率调度
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

if config.scheduler_type == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
elif config.scheduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

print(f"优化器: Adam(lr={config.learning_rate}, weight_decay={config.weight_decay})")
print(f"学习率调度: {config.scheduler_type}")

# ============================================================
# 5. 训练循环（含验证、早停、最佳模型保存）
# ============================================================
print("\n" + "=" * 60)
print("5. 开始训练")
print("=" * 60)

history = defaultdict(list)
best_val_acc = 0.0
early_stop_counter = 0
best_model_path = os.path.join(config.save_dir, 'best_model.pth')

for epoch in range(config.num_epochs):
    # ---- 训练 ----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(config.device), labels.to(config.device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    scheduler.step()

    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total

    # ---- 验证 ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total

    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    elapsed = time.time() - start_time
    print(f"Epoch [{epoch+1:2d}/{config.num_epochs}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% | "
          f"lr={optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': vars(config),
        }, best_model_path)
        print(f"  -> 新最佳验证准确率 {val_acc:.1f}%，模型已保存")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= config.early_stop_patience:
            print(f"  -> 早停！连续 {config.early_stop_patience} 轮没有提升")
            break

print(f"\n训练完成！最佳验证准确率: {best_val_acc:.1f}%")

# ============================================================
# 6. 测试评估
# ============================================================
print("\n" + "=" * 60)
print("6. 测试集评估")
print("=" * 60)

# 加载最佳模型
checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(config.device), labels.to(config.device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100. * test_correct / test_total
print(f"测试集准确率: {test_acc:.1f}%")

# 混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(all_labels, all_preds)

print(f"\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=classes, digits=3))

# ============================================================
# 7. 可视化
# ============================================================
print("\n" + "=" * 60)
print("7. 可视化训练结果")
print("=" * 60)

fig = plt.figure(figsize=(16, 10))

# 图1：损失曲线
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(history['train_loss'], 'b-', label='Train')
ax1.plot(history['val_loss'], 'r-', label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：准确率曲线
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(history['train_acc'], 'b-', label='Train')
ax2.plot(history['val_acc'], 'r-', label='Val')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：学习率曲线
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(history['lr'], 'g-')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('Learning Rate Schedule')
ax3.grid(True, alpha=0.3)

# 图4：混淆矩阵
ax4 = fig.add_subplot(2, 3, 4)
im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax4.set_title('Confusion Matrix')
ax4.set_xticks(range(10))
ax4.set_yticks(range(10))
ax4.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
ax4.set_yticklabels(classes, fontsize=8)
# 在格子里标数字
thresh = cm.max() / 2.
for i in range(10):
    for j in range(10):
        ax4.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=7)
ax4.set_ylabel('True Label')
ax4.set_xlabel('Predicted Label')
plt.colorbar(im, ax=ax4)

# 图5-6：预测示例
images, labels = next(iter(test_loader))
images_gpu = images.to(config.device)
with torch.no_grad():
    outputs = model(images_gpu)
    _, predicted = outputs.max(1)

for i in range(8):
    ax = fig.add_subplot(2, 8, 9 + i)
    img = images[i].permute(1, 2, 0).numpy()
    img = img * np.array(cifar_std) + np.array(cifar_mean)
    img = img.clip(0, 1)
    ax.imshow(img)
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f'{classes[predicted[i]]}\n({classes[labels[i]]})', color=color, fontsize=7)
    ax.axis('off')

plt.suptitle(f'CIFAR-10 Classification | {config.model_name} | Test Acc: {test_acc:.1f}%', fontsize=14)
plt.tight_layout()
plt.savefig('10_complete_project_result.png', dpi=150)
print("结果图已保存为: 10_complete_project_result.png")

# ============================================================
# 8. 推理函数（单张图片预测）
# ============================================================
print("\n" + "=" * 60)
print("8. 推理函数示例")
print("=" * 60)

def predict_image(model, image_tensor, classes, device):
    """单张图片预测，返回预测类别和概率"""
    model.eval()
    with torch.no_grad():
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
        return classes[predicted.item()], confidence.item(), probs[0].cpu().numpy()

# 测试推理函数
img, label = test_dataset[0]
pred_class, confidence, all_probs = predict_image(model, img, classes, config.device)
print(f"真实类别: {classes[label]}")
print(f"预测类别: {pred_class} (置信度: {confidence*100:.1f}%)")
print(f"各类别概率:")
for cls, prob in zip(classes, all_probs):
    bar = '█' * int(prob * 30)
    print(f"  {cls:6s}: {prob*100:5.1f}% {bar}")

print("\n" + "=" * 60)
print("完整项目实战完成！")
print("=" * 60)
print("""
这个项目模板包含了深度学习项目的所有标准组件：
1. 配置管理 —— 超参数集中管理，方便调参
2. 数据准备 —— Dataset + DataLoader + 数据增强 + 训练/验证/测试划分
3. 模型构建 —— 支持多种模型切换
4. 训练循环 —— 验证、早停、学习率调度、最佳模型保存
5. 测试评估 —— 混淆矩阵、分类报告
6. 推理函数 —— 单张图片预测，可直接部署

你可以基于这个模板修改：
- 换数据集：修改数据加载部分
- 换模型：在 build_model 里添加新模型
- 换任务：修改损失函数和输出层
""")
