"""
第7课：自定义数据集与 DataLoader
==================================
实际项目中，你需要用自己的数据训练模型。
PyTorch 提供了 Dataset 和 DataLoader 来优雅地处理数据加载。

核心概念：
- Dataset：定义数据从哪来、怎么取、有多少条
- DataLoader：自动分批、打乱、并行加载
- transforms：数据预处理和增强流水线

运行方式：
    conda activate torch-gpu
    python 07_dataset_dataloader.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("1. 最简单的自定义 Dataset")
print("=" * 60)

# 继承 Dataset，必须实现 __len__ 和 __getitem__
class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """返回第 idx 个样本"""
        return self.data[idx], self.labels[idx]

# 创建数据
data = torch.randn(100, 10)  # 100个样本，每个10个特征
labels = torch.randint(0, 2, (100,))  # 100个二分类标签

dataset = SimpleDataset(data, labels)
print(f"数据集大小: {len(dataset)}")
print(f"第0个样本: data.shape={dataset[0][0].shape}, label={dataset[0][1]}")

print("\n" + "=" * 60)
print("2. DataLoader：分批、打乱、并行")
print("=" * 60)

dataloader = DataLoader(
    dataset,
    batch_size=16,      # 每批16个样本
    shuffle=True,        # 打乱顺序
    num_workers=0,       # 并行加载进程数（Windows 上设 0 更稳定）
    drop_last=True,      # 最后不足一批就丢弃
)

print(f"批大小: 16, 批数: {len(dataloader)}")
for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"  批 {batch_idx}: data.shape={batch_data.shape}, labels.shape={batch_labels.shape}")
    if batch_idx >= 2:
        break

print("\n" + "=" * 60)
print("3. 从文件夹加载图片的标准 Dataset")
print("=" * 60)
print("""
假设目录结构：
    my_dataset/
    ├── class_A/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── class_B/
    │   ├── img3.jpg
    │   └── img4.jpg
    └── ...
""")

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        # 遍历所有类别文件夹，收集图片路径和标签
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(cls_dir, img_name)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

print("ImageFolderDataset 类已定义（需要实际图片目录才能运行）")
print("PyTorch 也内置了 torchvision.datasets.ImageFolder，功能一样")

print("\n" + "=" * 60)
print("4. 从 CSV 文件加载数据")
print("=" * 60)

import csv

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            for row in reader:
                # 假设最后一列是标签，前面是特征
                features = list(map(float, row[:-1]))
                label = int(row[-1])
                self.data.append((features, label))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

print("CSVDataset 类已定义")

print("\n" + "=" * 60)
print("5. 数据增强 transforms 详解")
print("=" * 60)

# 创建一个示例图片
demo_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

# 常用 transforms
transform_list = [
    ("原始", transforms.Compose([transforms.ToTensor()])),
    ("Resize", transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])),
    ("RandomCrop", transforms.Compose([transforms.RandomCrop(32), transforms.ToTensor()])),
    ("RandomHorizontalFlip", transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()])),
    ("RandomRotation", transforms.Compose([transforms.RandomRotation(30), transforms.ToTensor()])),
    ("ColorJitter", transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5), transforms.ToTensor()])),
    ("Normalize", transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])),
]

print("常用数据增强操作：")
for name, _ in transform_list:
    print(f"  - {name}")

# 可视化数据增强效果
fig, axes = plt.subplots(1, len(transform_list), figsize=(20, 3))
for ax, (name, transform) in zip(axes, transform_list):
    img_t = transform(demo_img)
    if img_t.shape[0] == 3:
        # 反标准化用于显示
        img_np = img_t.permute(1, 2, 0).numpy()
        if name == "Normalize":
            img_np = img_np * 0.5 + 0.5
        img_np = img_np.clip(0, 1)
        ax.imshow(img_np)
    else:
        ax.imshow(img_t.squeeze().numpy(), cmap='gray')
    ax.set_title(name, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('07_transforms_demo.png', dpi=150)
print("\n数据增强效果图已保存为: 07_transforms_demo.png")

print("\n" + "=" * 60)
print("6. 完整示例：合成数据 + 自定义 Dataset + 训练")
print("=" * 60)

# 合成一个二分类数据集（类似第4课的逻辑回归数据）
class SyntheticDataset(Dataset):
    def __init__(self, n_samples=500, n_features=20, n_classes=2):
        torch.manual_seed(42)
        self.n_samples = n_samples
        # 生成可分的数据
        self.data = torch.randn(n_samples, n_features)
        self.labels = torch.randint(0, n_classes, (n_samples,))
        # 让两类数据有差异
        self.data[self.labels == 0] += 1.0
        self.data[self.labels == 1] -= 1.0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集和 DataLoader
dataset = SyntheticDataset(n_samples=500, n_features=20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

# 简单的分类模型
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(10):
    model.train()
    train_loss = 0
    for data, labels in train_loader:
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Epoch [{epoch+1}/10] loss={train_loss/len(train_loader):.4f} "
          f"val_acc={100.*correct/total:.1f}%")

print("\n" + "=" * 60)
print("7. 关键技巧总结")
print("=" * 60)
print("""
1. Dataset 必须实现 __len__ 和 __getitem__
2. DataLoader 参数：
   - batch_size: 批大小
   - shuffle: 训练时 True，测试时 False
   - num_workers: 并行加载，Windows 上建议 0，Linux 上可用 2-4
   - drop_last: 最后不足一批时是否丢弃
   - pin_memory: GPU 训练时设 True，加速数据传输

3. 数据划分：torch.utils.data.random_split 或手动索引

4. transforms.Compose：把多个预处理操作串成流水线

5. 训练集做数据增强，验证/测试集只做基础预处理（Resize + Normalize）
""")

print("\n" + "=" * 60)
print("第7课完成！自定义数据集是实际项目的必备技能。")
print("下一课：模型保存与加载")
print("=" * 60)
