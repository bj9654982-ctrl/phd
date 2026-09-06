# PyTorch 学习指南

> 从零开始，循序渐进掌握 PyTorch 深度学习框架

## 环境准备

```bash
# 激活 conda 环境
conda activate torch-gpu

# 验证安装
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

当前环境：
- Python 3.11
- PyTorch 2.5.1 + CUDA 12.1
- GPU: RTX 3060 Laptop (12GB)

## 课程列表

| 序号 | 文件 | 主题 | 核心知识点 |
|------|------|------|-----------|
| 01 | `01_tensor_basics.py` | 张量基础 | 创建、运算、变形、索引、GPU、NumPy互转 |
| 02 | `02_autograd.py` | 自动求导 | requires_grad、backward、梯度清零、no_grad |
| 03 | `03_linear_regression.py` | 线性回归 | 完整训练流程：数据→模型→损失→优化器→训练 |
| 04 | `04_logistic_regression.py` | 逻辑回归 | sigmoid、BCE损失、二分类、决策边界 |
| 05 | `05_mlp.py` | 多层感知机 | ReLU、Dropout、CrossEntropy、MNIST、Adam |
| 06 | `06_cnn.py` | 卷积神经网络 | Conv2d、Pooling、BatchNorm、CIFAR-10、数据增强 |
| 07 | `07_dataset_dataloader.py` | 自定义数据集 | Dataset、DataLoader、transforms、CSV/图片加载 |
| 08 | `08_save_load.py` | 模型保存加载 | state_dict、checkpoint、best model、CPU/GPU加载 |
| 09 | `09_transfer_learning.py` | 迁移学习 | 预训练模型、特征提取、微调、差分学习率 |
| 10 | `10_complete_project.py` | 完整项目实战 | 配置管理、早停、学习率调度、混淆矩阵、推理函数 |

## 学习路径

### 第一阶段：基础（第1-3课）
理解张量和自动求导，掌握最基本的训练流程。

```bash
python 01_tensor_basics.py    # 张量操作
python 02_autograd.py          # 自动求导
python 03_linear_regression.py # 第一个训练流程
```

### 第二阶段：分类任务（第4-6课）
从二分类到多分类，从全连接网络到卷积网络。

```bash
python 04_logistic_regression.py  # 二分类
python 05_mlp.py                   # 神经网络 + MNIST
python 06_cnn.py                   # CNN + CIFAR-10
```

### 第三阶段：工程实践（第7-10课）
自定义数据、模型管理、迁移学习、完整项目。

```bash
python 07_dataset_dataloader.py   # 自定义数据集
python 08_save_load.py             # 模型保存加载
python 09_transfer_learning.py     # 迁移学习
python 10_complete_project.py      # 完整项目
```

## 核心概念速查

### 训练循环模板
```python
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, labels)

        # 反向传播 + 参数更新
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
```

### 常用层
| 层 | 用途 |
|----|------|
| `nn.Linear(in, out)` | 全连接层 |
| `nn.Conv2d(in, out, k, pad)` | 2D卷积 |
| `nn.MaxPool2d(k)` | 最大池化 |
| `nn.BatchNorm2d(n)` | 批归一化 |
| `nn.Dropout(p)` | Dropout |
| `nn.ReLU()` | ReLU激活 |

### 常用损失函数
| 损失函数 | 适用场景 |
|---------|---------|
| `nn.MSELoss()` | 回归 |
| `nn.BCELoss()` | 二分类（输出已过sigmoid） |
| `nn.BCEWithLogitsLoss()` | 二分类（输出未过sigmoid） |
| `nn.CrossEntropyLoss()` | 多分类 |

### 常用优化器
| 优化器 | 特点 |
|--------|------|
| `torch.optim.SGD` | 经典，泛化好，需调学习率 |
| `torch.optim.Adam` | 自适应学习率，收敛快，最常用 |
| `torch.optim.AdamW` | Adam + 权重衰减，效果更好 |

## 调试技巧

1. **打印形状**：在 forward 里打印 `x.shape`，确认维度变化
2. **检查梯度**：`param.grad` 查看梯度是否为 None 或全零
3. **小数据测试**：先用少量数据跑通整个流程
4. **过拟合小批量**：让模型过拟合一个小批量，验证模型能力
5. **可视化中间层**：用 hook 或直接打印特征图

## 推荐资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 中文文档](https://pytorch-cn.readthedocs.io/)
- 《深度学习》（花书）- 理论基础
- 《动手学深度学习》- 实战导向

## 下一步

学完这10课后，可以尝试：
1. **自己的数据集**：用第7课的方法加载你的数据
2. **更复杂的模型**：ResNet、ViT、U-Net 等
3. **其他任务**：目标检测、语义分割、生成模型、NLP
4. **读论文复现**：找一篇感兴趣的论文，复现其方法
