"""
第4课：逻辑回归 —— 二分类问题
================================
线性回归预测的是连续值（如房价），逻辑回归预测的是类别（是/否）。

核心改动：
1. 模型输出加 sigmoid 激活函数，把值压缩到 [0,1]，表示概率
2. 损失函数从 MSE 换成 BCE（二元交叉熵）
3. 评估指标从 MSE 换成准确率（Accuracy）

运行方式：
    conda activate torch-gpu
    python 04_logistic_regression.py
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("1. 生成二分类数据")
print("=" * 60)

torch.manual_seed(42)

# 生成两类数据，每类 100 个样本，每个样本 2 个特征
# 类别0：中心在 (2, 2)
# 类别1：中心在 (-2, -2)
n_samples = 100

# 类别0
x0 = torch.randn(n_samples, 2) + torch.tensor([2.0, 2.0])
y0 = torch.zeros(n_samples, 1)

# 类别1
x1 = torch.randn(n_samples, 2) + torch.tensor([-2.0, -2.0])
y1 = torch.ones(n_samples, 1)

# 合并
x_train = torch.cat([x0, x1], dim=0)  # [200, 2]
y_train = torch.cat([y0, y1], dim=0)  # [200, 1]

# 打乱顺序
perm = torch.randperm(200)
x_train = x_train[perm]
y_train = y_train[perm]

print(f"数据形状: x={x_train.shape}, y={y_train.shape}")
print(f"类别0数量: {(y_train==0).sum().item()}, 类别1数量: {(y_train==1).sum().item()}")

print("\n" + "=" * 60)
print("2. 定义逻辑回归模型")
print("=" * 60)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # 输出1个值

    def forward(self, x):
        logits = self.linear(x)       # 线性输出，范围 (-inf, +inf)
        prob = torch.sigmoid(logits)   # sigmoid 压缩到 (0, 1)
        return prob

model = LogisticRegression(input_dim=2)
print(f"模型:\n{model}")

print("\n" + "=" * 60)
print("3. 损失函数与优化器")
print("=" * 60)

# BCE Loss: Binary Cross Entropy
# 公式: loss = -[y*log(p) + (1-y)*log(1-p)]
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print(f"损失函数: {criterion}")
print(f"优化器: SGD(lr=0.1)")

print("\n" + "=" * 60)
print("4. 训练循环")
print("=" * 60)

num_epochs = 200
loss_history = []
acc_history = []

for epoch in range(num_epochs):
    # 前向
    y_prob = model(x_train)

    # 损失
    loss = criterion(y_prob, y_train)

    # 反向 + 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算准确率：概率 > 0.5 判为类别1
    y_pred = (y_prob > 0.5).float()
    accuracy = (y_pred == y_train).float().mean()

    loss_history.append(loss.item())
    acc_history.append(accuracy.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"loss={loss.item():.4f}  acc={accuracy.item()*100:.1f}%")

print(f"\n最终准确率: {acc_history[-1]*100:.1f}%")

print("\n" + "=" * 60)
print("5. 可视化决策边界")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 图1：训练数据
class0 = x_train[y_train.squeeze() == 0]
class1 = x_train[y_train.squeeze() == 1]
axes[0].scatter(class0[:, 0], class0[:, 1], c='blue', alpha=0.6, label='Class 0')
axes[0].scatter(class1[:, 0], class1[:, 1], c='red', alpha=0.6, label='Class 1')
axes[0].set_title('Training Data')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：决策边界
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    probs = model(grid).numpy().reshape(xx.shape)

axes[1].contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
axes[1].contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
axes[1].scatter(class0[:, 0], class0[:, 1], c='blue', alpha=0.6, s=20)
axes[1].scatter(class1[:, 0], class1[:, 1], c='red', alpha=0.6, s=20)
axes[1].set_title('Decision Boundary')
axes[1].grid(True, alpha=0.3)

# 图3：训练曲线
axes[2].plot(loss_history, label='Loss')
axes[2].plot(acc_history, label='Accuracy')
axes[2].set_xlabel('Epoch')
axes[2].set_title('Training History')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_logistic_regression_result.png', dpi=150)
print("结果图已保存为: 04_logistic_regression_result.png")

print("\n" + "=" * 60)
print("6. 理解 sigmoid 和 BCE")
print("=" * 60)
print("""
sigmoid(x) = 1 / (1 + e^(-x))
  - 输入任意实数，输出 (0, 1)
  - 可以解释为"正类的概率"

BCE 损失 = -[y*log(p) + (1-y)*log(1-p)]
  - y=1 时：loss = -log(p)，p 越接近1损失越小
  - y=0 时：loss = -log(1-p)，p 越接近0损失越小
  - 预测越自信且正确，损失越小；预测错误会有很大惩罚
""")

# 演示 sigmoid
x_demo = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
print(f"sigmoid 演示:")
for x in x_demo:
    print(f"  sigmoid({x.item():.1f}) = {torch.sigmoid(x).item():.4f}")

print("\n" + "=" * 60)
print("第4课完成！逻辑回归是分类任务的基础。")
print("下一课：多层感知机（MLP）—— 第一个真正的神经网络")
print("=" * 60)
