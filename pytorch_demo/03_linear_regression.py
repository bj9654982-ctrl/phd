"""
第3课：线性回归 —— 第一个完整的训练流程
==========================================
这一课把前两课的知识串起来，实现一个完整的机器学习训练流程：

1. 准备数据
2. 定义模型（y = wx + b）
3. 定义损失函数（MSE）
4. 选择优化器（SGD）
5. 训练循环（前向 -> 算损失 -> 反向 -> 更新参数）
6. 评估结果

运行方式：
    conda activate torch-gpu
    python 03_linear_regression.py
"""

import torch
import matplotlib
matplotlib.use('Agg')  # 无界面环境用 Agg 后端
import matplotlib.pyplot as plt

print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 真实参数（我们假装不知道，让模型自己学出来）
w_true = 2.5
b_true = 1.0

# 生成模拟数据：y = 2.5x + 1.0 + 噪声
torch.manual_seed(42)  # 固定随机种子，结果可复现
x_train = torch.linspace(0, 10, 100).unsqueeze(1)  # [100, 1]
noise = torch.randn_like(x_train) * 1.0  # 高斯噪声
y_train = w_true * x_train + b_true + noise

print(f"训练数据: x.shape={x_train.shape}, y.shape={y_train.shape}")
print(f"前5个样本:")
for i in range(5):
    print(f"  x={x_train[i].item():.2f}, y={y_train[i].item():.2f}")

print("\n" + "=" * 60)
print("2. 定义模型")
print("=" * 60)

# 方式一：手动定义参数（学习用，理解底层）
class LinearRegressionManual:
    def __init__(self):
        # 可学习参数，随机初始化
        self.w = torch.randn(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return x * self.w + self.b

    def parameters(self):
        return [self.w, self.b]

# 方式二：用 nn.Module（PyTorch 标准写法，推荐）
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        # nn.Linear 内部自动创建 w 和 b
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 使用标准写法
model = LinearRegression()
print(f"模型结构:\n{model}")
print(f"\n初始参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data.squeeze().item():.4f}")

print("\n" + "=" * 60)
print("3. 定义损失函数和优化器")
print("=" * 60)

# 均方误差损失：MSE = mean((y_pred - y_true)^2)
criterion = nn.MSELoss()

# SGD 优化器：随机梯度下降
# lr = learning rate 学习率，控制每次更新的步长
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

print("\n" + "=" * 60)
print("4. 训练循环（核心！）")
print("=" * 60)

num_epochs = 500
loss_history = []
w_history = []
b_history = []

for epoch in range(num_epochs):
    # ---- 前向传播 ----
    y_pred = model(x_train)

    # ---- 计算损失 ----
    loss = criterion(y_pred, y_train)

    # ---- 反向传播 + 参数更新 ----
    optimizer.zero_grad()  # 1. 清零梯度（必须！否则会累加）
    loss.backward()        # 2. 反向传播，计算梯度
    optimizer.step()       # 3. 更新参数：w = w - lr * grad

    # 记录历史
    loss_history.append(loss.item())
    w_history.append(model.linear.weight.item())
    b_history.append(model.linear.bias.item())

    # 每 50 轮打印一次
    if (epoch + 1) % 50 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"loss={loss.item():.4f}  w={w:.4f}  b={b:.4f}")

print(f"\n训练完成！")
print(f"真实参数: w={w_true}, b={b_true}")
print(f"学到参数: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}")

print("\n" + "=" * 60)
print("5. 可视化训练过程")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 图1：损失曲线
axes[0].plot(loss_history)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (MSE)')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# 图2：参数收敛过程
axes[1].plot(w_history, label='w (learned)')
axes[1].plot(b_history, label='b (learned)')
axes[1].axhline(y=w_true, color='r', linestyle='--', label=f'w_true={w_true}')
axes[1].axhline(y=b_true, color='g', linestyle='--', label=f'b_true={b_true}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Value')
axes[1].set_title('Parameter Convergence')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3：拟合结果
axes[2].scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, s=20, label='Training data')
x_line = torch.linspace(0, 10, 100).unsqueeze(1)
with torch.no_grad():
    y_line = model(x_line)
axes[2].plot(x_line.numpy(), y_line.numpy(), 'r-', linewidth=2, label='Fitted line')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Linear Regression Fit')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_linear_regression_result.png', dpi=150)
print("训练过程图已保存为: 03_linear_regression_result.png")

print("\n" + "=" * 60)
print("6. 用训练好的模型做预测")
print("=" * 60)

model.eval()  # 切换到评估模式
with torch.no_grad():
    x_test = torch.tensor([[5.0], [7.5], [10.0]])
    y_pred = model(x_test)
    for i in range(len(x_test)):
        y_true_val = w_true * x_test[i].item() + b_true
        print(f"  x={x_test[i].item():.1f}  "
              f"预测={y_pred[i].item():.2f}  "
              f"真实={y_true_val:.2f}  "
              f"误差={abs(y_pred[i].item()-y_true_val):.4f}")

print("\n" + "=" * 60)
print("第3课完成！你已经掌握了完整的训练流程：")
print("  数据 -> 模型 -> 损失 -> 优化器 -> 训练循环 -> 评估")
print("这个流程适用于所有深度学习任务！")
print("下一课：逻辑回归（二分类问题）")
print("=" * 60)
