"""
第2课：自动求导（Autograd）—— 深度学习的基石
================================================
神经网络训练的核心是"反向传播"，需要计算损失对每个参数的梯度。
PyTorch 的 autograd 会自动追踪计算图并计算梯度，你不需要手动求导。

关键概念：
- requires_grad=True：告诉 PyTorch 追踪这个张量的计算
- backward()：反向传播，自动计算梯度
- .grad：保存计算出的梯度
- with torch.no_grad()：推理时禁用梯度计算，省内存

运行方式：
    conda activate torch-gpu
    python 02_autograd.py
"""

import torch

print("=" * 60)
print("1. 最简单的例子：y = x^2，求 dy/dx")
print("=" * 60)

# 创建一个需要追踪梯度的张量
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}, requires_grad={x.requires_grad}")

# 前向计算：y = x^2
y = x ** 2
print(f"y = x^2 = {y}")

# 反向传播：计算 dy/dx
y.backward()
print(f"dy/dx = 2x = {x.grad}")  # 应该是 6.0

print("\n" + "=" * 60)
print("2. 多变量复合函数：z = (x + y)^2，求 dz/dx 和 dz/dy")
print("=" * 60)

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = (x + y) ** 2
print(f"z = (x+y)^2 = ({x.item()}+{y.item()})^2 = {z.item()}")

z.backward()
print(f"dz/dx = 2(x+y) = {x.grad}")   # 2*(2+3)=10
print(f"dz/dy = 2(x+y) = {y.grad}")   # 2*(2+3)=10

print("\n" + "=" * 60)
print("3. 向量/矩阵的梯度（必须指定 grad_tensors）")
print("=" * 60)

# backward() 默认只能对标量（0维）调用
# 如果输出是向量，需要传入梯度参数
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y = [1, 4, 9]，是向量

# 方法1：先求和变成标量
y.sum().backward()
print(f"y=x^2, sum后反向传播, x.grad={x.grad}")  # [2, 4, 6]

# 方法2：传入 grad_tensors（等价于对每个元素加权求和）
x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y2 = x2 ** 2
y2.backward(torch.tensor([1.0, 1.0, 1.0]))
print(f"传入全1梯度, x2.grad={x2.grad}")  # 同样 [2, 4, 6]

print("\n" + "=" * 60)
print("4. 线性回归的梯度计算（手动实现）")
print("=" * 60)
print("模型: y = w*x + b")
print("损失: MSE = (y_pred - y_true)^2")
print("目标: 求 d(loss)/dw 和 d(loss)/db")

# 真实参数
w_true = 2.0
b_true = 1.0

# 模拟数据
x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = w_true * x_data + b_true  # [3, 5, 7]

# 可学习参数（随机初始化）
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 前向传播
y_pred = w * x_data + b
print(f"\n预测值: {y_pred}")
print(f"真实值: {y_data}")

# 计算 MSE 损失
loss = ((y_pred - y_data) ** 2).mean()
print(f"损失: {loss.item():.4f}")

# 反向传播
loss.backward()
print(f"\nd(loss)/dw = {w.grad.item():.4f}")
print(f"d(loss)/db = {b.grad.item():.4f}")

# 验证：手动计算梯度
# loss = mean((w*x+b - y)^2)
# d(loss)/dw = mean(2*(w*x+b-y)*x)
# d(loss)/db = mean(2*(w*x+b-y))
manual_dw = (2 * (y_pred - y_data) * x_data).mean()
manual_db = (2 * (y_pred - y_data)).mean()
print(f"\n手动验证 d(loss)/dw = {manual_dw.item():.4f}")
print(f"手动验证 d(loss)/db = {manual_db.item():.4f}")

print("\n" + "=" * 60)
print("5. 梯度清零（重要！）")
print("=" * 60)

# PyTorch 默认会累积梯度，所以每次反向传播前要清零
w = torch.tensor(3.0, requires_grad=True)

# 第一次
y1 = w ** 2
y1.backward()
print(f"第一次后 w.grad = {w.grad}")  # 6

# 第二次（不清零的话会累加）
y2 = w ** 3
y2.backward()
print(f"第二次后 w.grad = {w.grad}")  # 6 + 27 = 33，累加了！

# 正确做法：清零后再计算
w.grad.zero_()  # 原地清零
y3 = w ** 2
y3.backward()
print(f"清零后第三次 w.grad = {w.grad}")  # 6

print("\n" + "=" * 60)
print("6. 禁用梯度计算（推理/评估时用）")
print("=" * 60)

x = torch.tensor(2.0, requires_grad=True)

# 正常模式：会追踪计算图
with torch.no_grad():
    y = x ** 2
    print(f"no_grad 模式下 y.requires_grad = {y.requires_grad}")
    # 这里不能调用 y.backward()，因为没有追踪计算图

# 另一种方式：detach()
y2 = x ** 2
y2_detached = y2.detach()
print(f"detach 后 requires_grad = {y2_detached.requires_grad}")

print("\n" + "=" * 60)
print("7. 计算图可视化理解")
print("=" * 60)
print("""
前向传播构建计算图：

    x (requires_grad=True)
    |
    v
  x * x  (乘法节点)
    |
    v
    y = x^2
    |
    v
  backward() 反向传播
    |
    v
  x.grad = dy/dx = 2x

每次 forward 构建新图，backward 后释放图。
如果需要多次 backward，设置 retain_graph=True。
""")

# 演示 retain_graph
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)  # 保留计算图
print(f"第一次 backward, x.grad={x.grad}")
x.grad.zero_()
y.backward()  # 再次反向传播
print(f"第二次 backward, x.grad={x.grad}")

print("\n" + "=" * 60)
print("第2课完成！自动求导是神经网络训练的核心。")
print("下一课：用自动求导实现完整的线性回归训练")
print("=" * 60)
