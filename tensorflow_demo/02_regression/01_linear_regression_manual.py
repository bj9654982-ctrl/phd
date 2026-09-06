"""
第三课：线性回归 —— 用 TensorFlow 训练第一个模型
==================================================
前面学了：
  - 张量：数据载体
  - GradientTape：自动求导

现在把它们拼起来，训练一个线性回归模型：
  y = w * x + b

步骤：
  1. 造一批带噪声的假数据（真实 w=2, b=1）
  2. 定义可训练参数 w, b（初始化为随机值）
  3. 前向计算：y_pred = w*x + b
  4. 计算损失：loss = mean((y_pred - y_true)²)
  5. 反向传播：求 loss 对 w, b 的梯度
  6. 更新参数：w = w - lr * grad_w
  7. 重复 3-6 很多次
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print("1. 准备数据（真实关系 y = 2x + 1 + 噪声）")
print("=" * 60)

# 用 NumPy 造 100 个数据点
np.random.seed(42)
X_np = np.random.uniform(-10, 10, size=(100, 1)).astype(np.float32)
noise = np.random.normal(0, 1, size=(100, 1)).astype(np.float32)
y_np = 2.0 * X_np + 1.0 + noise  # 真实 w=2, b=1

# 转成 TF 张量
X = tf.constant(X_np)
y_true = tf.constant(y_np)

print(f"X shape: {X.shape}, y shape: {y_true.shape}")
print(f"前5个样本: x={X[:5].numpy().flatten()}, y={y_true[:5].numpy().flatten()}")

print()
print("=" * 60)
print("2. 定义可训练参数")
print("=" * 60)

# 随机初始化，模型要自己学到 w≈2, b≈1
w = tf.Variable(tf.random.normal([1], mean=0, stddev=1.0), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

print(f"初始 w = {w.numpy()}, b = {b.numpy()}")

print()
print("=" * 60)
print("3. 训练循环")
print("=" * 60)

learning_rate = 0.01
epochs = 200

for epoch in range(epochs):
    # --- 前向传播 + 计算损失 ---
    with tf.GradientTape() as tape:
        y_pred = w * X + b
        loss = tf.reduce_mean((y_pred - y_true) ** 2)  # MSE

    # --- 反向传播：求梯度 ---
    grads = tape.gradient(loss, [w, b])

    # --- 更新参数（梯度下降）---
    w.assign_sub(learning_rate * grads[0])
    b.assign_sub(learning_rate * grads[1])

    # 每 20 步打印一次
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | loss={loss.numpy():.4f} | w={w.numpy()[0]:.4f} | b={b.numpy()[0]:.4f}")

print()
print("=" * 60)
print("4. 训练结果")
print("=" * 60)
print(f"学到的 w = {w.numpy()[0]:.4f} (真实: 2.0)")
print(f"学到的 b = {b.numpy()[0]:.4f} (真实: 1.0)")
print(f"最终损失 = {loss.numpy():.4f}")

print()
print("=" * 60)
print("5. 用训练好的模型做预测")
print("=" * 60)
x_test = tf.constant([[0.0], [5.0], [-3.0]])
y_test_pred = w * x_test + b
for i in range(3):
    print(f"x={x_test[i].numpy()[0]:.1f} -> y_pred={y_test_pred[i].numpy()[0]:.4f} (真实≈{2*x_test[i].numpy()[0]+1:.1f})")

print()
print("=" * 60)
print("第三课完成！你已经手动实现了一个完整的训练循环。")
print("接下来我们会用 Keras 高层 API 简化这个过程。")
print("=" * 60)
