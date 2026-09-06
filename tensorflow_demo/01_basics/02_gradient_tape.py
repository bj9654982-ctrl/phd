"""
第二课：自动求导 GradientTape —— 深度学习的引擎
==================================================
为什么需要自动求导？
  训练神经网络 = 不断调整参数让损失变小
  怎么调？沿梯度反方向走（梯度下降）
  梯度 = 损失对参数的导数

手动求导太麻烦，TensorFlow 用 GradientTape 自动算。

核心思想：
  - 在 "tape" 上下文里做运算，TF 会记录每一步
  - 结束后用 tape.gradient(loss, variables) 一键求导
"""

import tensorflow as tf

print("=" * 60)
print("1. 最简单的例子：求 f(x) = x² 的导数")
print("=" * 60)
# f(x) = x²,  f'(x) = 2x
# x = 3.0 时，导数应该是 6.0

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2  # 前向计算，tape 记录

dy_dx = tape.gradient(y, x)  # 求 dy/dx
print(f"x = {x.numpy()}")
print(f"y = x² = {y.numpy()}")
print(f"dy/dx = 2x = {dy_dx.numpy()}  (期望: 6.0)")

print()
print("=" * 60)
print("2. 多变量：f(x, w, b) = w*x + b，对 w 和 b 求导")
print("=" * 60)
# 这就是线性模型 y = wx + b
# 对 w 求导 = x
# 对 b 求导 = 1

w = tf.Variable(2.0)
b = tf.Variable(1.0)
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    y = w * x + b

gradients = tape.gradient(y, [w, b])
print(f"y = w*x + b = {w.numpy()}*{x.numpy()} + {b.numpy()} = {y.numpy()}")
print(f"dy/dw = x = {gradients[0].numpy()}  (期望: 3.0)")
print(f"dy/db = 1 = {gradients[1].numpy()}  (期望: 1.0)")

print()
print("=" * 60)
print("3. 真实场景：计算损失对模型参数的梯度")
print("=" * 60)
# 假设我们有一个线性模型 y_pred = w*x + b
# 真实值 y_true = 5.0
# 损失 = (y_pred - y_true)²  (均方误差)

w = tf.Variable(1.0)
b = tf.Variable(0.0)
x = tf.constant(2.0)
y_true = tf.constant(5.0)

with tf.GradientTape() as tape:
    y_pred = w * x + b
    loss = (y_pred - y_true) ** 2

grads = tape.gradient(loss, [w, b])
print(f"当前: w={w.numpy()}, b={b.numpy()}")
print(f"预测: y_pred = {y_pred.numpy()}, 真实: y_true = {y_true.numpy()}")
print(f"损失: loss = {loss.numpy()}")
print(f"d(loss)/dw = {grads[0].numpy()}")
print(f"d(loss)/db = {grads[1].numpy()}")

# 用梯度下降更新参数（学习率 0.1）
lr = 0.1
w.assign_sub(lr * grads[0])  # w = w - lr * d(loss)/dw
b.assign_sub(lr * grads[1])
print(f"\n更新一步后: w={w.numpy():.4f}, b={b.numpy():.4f}")
print(f"新预测: y_pred = {w.numpy() * x.numpy() + b.numpy():.4f} (更接近5了)")

print()
print("=" * 60)
print("4. 高阶导数（嵌套 GradientTape）")
print("=" * 60)
# f(x) = x³, 一阶导=3x², 二阶导=6x
x = tf.Variable(2.0)

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 3
    dy_dx = tape1.gradient(y, x)
d2y_dx2 = tape2.gradient(dy_dx, x)

print(f"x = {x.numpy()}")
print(f"y = x³ = {y.numpy()}")
print(f"dy/dx = 3x² = {dy_dx.numpy()}  (期望: 12.0)")
print(f"d²y/dx² = 6x = {d2y_dx2.numpy()}  (期望: 12.0)")

print()
print("=" * 60)
print("5. 注意：默认只跟踪 Variable")
print("=" * 60)
# constant 默认不被跟踪，需要 watch

x = tf.constant(3.0)  # 不是 Variable
with tf.GradientTape() as tape:
    tape.watch(x)  # 手动告诉 tape 跟踪这个 constant
    y = x ** 2
dy_dx = tape.gradient(y, x)
print(f"对 constant 求导需要 watch: dy/dx = {dy_dx.numpy()}")

print()
print("=" * 60)
print("第二课完成！自动求导是训练所有神经网络的基础。")
print("=" * 60)
