"""
第一课：TensorFlow 核心概念 —— 张量 (Tensor)
============================================
什么是张量？
  - 0维张量 = 标量 (scalar)，比如一个数字 3.14
  - 1维张量 = 向量 (vector)，比如 [1, 2, 3]
  - 2维张量 = 矩阵 (matrix)，比如 [[1,2],[3,4]]
  - 3维及以上 = 高维张量，比如一批图片 (batch, height, width, channel)

和 NumPy 的 ndarray 几乎一样，区别是：
  - tf.Tensor 可以在 GPU/TPU 上运行
  - tf.Tensor 支持自动求导 (GradientTape)
  - tf.Tensor 不可变 (immutable)，要改值得用 tf.Variable
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print("1. 创建张量")
print("=" * 60)

# 标量 (0维)
scalar = tf.constant(3.14)
print(f"标量: {scalar}, shape={scalar.shape}, dtype={scalar.dtype}")

# 向量 (1维)
vector = tf.constant([1, 2, 3, 4])
print(f"向量: {vector}, shape={vector.shape}")

# 矩阵 (2维)
matrix = tf.constant([[1, 2], [3, 4]])
print(f"矩阵:\n{matrix}")

# 3维张量 (比如单张灰度图: height, width, channel)
img = tf.zeros((28, 28, 1))
print(f"模拟图片张量 shape: {img.shape}")

# 4维张量 (一批图片: batch, height, width, channel)
batch = tf.ones((32, 28, 28, 3))
print(f"一批图片 shape: {batch.shape}  (32张28x28彩色图)")

print()
print("=" * 60)
print("2. 常用创建方式")
print("=" * 60)

print(f"全0: {tf.zeros((2,3))}")
print(f"全1: {tf.ones((2,3))}")
print(f"随机正态: {tf.random.normal((2,2), mean=0, stddev=1)}")
print(f"随机均匀: {tf.random.uniform((2,2), minval=0, maxval=1)}")
print(f"序列: {tf.range(1, 10, 2)}")

print()
print("=" * 60)
print("3. 张量运算 (和 NumPy 几乎一样)")
print("=" * 60)

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print(f"a + b =\n{a + b}")
print(f"a * b (逐元素) =\n{a * b}")
print(f"a @ b (矩阵乘法) =\n{a @ b}")
print(f"tf.reduce_sum(a) = {tf.reduce_sum(a)}")
print(f"tf.reduce_mean(a) = {tf.reduce_mean(a)}")
print(f"tf.transpose(a) =\n{tf.transpose(a)}")

print()
print("=" * 60)
print("4. 和 NumPy 互转")
print("=" * 60)

np_arr = np.array([[1, 2], [3, 4]])
tf_tensor = tf.convert_to_tensor(np_arr)
back_to_np = tf_tensor.numpy()

print(f"NumPy -> TF: {type(tf_tensor)}")
print(f"TF -> NumPy: {type(back_to_np)}")
print(f"值相等: {np.array_equal(np_arr, back_to_np)}")

print()
print("=" * 60)
print("5. Variable (可变张量，模型参数用这个)")
print("=" * 60)

# 模型的权重都是 Variable，可以在训练中更新
w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
print(f"初始 w:\n{w}")
w.assign(w * 2)  # 原地修改
print(f"w.assign(w*2) 后:\n{w}")
print(f"w + 1 (返回新Tensor，不改变w):\n{w + 1}")
print(f"此时 w 没变:\n{w}")

print()
print("=" * 60)
print("第一课完成！你已经掌握了 TensorFlow 的数据基础。")
print("=" * 60)
