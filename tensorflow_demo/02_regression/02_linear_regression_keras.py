"""
第四课：Keras 高层 API —— 5 行代码搞定线性回归
==================================================
上一课我们手动写了训练循环（前向->求导->更新），
实际工作中用 Keras 高层 API，几行代码就搞定。

Keras 是 TensorFlow 的官方高层接口，核心三步：
  1. model = tf.keras.Sequential([...])  搭模型
  2. model.compile(optimizer, loss)       配置训练
  3. model.fit(X, y, epochs=...)          开练
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print("1. 准备数据（和上一课一样）")
print("=" * 60)

np.random.seed(42)
X = np.random.uniform(-10, 10, size=(100, 1)).astype(np.float32)
noise = np.random.normal(0, 1, size=(100, 1)).astype(np.float32)
y = 2.0 * X + 1.0 + noise

print(f"X shape: {X.shape}, y shape: {y.shape}")

print()
print("=" * 60)
print("2. 搭建模型（Keras Sequential）")
print("=" * 60)

# Dense 层 = 全连接层 = y = Wx + b
# units=1 表示输出1个值，input_shape=(1,) 表示输入1个特征
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,), name="dense")
])

# 看看模型结构
model.summary()

# 查看初始参数
w, b = model.layers[0].get_weights()
print(f"\n初始权重 w = {w.flatten()}, 偏置 b = {b.flatten()}")

print()
print("=" * 60)
print("3. 编译模型（指定优化器和损失函数）")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),  # 随机梯度下降
    loss="mse"  # 均方误差，等价于 tf.reduce_mean((y_pred-y_true)**2)
)
print("优化器: SGD(lr=0.01)")
print("损失函数: MSE (均方误差)")

print()
print("=" * 60)
print("4. 训练（fit 一行搞定所有循环）")
print("=" * 60)

history = model.fit(
    X, y,
    epochs=200,
    batch_size=32,  # 每次喂32个样本
    verbose=0       # 0=不打印进度, 1=进度条, 2=每epoch一行
)

# 打印最后几个 loss
print(f"初始 loss: {history.history['loss'][0]:.4f}")
print(f"最终 loss: {history.history['loss'][-1]:.4f}")

print()
print("=" * 60)
print("5. 训练结果")
print("=" * 60)

w, b = model.layers[0].get_weights()
print(f"学到的 w = {w.flatten()[0]:.4f} (真实: 2.0)")
print(f"学到的 b = {b.flatten()[0]:.4f} (真实: 1.0)")

print()
print("=" * 60)
print("6. 预测")
print("=" * 60)

x_test = np.array([[0.0], [5.0], [-3.0]], dtype=np.float32)
y_pred = model.predict(x_test, verbose=0)
for i in range(3):
    print(f"x={x_test[i][0]:.1f} -> y_pred={y_pred[i][0]:.4f} (真实≈{2*x_test[i][0]+1:.1f})")

print()
print("=" * 60)
print("7. 保存和加载模型")
print("=" * 60)

# 保存整个模型（结构+权重+优化器状态）
model.save("linear_model.keras")
print("模型已保存为 linear_model.keras")

# 加载模型
loaded_model = tf.keras.models.load_model("linear_model.keras")
y_loaded = loaded_model.predict(x_test, verbose=0)
print(f"加载后预测一致: {np.allclose(y_pred, y_loaded)}")

print()
print("=" * 60)
print("第四课完成！Keras 把训练循环封装好了，")
print("你只需要关心模型结构和超参数。")
print("=" * 60)
