"""
第五课：全连接神经网络 —— MNIST 手写数字识别
==================================================
从回归到分类：
  - 回归：预测一个连续值（如房价）
  - 分类：预测一个类别（如数字0-9）

MNIST 数据集：
  - 70000 张 28x28 灰度手写数字图片
  - 60000 训练 + 10000 测试
  - 每张图标注了 0-9 的数字

网络结构：
  输入(784) -> 隐藏层(128, ReLU) -> 隐藏层(64, ReLU) -> 输出(10, Softmax)
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print("1. 加载 MNIST 数据集")
print("=" * 60)

# Keras 内置数据集，第一次会自动下载
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"训练集: x={x_train.shape}, y={y_train.shape}")
print(f"测试集: x={x_test.shape}, y={y_test.shape}")
print(f"像素值范围: [{x_train.min()}, {x_train.max()}]")
print(f"标签示例: {y_train[:10]}")

# 数据预处理：归一化到 [0,1]，展平 28x28 -> 784
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"\n归一化后范围: [{x_train.min()}, {x_train.max()}]")

print()
print("=" * 60)
print("2. 搭建全连接神经网络")
print("=" * 60)

model = tf.keras.Sequential([
    # 输入层：把 28x28 的图片展平成 784 维向量
    tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),

    # 第一个隐藏层：128个神经元，ReLU激活
    tf.keras.layers.Dense(128, activation="relu", name="hidden1"),

    # Dropout：训练时随机丢弃20%神经元，防止过拟合
    tf.keras.layers.Dropout(0.2, name="dropout1"),

    # 第二个隐藏层：64个神经元
    tf.keras.layers.Dense(64, activation="relu", name="hidden2"),

    # 输出层：10个类别，softmax 转成概率分布
    tf.keras.layers.Dense(10, activation="softmax", name="output")
])

model.summary()

print()
print("=" * 60)
print("3. 编译模型")
print("=" * 60)

model.compile(
    optimizer="adam",  # Adam 是最常用的优化器，自适应学习率
    loss="sparse_categorical_crossentropy",  # 整数标签用这个
    metrics=["accuracy"]  # 同时监控准确率
)
print("优化器: Adam")
print("损失: sparse_categorical_crossentropy (标签是整数0-9)")
print("评估: accuracy (准确率)")

print()
print("=" * 60)
print("4. 训练")
print("=" * 60)

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,  # 拿10%训练数据做验证
    verbose=2
)

print()
print("=" * 60)
print("5. 在测试集上评估")
print("=" * 60)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

print()
print("=" * 60)
print("6. 预测示例")
print("=" * 60)

# 拿前5张测试图预测
predictions = model.predict(x_test[:5], verbose=0)
for i in range(5):
    predicted_digit = np.argmax(predictions[i])
    confidence = predictions[i][predicted_digit]
    true_digit = y_test[i]
    status = "✓" if predicted_digit == true_digit else "✗"
    print(f"图片{i}: 预测={predicted_digit} (置信度{confidence:.2%}) 真实={true_digit} {status}")

print()
print("=" * 60)
print("第五课完成！你已经训练了一个识别手写数字的神经网络。")
print("这个网络能达到 ~97-98% 的准确率。")
print("=" * 60)
