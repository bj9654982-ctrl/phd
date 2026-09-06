"""
第六课：卷积神经网络 CNN —— 图像识别的利器
==================================================
全连接网络 vs CNN：
  - 全连接：把图片拉平成向量，丢失了空间结构信息
  - CNN：用卷积核在图片上滑动，保留局部空间特征

CNN 核心组件：
  - Conv2D：卷积层，提取局部特征（边缘、纹理、图案）
  - MaxPooling2D：池化层，降维、扩大感受野
  - Flatten：把特征图拉平，接全连接层

对图像任务，CNN 通常比全连接网络准确率更高、参数更少。
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print("1. 加载并预处理 MNIST")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# CNN 需要 4D 输入: (batch, height, width, channels)
# MNIST 是灰度图，channels=1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]  # (60000, 28, 28) -> (60000, 28, 28, 1)
x_test = x_test[..., np.newaxis]

print(f"训练集 shape: {x_train.shape}")
print(f"测试集 shape: {x_test.shape}")

print()
print("=" * 60)
print("2. 搭建 CNN 模型")
print("=" * 60)

model = tf.keras.Sequential([
    # 第一个卷积块：32个3x3卷积核，ReLU激活
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),  # 28x28 -> 14x14

    # 第二个卷积块：64个卷积核
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),  # 14x14 -> 7x7

    # 第三个卷积块
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

    # 拉平 + 全连接分类头
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

print()
print("=" * 60)
print("3. 编译并训练")
print("=" * 60)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

print()
print("=" * 60)
print("4. 评估")
print("=" * 60)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"对比：全连接网络约 98%，CNN 通常能达到 99%+")

print()
print("=" * 60)
print("5. 可视化中间层输出（理解 CNN 在看什么）")
print("=" * 60)

# 构建一个能输出中间层的模型
layer_outputs = [layer.output for layer in model.layers[:6]]  # 前6层
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

# 拿一张测试图
img = x_test[0:1]
activations = activation_model.predict(img, verbose=0)

print(f"输入图片 shape: {img.shape}")
print(f"第1层(Conv2D_32)输出 shape: {activations[0].shape}  -> 32个特征图")
print(f"第2层(MaxPool)输出 shape: {activations[1].shape}")
print(f"第3层(Conv2D_64)输出 shape: {activations[2].shape}  -> 64个特征图")
print(f"第4层(MaxPool)输出 shape: {activations[3].shape}")
print(f"第5层(Conv2D_64)输出 shape: {activations[4].shape}")
print(f"第6层(Flatten)输出 shape: {activations[5].shape}")

print()
print("=" * 60)
print("6. 参数量对比")
print("=" * 60)

# 统计参数量
total_params = model.count_params()
conv_params = sum(layer.count_params() for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D))
dense_params = sum(layer.count_params() for layer in model.layers if isinstance(layer, tf.keras.layers.Dense))

print(f"总参数量: {total_params:,}")
print(f"卷积层参数: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
print(f"全连接层参数: {dense_params:,} ({dense_params/total_params*100:.1f}%)")
print(f"对比：上一课全连接网络约 100,000+ 参数")

print()
print("=" * 60)
print("第六课完成！CNN 是所有图像任务的基础，")
print("你的钢坯检测项目核心就是 CNN（或其变体）。")
print("=" * 60)
