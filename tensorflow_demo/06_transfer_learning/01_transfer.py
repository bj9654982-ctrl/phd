"""
第八课：迁移学习 —— 站在巨人肩膀上
==================================================
什么是迁移学习？
  - 用别人在大数据集（如 ImageNet，1400万张图）上预训练好的模型
  - 冻结前面的特征提取层，只训练最后的分类头
  - 用少量数据就能达到很好的效果

为什么需要迁移学习？
  - 你的钢坯缺陷数据可能只有几百几千张
  - 从头训练 CNN 容易过拟合
  - 预训练模型已经学会了通用视觉特征（边缘、纹理、形状）

常用预训练模型：
  - MobileNetV2：轻量，适合移动端/边缘设备
  - ResNet50：经典，准确率高
  - EfficientNet：效率和准确率平衡
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print("1. 加载预训练模型（MobileNetV2，在 ImageNet 上训练）")
print("=" * 60)

# 加载预训练的 MobileNetV2，不包含顶部的全连接层
# include_top=False 表示只取特征提取部分
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),  # 输入图片大小
    include_top=False,         # 不要最后的分类层
    weights="imagenet"         # 用 ImageNet 预训练权重
)

# 冻结基础模型，不训练它的权重
base_model.trainable = False

print(f"基础模型层数: {len(base_model.layers)}")
print(f"基础模型参数量: {base_model.count_params():,}")
print(f"特征输出 shape: {base_model.output_shape}")  # (None, 3, 3, 1280)

print()
print("=" * 60)
print("2. 在预训练模型上加分类头")
print("=" * 60)

inputs = tf.keras.Input(shape=(96, 96, 3))

# 数据增强（训练时随机变换，增加数据多样性，防止过拟合）
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)

# 预处理（MobileNetV2 需要的输入范围是 [-1, 1]）
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# 特征提取（冻结，不训练）
x = base_model(x, training=False)

# 全局平均池化：把 (3, 3, 1280) 变成 (1280,)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Dropout 防止过拟合
x = tf.keras.layers.Dropout(0.2)(x)

# 分类头：假设有 10 个类别
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

print()
print("=" * 60)
print("3. 可训练参数对比")
print("=" * 60)

total = model.count_params()
trainable = sum(w.shape.num_elements() for w in model.trainable_weights)
non_trainable = total - trainable

print(f"总参数量: {total:,}")
print(f"可训练参数: {trainable:,} ({trainable/total*100:.2f}%)")
print(f"冻结参数: {non_trainable:,} ({non_trainable/total*100:.2f}%)")
print(f"只训练最后的分类头，参数很少，少量数据就能训练")

print()
print("=" * 60)
print("4. 模拟数据演示（用随机数据模拟你的钢坯数据集）")
print("=" * 60)

# 模拟：1000 张训练图，10 个类别
np.random.seed(42)
X_train = np.random.rand(1000, 96, 96, 3).astype(np.float32)
y_train = np.random.randint(0, 10, size=(1000,))
X_val = np.random.rand(200, 96, 96, 3).astype(np.float32)
y_val = np.random.randint(0, 10, size=(200,))

print(f"模拟训练集: {X_train.shape}, {y_train.shape}")
print(f"模拟验证集: {X_val.shape}, {y_val.shape}")

print()
print("=" * 60)
print("5. 第一阶段：只训练分类头（特征提取）")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("训练 3 个 epoch（只更新分类头）...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    batch_size=32,
    verbose=2
)

print()
print("=" * 60)
print("6. 第二阶段：微调（解冻部分层，用更小的学习率）")
print("=" * 60)

# 解冻基础模型的最后 20 层
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"解冻后可训练层数: {sum(l.trainable for l in base_model.layers)}/{len(base_model.layers)}")

# 用更小的学习率微调（避免破坏预训练权重）
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 小100倍
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("微调 2 个 epoch（学习率 1e-5）...")
history_fine = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=32,
    verbose=2
)

print()
print("=" * 60)
print("7. 可用的预训练模型列表（Keras 内置）")
print("=" * 60)

models_info = [
    ("MobileNetV2", "3.5M", "轻量，移动端首选"),
    ("ResNet50", "25.6M", "经典，残差连接"),
    ("ResNet50V2", "25.6M", "ResNet 改进版"),
    ("EfficientNetB0", "5.3M", "效率高，参数少"),
    ("VGG16", "138M", "经典但笨重"),
    ("InceptionV3", "23.9M", "GoogleNet 变体"),
    ("DenseNet121", "8.1M", "密集连接"),
]

print(f"{'模型':<20} {'参数量':<10} {'特点':<20}")
print("-" * 50)
for name, params, desc in models_info:
    print(f"{name:<20} {params:<10} {desc:<20}")

print()
print("=" * 60)
print("8. 对你的钢坯检测项目的建议")
print("=" * 60)

print("""
1. 数据量小（<1万张）：
   - 用 MobileNetV2 / EfficientNetB0 预训练模型
   - 先冻结训练分类头，再微调
   - 一定要加数据增强

2. 缺陷检测（分类/定位）：
   - 分类：迁移学习 + 自定义分类头
   - 检测：用 TF Model Garden 的 DETR / YOLO 变体
   - 分割：用 U-Net 结构（编码器可用预训练 backbone）

3. 部署到产线：
   - 训练完后导出为 SavedModel
   - 用 TFLite 转换为边缘设备模型
   - 量化（INT8）进一步加速

4. 学习率策略：
   - 特征提取阶段：1e-3
   - 微调阶段：1e-5 ~ 1e-4
""")

print()
print("=" * 60)
print("第八课完成！迁移学习是工业图像项目的标配，")
print("你的钢坯检测项目强烈建议从这里开始。")
print("=" * 60)
