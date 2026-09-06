"""
第七课：自定义训练循环 —— 像 PyTorch 一样完全掌控
==================================================
Keras 的 model.fit() 很方便，但有时候你需要完全控制训练过程：
  - 自定义损失函数（如对抗训练、多任务损失）
  - 梯度裁剪、学习率预热、混合精度
  - GAN 训练（生成器和判别器交替训练）
  - 自定义评估指标

这时候就需要用 GradientTape 手写训练循环，和 PyTorch 的写法非常像。

对比：
  PyTorch: optimizer.zero_grad() -> loss.backward() -> optimizer.step()
  TF:      with GradientTape() -> tape.gradient() -> optimizer.apply_gradients()
"""

import tensorflow as tf
import numpy as np
import time

print("=" * 60)
print("1. 准备数据（用 tf.data 构建高效数据管道）")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# tf.data.Dataset：TF 的数据管道，类似 PyTorch 的 DataLoader
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000).batch(128).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

print(f"训练批次数: {len(train_dataset)}")
print(f"测试批次数: {len(test_dataset)}")

print()
print("=" * 60)
print("2. 定义模型（用 Keras 子类化，和 PyTorch 的 nn.Module 几乎一样）")
print("=" * 60)

class CNNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10)  # 输出 logits，不加 softmax

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)  # 返回 logits

model = CNNModel()
# 先 build 一下，让模型知道输入 shape
model.build((None, 28, 28, 1))
model.summary()

print()
print("=" * 60)
print("3. 定义损失函数、优化器、评估指标")
print("=" * 60)

# 损失函数：稀疏分类交叉熵（from_logits=True 因为模型输出是 logits）
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 评估指标（Keras 的 Metric 对象，会自动累积平均）
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")

print("损失: SparseCategoricalCrossentropy(from_logits=True)")
print("优化器: Adam(lr=1e-3)")

print()
print("=" * 60)
print("4. 定义训练步（核心：GradientTape）")
print("=" * 60)

# @tf.function：把这个函数编译成静态图，加速运行（类似 torch.compile）
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # 前向传播，training=True 会启用 Dropout、BatchNorm 等训练行为
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

    # 反向传播：求 loss 对所有可训练参数的梯度
    gradients = tape.gradient(loss, model.trainable_variables)

    # 更新参数
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 更新指标
    train_loss(loss)
    train_accuracy(labels, predictions)

    return loss

print("train_step 定义完成")
print("  - GradientTape 记录前向计算")
print("  - tape.gradient 求梯度")
print("  - optimizer.apply_gradients 更新参数")

print()
print("=" * 60)
print("5. 定义测试步（不需要求导）")
print("=" * 60)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_fn(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

print("test_step 定义完成（不需要 GradientTape）")

print()
print("=" * 60)
print("6. 训练循环（完全手写，和 PyTorch 一模一样的结构）")
print("=" * 60)

EPOCHS = 5

for epoch in range(EPOCHS):
    start = time.time()

    # 重置指标
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    # 训练
    for images, labels in train_dataset:
        train_step(images, labels)

    # 测试
    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) "
          f"| 训练 loss: {train_loss.result():.4f}, acc: {train_accuracy.result()*100:.2f}% "
          f"| 测试 loss: {test_loss.result():.4f}, acc: {test_accuracy.result()*100:.2f}%")

print()
print("=" * 60)
print("7. 进阶：梯度裁剪（防止梯度爆炸）")
print("=" * 60)

@tf.function
def train_step_with_clip(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    # 梯度裁剪：把梯度的全局范数限制在 1.0 以内
    gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, grad_norm

# 演示一步
sample_img, sample_label = next(iter(train_dataset))
loss, grad_norm = train_step_with_clip(sample_img, sample_label)
print(f"梯度裁剪演示: loss={loss:.4f}, 梯度全局范数={grad_norm:.4f}")

print()
print("=" * 60)
print("8. 进阶：手动查看梯度")
print("=" * 60)

with tf.GradientTape() as tape:
    pred = model(sample_img[:1], training=False)
    loss = loss_fn(sample_label[:1], pred)

grads = tape.gradient(loss, model.trainable_variables)
for var, grad in zip(model.trainable_variables, grads):
    if grad is not None:
        print(f"  {var.name}: shape={var.shape}, 梯度均值={tf.reduce_mean(tf.abs(grad)):.6f}")

print()
print("=" * 60)
print("第七课完成！你已经掌握了 TF 的自定义训练循环，")
print("可以像 PyTorch 一样完全掌控训练过程。")
print("=" * 60)
