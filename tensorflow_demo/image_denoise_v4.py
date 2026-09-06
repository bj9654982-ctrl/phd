"""
图像去噪 v4：用 Binary Crossentropy 损失
MSE 会让模型输出灰色均值，BCE 更适合 [0,1] 的像素值
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("1. 加载数据 + 加噪")
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

noise_factor = 0.3
np.random.seed(42)
x_train_noisy = np.clip(x_train + noise_factor * np.random.normal(0, 1, x_train.shape), 0, 1).astype('float32')
x_test_noisy = np.clip(x_test + noise_factor * np.random.normal(0, 1, x_test.shape), 0, 1).astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((x_train_noisy, x_train)).shuffle(60000).batch(256).prefetch(tf.data.AUTOTUNE)

print("2. 构建模型")
inputs = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D()(x)
outputs = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
model = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.BinaryCrossentropy()  # ← 关键：用BCE不用MSE

print("3. 训练 15 epoch")
loss_history = []
for epoch in range(15):
    epoch_loss = tf.keras.metrics.Mean()
    for batch_x, batch_y in train_ds:
        with tf.GradientTape() as tape:
            pred = model(batch_x, training=True)
            loss = loss_fn(batch_y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss(loss)
    avg = epoch_loss.result().numpy()
    loss_history.append(avg)
    print("Epoch %2d: BCE_loss=%.6f" % (epoch+1, avg))

print("4. 评估")
decoded = model.predict(x_test_noisy[:1000], verbose=0)
def psnr(a, b):
    mse = np.mean((a-b)**2)
    return 20*np.log10(1.0/np.sqrt(mse)) if mse > 0 else 999
p_noisy = np.mean([psnr(x_test[i], x_test_noisy[i]) for i in range(500)])
p_denoise = np.mean([psnr(x_test[i], decoded[i]) for i in range(500)])
print("加噪 PSNR: %.2f dB" % p_noisy)
print("去噪 PSNR: %.2f dB" % p_denoise)
print("提升: %.2f dB" % (p_denoise - p_noisy))

print("5. 生成图表")
# 去噪对比
n = 10
fig, axes = plt.subplots(3, n, figsize=(18, 6))
for i in range(n):
    axes[0,i].imshow(x_test[i].squeeze(), cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(x_test_noisy[i].squeeze(), cmap='gray'); axes[1,i].axis('off')
    axes[2,i].imshow(decoded[i].squeeze(), cmap='gray'); axes[2,i].axis('off')
axes[0,0].set_ylabel('原图', fontsize=13, rotation=0, labelpad=40)
axes[1,0].set_ylabel('加噪图', fontsize=13, rotation=0, labelpad=40)
axes[2,0].set_ylabel('去噪图', fontsize=13, rotation=0, labelpad=40)
plt.suptitle('图像去噪效果（加噪 %.1fdB → 去噪 %.1fdB，提升 %.1fdB）' %
             (p_noisy, p_denoise, p_denoise-p_noisy), fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_result.png', dpi=150, bbox_inches='tight')
plt.close()

# 训练曲线
plt.figure(figsize=(8,5))
plt.plot(range(1,16), loss_history, 'b-o', ms=5, lw=2)
plt.xlabel('Epoch'); plt.ylabel('BCE Loss')
plt.title('去噪模型训练曲线（Binary Crossentropy）', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_training.png', dpi=150, bbox_inches='tight')
plt.close()

# 加噪对比
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
for i in range(10):
    axes[0,i].imshow(x_test[i].squeeze(), cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(x_test_noisy[i].squeeze(), cmap='gray'); axes[1,i].axis('off')
axes[0,0].set_ylabel('原图', fontsize=12, rotation=0, labelpad=35)
axes[1,0].set_ylabel('加噪图', fontsize=12, rotation=0, labelpad=35)
plt.suptitle('原图 vs 加噪图（噪声 %.1f）' % noise_factor, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_input.png', dpi=150, bbox_inches='tight')
plt.close()

# 残差分析
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
idx = 3
axes[0].imshow(x_test[idx].squeeze(), cmap='gray'); axes[0].set_title('原图'); axes[0].axis('off')
axes[1].imshow(x_test_noisy[idx].squeeze(), cmap='gray'); axes[1].set_title('加噪图'); axes[1].axis('off')
axes[2].imshow(decoded[idx].squeeze(), cmap='gray'); axes[2].set_title('去噪图'); axes[2].axis('off')
err = np.abs(x_test[idx].squeeze() - decoded[idx].squeeze())
im = axes[3].imshow(err, cmap='hot'); axes[3].set_title('残差(越暗越好)'); axes[3].axis('off')
plt.colorbar(im, ax=axes[3], shrink=0.8)
plt.suptitle('单张图片去噪细节与残差', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_residual.png', dpi=150, bbox_inches='tight')
plt.close()

model.save(r'J:\phd\tensorflow_demo\denoiser_final.keras')
print("完成！")
