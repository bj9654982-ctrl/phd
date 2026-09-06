"""
图像去噪训练 Demo v2（U-Net 结构）
====================================
U-Net 的核心：编码器和解码器之间加跳跃连接(skip connection)，
把低层细节直接传给解码器，解决梯度消失和细节丢失问题。
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
x_train_noisy = np.clip(x_train + noise_factor * np.random.normal(0, 1, x_train.shape), 0, 1)
x_test_noisy = np.clip(x_test + noise_factor * np.random.normal(0, 1, x_test.shape), 0, 1)
print("噪声强度:", noise_factor)

print("2. 构建 U-Net 去噪模型")
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    return x

input_img = tf.keras.Input(shape=(28, 28, 1))

# 编码器（下采样）
c1 = conv_block(input_img, 32)           # (28,28,32)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)  # (14,14,32)
c2 = conv_block(p1, 64)                  # (14,14,64)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)  # (7,7,64)

# 瓶颈
bottleneck = conv_block(p2, 128)         # (7,7,128)

# 解码器（上采样）+ 跳跃连接
u1 = tf.keras.layers.UpSampling2D((2,2))(bottleneck)  # (14,14,128)
concat1 = tf.keras.layers.Concatenate()([u1, c2])     # 跳跃连接！
c3 = conv_block(concat1, 64)               # (14,14,64)
u2 = tf.keras.layers.UpSampling2D((2,2))(c3)          # (28,28,64)
concat2 = tf.keras.layers.Concatenate()([u2, c1])     # 跳跃连接！
c4 = conv_block(concat2, 32)               # (28,28,32)

output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c4)

model = tf.keras.Model(input_img, output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
model.summary()

print("3. 训练")
history = model.fit(
    x_train_noisy, x_train,
    epochs=20, batch_size=256, shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=2
)

# 训练曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history['loss'], 'b-o', ms=4, label='训练损失', lw=2)
ax.plot(history.history['val_loss'], 'r-o', ms=4, label='验证损失', lw=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('U-Net 去噪模型训练曲线', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_training.png', dpi=150, bbox_inches='tight')
plt.close()

print("4. 评估 + 可视化")
decoded = model.predict(x_test_noisy, verbose=0)

def psnr(a, b):
    mse = np.mean((a-b)**2)
    return 20*np.log10(1.0/np.sqrt(mse)) if mse > 0 else 999

p_noisy = np.mean([psnr(x_test[i], x_test_noisy[i]) for i in range(500)])
p_denoise = np.mean([psnr(x_test[i], decoded[i]) for i in range(500)])
print("加噪 PSNR: %.2f dB" % p_noisy)
print("去噪 PSNR: %.2f dB" % p_denoise)
print("提升: %.2f dB" % (p_denoise - p_noisy))

# 三行对比
n = 10
fig, axes = plt.subplots(3, n, figsize=(18, 6))
for i in range(n):
    axes[0,i].imshow(x_test[i].squeeze(), cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(x_test_noisy[i].squeeze(), cmap='gray'); axes[1,i].axis('off')
    axes[2,i].imshow(decoded[i].squeeze(), cmap='gray'); axes[2,i].axis('off')
axes[0,0].set_ylabel('原图', fontsize=13, rotation=0, labelpad=40)
axes[1,0].set_ylabel('加噪图', fontsize=13, rotation=0, labelpad=40)
axes[2,0].set_ylabel('去噪图', fontsize=13, rotation=0, labelpad=40)
plt.suptitle('U-Net 图像去噪效果（加噪 %.1fdB → 去噪 %.1fdB，提升 %.1fdB）' %
             (p_noisy, p_denoise, p_denoise-p_noisy), fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_result.png', dpi=150, bbox_inches='tight')
plt.close()

# 加噪对比图
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
for i in range(10):
    axes[0,i].imshow(x_test[i].squeeze(), cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(x_test_noisy[i].squeeze(), cmap='gray'); axes[1,i].axis('off')
axes[0,0].set_ylabel('原图', fontsize=12, rotation=0, labelpad=35)
axes[1,0].set_ylabel('加噪图', fontsize=12, rotation=0, labelpad=35)
plt.suptitle('原图 vs 加噪图（噪声强度 %.1f）' % noise_factor, fontsize=14, fontweight='bold')
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

model.save(r'J:\phd\tensorflow_demo\denoiser_unet.keras')
print("完成！文件保存在 J:\\phd\\tensorflow_demo\\")
