"""
图像去噪训练 Demo（MNIST 灰度图版）
====================================
去噪自编码器 (Denoising Autoencoder)：
输入 = 带噪声的图片 → 编码器压缩 → 解码器重建 → 输出 = 干净图片
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("1. 加载 MNIST 数据")
print("=" * 60)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., np.newaxis]  # (60000,28,28) -> (60000,28,28,1)
x_test = x_test[..., np.newaxis]
print("训练集:", x_train.shape)
print("测试集:", x_test.shape)

print()
print("=" * 60)
print("2. 加高斯噪声")
print("=" * 60)
noise_factor = 0.5  # 噪声强度（大一点效果明显）

def add_noise(images):
    noisy = images + noise_factor * np.random.normal(0, 1, size=images.shape)
    return np.clip(noisy, 0., 1.)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)
print("噪声强度 sigma =", noise_factor)

# 展示加噪效果
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
for i in range(10):
    axes[0, i].imshow(x_test[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(x_test_noisy[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('原图', fontsize=12, rotation=0, labelpad=35)
axes[1, 0].set_ylabel('加噪图', fontsize=12, rotation=0, labelpad=35)
plt.suptitle('原图 vs 加噪图（噪声强度 %.1f）' % noise_factor, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_input.png', dpi=150, bbox_inches='tight')
plt.close()
print("加噪对比图已保存")

print()
print("=" * 60)
print("3. 构建去噪自编码器")
print("=" * 60)

input_img = tf.keras.Input(shape=(28, 28, 1))

# 编码器
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)   # 28->14
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)   # 14->7
encoded = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)

# 解码器
x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2,2))(x)   # 7->14
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2,2))(x)   # 14->28
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
decoded = tf.keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
autoencoder.summary()

print()
print("=" * 60)
print("4. 训练（输入=加噪图，标签=干净图）")
print("=" * 60)

history = autoencoder.fit(
    x_train_noisy, x_train,
    epochs=15,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=2
)

# 训练曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history['loss'], 'b-o', ms=4, label='训练损失', lw=2)
ax.plot(history.history['val_loss'], 'r-o', ms=4, label='验证损失', lw=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('去噪自编码器训练曲线', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("训练曲线已保存")

print()
print("=" * 60)
print("5. 去噪效果对比")
print("=" * 60)

decoded_imgs = autoencoder.predict(x_test_noisy, verbose=0)

# PSNR
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

psnr_noisy = np.mean([psnr(x_test[i], x_test_noisy[i]) for i in range(500)])
psnr_denoised = np.mean([psnr(x_test[i], decoded_imgs[i]) for i in range(500)])
print("加噪图 PSNR: %.2f dB" % psnr_noisy)
print("去噪图 PSNR: %.2f dB" % psnr_denoised)
print("提升: %.2f dB" % (psnr_denoised - psnr_noisy))

# 三行对比
n = 10
fig, axes = plt.subplots(3, n, figsize=(18, 6))
for i in range(n):
    axes[0, i].imshow(x_test[i].squeeze(), cmap='gray'); axes[0, i].axis('off')
    axes[1, i].imshow(x_test_noisy[i].squeeze(), cmap='gray'); axes[1, i].axis('off')
    axes[2, i].imshow(decoded_imgs[i].squeeze(), cmap='gray'); axes[2, i].axis('off')
axes[0, 0].set_ylabel('原图', fontsize=13, rotation=0, labelpad=40)
axes[1, 0].set_ylabel('加噪图', fontsize=13, rotation=0, labelpad=40)
axes[2, 0].set_ylabel('去噪图', fontsize=13, rotation=0, labelpad=40)
plt.suptitle('图像去噪效果（加噪 %.1fdB → 去噪 %.1fdB，提升 %.1fdB）' %
             (psnr_noisy, psnr_denoised, psnr_denoised-psnr_noisy),
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("去噪对比图已保存")

# 逐像素误差图
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
idx = 0
axes[0].imshow(x_test[idx].squeeze(), cmap='gray'); axes[0].set_title('原图'); axes[0].axis('off')
axes[1].imshow(x_test_noisy[idx].squeeze(), cmap='gray'); axes[1].set_title('加噪图'); axes[1].axis('off')
axes[2].imshow(decoded_imgs[idx].squeeze(), cmap='gray'); axes[2].set_title('去噪图'); axes[2].axis('off')
error = np.abs(x_test[idx].squeeze() - decoded_imgs[idx].squeeze())
im = axes[3].imshow(error, cmap='hot'); axes[3].set_title('残差(原图-去噪)'); axes[3].axis('off')
plt.colorbar(im, ax=axes[3], shrink=0.8)
plt.suptitle('单张图片去噪细节与残差分析', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_residual.png', dpi=150, bbox_inches='tight')
plt.close()
print("残差分析图已保存")

print()
print("=" * 60)
print("全部完成！生成文件：")
print("  denoise_input.png     原图vs加噪图")
print("  denoise_training.png  训练曲线")
print("  denoise_result.png    去噪效果对比(10组)")
print("  denoise_residual.png  残差分析")
print("  denoiser_mnist.keras  训练好的模型")
print("=" * 60)

autoencoder.save(r'J:\phd\tensorflow_demo\denoiser_mnist.keras')
