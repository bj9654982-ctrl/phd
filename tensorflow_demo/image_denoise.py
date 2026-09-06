"""
图像去噪训练 Demo
==================
用 CIFAR-10 数据集训练一个去噪自编码器 (Denoising Autoencoder)
模型学习：输入带噪声的图片 → 输出干净的图片
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("1. 加载 CIFAR-10 数据")
print("=" * 60)
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print("训练集:", x_train.shape)
print("测试集:", x_test.shape)

print()
print("=" * 60)
print("2. 给图片加高斯噪声")
print("=" * 60)
noise_factor = 0.15  # 噪声强度

def add_noise(images):
    """给图片加高斯噪声，并裁剪到[0,1]"""
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0., 1.)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)
print("噪声强度: sigma =", noise_factor)

# 展示加噪效果
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i in range(8):
    axes[0, i].imshow(x_test[i])
    axes[0, i].set_title('原图', fontsize=10)
    axes[0, i].axis('off')
    axes[1, i].imshow(x_test_noisy[i])
    axes[1, i].set_title('加噪图', fontsize=10)
    axes[1, i].axis('off')
plt.suptitle('原图 vs 加噪图（噪声强度 %.2f）' % noise_factor, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_input.png', dpi=150, bbox_inches='tight')
plt.close()
print("加噪对比图已保存: denoise_input.png")

print()
print("=" * 60)
print("3. 构建去噪自编码器模型")
print("=" * 60)

def build_denoiser():
    input_img = tf.keras.Input(shape=(32, 32, 3))

    # ===== 编码器：压缩特征 =====
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)   # 32->16
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)   # 16->8
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 8->4 (瓶颈)

    # ===== 解码器：重建图像 =====
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)   # 4->8
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)   # 8->16
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)   # 16->32
    decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.Model(input_img, decoded)
    return autoencoder

model = build_denoiser()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

print()
print("=" * 60)
print("4. 训练（输入=带噪图，标签=干净图）")
print("=" * 60)

history = model.fit(
    x_train_noisy, x_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=2
)

# 训练曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history['loss'], 'b-', label='训练损失', linewidth=2)
ax.plot(history.history['val_loss'], 'r-', label='验证损失', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('去噪自编码器训练曲线', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("训练曲线已保存: denoise_training.png")

print()
print("=" * 60)
print("5. 去噪效果对比")
print("=" * 60)

# 用训练好的模型去噪
decoded_imgs = model.predict(x_test_noisy, verbose=0)

# 计算 PSNR（峰值信噪比，越高越好）
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

psnr_noisy = np.mean([psnr(x_test[i], x_test_noisy[i]) for i in range(100)])
psnr_denoised = np.mean([psnr(x_test[i], decoded_imgs[i]) for i in range(100)])
print("加噪图平均 PSNR: %.2f dB" % psnr_noisy)
print("去噪图平均 PSNR: %.2f dB" % psnr_denoised)
print("PSNR 提升: %.2f dB" % (psnr_denoised - psnr_noisy))

# 展示 3 行对比：原图 / 加噪 / 去噪
n = 10
fig, axes = plt.subplots(3, n, figsize=(20, 6.5))
for i in range(n):
    # 原图
    axes[0, i].imshow(x_test[i])
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel('原图', fontsize=13, rotation=0, labelpad=40)
    # 加噪
    axes[1, i].imshow(x_test_noisy[i])
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_ylabel('加噪图', fontsize=13, rotation=0, labelpad=40)
    # 去噪
    axes[2, i].imshow(decoded_imgs[i])
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_ylabel('去噪图', fontsize=13, rotation=0, labelpad=40)

plt.suptitle('图像去噪效果对比（加噪 PSNR=%.1fdB → 去噪 PSNR=%.1fdB）' % (psnr_noisy, psnr_denoised),
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(r'J:\phd\tensorflow_demo\denoise_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("去噪对比图已保存: denoise_result.png")

print()
print("=" * 60)
print("6. 保存模型")
print("=" * 60)
model.save(r'J:\phd\tensorflow_demo\denoiser_model.keras')
print("模型已保存: denoiser_model.keras")

print()
print("全部完成！生成的文件：")
print("  - denoise_input.png    原图vs加噪图")
print("  - denoise_training.png 训练曲线")
print("  - denoise_result.png   去噪效果对比")
print("  - denoiser_model.keras 训练好的模型")
