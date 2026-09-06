import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== Prewitt 边缘检测处理函数 =====================
def prewitt_detect(img_gray):
    # Prewitt 水平梯度模板 Gx：检测垂直边缘
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Prewitt 垂直梯度模板 Gy：检测水平边缘
    kernel_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    # 分别卷积，保留正负值
    g_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_x)
    g_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_y)

    # 计算梯度幅值（精确法：欧氏距离）
    magnitude = np.sqrt(g_x**2 + g_y**2)

    # 归一化到 0-255 方便显示
    def norm(x):
        return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 返回：Gx响应绝对值、Gy响应绝对值、梯度幅值
    return norm(np.abs(g_x)), norm(np.abs(g_y)), norm(magnitude)

# ===================== 1. 读取两张对比图像并转灰度 =====================
img1 = cv2.imread("./leftwhite.png")
img2 = cv2.imread("./topwhite.png")


img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ===================== 2. 分别对两张图做 Prewitt 边缘检测 =====================
gx1, gy1, mag1 = prewitt_detect(img1_gray)
gx2, gy2, mag2 = prewitt_detect(img2_gray)

# ===================== 3. 对比可视化（2行4列，上下一一对应） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(16, 8))

# ---------- 第一行：图1的结果 ----------
plt.subplot(2, 4, 1)
plt.title("图1 原始灰度图")
plt.imshow(img1_gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title("图1 Gx 水平梯度（垂直边缘）")
plt.imshow(gx1, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("图1 Gy 垂直梯度（水平边缘）")
plt.imshow(gy1, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("图1 Prewitt 梯度幅值")
plt.imshow(mag1, cmap='gray')
plt.axis('off')

# ---------- 第二行：图2的结果 ----------
plt.subplot(2, 4, 5)
plt.title("图2 原始灰度图")
plt.imshow(img2_gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.title("图2 Gx 水平梯度（垂直边缘）")
plt.imshow(gx2, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.title("图2 Gy 垂直梯度（水平边缘）")
plt.imshow(gy2, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.title("图2 Prewitt 梯度幅值")
plt.imshow(mag2, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 可选：保存结果
# cv2.imwrite("图1_Prewitt边缘.png", mag1)
# cv2.imwrite("图2_Prewitt边缘.png", mag2)
