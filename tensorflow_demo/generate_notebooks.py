"""生成 TensorFlow 可视化学习 Notebook"""
import nbformat as nbf
import os

output_dir = r"J:\phd\tensorflow_demo\notebooks"
os.makedirs(output_dir, exist_ok=True)

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

def save_nb(cells, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    }
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  生成: {name}")

# ============================================================
# Notebook 1: 张量与自动求导
# ============================================================
cells1 = [
    md("# 第一课：张量与自动求导（可视化版）\n\n"
       "**学习目标**：理解 TF 的数据结构（张量）和核心引擎（自动求导），全部用图表展示。\n\n"
       "> 对应 PyTorch：`torch.Tensor` + `torch.autograd`"),

    code("import tensorflow as tf\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "%matplotlib inline\n"
         "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'\n"
         "print('TF版本:', tf.__version__)"),

    md("## 1. 张量的维度\n\n"
       "0维=标量，1维=向量，2维=矩阵，3维+ =高维张量（图片批次）。"),

    code("scalar = tf.constant(3.14)\n"
         "vector = tf.constant([1, 2, 3, 4])\n"
         "matrix = tf.constant([[1,2,3],[4,5,6],[7,8,9]])\n"
         "\n"
         "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
         "axes[0].bar(['scalar'], [scalar.numpy()], color='steelblue')\n"
         "axes[0].set_title('0D 标量: %.2f' % scalar.numpy())\n"
         "axes[0].set_ylim(0, 4)\n"
         "axes[1].bar(range(4), vector.numpy(), color='coral')\n"
         "axes[1].set_title('1D 向量 shape=%s' % str(vector.shape))\n"
         "axes[1].set_xticks(range(4))\n"
         "im = axes[2].imshow(matrix.numpy(), cmap='YlOrRd')\n"
         "axes[2].set_title('2D 矩阵 shape=%s' % str(matrix.shape))\n"
         "for i in range(3):\n"
         "    for j in range(3):\n"
         "        axes[2].text(j, i, matrix[i,j].numpy(), ha='center', va='center')\n"
         "plt.colorbar(im, ax=axes[2])\n"
         "plt.tight_layout(); plt.show()"),

    md("### 1.1 模拟一批图片（4D张量）\n\n"
       "图片数据格式：`(batch, height, width, channel)`"),

    code("batch = tf.random.normal((8, 64, 64, 3))\n"
         "fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n"
         "for i, ax in enumerate(axes.flat):\n"
         "    img = batch[i].numpy()\n"
         "    img = (img - img.min()) / (img.max() - img.min())\n"
         "    ax.imshow(img); ax.set_title('图片 %d' % (i+1)); ax.axis('off')\n"
         "plt.suptitle('4D张量 shape=%s (batch, H, W, C)' % str(batch.shape), fontsize=14)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 2. 张量运算可视化"),

    code("a = tf.constant([[1,2],[3,4]], dtype=tf.float32)\n"
         "b = tf.constant([[5,6],[7,8]], dtype=tf.float32)\n"
         "all_mats = [('a',a), ('b',b), ('a+b',a+b), ('a*b',a*b), ('a@b',a@b)]\n"
         "fig, axes = plt.subplots(1, 5, figsize=(18, 4))\n"
         "for ax, (title, mat) in zip(axes, all_mats):\n"
         "    im = ax.imshow(mat.numpy(), cmap='coolwarm', vmin=-5, vmax=60)\n"
         "    ax.set_title(title)\n"
         "    for i in range(2):\n"
         "        for j in range(2):\n"
         "            ax.text(j, i, '%.0f'%mat[i,j].numpy(), ha='center', va='center', fontsize=12)\n"
         "plt.colorbar(im, ax=axes, shrink=0.8)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 3. 自动求导 GradientTape\n\n"
       "在 `tape` 上下文里做运算，TF 自动记录，结束后 `tape.gradient()` 一键求导。\n\n"
       "> 对应 PyTorch：`loss.backward()`"),

    md("### 3.1 f(x) = x² 的导数（函数曲线 + 切线）"),

    code("x = tf.Variable(3.0)\n"
         "with tf.GradientTape() as tape:\n"
         "    y = x ** 2\n"
         "dy_dx = tape.gradient(y, x)\n"
         "\n"
         "x_range = np.linspace(-5, 5, 100)\n"
         "x0, y0, slope = x.numpy(), y.numpy(), dy_dx.numpy()\n"
         "tangent = slope * (x_range - x0) + y0\n"
         "\n"
         "plt.figure(figsize=(8, 5))\n"
         "plt.plot(x_range, x_range**2, 'b-', lw=2, label='f(x)=x²')\n"
         "plt.plot(x_range, tangent, 'r--', lw=2, label='切线(斜率=%.1f)'%slope)\n"
         "plt.scatter([x0],[y0], color='red', s=100, zorder=5, label='点(%.1f,%.1f)'%(x0,y0))\n"
         "plt.title('f(x)=x² 在 x=%.1f 的导数=%.1f (数学解:2x=%.1f)'%(x0,slope,2*x0))\n"
         "plt.legend(); plt.grid(alpha=0.3); plt.xlabel('x'); plt.ylabel('y')\n"
         "plt.axhline(0,color='k',lw=0.5); plt.axvline(0,color='k',lw=0.5)\n"
         "plt.show()"),

    md("### 3.2 梯度下降全过程可视化"),

    code("x = tf.Variable(4.0)\n"
         "lr = 0.1\n"
         "traj = []\n"
         "for step in range(20):\n"
         "    with tf.GradientTape() as tape:\n"
         "        y = x ** 2\n"
         "    grad = tape.gradient(y, x)\n"
         "    traj.append((x.numpy(), y.numpy()))\n"
         "    x.assign_sub(lr * grad)\n"
         "\n"
         "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
         "xr = np.linspace(-4.5, 4.5, 100)\n"
         "axes[0].plot(xr, xr**2, 'b-', alpha=0.5, label='f(x)=x²')\n"
         "axes[0].plot([t[0] for t in traj], [t[1] for t in traj], 'ro-', ms=5, label='下降轨迹')\n"
         "axes[0].scatter([0],[0], color='green', s=200, marker='*', zorder=5, label='最小值x=0')\n"
         "axes[0].set_title('函数曲线上的梯度下降轨迹')\n"
         "axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')\n"
         "axes[1].plot(range(len(traj)), [t[0] for t in traj], 'b-o', ms=4, label='x值')\n"
         "axes[1].plot(range(len(traj)), [t[1] for t in traj], 'r-s', ms=4, label='损失')\n"
         "axes[1].axhline(0, color='g', ls='--', alpha=0.5)\n"
         "axes[1].set_title('x和损失随步数变化')\n"
         "axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xlabel('步数')\n"
         "plt.tight_layout(); plt.show()\n"
         "print('最终 x=%.4f, 损失=%.6f' % (x.numpy(), x.numpy()**2))"),

    md("### 3.3 多变量求导：y = wx + b"),

    code("w = tf.Variable(2.0); b = tf.Variable(1.0); xv = 3.0\n"
         "with tf.GradientTape() as tape:\n"
         "    yp = w * xv + b\n"
         "grads = tape.gradient(yp, [w, b])\n"
         "\n"
         "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
         "wr = np.linspace(0, 4, 50)\n"
         "axes[0].plot(wr, wr*xv+b.numpy(), 'b-', lw=2)\n"
         "axes[0].scatter([w.numpy()], [yp.numpy()], color='red', s=100, zorder=5)\n"
         "axes[0].annotate('dy/dw=%.1f\\n(=x=%.1f)'%(grads[0].numpy(),xv), xy=(w.numpy(),yp.numpy()),\n"
         "                 xytext=(2.5,5), arrowprops=dict(arrowstyle='->'))\n"
         "axes[0].set_title('w对输出的影响'); axes[0].set_xlabel('w'); axes[0].set_ylabel('y=wx+b')\n"
         "axes[0].grid(alpha=0.3)\n"
         "br = np.linspace(-2, 4, 50)\n"
         "axes[1].plot(br, w.numpy()*xv+br, 'g-', lw=2)\n"
         "axes[1].scatter([b.numpy()], [yp.numpy()], color='red', s=100, zorder=5)\n"
         "axes[1].annotate('dy/db=%.1f\\n(恒等于1)'%grads[1].numpy(), xy=(b.numpy(),yp.numpy()),\n"
         "                 xytext=(1.5,5), arrowprops=dict(arrowstyle='->'))\n"
         "axes[1].set_title('b对输出的影响'); axes[1].set_xlabel('b'); axes[1].set_ylabel('y=wx+b')\n"
         "axes[1].grid(alpha=0.3)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 4. Variable（可变参数）vs constant（不可变）"),

    code("v = tf.Variable([1.0,2.0,3.0])\n"
         "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
         "axes[0].bar(['v1','v2','v3'], v.numpy(), color='steelblue')\n"
         "axes[0].set_title('Variable初始值'); axes[0].set_ylim(0,8)\n"
         "v.assign(v * 2)\n"
         "axes[1].bar(['v1','v2','v3'], v.numpy(), color='coral')\n"
         "axes[1].set_title('v.assign(v*2)后'); axes[1].set_ylim(0,8)\n"
         "c = tf.constant([1.0,2.0,3.0])\n"
         "axes[2].bar(['c1','c2','c3'], (c+1).numpy(), color='green')\n"
         "axes[2].set_title('constant c+1返回新张量\\n(原c不变)'); axes[2].set_ylim(0,8)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 第一课小结\n\n"
       "| 概念 | 作用 | 对应 PyTorch |\n|---|---|---|\n"
       "| `tf.constant` | 不可变张量 | `torch.tensor` |\n"
       "| `tf.Variable` | 可变参数 | `nn.Parameter` |\n"
       "| `tf.GradientTape` | 自动求导 | `loss.backward()` |\n"
       "| `tape.gradient(loss,vars)` | 计算梯度 | `param.grad` |\n"
       "| `optimizer.apply_gradients` | 更新参数 | `optimizer.step()` |\n\n"
       "**下一课**：线性回归，可视化拟合全过程。"),
]
save_nb(cells1, "01_张量与自动求导.ipynb")

# ============================================================
# Notebook 2: 线性回归可视化
# ============================================================
cells2 = [
    md("# 第二课：线性回归（可视化拟合全过程）\n\n"
       "**学习目标**：从零训练线性模型 `y=wx+b`，用图表展示数据、损失曲线、参数收敛、拟合过程。"),

    code("import tensorflow as tf\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "%matplotlib inline\n"
         "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'"),

    md("## 1. 准备数据（真实关系 y=2x+1 + 噪声）"),

    code("np.random.seed(42)\n"
         "X_np = np.random.uniform(-10, 10, (100,1)).astype(np.float32)\n"
         "noise = np.random.normal(0, 1.5, (100,1)).astype(np.float32)\n"
         "y_np = 2.0 * X_np + 1.0 + noise\n"
         "X = tf.constant(X_np); y_true = tf.constant(y_np)\n"
         "\n"
         "plt.figure(figsize=(8,5))\n"
         "plt.scatter(X_np, y_np, alpha=0.6, label='带噪声数据')\n"
         "xl = np.linspace(-10,10,100)\n"
         "plt.plot(xl, 2*xl+1, 'r-', lw=2, label='真实 y=2x+1')\n"
         "plt.xlabel('x'); plt.ylabel('y'); plt.title('训练数据(100样本)')\n"
         "plt.legend(); plt.grid(alpha=0.3); plt.show()"),

    md("## 2. 手动训练循环"),

    code("w = tf.Variable(tf.random.normal([1], mean=0, stddev=1.0))\n"
         "b = tf.Variable(tf.zeros([1]))\n"
         "lr = 0.01\n"
         "hist = {'w':[], 'b':[], 'loss':[]}\n"
         "snaps = []\n"
         "for epoch in range(200):\n"
         "    with tf.GradientTape() as tape:\n"
         "        y_pred = w * X + b\n"
         "        loss = tf.reduce_mean((y_pred - y_true)**2)\n"
         "    grads = tape.gradient(loss, [w, b])\n"
         "    w.assign_sub(lr * grads[0])\n"
         "    b.assign_sub(lr * grads[1])\n"
         "    hist['w'].append(w.numpy()[0]); hist['b'].append(b.numpy()[0]); hist['loss'].append(loss.numpy())\n"
         "    if epoch % 20 == 0:\n"
         "        snaps.append((epoch, w.numpy()[0], b.numpy()[0]))"),

    md("### 2.1 损失曲线"),

    code("plt.figure(figsize=(10,5))\n"
         "plt.plot(hist['loss'], 'b-', lw=1.5)\n"
         "plt.xlabel('Epoch'); plt.ylabel('MSE Loss')\n"
         "plt.title('训练损失曲线 (最终loss=%.4f)'%hist['loss'][-1])\n"
         "plt.yscale('log'); plt.grid(alpha=0.3); plt.show()"),

    md("### 2.2 参数收敛轨迹"),

    code("fig, axes = plt.subplots(1, 2, figsize=(14,5))\n"
         "axes[0].plot(hist['w'], 'b-', lw=1.5, label='学到的w')\n"
         "axes[0].axhline(2.0, color='r', ls='--', label='真实w=2.0')\n"
         "axes[0].set_title('w收敛 (最终=%.3f)'%hist['w'][-1])\n"
         "axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlabel('Epoch')\n"
         "axes[1].plot(hist['b'], 'g-', lw=1.5, label='学到的b')\n"
         "axes[1].axhline(1.0, color='r', ls='--', label='真实b=1.0')\n"
         "axes[1].set_title('b收敛 (最终=%.3f)'%hist['b'][-1])\n"
         "axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xlabel('Epoch')\n"
         "plt.tight_layout(); plt.show()"),

    md("### 2.3 拟合过程快照（红线逐步贴近绿线）"),

    code("fig, axes = plt.subplots(2, 5, figsize=(18,7))\n"
         "for idx, (ep, ws, bs) in enumerate(snaps):\n"
         "    ax = axes.flat[idx]\n"
         "    ax.scatter(X_np, y_np, alpha=0.3, s=20)\n"
         "    ax.plot(xl, ws*xl+bs, 'r-', lw=2)\n"
         "    ax.plot(xl, 2*xl+1, 'g--', lw=1.5, alpha=0.7)\n"
         "    ax.set_title('Epoch %d\\nw=%.2f,b=%.2f'%(ep,ws,bs))\n"
         "    ax.set_xlim(-10,10); ax.set_ylim(-20,25); ax.grid(alpha=0.3)\n"
         "axes.flat[0].legend(['数据','模型','真实'], loc='upper left', fontsize=8)\n"
         "plt.suptitle('拟合过程：红线(模型)逐步贴近绿线(真实)', fontsize=14)\n"
         "plt.tight_layout(); plt.show()"),

    md("### 2.4 损失等高线 + 梯度下降轨迹"),

    code("wg = np.linspace(-1,5,100); bg = np.linspace(-3,5,100)\n"
         "W, B = np.meshgrid(wg, bg)\n"
         "Loss = np.mean((W[...,None]*X_np + B[...,None] - y_np)**2, axis=2)\n"
         "\n"
         "plt.figure(figsize=(8,6))\n"
         "cs = plt.contour(W, B, Loss, levels=30, cmap='viridis')\n"
         "plt.colorbar(cs, label='Loss')\n"
         "plt.plot(hist['w'], hist['b'], 'ro-', ms=3, lw=1, label='梯度下降轨迹')\n"
         "plt.scatter([2.0],[1.0], color='yellow', s=200, marker='*', zorder=5, edgecolors='k', label='最优点(2,1)')\n"
         "plt.scatter([hist['w'][0]],[hist['b'][0]], color='white', s=100, zorder=5, edgecolors='k', label='起点')\n"
         "plt.xlabel('w'); plt.ylabel('b'); plt.title('损失等高线上的梯度下降轨迹')\n"
         "plt.legend(); plt.show()"),

    md("## 3. Keras 版本（5行代码）"),

    code("model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])\n"
         "model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss='mse')\n"
         "model.fit(X_np, y_np, epochs=200, verbose=0, batch_size=32)\n"
         "wk, bk = model.layers[0].get_weights()\n"
         "\n"
         "plt.figure(figsize=(8,5))\n"
         "plt.scatter(X_np, y_np, alpha=0.3, label='数据')\n"
         "plt.plot(xl, hist['w'][-1]*xl+hist['b'][-1], 'r-', lw=2,\n"
         "         label='手动版: w=%.3f,b=%.3f'%(hist['w'][-1],hist['b'][-1]))\n"
         "plt.plot(xl, wk[0][0]*xl+bk[0], 'g--', lw=2,\n"
         "         label='Keras版: w=%.3f,b=%.3f'%(wk[0][0],bk[0]))\n"
         "plt.title('手动实现 vs Keras高层API')\n"
         "plt.legend(); plt.grid(alpha=0.3); plt.xlabel('x'); plt.ylabel('y')\n"
         "plt.show()"),

    md("## 第二课小结\n\n"
       "- 训练四步：前向 → 损失 → 梯度 → 更新\n"
       "- 三大可视化工具：损失曲线、参数轨迹、等高线图\n"
       "- Keras `model.fit()` 封装了全部循环\n\n"
       "**下一课**：CNN 图像分类，可视化特征图。"),
]
save_nb(cells2, "02_线性回归可视化.ipynb")

# ============================================================
# Notebook 3: CNN图像分类可视化
# ============================================================
cells3 = [
    md("# 第三课：CNN 图像分类（可视化特征图与训练过程）\n\n"
       "**学习目标**：用 CNN 做 MNIST 手写数字识别，可视化图片、预测、特征图、训练曲线。"),

    code("import tensorflow as tf\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "%matplotlib inline\n"
         "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'"),

    md("## 1. 加载并可视化 MNIST 数据"),

    code("(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()\n"
         "x_train = x_train.astype('float32')/255.0\n"
         "x_test = x_test.astype('float32')/255.0\n"
         "x_train = x_train[..., np.newaxis]\n"
         "x_test = x_test[..., np.newaxis]\n"
         "\n"
         "# 展示前20张\n"
         "fig, axes = plt.subplots(2, 10, figsize=(16, 4))\n"
         "for i, ax in enumerate(axes.flat):\n"
         "    ax.imshow(x_train[i].squeeze(), cmap='gray')\n"
         "    ax.set_title('标签:%d'%y_train[i]); ax.axis('off')\n"
         "plt.suptitle('MNIST 训练样本（28x28灰度图）', fontsize=14)\n"
         "plt.tight_layout(); plt.show()\n"
         "\n"
         "# 标签分布\n"
         "plt.figure(figsize=(8,4))\n"
         "plt.hist(y_train, bins=range(11), rwidth=0.8, align='left', color='steelblue', edgecolor='k')\n"
         "plt.xticks(range(10)); plt.xlabel('数字'); plt.ylabel('样本数')\n"
         "plt.title('训练集标签分布（60000张）'); plt.grid(alpha=0.3, axis='y')\n"
         "plt.show()"),

    md("## 2. 搭建 CNN 模型"),

    code("model = tf.keras.Sequential([\n"
         "    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), name='conv1'),\n"
         "    tf.keras.layers.MaxPooling2D((2,2), name='pool1'),\n"
         "    tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2'),\n"
         "    tf.keras.layers.MaxPooling2D((2,2), name='pool2'),\n"
         "    tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv3'),\n"
         "    tf.keras.layers.Flatten(name='flatten'),\n"
         "    tf.keras.layers.Dense(64, activation='relu', name='dense1'),\n"
         "    tf.keras.layers.Dense(10, activation='softmax', name='output')\n"
         "])\n"
         "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
         "model.summary()"),

    md("## 3. 训练并可视化过程"),

    code("history = model.fit(x_train, y_train, epochs=5, batch_size=128,\n"
         "                     validation_split=0.1, verbose=0)\n"
         "\n"
         "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
         "axes[0].plot(history.history['loss'], 'b-o', ms=4, label='训练损失')\n"
         "axes[0].plot(history.history['val_loss'], 'r-o', ms=4, label='验证损失')\n"
         "axes[0].set_title('损失曲线'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)\n"
         "axes[1].plot(history.history['accuracy'], 'b-o', ms=4, label='训练准确率')\n"
         "axes[1].plot(history.history['val_accuracy'], 'r-o', ms=4, label='验证准确率')\n"
         "axes[1].set_title('准确率曲线'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)\n"
         "plt.tight_layout(); plt.show()\n"
         "\n"
         "test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)\n"
         "print('测试集准确率: %.2f%%' % (test_acc*100))"),

    md("## 4. 预测结果可视化（正确 vs 错误）"),

    code("preds = model.predict(x_test, verbose=0)\n"
         "pred_labels = np.argmax(preds, axis=1)\n"
         "\n"
         "# 找一些正确和错误的样本\n"
         "correct = np.where(pred_labels == y_test)[0]\n"
         "wrong = np.where(pred_labels != y_test)[0]\n"
         "\n"
         "fig, axes = plt.subplots(2, 8, figsize=(16, 5))\n"
         "for i in range(8):\n"
         "    idx = correct[i]\n"
         "    axes[0,i].imshow(x_test[idx].squeeze(), cmap='gray')\n"
         "    axes[0,i].set_title('预测:%d\\n真实:%d'%(pred_labels[idx],y_test[idx]), color='green', fontsize=9)\n"
         "    axes[0,i].axis('off')\n"
         "for i in range(min(8, len(wrong))):\n"
         "    idx = wrong[i]\n"
         "    axes[1,i].imshow(x_test[idx].squeeze(), cmap='gray')\n"
         "    axes[1,i].set_title('预测:%d\\n真实:%d'%(pred_labels[idx],y_test[idx]), color='red', fontsize=9)\n"
         "    axes[1,i].axis('off')\n"
         "axes[0,0].set_ylabel('正确预测', rotation=0, labelpad=40, fontsize=12)\n"
         "axes[1,0].set_ylabel('错误预测', rotation=0, labelpad=40, fontsize=12)\n"
         "plt.suptitle('预测结果展示（上：正确，下：错误）', fontsize=14)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 5. 预测概率分布（模型有多确信？）"),

    code("idx = 0\n"
         "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
         "axes[0].imshow(x_test[idx].squeeze(), cmap='gray')\n"
         "axes[0].set_title('输入图片（真实标签:%d）'%y_test[idx]); axes[0].axis('off')\n"
         "colors = ['green' if i==pred_labels[idx] else 'steelblue' for i in range(10)]\n"
         "axes[1].bar(range(10), preds[idx], color=colors)\n"
         "axes[1].set_xticks(range(10)); axes[1].set_xlabel('数字'); axes[1].set_ylabel('概率')\n"
         "axes[1].set_title('预测概率分布（最高=预测:%d, 置信度=%.1f%%）'%(pred_labels[idx],preds[idx][pred_labels[idx]]*100))\n"
         "axes[1].grid(alpha=0.3, axis='y')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 6. CNN 特征图可视化（模型在看什么？）\n\n"
       "每一层卷积核都在提取不同的特征：边缘、纹理、图案。"),

    code("# 构建能输出中间层的模型\n"
         "layer_outputs = [model.get_layer(name).output for name in ['conv1','pool1','conv2','pool2','conv3']]\n"
         "activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)\n"
         "\n"
         "img = x_test[0:1]  # 选一张'7'\n"
         "activations = activation_model.predict(img, verbose=0)\n"
         "layer_names = ['conv1(32核)', 'pool1', 'conv2(64核)', 'pool2', 'conv3(64核)']\n"
         "\n"
         "fig, axes = plt.subplots(5, 8, figsize=(16, 10))\n"
         "for row, (name, act) in enumerate(zip(layer_names, activations)):\n"
         "    for col in range(8):\n"
         "        ax = axes[row, col]\n"
         "        if col < act.shape[-1]:\n"
         "            ax.imshow(act[0, :, :, col], cmap='viridis')\n"
         "        ax.axis('off')\n"
         "        if col == 0:\n"
         "            ax.set_ylabel(name, rotation=0, labelpad=50, fontsize=10, va='center')\n"
         "plt.suptitle('CNN 各层特征图（每行展示前8个通道）', fontsize=14)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 7. 卷积核可视化"),

    code("conv1_weights = model.get_layer('conv1').get_weights()[0]\n"
         "fig, axes = plt.subplots(4, 8, figsize=(12, 6))\n"
         "for i, ax in enumerate(axes.flat):\n"
         "    if i < conv1_weights.shape[-1]:\n"
         "        kernel = conv1_weights[:, :, 0, i]\n"
         "        ax.imshow(kernel, cmap='gray')\n"
         "    ax.axis('off')\n"
         "plt.suptitle('第一层32个卷积核（3x3）', fontsize=14)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 第三课小结\n\n"
       "- CNN = 卷积层（提特征）+ 池化层（降维）+ 全连接层（分类）\n"
       "- 特征图可视化能帮你理解模型学到了什么\n"
       "- 训练曲线（损失/准确率）是诊断过拟合的关键\n\n"
       "**下一课**：迁移学习，用预训练模型做你的钢坯数据。"),
]
save_nb(cells3, "03_CNN图像分类可视化.ipynb")

# ============================================================
# Notebook 4: 迁移学习
# ============================================================
cells4 = [
    md("# 第四课：迁移学习（预训练模型 + 数据增强可视化）\n\n"
       "**学习目标**：用 ImageNet 预训练的 MobileNetV2，通过迁移学习快速适配新任务。这是你钢坯检测项目的标配。"),

    code("import tensorflow as tf\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "%matplotlib inline\n"
         "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'"),

    md("## 1. 加载预训练模型（MobileNetV2）"),

    code("base_model = tf.keras.applications.MobileNetV2(\n"
         "    input_shape=(96,96,3), include_top=False, weights='imagenet')\n"
         "base_model.trainable = False\n"
         "print('层数:', len(base_model.layers))\n"
         "print('参数量: {:,}'.format(base_model.count_params()))\n"
         "print('特征输出shape:', base_model.output_shape)"),

    md("## 2. 数据增强可视化\n\n"
       "数据少的时候，数据增强是防止过拟合的关键。"),

    code("# 造一张模拟图片\n"
         "np.random.seed(42)\n"
         "sample_img = np.random.rand(96, 96, 3).astype(np.float32)\n"
         "# 加一些结构（模拟缺陷）\n"
         "sample_img[30:50, 30:50] = 0.9\n"
         "sample_img[60:70, 20:80] = 0.1\n"
         "\n"
         "data_augmentation = tf.keras.Sequential([\n"
         "    tf.keras.layers.RandomFlip('horizontal'),\n"
         "    tf.keras.layers.RandomRotation(0.15),\n"
         "    tf.keras.layers.RandomZoom(0.15),\n"
         "    tf.keras.layers.RandomContrast(0.3),\n"
         "])\n"
         "\n"
         "fig, axes = plt.subplots(2, 5, figsize=(14, 6))\n"
         "axes[0,0].imshow(sample_img); axes[0,0].set_title('原图'); axes[0,0].axis('off')\n"
         "for i, ax in enumerate(axes.flat[1:]):\n"
         "    augmented = data_augmentation(sample_img[np.newaxis], training=True)[0].numpy()\n"
         "    ax.imshow(np.clip(augmented, 0, 1))\n"
         "    ax.set_title('增强%d'%(i+1)); ax.axis('off')\n"
         "plt.suptitle('数据增强效果（同一张图的不同变换）', fontsize=14)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 3. 构建迁移学习模型"),

    code("inputs = tf.keras.Input(shape=(96,96,3))\n"
         "x = data_augmentation(inputs)\n"
         "x = tf.keras.applications.mobilenet_v2.preprocess_input(x)\n"
         "x = base_model(x, training=False)\n"
         "x = tf.keras.layers.GlobalAveragePooling2D()(x)\n"
         "x = tf.keras.layers.Dropout(0.2)(x)\n"
         "outputs = tf.keras.layers.Dense(10, activation='softmax')(x)\n"
         "model = tf.keras.Model(inputs, outputs)\n"
         "\n"
         "total = model.count_params()\n"
         "trainable = sum(w.shape.num_elements() for w in model.trainable_weights)\n"
         "print('总参数: {:,}'.format(total))\n"
         "print('可训练: {:,} ({:.2f}%)'.format(trainable, trainable/total*100))\n"
         "print('冻结: {:,} ({:.2f}%)'.format(total-trainable, (total-trainable)/total*100))"),

    md("## 4. 模拟数据训练（用随机数据模拟你的钢坯数据集）"),

    code("np.random.seed(42)\n"
         "X_tr = np.random.rand(500, 96, 96, 3).astype(np.float32)\n"
         "y_tr = np.random.randint(0, 10, 500)\n"
         "X_val = np.random.rand(100, 96, 96, 3).astype(np.float32)\n"
         "y_val = np.random.randint(0, 10, 100)\n"
         "\n"
         "model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),\n"
         "              loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
         "hist1 = model.fit(X_tr, y_tr, validation_data=(X_val,y_val),\n"
         "                  epochs=5, batch_size=32, verbose=0)\n"
         "\n"
         "fig, axes = plt.subplots(1, 2, figsize=(14,5))\n"
         "axes[0].plot(hist1.history['loss'], 'b-o', ms=4, label='训练')\n"
         "axes[0].plot(hist1.history['val_loss'], 'r-o', ms=4, label='验证')\n"
         "axes[0].set_title('第一阶段：只训练分类头 (lr=1e-3)')\n"
         "axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlabel('Epoch')\n"
         "axes[1].plot(hist1.history['accuracy'], 'b-o', ms=4, label='训练')\n"
         "axes[1].plot(hist1.history['val_accuracy'], 'r-o', ms=4, label='验证')\n"
         "axes[1].set_title('准确率')\n"
         "axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xlabel('Epoch')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 5. 微调（解冻部分层，用更小学习率）"),

    code("base_model.trainable = True\n"
         "fine_tune_at = len(base_model.layers) - 20\n"
         "for layer in base_model.layers[:fine_tune_at]:\n"
         "    layer.trainable = False\n"
         "\n"
         "model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),\n"
         "              loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
         "hist2 = model.fit(X_tr, y_tr, validation_data=(X_val,y_val),\n"
         "                  epochs=3, batch_size=32, verbose=0)\n"
         "\n"
         "# 合并两阶段曲线\n"
         "all_loss = hist1.history['loss'] + hist2.history['loss']\n"
         "all_val_loss = hist1.history['val_loss'] + hist2.history['val_loss']\n"
         "all_acc = hist1.history['accuracy'] + hist2.history['accuracy']\n"
         "all_val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']\n"
         "\n"
         "fig, axes = plt.subplots(1, 2, figsize=(14,5))\n"
         "axes[0].plot(all_loss, 'b-o', ms=4, label='训练损失')\n"
         "axes[0].plot(all_val_loss, 'r-o', ms=4, label='验证损失')\n"
         "axes[0].axvline(x=len(hist1.history['loss'])-0.5, color='g', ls='--', label='开始微调')\n"
         "axes[0].set_title('两阶段训练损失曲线')\n"
         "axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlabel('Epoch')\n"
         "axes[1].plot(all_acc, 'b-o', ms=4, label='训练准确率')\n"
         "axes[1].plot(all_val_acc, 'r-o', ms=4, label='验证准确率')\n"
         "axes[1].axvline(x=len(hist1.history['accuracy'])-0.5, color='g', ls='--', label='开始微调')\n"
         "axes[1].set_title('两阶段准确率曲线')\n"
         "axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xlabel('Epoch')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 6. 预训练模型特征提取可视化\n\n"
       "把一张图通过预训练模型，看看提取的特征是什么样的。"),

    code("# 用模拟图片\n"
         "demo_img = np.random.rand(1, 96, 96, 3).astype(np.float32)\n"
         "demo_img[0, 30:50, 30:50] = 0.9\n"
         "\n"
         "preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(demo_img)\n"
         "features = base_model(preprocessed, training=False)\n"
         "\n"
         "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
         "axes[0].imshow(demo_img[0]); axes[0].set_title('输入图片'); axes[0].axis('off')\n"
         "# 展示特征图的前16个通道（3x3空间）\n"
         "feat_vis = np.zeros((3*4, 3*4))\n"
         "for i in range(16):\n"
         "    r, c = i//4, i%4\n"
         "    feat_vis[r*3:(r+1)*3, c*3:(c+1)*3] = features[0,:,:,i]\n"
         "axes[1].imshow(feat_vis, cmap='viridis'); axes[1].set_title('MobileNetV2提取的特征(前16通道)'); axes[1].axis('off')\n"
         "plt.tight_layout(); plt.show()\n"
         "print('特征图shape:', features.shape)"),

    md("## 7. 可用预训练模型对比"),

    code("models_info = [\n"
         "    ('MobileNetV2', 3.5, '轻量，移动端/边缘设备首选'),\n"
         "    ('EfficientNetB0', 5.3, '效率高，参数少'),\n"
         "    ('ResNet50', 25.6, '经典残差网络'),\n"
         "    ('ResNet50V2', 25.6, 'ResNet改进版'),\n"
         "    ('DenseNet121', 8.1, '密集连接'),\n"
         "    ('InceptionV3', 23.9, 'GoogleNet变体'),\n"
         "    ('VGG16', 138, '经典但笨重'),\n"
         "]\n"
         "names = [m[0] for m in models_info]\n"
         "params = [m[1] for m in models_info]\n"
         "\n"
         "fig, ax = plt.subplots(figsize=(10, 5))\n"
         "bars = ax.barh(names, params, color='steelblue')\n"
         "ax.set_xlabel('参数量 (M)'); ax.set_title('Keras内置预训练模型参数量对比')\n"
         "for bar, (_, _, desc) in zip(bars, models_info):\n"
         "    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, desc, va='center', fontsize=9)\n"
         "ax.set_xlim(0, 160); ax.grid(alpha=0.3, axis='x')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 第四课小结\n\n"
       "**迁移学习两阶段**：\n"
         "1. **特征提取**：冻结 backbone，只训练分类头（lr=1e-3）\n"
         "2. **微调**：解冻顶部几层，用小学习率（lr=1e-5）精调\n\n"
       "**对你的钢坯检测项目**：\n"
         "- 数据少 → 必须用迁移学习 + 数据增强\n"
         "- 边缘部署 → 选 MobileNetV2 / EfficientNetB0\n"
         "- 训练完导出 SavedModel → TFLite → 量化部署\n\n"
         "全部课程完成！你已经掌握了 TF 的核心用法，可以开始你的项目了。"),
]
save_nb(cells4, "04_迁移学习可视化.ipynb")

print("\n全部4个Notebook生成完成！")
print("目录:", output_dir)
