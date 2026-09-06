# TensorFlow 学习教程

## 环境信息

- **Python 环境**: `torch-gpu` (Python 3.11.15)
- **TensorFlow 版本**: 2.17.0
- **Keras 版本**: 3.15.1
- **运行方式**: CPU (TF 2.11+ Windows 原生不支持 GPU，学习足够用)

## 运行命令

所有代码用 `torch-gpu` 环境的 Python 运行：

```bash
C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe <脚本路径>
```

例如：
```bash
C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe 01_basics\01_tensor.py
```

## 课程目录

### 阶段一：核心概念

| 序号 | 文件 | 内容 | 对应 PyTorch |
|------|------|------|-------------|
| 1 | `01_basics/01_tensor.py` | 张量创建、运算、Variable | `torch.Tensor`, `nn.Parameter` |
| 2 | `01_basics/02_gradient_tape.py` | 自动求导 GradientTape | `torch.autograd`, `.backward()` |

### 阶段二：线性回归

| 序号 | 文件 | 内容 |
|------|------|------|
| 3 | `02_regression/01_linear_regression_manual.py` | 手动写训练循环（前向->求导->更新） |
| 4 | `02_regression/02_linear_regression_keras.py` | Keras 高层 API（5行代码搞定） |

### 阶段三：神经网络

| 序号 | 文件 | 内容 | 准确率 |
|------|------|------|--------|
| 5 | `03_neural_network/01_mnist_dense.py` | 全连接网络做 MNIST 分类 | ~98% |
| 6 | `04_cnn/01_mnist_cnn.py` | 卷积神经网络 CNN | ~99% |

### 阶段四：进阶

| 序号 | 文件 | 内容 |
|------|------|------|
| 7 | `05_custom_training/01_custom_loop.py` | 自定义训练循环（类似 PyTorch 风格）+ 梯度裁剪 |
| 8 | `06_transfer_learning/01_transfer.py` | 迁移学习（MobileNetV2 预训练 + 微调） |

## 学习建议

1. **按顺序学**：每一课都建立在前一课的基础上
2. **边跑边改**：运行后试着修改参数，观察结果变化
3. **对比 PyTorch**：你有 PyTorch 基础，重点理解 TF 的思维差异
4. **重点掌握**：
   - GradientTape 的用法（和 PyTorch backward 的区别）
   - Keras Sequential / Model 子类化
   - `model.fit()` vs 自定义循环
   - `tf.data.Dataset` 数据管道
   - 迁移学习两阶段训练

## TF vs PyTorch 速查

| 操作 | PyTorch | TensorFlow |
|------|---------|-----------|
| 张量 | `torch.tensor([1,2])` | `tf.constant([1,2])` |
| 可训练参数 | `nn.Parameter` | `tf.Variable` |
| 求导 | `loss.backward()` | `with tf.GradientTape() as t: g=t.gradient(loss,vars)` |
| 优化器更新 | `optimizer.step()` | `optimizer.apply_gradients(zip(grads,vars))` |
| 模型定义 | `class Net(nn.Module): def forward` | `class Net(tf.keras.Model): def call` |
| 训练 | 手动循环 | `model.fit()` 或手动循环 |
| 设备 | `.to('cuda')` | 自动，或 `with tf.device('/GPU:0')` |
| 数据 | `DataLoader` | `tf.data.Dataset` |

## 注意事项

- 运行时会看到 oneDNN / CPU feature 提示，这是正常信息，不是错误
- 第一次运行 MNIST 会自动下载数据集（约 11MB）
- 第一次运行迁移学习会自动下载 MobileNetV2 权重（约 9MB）
- numpy 被降级到 1.26.4（TF 2.17 要求），opencv 可能有版本警告但不影响使用
