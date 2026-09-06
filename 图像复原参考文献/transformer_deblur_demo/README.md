# Transformer 去模糊直观演示（PyTorch版）

一个用于直观理解 **Transformer 自注意力机制为什么能去模糊** 的最小演示工程。支持用户传入自定义图片，直观看到模糊→还原的全过程。支持 CUDA GPU 加速。

## 工程结构

```
transformer_deblur_demo/
├── README.md                          # 本说明文档
├── requirements.txt                   # Python 依赖
├── transformer_deblur_demo.py        # PyTorch 版本（完整 Transformer 网络，含训练过程）
├── deblur_model.pt                    # 训练好的模型权重（运行后生成）
├── blurred_input.png                  # 生成的运动模糊图像（运行后生成）
├── restored_result.png                # 复原后的图像（运行后生成）
└── transformer_deblur_result.png     # 完整可视化对比图（运行后生成）
```

## 快速开始

### 直接运行（默认使用配置区域的图片）

```bash
python transformer_deblur_demo.py
```

程序会自动使用配置区域中 `IMAGE_PATH` 指定的图片。如果该文件不存在，会自动回退到内置合成图像。

### 推荐方式：直接修改代码中的配置变量

脚本的 `import` 之后有一个**配置区域**，所有参数都用变量定义，直接在 PyCharm 里改变量就行，无需命令行参数：

```python
# ============================================================
# 配置区域：直接在这里修改参数，无需命令行参数
# ============================================================
IMAGE_PATH = "building.jpg"                        # 输入图片路径（改成你自己的图片路径）
OUTPUT_PATH = "transformer_deblur_result.png"    # 输出结果图路径
BLURRED_OUTPUT_PATH = "blurred_input.png"         # 生成的模糊图像保存路径
RESTORED_OUTPUT_PATH = "restored_result.png"      # 复原后的图像保存路径
MODEL_SAVE_PATH = "deblur_model.pt"               # 训练好的模型权重保存路径
LOAD_MODEL = False                                  # 是否加载已保存的模型（True则跳过训练）
TARGET_PSNR = 32.57                                 # 目标PSNR，达到此值则提前停止（None则不启用）
IMAGE_SIZE = 128                                   # 图像处理尺寸（建议64-256，越大越慢）
MOTION_LENGTH = 15                                  # 运动模糊长度（像素，越大模糊越严重）
MOTION_ANGLE = 28.0                                 # 运动模糊角度（度）
TRAIN_STEPS = 500                                  # 训练步数（越多恢复越充分但越慢）
LEARNING_RATE = 1e-3                               # 学习率
DIM = 32                                            # 特征维度（越大模型容量越大）
NUM_BLOCKS = 4                                      # Transformer Block数量
WINDOW_SIZE = 8                                     # 自注意力窗口大小（必须能整除 IMAGE_SIZE）
NUM_HEADS = 4                                       # 注意力头数（必须能整除 DIM）
# ============================================================
```

> 命令行参数仍然保留，可以临时覆盖配置区域的变量值。

### 使用用户自定义图片（命令行方式）

```bash
python transformer_deblur_demo.py --image "你的图片路径.jpg"
```

支持 jpg、png、bmp、webp 等常见图片格式。程序会自动：
1. 居中裁剪为正方形并缩放到指定尺寸
2. 自动施加运动模糊
3. 用 Transformer 去模糊
4. 生成可视化对比图（原图 → 模糊 → 训练过程 → 最终还原 → 注意力图）

## 完整命令行参数

```bash
python transformer_deblur_demo.py \
  --image "test.jpg" \      # 用户图片路径
  --size 128 \              # 图像处理尺寸（默认128，建议64-256）
  --motion-length 15 \      # 运动模糊长度（像素）
  --motion-angle 28.0 \     # 运动模糊角度（度）
  --steps 500 \             # 训练步数（默认500，越多越充分）
  --lr 0.001 \              # 学习率
  --dim 32 \                # 特征维度
  --blocks 4 \              # Transformer Block数量
  --window 8 \              # 自注意力窗口大小（必须能整除size）
  --heads 4 \               # 注意力头数（必须能整除dim）
  --output "result.png"     # 输出图片路径
```

## 运行模式

| 模式 | LOAD_MODEL 设置 | 行为 |
|------|----------------|------|
| **训练模式**（默认） | `False` | 从头训练，训练完自动保存 `deblur_model.pt` |
| **加载模式** | `True` | 如果 `deblur_model.pt` 存在，直接加载权重跳过训练，秒出结果 |
| **早停模式** | `TARGET_PSNR = 32.57` | 训练中PSNR超过目标值则自动停止 |

## 网络结构

```
输入模糊图像 (B, 3, H, W)
    ↓
Patch Embedding (Conv2d, 3→DIM)
    ↓
Transformer Block × NUM_BLOCKS
    ├── LayerNorm
    ├── 窗口自注意力 (Window Multi-Head Self-Attention)
    ├── 残差连接
    ├── LayerNorm
    ├── FFN (Linear → GELU → Linear)
    └── 残差连接
    ↓
重建头 (Conv2d → GELU → Conv2d, DIM→3)
    ↓
残差连接: 输出 = 输入 + 残差
    ↓
复原图像 (B, 3, H, W)
```

## 可视化结果说明

运行后生成的 `transformer_deblur_result.png` 包含 3 行：

| 行 | 内容 |
|----|------|
| 第1行 | 原图 → 模糊输入 → 训练第1/10/50步 → 最终输出 |
| 第2行 | 注意力图说明 + 4个不同位置像素的注意力分布 |
| 第3行 | 训练损失曲线 + 中心区域放大对比 + 总结 |

## 典型结果

```
模糊输入 PSNR:   14.34 dB
恢复输出 PSNR:   15.07 dB（20步训练）
PSNR 提升:       +0.73 dB
```

> 注：这是**极简演示**，目的是直观感受自注意力的全局参考机制。真实的 Transformer 去模糊网络（如 Uformer、Restormer）经过大规模数据训练后，在 GoPro 数据集上 PSNR 通常可达 32+ dB。

## GPU 加速

脚本自动检测 CUDA：
- 有 NVIDIA 显卡且安装 CUDA 版 PyTorch → 自动使用 GPU
- 否则 → 使用 CPU

运行时第一行会打印 `使用设备: cuda` 或 `使用设备: cpu`。

## 使用技巧

1. **图片选择**：选择有清晰边缘、文字、纹理的图片效果更明显（如风景、建筑、文字截图）
2. **尺寸调整**：默认128速度快，256效果更好但更慢
3. **模糊强度**：`MOTION_LENGTH` 控制模糊程度，15-25适合演示
4. **训练步数**：`TRAIN_STEPS` 设为1000-2000效果更好，但需要更长时间
5. **窗口大小**：`WINDOW_SIZE` 必须能整除 `IMAGE_SIZE`（如size=128, window=8或16）
6. **复用模型**：训练一次后把 `LOAD_MODEL = True`，后续直接加载模型秒出结果

## 常见问题

**Q: 中文字体显示为方块？**
A: 脚本已配置Windows自带的SimHei和Microsoft YaHei字体。如果仍有问题，请确保系统安装了中文字体。

**Q: 每次结果不一样？**
A: 神经网络权重初始化是随机的，每次训练结果会有差异，但趋势一致。

**Q: 加载模型失败？**
A: 加载模型时，模型结构参数（DIM、NUM_BLOCKS、WINDOW_SIZE、NUM_HEADS）必须和训练时一致，否则会加载失败。如果改了这些参数，需要把 `LOAD_MODEL` 改回 `False` 重新训练。

**Q: 训练很久也达不到目标PSNR？**
A: 这个极简网络（4个Block、32维特征、单张图片）可能很难达到32.57 dB。可以调大模型和训练步数，或者把 `TARGET_PSNR = None` 关闭早停。

## 相关论文

- **Uformer** (CVPR 2022)：U-shaped Transformer for image restoration
- **Restormer** (CVPR 2022)：Efficient Transformer for high-resolution image restoration
- **SwinIR** (ICCVW 2021)：Swin Transformer for image restoration
- **融合非局部特征表示的模糊图像复原**（中国图象图形学报 2024）：洪汉玉团队，NLFR 模块

## 环境要求

- Python 3.8+
- torch（支持 CUDA）
- numpy
- matplotlib
- Pillow

## License

MIT
