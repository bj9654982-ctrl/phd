# Transformer 模糊图像复原 Demo

这是一个可训练、可推理的轻量级图像去模糊工程。模型采用 Restormer 风格的 Transformer U-Net：

- 多尺度编码器—解码器负责不同尺度的模糊；
- 转置自注意力在全图空间上统计相关性；
- 深度卷积前馈网络补充局部边缘与纹理；
- 输出采用全局残差学习；
- 训练损失为像素域 L1 损失与频域损失的组合。

> `demo.py` 会在运行时现场训练，因此不需要提前准备 checkpoint。只有使用 `infer.py` 复原任意真实模糊图时，才需要先用 `train.py` 得到训练权重。

## 0. 最简单的效果演示

直接运行 `demo.py` 即可。当前版本只使用 NVIDIA CUDA，不包含 CPU 推理或自动回退。默认情况下它会：

1. 读取 `IMAGE_PATH` 指定的清晰图片；
2. 合成运动模糊；
3. 在 GPU 上现场训练一个小型 Transformer；
4. 输出清晰图、模糊图、复原图和横向对比图；
5. 打印复原前后的 PSNR。

```powershell
C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe demo.py
```

结果保存在 `simple_demo_outputs`。在 `demo.py` 顶部修改清晰图片路径：

```python
IMAGE_PATH = r"D:\images\my_sharp_image.png"
```

这里应填写一张清晰图像，因为教学 Demo 需要它作为真值并自动生成对应模糊图。相对路径以工程目录为基准。这个单图现场训练实验用于直观展示 Transformer 能够学习逆模糊映射，不代表模型对任意真实模糊图像的泛化性能。

## 1. 推荐环境

本机已经存在可用环境：

```powershell
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n torch-gpu python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

检测结果为 Python 3.11、PyTorch 2.5.1+cu121，能够使用 RTX 3060 Laptop GPU。因此无需修改 base 环境，后续命令直接使用 `torch-gpu`。如果 CUDA 不可用，程序会直接报错，不会切换到 CPU。

如果需要独立环境，也可以参考 `environment.yml`，然后根据显卡和 CUDA 版本安装对应的 PyTorch。

## 2. 准备训练数据

`--train-dir` 应指向包含清晰图像的目录，支持 PNG、JPG、JPEG、BMP、TIFF 和 WebP。训练代码会在线生成随机高斯模糊、随机方向运动模糊和轻微噪声，因此不要求事先准备模糊—清晰配对。

例如：

```powershell
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n torch-gpu python train.py `
  --train-dir "D:\datasets\sharp_images" `
  --output-dir checkpoints `
  --epochs 20 `
  --batch-size 4 `
  --patch-size 128
```

训练结束后会生成：

- `checkpoints/best.pt`：当前最低训练损失模型；
- `checkpoints/last.pt`：最后一轮模型。

显存不足时，首先减小 `--batch-size`，然后减小 `--patch-size` 或使用较小模型：

```powershell
python train.py --train-dir "D:\datasets\sharp_images" --batch-size 1 --patch-size 96 --embed-dim 16 --depths 1 1 2 --heads 1 2 4
```

## 3. 复原任意真实模糊图像

简单演示无需执行本节。`demo.py` 是单图现场训练实验，需要清晰图作为真值；如果只有一张真实模糊图，没有清晰真值，就必须先在一批训练数据上训练模型，再使用 `infer.py`：

```powershell
C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe infer.py `
  --input "D:\images\my_blurred_image.png" `
  --checkpoint "checkpoints\best.pt" `
  --output "outputs\my_restored_image.png"
```

`--input` 也可以指向包含多张图像的目录。输入尺寸不要求是固定倍数，模型会自动填充并在输出时裁剪回原始尺寸。

## 4. 不使用私人数据的端到端流程测试

先生成程序化的清晰图像：

```powershell
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n torch-gpu python make_toy_dataset.py --count 64 --size 160
```

再训练一个小模型验证流程：

```powershell
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n torch-gpu python train.py `
  --train-dir data\toy_sharp `
  --epochs 3 `
  --batch-size 4 `
  --patch-size 96 `
  --embed-dim 16 `
  --depths 1 1 2 `
  --heads 1 2 4
```

程序化数据只能验证代码流程，不能替代真实图像数据训练。

## 5. 测试

```powershell
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n torch-gpu python -m unittest discover -s tests -v
```

测试覆盖 CUDA 上的任意尺寸前向传播、模糊核归一化、在线模糊生成和图像读写。

## 工程结构

```text
deblur_transformer/
  model.py       Transformer U-Net、注意力与前馈网络
  data.py        清晰图像数据集和在线模糊生成
  io_utils.py    图像读写与路径扫描
train.py         训练入口
infer.py         推理入口
demo.py          无需预训练权重的单图效果演示
make_toy_dataset.py
tests/test_smoke.py
requirements.txt
environment.yml
```

## 适用范围与限制

这是教学和研究原型，不是通用商业去模糊模型。最终效果取决于训练数据与实际模糊类型是否匹配。当前合成训练退化包括高斯模糊、线性运动模糊和轻微高斯噪声；如果目标图像包含空间变化模糊、散焦、压缩伪影或真实相机复杂退化，应使用对应的真实配对数据扩展训练流程。
