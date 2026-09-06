# 动态稀疏非局部去模糊论文工程

本工程用于逐步完成一篇与《融合非局部特征表示的模糊图像复原》相近层次的图像复原论文。目标是形成可验证、可复现、能够投稿的完整工作，不以“换模块后只展示一张效果图”为完成标准。

## 当前状态

- 当前阶段：P1 与合成几何门槛已通过；第一版 MT-NLR 在真实数据 P2 中未超过 B0，已停止扩展，准备建立全训练集强基线
- 参考论文：华夏等，2024，《融合非局部特征表示的模糊图像复原》
- 已审计研究假设：运动轨迹条件非局部特征表示
- 阶段模块名：MT-NLR（Motion-Trajectory-conditioned Nonlocal Representation）
- 注意：该模块在受控合成实验有效，但在 GoPro 迷你真实数据 P2 中未超过 B0，不能作为已成立的论文贡献

## 当前工作假设

固定窗口、固定采样块和通用 Top-K 稀疏注意力都不能针对空间变化运动模糊分配合适的感受野。候选模块根据局部模糊方向、长度和置信度，沿可能的运动轨迹选择非局部特征；清晰或方向不可靠区域保留局部旁路。第一版使用固定采样点数，只让方向和几何跨度自适应，以保证实际计算量可复现。

这不是最终创新结论。DRSformer、AST、TEAFormer、Concertormer、MISC Filter 和 MDT 都是必须避开的近邻工作。

## 第一条运行命令

```powershell
C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe scripts\check_environment.py
```

脚本只接受 CUDA 环境，不提供 CPU 训练回退。

完整训练集开发基线（耗时任务，建议在独立 PowerShell 中执行）：

```powershell
powershell -ExecutionPolicy Bypass -File "J:\phd\动态稀疏非局部去模糊论文\scripts\run_p3_full_baseline.ps1"
```

该脚本使用全部 2103 对训练图像、100 对固定开发验证图像和 30000 步训练；输出到 `results/runs/P3_baseline_full_v0/`。运行结束后先检查 `summary.json`，不自动启动新方法训练。

基线单元测试与单 batch 过拟合：

```powershell
C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe -m unittest discover -s tests -v
C:\Users\Administrator\miniconda3\envs\torch-gpu\python.exe scripts\run_overfit_smoke.py --steps 400 --size 64 --width 16
```

当前已验证结果：17 个 CUDA/数据/轨迹采样测试通过；合成单 batch PSNR 从 14.40 dB 提升到 37.20 dB；GoPro 迷你基线相对模糊输入提升 0.285 dB；稠密 E0 方向 MAE 6.315°、输入尺度长度 MAE 1.512 像素；真实 P2 的最好 MT-NLR 仍比 B0 低 0.022 dB。以上均为阶段闸门结果，不能替代公开协议上的论文主结果。

## 目录职责

```text
configs/   训练与模型配置；每次正式实验必须保存配置
data/      数据目录说明；原始数据不提交到 Git
docs/      路线、查新矩阵、实验日志和决策记录
paper/     论文正文、图表清单和投稿材料
results/   指标、曲线、可视化和复杂度结果
scripts/   环境、训练、测试和统计入口脚本
src/       数据、模型、损失和评测的实现
tests/     单元测试与最小过拟合测试
```

## 不可跳过的规则

1. 先冻结评价协议，再比较模型。
2. 先复现基线，再实现新模块。
3. 新旧模型使用相同数据、裁剪、训练步数、随机种子和测试脚本。
4. 每个结论必须对应消融实验或可视化证据。
5. 不用测试集选择超参数。
6. 不只报告最好一次结果；关键实验至少运行 3 个随机种子。
7. 不将参数量增加带来的收益包装成模块收益。
8. 失败实验同样记录，避免反复走同一条路。
