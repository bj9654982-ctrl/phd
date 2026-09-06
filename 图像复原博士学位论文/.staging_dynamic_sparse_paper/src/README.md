# 源码目录

计划模块：

- `data/`：成对模糊—清晰数据加载与固定评价协议；
- `models/`：基线和 MC-DNFR 候选模块；
- `losses/`：Charbonnier、频域和候选几何约束；
- `metrics/`：PSNR、SSIM、LPIPS、参数量、MACs 和计时；
- `engine/`：CUDA-only 训练、验证与断点恢复。

P0 查新结束前不创建实际模型实现。

