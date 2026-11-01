# 分支 B2 文献侦察清单

**分支名**：传统全参考(FR)/半参考(RR)客观质量指标，以及学习型感知 FR 指标（LPIPS/DISTS 谱系）
**生成时间**：2026-10-20
**正式核验文献**：24 篇（VERIFIED=24；其中 A 级 12 篇、B 级 9 篇、C 级 3 篇）
**2023—2026 年新文献**：6 篇（B2-019～B2-024）

---

## 一、检索式记录（全部逐条执行）

**第零步 原词直搜（中文+英文）**
1. `全参考图像质量评价 SSIM 综述`（中文）
2. `full reference image quality assessment metric survey`（英文）
3. `learning-based full-reference image quality assessment 2024 2025`（英文·新进展）

**视角 1 信号保真度基线**
4. `Wang Bovik Mean squared error Love it or leave it signal fidelity`

**视角 2 HVS 与信息论早期**
5. `Sheikh Bovik Image information and visual quality VIF`
6. `Chandler Hemami VSNR wavelet visual signal-to-noise ratio natural images`

**视角 3 SSIM 谱系**
7. `Wang Simoncelli Bovik Multi-scale structural similarity for image quality assessment Asilomar`
8. `Wang Li Information content weighting perceptual image quality assessment IW-SSIM`
9. `complex wavelet structural similarity image quality assessment CW-SSIM Wang Simoncelli`

**视角 4 特征/梯度/显著性类**
10. `Zhang Lei FSIM feature similarity index image quality assessment`
11. `Xue Gradient Magnitude Similarity Deviation GMSD image quality assessment`
12. `Zhang VSI visual saliency-induced index perceptual image quality assessment`
13. `RFSIM Riesz transform feature based image quality assessment metric`
14. `Larson Chandler Most Apparent Distortion MAD full-reference image quality assessment`
15. `Reisenhofer Haar wavelet-based perceptual similarity index HaarPSI image quality assessment`
16. `SR-SIM spectral residual image quality assessment index`

**视角 5 半参考 RR**
17. `Soundararajan Bovik RRED indices reduced reference quality assessment natural images`

**视角 6 学习型感知 FR（LPIPS/DISTS 谱系）**
18. `Zhang Isola Efros Unreasonable Effectiveness of Deep Features as a Perceptual Metric LPIPS`
19. `Prashnani PieAPP Perceptual Image-Error Assessment through Pairwise Preference`
20. `DeepWSD deep weighted spatial distances perceptual quality metric`
21. `Bhardwaj Perceptual Index Metric PIM full reference image quality`

**视角 7 2023—2026 新指标（重点）**
22. `full-reference image quality assessment 2025 transformer state-of-the-art PIPAL`
23. `vision language model CLIP perceptual quality metric full-reference image 2024`

**视角 8 中文锚点**
24. `程茹秋 余烨 图像与视频质量评价综述 中国图象图形学报`

**饱和判据说明**：覆盖达标（种子 21 条全部命中并逐条核验）✓；主题饱和（基线/HVS信息论/SSIM谱系/特征梯度显著性/RR/学习型感知FR/新进展 7 个类别均落位）✓；引用闭环（SSIM、FSIM、LPIPS、DISTS 的共引文献 IFC/VIF/MS-SSIM/GMSD/PieAPP/A-DISTS 已全部捞入）✓；时间跨度覆盖（2003—2024 无断层）✓。满足 ≥3 条停止检索。

---

## 二、正式文献清单（按主题分组）

### 主题 A：信号保真度基线

**[B2-001] Wang & Bovik (2009)** — *Mean Squared Error: Love It or Leave It? [A new look at signal fidelity measures]*, IEEE Signal Processing Magazine, 26(1). — 核验：VERIFIED；source_quality=A；authority=classic；is_seminal=yes；角色：奠基/问题提出；fr_type=FR。关键发现：MSE/PSNR 在感知信号上长期主导但性能弱，需从视觉感知角度设计度量。方法：立场性专论，不提出新指标。url: https://sites.units.it/ramponi/teaching/DIP/DIPmaterials/z08_zz_mse_bovik09.pdf

### 主题 B：HVS 与信息论早期

**[B2-002] Sheikh, Bovik & De Veciana (2005)** — *An Information Fidelity Criterion for Image Quality Assessment Using Natural Scene Statistics*, IEEE TIP, 14(12). — VERIFIED；A；high_citation；seminal；FR。关键发现：以 NSS 建模参考-失真互信息，无参数、优于当时主流方法。方法：高斯尺度混合 NSS + 信息保真度准则。url: https://users.ece.utexas.edu/~gustavo/papers/de%20Veciana1251902020.pdf

**[B2-003] Sheikh & Bovik (2006)** — *Image Information and Visual Quality*, IEEE TIP, 15(2). — VERIFIED；A；high_citation；seminal；FR。关键发现：VIF 将参考-失真互信息与 HVS 通道结合，多失真类型上优于同期指标。方法：子带 NSS + 退化模型 + HVS 模型的信息比。url: http://live.ece.utexas.edu/publications/2004/hrs_imginfo_icassp_2004.pdf

### 主题 C：结构相似性（SSIM）谱系

**[B2-004] Wang, Bovik, Sheikh & Simoncelli (2004)** — *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP, 13(4). — VERIFIED；A；classic；seminal；FR。关键发现：HVS 高度适应提取结构信息；亮度×对比度×结构三分量局部乘积优于像素误差。方法：局部窗口 l·c·s 乘积 + 全局均值池化。url: https://www.docketalarm.com/cases/PTAB/IPR2025-01223/Amazon.com_Inc/docs/06-30-2025-Petitioner/Exhibit-1035-Exhibit_1035_Zhou_Wang_et_al,_Image_Quality_Assessment_From_Error_.pdf

**[B2-005] Wang, Simoncelli & Bovik (2003)** — *Multi-Scale Structural Similarity for Image Quality Assessment*, IEEE Asilomar. — VERIFIED；A；classic；seminal；FR。关键发现：多尺度加权融合对观看条件更鲁棒。方法：5 级高斯金字塔，对比度/结构跨尺度加权。url: https://videoclarity.com/PDF/MS-SSIMAsilomar.pdf

**[B2-006] Wang & Simoncelli (2005)** — *Translation Insensitive Image Similarity in Complex Wavelet Domain (CW-SSIM)*, IEEE ICASSP. — VERIFIED；B；high_citation；FR。关键发现：一致相位漂移不改变结构；对平移/小旋转鲁棒且免配准。方法：复小波域系数模与相位一致性比较。url: http://atc.umh.es/gatcom/bin/oqam/Referencias/Wang_Simoncelli_2005c.pdf
（注：另有 Sampat et al., IEEE TIP 2009 期刊扩展版。）

**[B2-007] Wang & Li (2011)** — *Information Content Weighting for Perceptual Image Quality Assessment*, IEEE TIP, 20(5). — VERIFIED；A；high_citation；FR。关键发现：最优池化权重应正比于局部信息含量；IW-MSSSIM 最佳，加权后 PSNR 也可具竞争力。方法：NSS 自信息作池化权重。url: http://ece.uwaterloo.ca/~z70wang/publications/InfoWeight_v3.pdf

### 主题 D：特征 / 梯度 / 显著性类

**[B2-008] Zhang, Zhang, Mou & Zhang (2011)** — *FSIM: A Feature Similarity Index for Image Quality Assessment*, IEEE TIP, 20(8). — VERIFIED；A；high_citation；seminal；FR。关键发现：相位一致性(主)+梯度幅值(辅)互补，6 库验证优于同期指标。方法：PC×GM 局部相似，PC 加权池化。作者勘误：第一作者 Lin Zhang（香港理工）。url: https://www4.comp.polyu.edu.hk/~cslzhang/paper/PCG_IQA.pdf

**[B2-009] Xue, Zhang, Mou & Bovik (2014)** — *Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index*, IEEE TIP, 23(2). — VERIFIED；A；high_citation；FR。关键发现：GMS 图的标准差（非均值）作池化，极低成本达到有竞争力精度。方法：Sobel 梯度→逐点 GMS→标准差池化。url: https://arxiv.org/pdf/1308.03052

**[B2-010] Zhang, Shen & Li (2014)** — *VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment*, IEEE TIP, 23(10). — VERIFIED；A；high_citation；FR。关键发现：显著性既作局部特征又作池化权重，4 库预测精度优于同期。方法：梯度+显著性图，显著性加权池化。url: https://cslinzhang.github.io/home/files/VSI.pdf

**[B2-011] Larson & Chandler (2010)** — *Most Apparent Distortion: Full-Reference Image Quality Assessment and the Role of Strategy*, Journal of Electronic Imaging, 19(1). — VERIFIED；B；high_citation；FR。关键发现：HVS 对近阈值失真用检测策略、对明显失真用外观策略，双策略融合。方法：掩蔽估计+空间频域局部统计，加权融合。url: https://openreview.net/forum?id=hW45VCeeKO

**[B2-012] Reisenhofer, Bosse, Kutyniok & Wiegand (2018)** — *A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment (HaarPSI)*, Signal Processing: Image Communication, 61. — VERIFIED；B；high_citation；FR。关键发现：Haar 小波系数足以度量局部感知相似，复杂度低、4 库相关度超 SSIM/FSIM/VSI。方法：Haar 小波分解→水平/垂直系数相似→加权平均+非线性校正。url: https://arxiv.org/pdf/1607.06140

### 主题 E：半参考（RR）

**[B2-013] Soundararajan & Bovik (2012)** — *RRED Indices: Reduced Reference Entropic Differencing for Image Quality Assessment*, IEEE TIP, 21(2). DOI: 10.1109/TIP.2011.2166082. — VERIFIED；A；high_citation；seminal（RR 路线）；RR。关键发现：小波系数熵差度量参考-失真信息变化，所需参考信息量从近乎全参考到单标量可调。方法：感知加权小波熵差家族（RRED-1/2 等）。url: https://dl.acm.org/doi/10.1109/TIP.2011.2166082

### 主题 F：学习型感知 FR（LPIPS/DISTS 谱系）

**[B2-014] Zhang, Isola, Efros, Shechtman & Wang (2018)** — *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)*, CVPR 2018. — VERIFIED；A；classic；seminal；FR。关键发现：BAPPS 上预训练深度特征线性加权距离大幅超过传统指标，跨架构/跨监督方式成立。方法：多层特征图通道归一化后按层加权 L2 距离。作者勘误：第一作者 Richard Zhang（UC Berkeley），非他人。url: https://arxiv.org/pdf/1801.03924

**[B2-015] Prashnani, Cai, Mostofi & Sen (2018)** — *PieAPP: Perceptual Image-Error Assessment through Pairwise Preference*, CVPR 2018. — VERIFIED；B；high_citation；FR。关键发现：成对偏好学习，推理时单张失真图+参考图输出感知误差，二分类错误率较此前 SOTA 降近 3 倍。方法：孪生 CNN 局部 patch 特征+成对比较网络+概率框架回归。url: https://web3.arxiv.org/pdf/1806.02067

**[B2-016] Ding, Ma, Wang & Simoncelli (2022)** — *Image Quality Assessment: Unifying Structure and Texture Similarity (DISTS)*, IEEE TPAMI, 44(7). — VERIFIED；A；high_citation；seminal；FR。关键发现：显式分离结构相似（特征图相关）与纹理相似（空间均值统计相关），容忍纹理重采样而保持结构敏感。方法：CNN 多尺度过完备嵌入+相关型距离，无监督纹理项+有监督主观分数联合优化。url: https://arxiv.org/pdf/2004.07728

**[B2-017] Ding, Liu, Zou, Wang & Ma (2021)** — *Locally Adaptive Structure and Texture Similarity for Image Quality Assessment (A-DISTS)*, ACM MM 2021. — VERIFIED；B；core_journal；FR。关键发现：DISTS 是全局度量；以 dispersion index 定位纹理区域并自适应池化，10 库提升相关性且免主观分数监督。方法：单一统计特征估计 patch 为纹理的概率，自适应加权局部结构/纹理测量。url: https://arxiv.org/pdf/2110.08521

**[B2-018] Liao, Chen, Zhu, Wang, Zhou & Kwong (2022)** — *DeepWSD: Projecting Degradations in Perceptual Space to Wasserstein Distance in Deep Feature Space*, ACM MM 2022. — VERIFIED；B；core_journal；FR。关键发现：从确定性特征距离转向分布视角，VGG 逐层 1D Wasserstein 距离对失真可解释性更好。方法：预训练 VGG 逐层特征分布内 W1 距离，层间加权聚合。url: https://arxiv.org/pdf/2208.03323

### 主题 G：2023—2026 新 FR/RR 指标（重点）

**[B2-019] Severo, Theis & Ballé (2024)** — *The Unreasonable Effectiveness of Linear Prediction as a Perceptual Metric (LASI)*, ICLR 2024. — VERIFIED；B；top_journal；FR。关键发现：无需训练数据/深度特征，仅靠推理时像素级 WLS 线性预测嵌入，性能与 LPIPS/PIM 竞争，成本近 MS-SSIM。方法：像素域加权最小二乘自回归嵌入，嵌入空间距离即相似度。url: https://proceedings.iclr.cc/paper_files/paper/2024/file/00358de35a101a372ea0412bed913c86-Paper-Conference.pdf

**[B2-020] Tian, Chen, Wang & Kwong (2023)** — *Towards Thousands to One Reference: Can We Trust the Reference Image for Quality Assessment? (FLRE)*, IEEE TMM, 25. — VERIFIED；B；core_journal；FR。关键发现：近阈值感知意味着同场景存在大量优质参考；在等质量特征空间搜索伪参考再做深度 FR 度量可显著提升性能。方法：PNTE 预测等质量 map + PRS 最小距离搜索最佳伪参考特征，作插件叠加现有深度 FR 模型。url: https://ieeexplore.ieee.org/document/10234614

**[B2-021] Tang, Chen, Li, Zhong, Zhang & Zhang (2023)** — *Unifying Dual-Attention and Siamese Transformer Network for Full-Reference Image Quality Assessment (UniDASTN)*, ACM TOMM, 19(5s). DOI: 10.1145/3597434. — VERIFIED；B；core_journal；FR。关键发现：孪生 Transformer 空间注意力+通道/空间双注意力可同时敏感传统失真与 GAN 失真。方法：孪生 Transformer 特征融合+MSE/双向 KL/排序联合损失。url: https://dl.acm.org/doi/pdf/10.1145/3597434

**[B2-022] Kim & Yim (2024)** — *Swin Transformer Fusion Network for Image Quality Assessment (STFN)*, IEEE Access, 12. — VERIFIED；C；FR。关键发现：融合 Swin Transformer 前两阶段特征，更小训练成本超过此前 SOTA FR 方法。方法：逆 patch merging 对齐两阶段尺寸+mediator block 融合+质量回归。url: https://xplorestaging.ieee.org/ielx7/6287639/10380310/10473038.pdf?arnumber=10473038

**[B2-023] He, Wang, Wang, Li, Fang, Sun & Ma (2024)** — *Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures (MS-SWD)*, arXiv:2407.10181（预印本，未证实录用）. — VERIFIED；C；FR。关键发现：多尺度 sliced Wasserstein 距离在非局部相似 patch 集合间比较，免训练、对图像错位鲁棒。方法：多尺度 patch 集合间 W2 距离，可作损失函数。url: https://web3.arxiv.org/pdf/2407.10181

**[B2-024] Yuan, Yang, Li, Cao, Che, Lin & Cao (2024)** — *PKU-AIGIQA-4K: A Perceptual Quality Assessment Database for Both Text-to-Image and Image-to-Image AI-Generated Images*, arXiv:2404.18409（投稿中）. — VERIFIED；C；FR。关键发现：AIGIQA 新范式——FR 参考图像=image prompt；提出 FR-AIGCIQA/PR-AIGCIQA 基准方法。方法：预训练模型提取 prompt 参考与待评图像特征并比较。url: https://arxiv.org/pdf/2404.18409

---

## 三、待确认 / 暂缓条目（pending）

| 条目 | 状态 | 说明 |
|---|---|---|
| VSNR (Chandler & Hemami, IEEE TIP 2007) | VERIFIED-deferred | 种子已核验；因 24 篇预算暂缓，综合阶段可按 HVS 早期需要补入 |
| RFSIM (Zhang, Zhang, Mou, ICIP 2010) | VERIFIED-deferred | 种子已核验；ICIP 会议版权威度低于 TIP |
| SR-SIM (Zhang & Li, ICIP 2012) | VERIFIED-deferred | 种子已核验；ICIP 会议版 |
| 图像与视频质量评价综述（程茹秋等, 中国图象图形学报 2022, 27(5):1410-1429） | VERIFIED-deferred | 目标期刊综述，DOI:10.11834/jig.210314，已读摘要+部分正文，强烈建议相关工作节引用 |
| PIM (Bhardwaj et al., NeurIPS 2020) | VERIFIED-deferred | 非种子；无监督信息论感知度量 |
| Understanding and Simplifying Perceptual Distances (Amir & Weiss, CVPR 2021) | VERIFIED-deferred | MMD 理论解释 |
| MIPM (Yao et al., Springer 2024) | VERIFIED-deferred | 错位鲁棒 FR 新指标 |
| CLIP-IQA (Wang, Chan, Loy, AAAI 2023) | out-of-scope | NR 零样本范式，不属 FR 分支 |
| NTIRE 2021 Challenge on Perceptual IQA (Gu et al., CVPRW 2021) | VERIFIED-deferred | PIPAL 背景 |

---

## 四、盲区记录

1. **视觉基础模型/VLM 作为 FR 特征骨干的公认新指标（2025—2026）**：未检索到代表作。CLIP 类工作（CLIP-IQA）均为 NR/零测范式，不属 FR 边界。若综合阶段需要"基础模型 FR"论述，建议在 B3（NR/学习型）分支联合补搜。
2. **RR-IQA 2023—2026 新指标**：除 RRED 谱系外，新 RR 指标（如 S4RR, Zhang/Phan/Chandler, SPIC 2020 前后）未达本分支收录线，记录于检索日志。
3. **C 级文献（B2-022/023/024）venue 未证实**：均为 arXiv/IEEE Access，正式投稿后应复核 DOI 与卷期。
4. **性能数字**：本清单全部为摘要层表述，未读全文，禁止在综述中直接引用具体 SROCC/PLCC 数值。
