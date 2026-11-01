# 文献检索清单 — 分支 B5：AIGC 生成图像的质量评价

> 生成时间：2026-10-20 ｜ 检索工具：scholar_search（+ web.fetch 精读 arXiv 摘要页）
> 核验标准：存在性 + 引述准确性（仅按摘要写）+ 权威性三级判定；纯预印本 source_quality=C 并注明。

## 一、检索式记录（宽→窄）

| 轮次 | 检索式 | 目的 |
|---|---|---|
| 第零步-中 | `AI生成图像 质量评价 数据库` | 原词直搜 |
| 第零步-英 | `AI-generated image quality assessment` | 原词直搜 |
| 第零步-英 | `text-to-image generation quality evaluation benchmark` | 原词直搜 |
| 种子 | `AGIQA-1K database AI-generated image quality assessment` | 库核验 |
| 种子 | `AIGCIQA2023 database AI-generated image quality assessment subjective` | 库核验 |
| 种子 | `ImageReward learning to evaluate text-to-image alignment human preferences Xu` | 奖励模型核验 |
| 种子 | `Human Preference Score v2 HPSv2 Wu text-to-image` | 奖励模型核验 |
| 种子 | `Pick-a-Pic open dataset preferences text-to-image Kirstain NeurIPS` | 奖励模型核验 |
| 种子 | `NTIRE 2024 Quality Assessment of AI-Generated Images challenge report` | 挑战赛核验 |
| 视角1 | `T2I-CompBench comprehensive benchmark text-to-image compositional Huang NeurIPS 2023` | 综合benchmark |
| 视角1 | `EvalCrafter comprehensive evaluator text-to-video Li 2023` | 综合benchmark（勘误：实为视频） |
| 视角1 | `VQAScore evaluating text-to-image generation via VQA 2024` | 对齐指标 |
| 视角2 | `GeoTrp geometric transformation AI-generated image quality assessment` / `"GeoTrp"` | 库核验（未命中） |
| 视角2 | `MP-IQDF multi-perspective AI-generated image quality assessment database` | 库核验（未命中） |
| 视角2 | `EPAIQA local editing AI-generated image quality assessment 2025` | 新库核验 |
| 视角3 | `geometric transformation no-reference AI-generated image quality assessment 2024 2025` | 补盲（GeoTrp 二轮） |
| 视角3 | `LAION aesthetic predictor v2 improved aesthetic score Schuhmann` | 美学打分核验 |
| 视角3 | `DiffusionGIQA diffusion generated image quality assessment database` | 存在性（不存在） |
| 视角4 | `T2IQA-1M text-to-image quality assessment database one million` | 存在性（不存在） |
| 视角4 | `EvalMuse AesBench text-to-image generation evaluation benchmark aesthetic` | 新benchmark核验 |
| 视角4 | `multi-preference score MPS text-to-image reward model` | MPS核验 |
| 视角4 | `AI-enhanced UGC image quality assessment AU-IQA ACM Multimedia 2025` | AU-IQA核验 |
| venue | web.fetch `https://arxiv.org/abs/2303.12618` | AGIQA-1K 版本信息 |

## 二、饱和判据判定

- 覆盖达标：AIGC 专用库 / 偏好奖励模型 / T2I 综合 benchmark / MLLM 评判 / 早期 GAN 奠基，5 个子方向均 ≥3 篇 ✅
- 新增衰减：最近 2 轮（视角3/4 补盲）新增有效文献 <10% ✅
- 主题饱和：种子清单中所有"必收"条目均已落位或记 dead end ✅
- 时间覆盖：2020（GIQA）—2022（LAION-5B）—2023—2024—2025 无断层 ✅
→ 停止检索。

---

## 三、正式文献清单（共 32 篇，经核验）

### 主题 A：早期 GAN 生成图像质量评价（奠基）

1. **Gu, Bao, Chen, Wen (2020)** "GIQA: Generated Image Quality Assessment," arXiv 2003.08932. — 核验：VERIFIED；角色：**奠基工作**；发现：首次把"单张生成图质量"作为独立课题，给出 learning-based/data-based 两类算法，可分离 realism 与 diversity；适合支撑：问题提出（生成式IQA与重建式IQA的分野起点）。

### 主题 B：AIGC 专用主观质量数据库

2. **Zhang Z. 等 (2023)** "A Perceptual Quality Assessment Exploration for AIGC Images"（**AGIQA-1K**）, arXiv 2303.12618（IEEE Xplore:10222021）. — VERIFIED；**seminal/库**；首个扩散模型AIGC感知质量库（1080图，MOS单维）。
3. **Li C. 等 (2023)** "**AGIQA-3K**: An Open Database for AI-Generated Image Quality Assessment," arXiv 2306.04717. — VERIFIED；**seminal/库**；感知质量+文生图对齐双维细粒度MOS；提出StairReward。
4. **Wang J. 等 (2023)** "**AIGCIQA2023**: A Large-scale IQA Database for AI Generated Images," arXiv 2307.00211. — VERIFIED；**seminal/库**；质量/真实性/对应性三视角（2000+图，6模型×100 prompt）。
5. **Wang J. 等 (2024)** "Quality Assessment for AI Generated Images with Instruction Tuning"（**AIGCIQA2023+ / MINT-IQA**）, arXiv 2405.07346. — VERIFIED；代表路线；偏好解释文本+指令微调多视角打分。
6. **Yuan J. 等 (2023)** "**PKU-I2IQA**: An Image-to-Image Quality Assessment Database," arXiv 2311.15556. — VERIFIED；库；首个图生图（I2I）AIGC质量库，含NR/FR双基线。
7. **Yuan J. 等 (2024)** "**PKU-AIGIQA-4K**," arXiv 2404.18409. — VERIFIED；库；T2I+I2I混合场景，NR/FR/PR三方法。
8. **Li C. 等 (2024)** "**AIGIQA-20K**: A Large Database for AI-Generated Image Quality Assessment," arXiv 2404.03407. — VERIFIED；库；20000图/420000评分，15模型，NTIRE2024指定用库。**勘误：父任务种子名"AIGCIQA-20K"有误，实际为 AIGIQA-20K。**
9. **Li C. 等 (2024)** "**NTIRE 2024 Quality Assessment of AI-Generated Content Challenge**," CVPRW 2024 (arXiv 2404.16687). — VERIFIED；**挑战赛**；图像赛道（318注册/16队）+视频赛道。
10. **Chen Z. 等 (2023)** "Study of Subjective and Objective Naturalness Assessment of AI-Generated Images"（**AGIN / JOINT**）, arXiv 2312.05476（IEEE TCSVT）. — VERIFIED；库；6049图/907350评价；整体自然度≈0.145×技术分+0.769×理性分（摘要层）。
11. **Tian Y. 等 (2024)** "AI-generated Image Quality Assessment in Visual Communication"（**AIGI-VC**）, arXiv 2412.15677. — VERIFIED；库；广告视觉传播场景（2500图，14主题×8情绪）。
12. **Qian J. 等 (2025)** "Towards Explainable Partial-AIGC Image Quality Assessment"（**EPAIQA-15K**）, arXiv 2504.09291. — VERIFIED；库；局部AI编辑图首个大规模库（15000图/300K评分），LMM三阶段：定位→打分→解释。
13. **Wang S. 等 (2025)** "**AU-IQA**: Benchmark Dataset for AI-Enhanced UGC," ACM MM 2025 (arXiv 2508.05016). — VERIFIED；库；AI增强（超分/低照/去噪）UGC，4800图。
14. **Wang J. 等 (2025)** "**LMM4LMM** / EvalMi-50K," arXiv 2504.08358. — VERIFIED；库；50400图/24模型/100K MOS/50K QA，MLLM多维评测指标。
15. **Qu B. 等 (2024)** "Bringing Textual Prompt to AI-Generated Image Quality Assessment"（**IP-IQA**）, arXiv 2403.18714. — VERIFIED；代表路线；指出MOS耦合图-文对应关系，图像+prompt多模态在AGIQA-1K/3K上SOTA。

### 主题 C：人类偏好数据集与奖励模型

16. **Xu J. 等 (2023)** "**ImageReward**: Learning and Evaluating Human Preferences for T2I," NeurIPS 2023. — VERIFIED；**seminal/奖励模型**；137K专家成对比较；ReFL对齐算法。
17. **Wu X. 等 (2023)** "**HPS v2**," arXiv 2306.09341（v1: arXiv 2303.14420）. — VERIFIED；**seminal/奖励模型**；HPD v2 798090偏好选择，CLIP微调打分模型。
18. **Ma Y. 等 (2025)** "**HPSv3**: Towards Wide-Spectrum Human Preference Score," arXiv 2508.03789. — VERIFIED；奖励模型；HPDv3 1.08M对/1.17M比较，VLM+不确定性排序损失，CoHP精修。
19. **Kirstain Y. 等 (2023)** "**Pick-a-Pic** / PickScore," NeurIPS 2023 D&B. — VERIFIED；**seminal/奖励模型**；开放真实用户偏好数据集，PickScore超人类预测偏好。
20. **Zhang S. 等 (2024)** "Learning Multi-dimensional Human Preference for T2I"（**MPS / MHP**）, CVPR 2024. — VERIFIED；奖励模型；918315四维（美学/语义对齐/细节/总体）偏好。
21. **Wu X. 等 (2024)** "MLLMs Make T2I Align Better"（**VisionPrefer / VP-Score**）, NeurIPS 2024. — VERIFIED；奖励模型；MLLM合成1.2M四维偏好数据。

### 主题 D：文生图/文生视觉综合评测 benchmark 与对齐指标

22. **Lin Z. 等 (2024)** "Evaluating Text-to-Visual Generation with Image-to-Text Generation"（**VQAScore**）, arXiv 2404.01291. — VERIFIED；代表路线；VQA "Yes"概率对齐分，8基准SOTA。
23. **Li B. 等 (2024)** "**GenAI-Bench**（含GenAI-Rank）," arXiv 2406.13743. — VERIFIED；benchmark；1600组合prompt/15000人评；VQAScore排序选图优于PickScore/HPSv2/ImageReward 2-3倍。
24. **Hu Y. 等 (2023)** "**TIFA**," ICCV 2023. — VERIFIED；benchmark；QA对式忠实度评测，计数/空间关系/组合是薄弱环节。
25. **Huang K. 等 (2023)** "**T2I-CompBench**," NeurIPS 2023 D&B. — VERIFIED；**seminal/benchmark**；6000组合prompt，属性绑定/关系/复杂组合三类。
26. **Ku M. 等 (2024)** "**ImagenHub**," ICLR 2024. — VERIFIED；benchmark；**传统自动指标与人类Spearman<0.2的关键失效证据**；30模型统一评测。
27. **Lee T. 等 (2023)** "**HEIM**," NeurIPS 2023 D&B. — VERIFIED；benchmark；12维62场景评测26个T2I模型。
28. **Ku M. 等 (2024)** "**VIEScore**," ACL 2024. — VERIFIED；代表路线；免训练MLLM评测，GPT-4o与人评Spearman≈0.4（人-人0.45，摘要层）。
29. **Huang Y. 等 (2024)** "**AesBench**," arXiv 2401.08276. — VERIFIED；benchmark；专家标注美学感知EAPD库，MLLM美学能力4维评测。
30. **Schuhmann C. 等 (2022)** "**LAION-5B**," NeurIPS 2022 D&B. — VERIFIED；背景数据库；LAION-Aesthetics子集与LAP美学预测器的数据生态出处。

### 主题 E：综述与立场文献

31. **Tian Y., Liu Y., Wang S., Kwong S. (2025)** "Quality Assessment for Text-to-Image Generation: A Survey," IEEE Multimedia. — VERIFIED；综述；主客观路线+数据集/指标全景。
32. **Ghildyal A. 等 (2024，含 Bovik)** "Quality Prediction of AI Generated Images and Videos: Emerging Trends and Opportunities," arXiv 2410.08534. — VERIFIED；局限证据；传统IQA/VQA面向"重建保真"而非"生成伪影"。

## 四、灰区 / dead-end 记录（不进正式清单）

- **GeoTrp**：精确名与多组近义词检索均无命中 → 疑似名称误传，灰区不使用。
- **MP-IQDF**：同上，未检索到对应文献。
- **DiffusionGIQA / DiffusionGIQA-23K、T2IQA / T2IQA-1M**：检索确认不存在此名称文献。
- **EvalCrafter**：实 Liu Yaofang 等（arXiv 2310.11440），对象是**文生视频**，非文生图——父任务描述有误。
- 相邻未纳入：T2I-CompBench++(2307.06350)、WISE(2503.07265)、MMIG-Bench(2505.19415)、QMI-Net/AIGI-IQAD(IEEE TMM)、FaceQ/F-Bench(2412.13155)、DiffDoctor(2501.12382)、LLaVA-Reward(2507.21391)、cFreD(2503.21721)、EvalMuse-40K(2412.18150)。

## 五、易混数据库勘误对照

| 名称 | 第一作者/团队 | 年份 | venue | 规模与维度 |
|---|---|---|---|---|
| **AGIQA-1K** | Zhang Zicheng / Zhai Guangtao（上海交大） | 2023 | arXiv 2303.12618（IEEE Xplore:10222021） | 1080图，**仅感知质量**单维 |
| **AGIQA-3K** | Li Chunyi / Zhai Guangtao（上海交大） | 2023 | arXiv 2306.04717 | 感知质量 **+ 文生图对齐**双维细粒度MOS，含StairReward |
| **AIGCIQA2023** | Wang Jiarui / Zhai Guangtao（上海交大） | 2023 | arXiv 2307.00211 | 2000+图，**质量/真实性/对应性三视角** |
| **AIGCIQA2023+** | Wang Jiarui 等 | 2024 | arXiv 2405.07346 | 在上者上加**偏好解释文本**，MINT-IQA |
| **AIGIQA-20K**（父任务误写"AIGCIQA-20K"） | Li Chunyi / Zhai Guangtao（上海交大） | 2024 | arXiv 2404.03407，NTIRE2024用库 | 20000图/420000评分，15模型 |
