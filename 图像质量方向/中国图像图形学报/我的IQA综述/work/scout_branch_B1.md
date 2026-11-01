# 文献检索清单 — 分支 B1：主观评价方法与质量数据库（自然图像/UGC）

- 分支：B1
- 分支名：主观评价方法与质量数据库（自然图像/UGC，不含 AIGC 数据库——AIGC 数据库由 B5 负责，本分支仅交叉提及）
- 检索日期：2026-10-20
- 检索工具：scholar_search（学术检索）+ web.fetch 备用
- 正式核验文献：26 篇（VERIFIED 26 / MINOR 0 / MAJOR 0 / UNVERIFIABLE 0 / PAYWALL 0）
- 2023—2026 年文献：8 篇（B1-019 ~ B1-026）
- 来源质量：A 级 16 篇 / B 级 9 篇 / C 级 1 篇
- seminal（奠基/高引经典）：B1-001、B1-002、B1-003、B1-004、B1-005、B1-006、B1-007、B1-008、B1-009、B1-010、B1-013、B1-015、B1-016、B1-018

---

## 一、检索式记录（全部逐条保留）

| # | 检索式 | 语言 | 用途 | 命中 | 有用 | 备注 |
|---|---|---|---|---|---|---|
| 1 | 图像质量 主观评价 数据库 | zh | 第零步原词直搜 | 6 | 5 | 命中 LIVEC/KonIQ/CID2013/Winkler |
| 2 | subjective image quality assessment database | en | 第零步原词直搜 | 6 | 5 | 命中 LIVE 统计评估/Waterloo/KonIQ |
| 3 | image quality assessment survey subjective methods MOS DMOS | en | 视角：综述方法论 | 6 | 3 | Mohammadi2014、Ma&Fang2021 |
| 4 | TID2013 Ponomarenko image quality database | en | 视角：经典库 | 6 | 3 | TID2013 期刊版+会议版 |
| 5 | CSIQ Categorical Image Quality database Larson Chandler | en | 视角：CSIQ 直搜 | 6 | 0 | 未命中，换关键词 |
| 6 | Most Apparent Distortion MAD Larson Chandler | en | 核验 CSIQ | 6 | 2 | CSIQ 伴随文献即 MAD 论文 |
| 7 | KADID-10k KADIS-700k Lin | en | 核验 KADID | 6 | 3 | QoMEX2019+DeepFL-IQA |
| 8 | LIVE multiply distorted Jayaraman | en | 核验 LIV-MD | 6 | 2 | Asilomar2012；附带 MDID2013 |
| 9 | SPAQ smartphone photography attribute quality | en | 视角：移动端首轮 | 6 | 3 | SCPQD2020、SQAD2023 |
| 10 | Perceptual quality assessment smartphone Fang CVPR | en | 核验 SPAQ 二轮 | 6 | 2 | 未直接命中 |
| 11 | SPAQ ... Fang 11125 images | en | 核验 SPAQ 三轮 | 6 | 4 | SPAQ 命中；附带 KonX/HRIQ |
| 12 | PIPAL perceptual image patch quality Gu | en | 核验 PIPAL | 6 | 2 | ECCV2020 |
| 13 | BAPPS learned perceptual image patch similarity Zhang | en | 核验 BAPPS | 6 | 2 | CVPR2018 LPIPS |
| 14 | Mantiuk pairwise comparison | en | 视角：成对比较 | 6 | 3 | CGF2012+配对比较指南 |
| 15 | crowdsourcing subjective image quality Ribeiro | en | 视角：众包 | 6 | 3 | crowdMOS ICIP2011 |
| 16 | TID2008 Ponomarenko evaluation objective metrics | en | 核验 TID2008 | 6 | 2 | EURASIP JIVP2009 |
| 17 | ITU-R BT.500 methodology subjective assessment | en | 视角：标准 | 6 | 2 | ITU 官方 BT.500-14(2019) |
| 18 | ITU-T P.910 subjective multimedia | en | 视角：标准 | 6 | 1 | ITU 官方 P.910 |
| 19 | VQEG multimedia test phase validation | en | 视角：VQEG | 6 | 2 | FRTV Phase II 报告(2003) |
| 20 | HDR image quality assessment database 2024 2025 | en | 视角：新库(HDR) | 6 | 4 | HDRC/TMHDR/HDR综述 |
| 21 | image quality database 2024 in-the-wild UGC blind | en | 视角：新库(in-the-wild) | 6 | 3 | CGIQA-6k/HRIQ/PaQ |
| 22 | image quality assessment database 2025 new benchmark | en | 视角：2025 查全 | 6 | 2 | VOS benchmark；AIGC 归 B5 |
| 23 | screen content image quality database SCTD | en | 视角：屏幕内容补盲 | 6 | 3 | SIQAD/SCD/IML-SCIQD |

## 二、检索饱和判据说明

满足 5 条中的 5 条，停止检索：
1. **覆盖达标**：方法论标准/众包/成对比较/经典合成失真库/真实 UGC 库/2023-2026 新库均 ≥2 篇。
2. **主题饱和**：合成失真、真实失真、HDR、移动端、CGI、跨分辨率、个体差异均有落位。
3. **引用闭环**：LIVE/TID/CSIQ/LIVEC 等奠基作均已捞入，后续工作共引关系闭合。
4. **时间跨度覆盖**：2006 奠基—2024 新库连续，无断层。
5. **新增衰减**：最后两轮新增有效文献占比 <10%。

---

## 三、正式文献清单（26 篇，经核验）

### 主题 A：主观实验方法论与标准

**[B1-001] ITU-R（2019），Recommendation ITU-R BT.500-14: Methodology for the subjective assessment of the quality of television pictures.** — 核验：VERIFIED；来源质量 A；authority_signal=official；角色：奠基工作；关键发现：主观电视图像质量评价实验方法的行业标准源头，规定观看条件/刺激方法/评分量表/数据处理。适合支撑：主观评价方法章节的标准依据。

**[B1-002] ITU-T（1996），Recommendation ITU-T P.910: Subjective video quality assessment methods for multimedia applications.** — 核验：VERIFIED；A；official；奠基工作；关键发现：规定多媒体应用 ACR 等主观质量评价方法，与 BT.500 并列为 MOS 评分范式双标准。适合支撑：MOS/DMOS 概念与评分量表。

**[B1-003] VQEG（2003），Final Report from the Video Quality Experts Group on the Validation of Objective Models of Video Quality Assessment, Phase II (FRTV).** — 核验：VERIFIED；A；official；奠基工作；关键发现：首次大规模跨实验室统一验证客观质量模型与主观数据关系，奠定 SROCC/PLCC 性能比较范式。适合支撑：性能评估范式、benchmark 方法。

**[B1-004] Mantiuk, Tomaszewska, Mantiuk（2012），Comparison of four subjective methods for image quality assessment, Computer Graphics Forum.** — 核验：VERIFIED；A；high_citation；奠基工作；关键发现：比较单刺激/双刺激/强制选择成对比较/相似性判断，结论成对比较测量方差最小、最准确。适合支撑：成对比较方法路线。

**[B1-005] Ribeiro, Florêncio, Nascimento（2011），Crowdsourcing subjective image quality evaluation, IEEE ICIP.** — 核验：VERIFIED；B；high_citation；奠基工作；关键发现：提出 crowdMOS 众包主观评价+恶意评分过滤机制，在 LIVE 数据上证明众包准确可重复。适合支撑：众包主观评价路线。

### 主题 B：经典合成失真质量数据库

**[B1-006] Sheikh, Sabir, Bovik（2006），A statistical evaluation of recent full reference image quality assessment algorithms, IEEE TIP.** — 核验：VERIFIED；A；high_citation；seminal；benchmark/dataset；关键发现：LIVE IQA 数据库出处，779 失真图×约 25000 次人类判断，当时除 VQEG 外最大主观图像质量研究。适合支撑：FR-IQA 主观基准奠基。

**[B1-007] Ponomarenko et al.（2009），TID2008 – A database for evaluation of full-reference visual quality assessment metrics.** — 核验：VERIFIED；A；high_citation；seminal；benchmark/dataset；关键发现：TID2008 库，1700 图（25 参考×17 失真×4 等级），三国 800+ 实验约 25.6 万次判断得 MOS。适合支撑：合成失真基准。

**[B1-008] Ponomarenko et al.（2015），Image database TID2013: Peculiarities, results and perspectives, Signal Processing: Image Communication.** — 核验：VERIFIED；A；high_citation；seminal；benchmark/dataset；关键发现：TID2013 库，3000 图（25 参考×24 失真×5 等级），五国 985 名观察者收集 MOS。适合支撑：大规模合成失真基准。

**[B1-009] Larson & Chandler（2010），Most apparent distortion: full-reference image quality assessment and the role of strategy, J. Electronic Imaging.** — 核验：VERIFIED；A；high_citation；seminal；benchmark/dataset；关键发现：CSIQ 数据库伴随文献，提出 HVS 检测式/外观式双策略模型。适合支撑：合成失真基准+HVS 策略模型。

**[B1-010] Jayaraman, Mittal, Moorthy, Bovik（2012），Objective image quality assessment of multiply distorted images, Asilomar.** — 核验：VERIFIED；B；high_citation；seminal；benchmark/dataset；关键发现：LIVE Multiply Distorted (LIV-MD) 库出处，首次系统提供多重失真叠加主观数据。适合支撑：多重失真基准。

**[B1-011] Ma et al.（2017），Waterloo Exploration Database: New challenges for image quality assessment models, IEEE TIP.** — 核验：VERIFIED；A；high_citation；benchmark/dataset；关键发现：4744 原始图+94880 失真图，提出免主观评分的 D-test/L-test/P-test 三准则。适合支撑：大规模内容多样性基准+无 MOS 测试范式。

**[B1-012] Lin, Hosu, Saupe（2019），KADID-10k: A large-scale artificially distorted IQA database, QoMEX.** — 核验：VERIFIED；A；high_citation；benchmark/dataset；关键发现：KADID-10k（81 原始×25 失真×5 等级，30 人众包 MOS）+KADIS-700k（14 万原始图×25 失真无标注）。适合支撑：深度学习时代合成失真基准。

### 主题 C：真实/UGC 失真质量数据库

**[B1-013] Ghadiyaram & Bovik（2017），Massive online crowdsourced study of subjective and objective picture quality, IEEE TIP.** — 核验：VERIFIED；A；high_citation；seminal；benchmark/dataset；关键发现：LIVE In-the-Wild (LIVEC) 库，1162 幅移动设备实拍真实失真图，8100+ 观察者众包 35 万+评分。适合支撑：首个大规模真实/UGC 主观数据库、在线众包范式标杆。

**[B1-014] Virtanen et al.（2015），CID2013: A database for evaluating no-reference image quality assessment algorithms, IEEE TIP.** — 核验：VERIFIED；A；high_citation；benchmark/dataset；关键发现：CID2013 库，480 图×79 相机设备，188 受试者；混合绝对类别评级对比较法，附清晰度/颗粒度/亮度/饱和度分项评分。适合支撑：真实相机多重失真 NR-IQA 基准。

**[B1-015] Hosu, Lin, Sziranyi, Saupe（2020），KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment, IEEE TIP.** — 核验：VERIFIED；A；high_citation；seminal；benchmark/dataset；关键发现：KonIQ-10k，10073 图×1459 众包工作者 120 万评分，首个生态效度 in-the-wild 库；KonCept512 在 LIVEC 上 SROCC 0.825（摘要层）。适合支撑：深度 BIQA 时代 in-the-wild 基准。

**[B1-016] Fang, Zhu, Zeng, Ma, Wang（2020），Perceptual quality assessment of smartphone photography, CVPR.** — 核验：VERIFIED；A；top_journal；seminal；benchmark/dataset；关键发现：SPAQ 库，11125 图×66 款手机，附质量+亮度/色彩/对比度/噪声/清晰度 5 属性+场景类别+EXIF。适合支撑：智能手机摄影质量、多属性标注范式。

**[B1-017] Gu et al.（2020），PIPAL: A large-scale image quality assessment dataset for perceptual image restoration, ECCV.** — 核验：VERIFIED；A；top_journal；benchmark/dataset；关键发现：PIPAL 数据集，含 GAN-based 恢复输出，113 万+人类判断，Elo 成对比较评分；现有 IQA 方法无法公平评价 GAN 恢复算法。适合支撑：感知图像恢复质量基准。

**[B1-018] Zhang, Isola, Efros, Shechtman, Wang（2018），The unreasonable effectiveness of deep features as a perceptual metric, CVPR.** — 核验：VERIFIED；A；high_citation；seminal；benchmark/dataset；关键发现：构建 BAPPS 人类感知相似性判断数据集，深度特征大幅超越经典度量。适合支撑：感知相似性/偏好数据集路线。

### 主题 D：2023—2026 新数据库与新方法

**[B1-019] Zhang Z. et al.（2023），Subjective and objective quality assessment for in-the-wild computer graphics images (CGIQA-6k), arXiv:2303.08050.** — 核验：VERIFIED；B；core_journal；最新进展；benchmark/dataset；关键发现：CGIQA-6k，6000 幅 in-the-wild CGI 受控实验室主观评分。适合支撑：CGI 专项主观质量数据库新方向。

**[B1-020] Wiedemann, Hosu, Su, Saupe（2023），KonX: Cross-resolution image quality assessment, Springer VCIBA.** — 核验：VERIFIED；B；core_journal；最新进展；benchmark/dataset；关键发现：KonX 跨分辨率库，首次揭示分辨率变化引起主观标签漂移与客观方法尺度偏置。适合支撑：分辨率敏感性新问题。

**[B1-021] Fang, Ignatov, Zamfir, Timofte（2023），SQAD: Automatic smartphone camera quality assessment and benchmarking, ICCV.** — 核验：VERIFIED；A；top_journal；最新进展；benchmark/dataset；关键发现：SQAD 库，29 款设备，六维度相机质量准则（分辨率/色彩/噪声/动态范围/PSF/混叠）。适合支撑：相机系统质量基准新方向。

**[B1-022] Liu, Ni, Chen, Wang, Kwong（2024a），HDRC: A subjective quality assessment database for compressed high dynamic range image, IJMLC.** — 核验：VERIFIED；B；core_journal；最新进展；benchmark/dataset；关键发现：HDRC 库，80 参考+400 幅 VVC 压缩失真 HDR 图，首个纳入 VVC 的 HDR-IQA 主观库。适合支撑：HDR 压缩质量评价新基准。

**[B1-023] Huang, Wan, Korhonen（2024），High resolution image quality database (HRIQ), arXiv:2401.16087.** — 核验：VERIFIED（预印本，source_quality=C）；最新进展；benchmark/dataset；关键发现：HRIQ 库，1120 幅 2880×2160 高分辨率图，实验室 MOS；低分辨率训练模型在高分辨率上精度受限。适合支撑：UHD/高分辨率 BIQA 新基准。

**[B1-024] Venkataramanan & Bovik（2024），Subjective quality assessment of compressed tone-mapped high dynamic range videos (LIVE-TMHDR), IEEE TIP.** — 核验：VERIFIED；A；top_journal；最新进展；benchmark/dataset；关键发现：LIVE-TMHDR 库，15000 条色调映射压缩序列，1600+观察者 75 万+标注。适合支撑：HDR 色调映射质量评价新库（视频方向，交叉参考）。

**[B1-025] Liu, Tian, Wang, Zhang, Kwong（2024b），Overview of high-dynamic-range image quality assessment, J. Imaging.** — 核验：VERIFIED；B；core_journal；综述；关键发现：综述 7 个主观 HDR-IQA 库+12 个客观度量，指出数据库规模多样性与通用感知映射两大改进方向。适合支撑：HDR 质量评价综述入口。

**[B1-026] Lou et al.（2024），A benchmark of variance of opinion scores in image quality assessment, ICIP.** — 核验：VERIFIED；B；core_journal；最新进展；方法论/基准；关键发现：提出观点分数方差(VOS)基准，研究 MOS 之外的观察者间个体差异。适合支撑：主观评分 beyond-MOS 的新视角。

---

## 四、待确认（pending）

无（本轮无 UNVERIFIABLE/PAYWALL 灰区文献进入待确认）。

## 五、补充核验文献（VERIFIED，因 26 篇名额未入正式清单）

| 文献 | 说明 |
|---|---|
| Winkler（2012）Analysis of public image and video databases for quality assessment, IEEE JSTSP | 数据库比较分析综述，建议综合阶段优先补入 |
| Ying et al.（2020）From Patches to Pictures (PaQ-2-PiQ), CVPR | 约 4 万真实图+12 万 patch+400 万判断大规模主观库，建议补入 |
| Pérez-Ortiz & Mantiuk（2017）配对比较数据分析指南, arXiv:1712.03686 | 方法论背景补充 |
| Mohammadi et al.（2014）Subjective and objective quality assessment of image: A survey | 综述背景补充 |
| Zhu et al.（2020）SCPQD2020, ICME Grand Challenge | 手机摄影多属性库背景补充 |
| He et al.（2020）Rethinking image aesthetics assessment (TAD66K), IEEE TIP | **勘误**：TAD66K 是美学数据集而非质量数据集 |
| Mantiuk et al.（2024）ColorVideoVDP (含 XR-DAVID), SIGGRAPH | 视频/显示方向交叉参考 |

## 六、AIGC 数据库交叉提及（归 B5，不展开）

检索中出现的 AIGC 质量数据库：AGIQA-1K、AGIQA-3K、AIGCIQA2023、PKU-AIGIQA-4K（Yuan et al., arXiv:2404.18409）等——按分工归 B5 负责，本分支仅在此列名交叉引用，不展开核验。

## 七、作者/归属勘误与争议

1. **TAD66K 归属勘误**：任务书将 TAD66K 列为"真实/UGC 图像质量数据库（Jiang 等，需核验）"。经核验，TAD66K 出自 He, Zhang, Xie, **Jiang**, Ming（北邮）2020 年 IEEE TIP 论文 *Rethinking Image Aesthetics Assessment*，是**图像美学（aesthetics）评价数据集**（47 主题、1200+人主题级标注），而非图像感知**质量（quality）**评价数据库。综述撰写时不应将其作为 IQA 质量数据库引用。
2. **CSIQ 数据库出处**：CSIQ（Categorical Image Quality）数据库的正式伴随文献是 Larson & Chandler 的 MAD 论文（J. Electronic Imaging 2010），数据库名"CSIQ"未出现在论文标题中——常见引用时易混淆为另一篇。
3. **KADIS-700k 出处**：KADIS-700k（无标注大规模合成失真集）的正式描述在 Lin/Hosu/Saupe 的 *DeepFL-IQA*（IEEE TMM）中，KADID-10k 的正式描述在 QoMEX 2019 论文中，两者应分开引用。
