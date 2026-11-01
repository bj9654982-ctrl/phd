# B6 分支清单：综述文献、评价协议、中文期刊工作与期刊投稿要求

> 检索人：Organizer（B6 路由）；检索日期 2026-10-20；检索工具 scholar_search（学术）+ general_search/web.fetch（期刊要求）。
> 本分支聚焦"综述/元方法"层面：IQA 综述（英文/中文）、评价协议与大规模比较方法学、目标期刊《中国图象图形学报》的前序 IQA 综述（用于定位），以及期刊投稿要求（另见 `journal_requirements.md`）。
> 状态标记：VERIFIED（标题/作者/期刊/年份多源一致）、MINOR（一项元数据待终稿核对）；C 级为 arXiv 纯预印本，不入核心证据。

## 一、检索式记录（全部 15 条，含期刊要求检索）

1. `perceptual image quality assessment survey`（英文综述种子直搜）
2. `image quality assessment deep learning survey 2024`（近年综述）
3. `AI-generated content image quality assessment survey`（AIGC 质量评价综述）
4. `statistical evaluation full reference image quality assessment algorithms Sheikh Bovik`（评价协议奠基）
5. `group MAD competition database full reference image quality assessment Ma`（gMAD 方法学）
6. `图像质量评价 综述 中国图象图形学报 深度学习`（中文综述/目标期刊前序）
7. `Automatic prediction of perceptual image and video quality Bovik Proceedings of the IEEE`（权威综述）
8. `multimodal large language model image quality assessment survey 2025 low-level vision`（MLLM 评价研究，交叉 B4/B5）
9. `无参考图像质量评价研究进展 方玉明 眭相杰 中国图象图形学报`（元数据核验）
10. `图像质量评价研究综述 从失真的角度 鄢杰斌 方玉明`（元数据核验）
11. `高敏娟 党宏社 全参考图像质量评价回顾与展望 电子学报 年 卷 期`（元数据核验）
12. `Manap Shao Non-distortion-specific no-reference image quality assessment survey Information Sciences year volume`（元数据核验）
13. general_search：`中国图象图形学报 投稿指南 综述 字数 参考文献 GB/T 7714 cjig.cn`
14. general_search：`中国图象图形学报 投稿模板 摘要 关键词 中图分类号 英文摘要 要求`
15. general_search：`中国图象图形学报 EI Compendex 收录 中文核心 CSCD 2025`

## 二、饱和判据（5 条核对）

- 第零步原词直搜：完成（"perceptual image quality assessment survey"、中文"图像质量评价 综述"）。
- 多视角：覆盖英文全景综述（4 篇）、NR 专题综述（2 篇）、Transformer 综述（1 篇）、AIGC/T2I 综述（3 篇）、中文综述（5 篇，含目标期刊 3 篇）、评价协议（3 篇）。
- 连续 3 轮新 query 无新增独立条目：第 10—12 轮仅核验元数据，无新文献，判定饱和。
- 目标期刊前序综述已查全：学报 2021 年 1 篇（方玉明，NR）、2022 年 2 篇（鄢杰斌、程茹秋），近年无更新的全景 IQA 综述（说明本综述定位窗口成立）。
- 交叉去重：Q-Bench+、MA-AGIQA、SC-AGIQA 为 MLLM/AIGC 方法论文，不属 B6 综述层，已移交 B4/B5 去重收录。

## 三、纳入清单（20 条）

### A. 英文综述（9 条）

| 编号 | 文献 | 年份 | 出处 | 等级 | 状态 |
|---|---|---|---|---|---|
| B6-001 | Lin & Kuo, Perceptual visual quality metrics: A survey | 2011 | JVCIR 22(4):297-312 | A | VERIFIED |
| B6-002 | Zhai & Min, Perceptual image quality assessment: a survey | 2020 | Science China Information Sciences 63(11) | A | VERIFIED |
| B6-003 | Manap & Shao, Non-distortion-specific NR-IQA: A survey | 2015 | Information Sciences 301:141-160 | A | VERIFIED |
| B6-004 | Rehman et al., IQA Vision Transformed: A Survey of Transformer Architectures in IQA | 2024 | IEEE Access | B | VERIFIED |
| B6-005 | Ma et al., A Survey on IQA: Insights, Analysis, and Future Outlook | 2025 | arXiv 2502.08540 | C | VERIFIED（预印本） |
| B6-006 | Wang, Blind Image Quality Assessment: A Brief Survey | 2023 | arXiv 2312.16551 | C | VERIFIED（预印本） |
| B6-007 | Tian et al., Quality Assessment for Text-to-Image Generation: A Survey | 2025 | IEEE Multimedia（早期访问） | A | MINOR（卷期待核） |
| B6-008 | Hartwig et al., A Survey on Quality Metrics for T2I Generation | 2024 | arXiv 2403.11821 | C | VERIFIED（预印本） |
| B6-009 | Ghildyal et al., Quality Prediction of AI Generated Images and Videos | 2024 | arXiv 2410.08534 | C | VERIFIED（预印本） |
| B6-019 | Xu et al., No-reference/blind IQA: a survey | 2017 | IETE Technical Review 34(3) | B | VERIFIED |
| B6-020 | Liu et al., Visual quality assessment: recent developments, coding applications and future trends | 2013 | APSIPA Trans. SIP 2:e4 | B | MINOR（DOI 待核） |

### B. 中文综述（5 条，其中 3 篇为目标期刊前序综述）

| 编号 | 文献 | 年份 | 出处 | 等级 | 状态 |
|---|---|---|---|---|---|
| B6-010 | 王志明，无参考图像质量评价综述 | 2015 | 自动化学报（DOI 10.16383/j.aas.2015.c140404） | A | MINOR（卷期页待核） |
| B6-011 | 方玉明等，无参考图像质量评价研究进展 | 2021 | 中国图象图形学报 26(2):265-286 | A | VERIFIED |
| B6-012 | 鄢杰斌等，图像质量评价研究综述——从失真的角度 | 2022 | 中国图象图形学报 27(5):1430-1466 | A | VERIFIED |
| B6-013 | 程茹秋等，图像与视频质量评价综述 | 2022 | 中国图象图形学报 27(5):1410-1429 | A | VERIFIED |
| B6-014 | 高敏娟等，全参考图像质量评价回顾与展望 | 2021 | 电子学报 49(11):2261-2272 | A | VERIFIED |

### C. 评价协议与大规模比较（3 条）

| 编号 | 文献 | 年份 | 出处 | 等级 | 状态 |
|---|---|---|---|---|---|
| B6-015 | Bovik, Automatic Prediction of Perceptual Image and Video Quality | 2013 | Proceedings of the IEEE 101(9) | A | VERIFIED |
| B6-016 | Sheikh, Sabir & Bovik, A Statistical Evaluation of Recent FR IQA Algorithms（LIVE 库） | 2006 | IEEE TIP 15(11) | A | VERIFIED |
| B6-017 | Ma et al., Group MAD Competition（gMAD，Waterloo Exploration 库） | 2016 | CVPR 2016 | A | VERIFIED |
| B6-018 | Zhang et al., A Comprehensive Evaluation of FR IQA Algorithms（7 库 11 指标横评） | 2012 | ICIP 2012 | B | VERIFIED |

## 四、对本综述定位的关键判断

1. **目标期刊存在直接对标的前序综述，但窗口明确**：方玉明等（2021）覆盖 2012—2020 NR-IQA；鄢杰斌等（2022）覆盖 2011—2021 并按失真类型组织；程茹秋等（2022）含视频。三篇均止步于 2021 年前后，**Transformer NR-IQA（2022—2023）、CLIP/MLLM 打分（2023—2024）、AIGC 质量评价（2023—2026）三大新板块未覆盖**——这正是本综述的增量定位，大纲"引言"中应明确引用并对比这三篇。
2. **英文综述可分三层引用**：Lin & Kuo（2011）、Zhai & Min（2020）、Bovik（2013）作为全景/经典综述；Manap & Shao（2015）、Xu et al.（2017）、Wang（2023）作为 NR 专题；Tian et al.（2025）、Hartwig et al.（2024）、Ghildyal et al.（2024）作为 AIGC 专题。
3. **评价协议层的三块基石**：Sheikh et al.（2006）确立 SROCC/PLCC 统计评测与 LIVE 库；Ma et al.（2016）gMAD 提供"对抗式选样"比较方法学与 Waterloo 大规模库；Zhang et al.（2012）提供多库横评范式。大纲"评价协议"节据此组织。
4. **预印本纪律**：B6-005/006/008/009 为 arXiv 预印本，仅作脉络参考；按期刊"参考文献须为正式出版物、全部提供 DOI"要求，终稿前须替换为正式录用版或降级处理。

## 五、移交 B4/B5 的交叉命中（不在 B6 计数）

- Q-Bench+（Zhang et al., arXiv 2402.07116）：MLLM 低层视觉基准，含 24 个 MLLM 与 7 个 IQA 库打分评测 → B4 收录。
- MA-AGIQA（Wang Puyi et al., ACM MM 2024, arXiv 2404.17762）：大模型辅助 AIGC 质量评价，MoE 融合语义与质量特征 → B5 收录。
- SC-AGIQA（Li Qiang et al., arXiv 2507.10432）：文本-视觉语义约束 AIGC 评价 → B5 收录。
- MLLM-IQA prompting 综合研究（Wu Tianhe, Ma Kede et al., arXiv 2403.10854）：MLLM 打分 prompting 系统研究 → B4 收录。
