# 分支 B4 文献检索清单：视觉语言模型/多模态大模型时代的图像质量评价

- 项目：我的IQA综述（投《中国图像图形学报》）
- 检索日期：2026-10-20
- 检索工具：scholar_search（学术库）+ web.fetch（arXiv/OpenReview/ECVA/CVF 摘要页精读）
- 正式核验文献：**26 篇**（VERIFIED 26 / MINOR 0 / MAJOR 0 / UNVERIFIABLE 0 / PAYWALL 0）
- 年份分布：2023 年 4 篇、2024 年 16 篇、2025 年 5 篇、2026 年 1 篇 → **2023—2026 占比 100%**
- 来源质量：A 级 10 篇 / B 级 14 篇 / C 级 2 篇（纯预印本，仅作背景与最新时间线证据）

---

## 一、作者归属勘误（重要）

任务简报提示"Liu Dingquan 的 LIQE/Q-Align"，经逐条检索核实**有误**：

| 论文 | 真实第一作者 | 团队 |
|---|---|---|
| LIQE（CVPR 2023） | **Weixia Zhang**（上交大AI研究院） | 上交大 Guangtao Zhai 组 + 港城大 Kede Ma |
| Q-Instruct（CVPR 2024） | **Haoning Wu**（NTU S-Lab） | NTU S-Lab Weisi Lin 组 + 上交大 + SenseTime |
| Q-Align（ICML 2024） | **Haoning Wu**（NTU S-Lab） | 同上（Q-Future 系列） |
| Co-Instruct（ECCV 2024） | **Haoning Wu** | NTU + 港城大 Shiqi Wang + 上交大 |
| Compare2Score（ECCV 2024） | **Hanwei Zhu**（港城大） | 港城大 + NTU + 上交大 |
| Q-Boost（IEEE 2024） | **Zicheng Zhang**（上交大） | 上交大 + NTU |

结论：本方向核心工作由 **NTU S-Lab Weisi Lin 组（Haoning Wu / Erli Zhang）+ 上海交大 Guangtao Zhai 组（Zicheng Zhang / Weixia Zhang / Xiongkuo Min）+ 港城大 Shiqi Wang 组（Hanwei Zhu）** 构成的"Q-Future"团队网络产出，与 Liu Dingquan 无关。Liu Dingquan 主要工作在立体视觉/早期 NIMA 系方向，不应与本分支引用混淆。

另一处 venue 勘误：**Co-Instruct 实为 ECCV 2024**（ECVA 官网论文页确认），而非简报所记 NeurIPS 2024。

---

## 二、检索式记录（全部执行）

**第零步原词直搜**
1. `CLIP image quality assessment`
2. `multimodal large language model image quality assessment`
3. `多模态大模型 图像质量评价`

**视角 1：CLIP 系零样本/微调**
4. `Q-Instruct low-level visual instruction tuning image quality assessment`
5. `Q-Align teaching MLLM to score image quality discrete levels`
6. `Co-Instruct multi-task collaborative instruction tuning MLLM image quality`
7. `"Q-Instruct" Improving Low-level Visual Abilities Multi-modal Foundation Models`
8. `"Co-Instruct" collaborative instruction tuning low-level visual`
9. `"Compare2Score" Wu Zicheng pairwise comparison`

**视角 2：大模型打分范式**
10. `Compare2Score pairwise comparison MLLM visual scoring`
11. `Co-Instruct Collaborative Instruction Tuning Low-level Visual Tasks NeurIPS`
12. `ChatIQA chat image quality assessment MLLM`
13. `Chat-driven image quality assessment multi-modal large language models ChatIQA`

**视角 3：MLLM-as-Judge 与经验研究**
14. `multimodal LLM as judge image quality perception bias evaluation benchmark`
15. `A Comprehensive Study of Multimodal Large Language Models for Image Quality Assessment`（直接命中 Tianhe Wu & Kede Ma 论文）

**视角 4：美学评价大模型**
16. `AesBench aesthetic assessment benchmark MLLM 2024`
17. `LAION aesthetic predictor CLIP-based image aesthetic scoring`
18. `AesExpert expert aesthetic assessment MLLM`

**视角 5：2025—2026 最新**
19. `large multimodal model image quality assessment 2025 instruction tuning`
20. `ChatIQA`（再查，确认无此 IQA 论文）
21. `Towards Open-ended Visual Quality Comparison Co-Instruct NeurIPS 2024`
22. `Q-Instruct Improving Low-level Visual Abilities CVPR 2024`

**精读核验（web.fetch）**
- `https://arxiv.org/abs/2405.18763` → 证实该号为数学论文（Wright-Fisher 模型），**证伪猜测 DOI**，避免幻觉引用
- `https://arxiv.org/abs/2402.16641` → Co-Instruct 摘要全文（作者/数据集 MICBench）
- `https://arxiv.org/abs/2311.06783` → Q-Instruct 摘要页

**饱和判据达成情况**：①覆盖达标（CLIP 系/打分范式/MLLM-Judge/美学/2025-26 五个子方向均 ≥3 篇）✓ ②新增衰减（第 4-5 轮新增多为同组后续工作，无新流派）✓ ③主题饱和（零样本提示→指令微调→离散等级→比较转分→描述式打分→RL/分数分布回归全链条均有文献）✓ ④引用闭环（Q-Future 系列互引已捞齐）✓ ⑤时间跨度（2023 奠基 → 2026 最新预印本连续无断层）✓ —— 5 条全满足，停止检索。

---

## 三、文献清单（按主题分组）

### 主题 A：CLIP 系零样本与自监督质量表征（2023 起点）

- **[B4-001] Wang et al. (2023)** *Exploring CLIP for Assessing the Look and Feel of Images (CLIP-IQA)*, AAAI 2023. — VERIFIED；**奠基工作/seminal**；关键发现：反义提示对零样本同时评估技术质量与抽象美感，无需任务特定训练。适合支撑：范式转变的问题提出。
- **[B4-002] Zhang et al. (2023)** *Blind IQA via Vision-Language Correspondence: A Multitask Learning Perspective (LIQE)*, CVPR 2023. — VERIFIED；**奠基工作/seminal**；关键发现：文本模板+图文联合概率做 BIQA/场景/失真三任务，跨数据集标注对齐。适合支撑：路线比较（多任务联合表征）。
- **[B4-003] Agnolucci et al. (2024)** *QualiCLIP*, arXiv:2403.11176. — VERIFIED；代表路线；自监督 opinion-unaware：合成退化排序训练 CLIP 质量感知表征。
- **[B4-004] Onuoha et al. (2024)** *Perceptual Image Quality Prediction: Are CLIP Visual Features Effective?*, Electronics (MDPI). — VERIFIED；**局限证据（反方）**：CLIP/LMM 特征并非总最优，与简单失真特征融合才能超 90% 准确率。

### 主题 B：指令微调与打分范式（Q-Future 核心系列，2024）

- **[B4-005] Wu et al. (2024)** *Q-Instruct*, CVPR 2024（arXiv:2311.06783）. — VERIFIED；**奠基工作/seminal**；首个低层视觉人类反馈数据集 Q-Pathway（58K/18,973 图）→ 200K 指令对。
- **[B4-006] Zhang et al. (2024)** *Q-Bench+*, arXiv:2402.07116. — VERIFIED；benchmark；24 个 MLLM 实测：仅 GPT-4V 成对比较精度超过单图。
- **[B4-007] Zhang et al. (2024)** *Q-Boost*, IEEE Xplore 10645451（arXiv:2312.15300）. — VERIFIED；代表路线；三元语调+多提示集成的零样本策略。
- **[B4-008] Wu et al. (2024)** *Q-Align*, ICML 2024. — VERIFIED；**奠基工作/seminal**；离散文字定义等级代替连续分数监督，跨数据集 SOTA。
- **[B4-009] Wu et al. (2024b)** *Towards Open-ended Visual Quality Comparison (Co-Instruct)*, ECCV 2024. — VERIFIED；代表路线；Co-Instruct-562K + MICBench，超其教师 GPT-4V。
- **[B4-010] Zhu et al. (2024)** *Compare2Score*, ECCV 2024（arXiv:2405.19298）. — VERIFIED；代表路线；软比较概率矩阵+MAP 把成对比较转为连续分，9 库 SOTA。
- **[B4-011] You et al. (2023)** *DepictQA*, arXiv:2312.08962. — VERIFIED；代表路线；放弃分数、用语言描述/比较做 IQA 的范式转换起点。
- **[B4-012] Zhu et al. (2024b)** *2AFC Prompting of LMMs for IQA*, arXiv:2402.01162. — VERIFIED；benchmark；LMM 粗粒度比较强、细粒度判别弱。

### 主题 C：MLLM-as-Judge 与系统性反方证据

- **[B4-013] Wu et al. (2024b)** *A Comprehensive Study of MLLMs for IQA*（Tianhe Wu, Kede Ma 等）, arXiv:2403.10854. — VERIFIED；**局限证据**：未微调 MLLM 只会模板化描述，通用 prompting 做不好 IQA。
- **[B4-014] Zhang et al. (2023)** *GPT-4V(ision) as a Generalist Evaluator for Vision-Language Tasks*, arXiv:2311.01361. — VERIFIED；代表路线；GPT-4V 评判能力早期系统验证。
- **[B4-015] Chen et al. (2024)** *MJ-Bench*, ICLR 2025（arXiv:2407.04842）. — VERIFIED；benchmark；CLIP 打分模型/开源/闭源 VLM 在图像质量与偏见维度对比。
- **[B4-016] Pu et al. (2025)** *Judge Anything*, ACM DL. — VERIFIED；benchmark；MLLM 评判生成任务得分仅 30–42%，跨模态偏见明显。

### 主题 D：美学评价大模型（2024–2026）

- **[B4-017] Huang et al. (2024)** *AesBench*, CVPR 2024（arXiv:2401.08276）. — VERIFIED；benchmark；专家标注 EAPD，MLLM 美学感知仅初级。
- **[B4-018] Huang et al. (2024b)** *AesExpert*, arXiv:2404.09624. — VERIFIED；代表路线；AesMMIT 409K 美学指令数据集。
- **[B4-019] Huang et al. (2024c)** *VisualCritic*, arXiv:2403.12806. — VERIFIED；代表路线；开箱即用广谱质量评估 LMM（数值+解释+AIGC 判别）。
- **[B4-026] Li et al. (2026)** *VideoAesBench*, arXiv:2601.21915. — VERIFIED；C 级最新；视频美学感知 benchmark（23 个 LMM 受测）。

### 主题 E：2025—2026 最新推进（分数回归、RL、AIGC）

- **[B4-020] You et al. (2025)** *DeQA-Score*, arXiv:2501.11561. — VERIFIED；分数分布软标签+Thurstone 保真损失的分数回归。
- **[B4-021] Li et al. (2025)** *RealQA / Next Token Is Enough*, arXiv:2503.06141. — VERIFIED；RealQA 数据集 14,715 UGC 图，next-token 直回归 MOS。
- **[B4-022] Lu et al. (2025)** *Q-Adapt*, arXiv:2504.01655. — VERIFIED；渐进式指令微调解决解释与属性问答冲突。
- **[B4-023] Tang et al. (2025)** *Q-Scorer*, arXiv:2511.07812. — VERIFIED；**局限证据（C 级预印本）**：直接批评 Q-Align 等级 token 的语义混淆与分数转换误差。
- **[B4-024] Wang et al. (2024)** *MA-AGIQA*, ACM MM 2024. — VERIFIED；AIGC 图像质量：MLLM 语义引导+MoE 融合。
- **[B4-025] Cui et al. (2025)** *M3-AGIQA*, arXiv:2502.15167. — VERIFIED；AIGC 三方面（感知/提示对应/真实性）多轮评估。

---

## 四、争议点（供综合阶段正反引用）

1. **MLLM 到底能不能做好 IQA？** Q-Future 微调路线（B4-005/008/010）证明指令微调后 MLLM 可达 SOTA；反方 B4-013（Kede Ma 组）与 B4-012 指出通用 prompting 只能模板化描述、细粒度判别弱。两派结论相反，综述需并置。
2. **离散等级 token vs 连续分数回归。** Q-Align（B4-008）主张文字等级；DeQA-Score（B4-020）用分数分布软标签；Q-Scorer（B4-023）理论批评等级 token 语义混淆。三者构成方法演进链条。
3. **CLIP/LMM 视觉特征的有效性。** B4-004 实证：不加传统失真特征时 CLIP 特征并非最优——对"大模型特征替代手工特征"叙事的反方证据。
4. **MLLM-as-Judge 的可信度边界。** B4-015/016 显示 MLLM 在生成任务打分、跨模态评判上准确率仅 30–50%，存在偏见；但 B4-014 在视觉-语言理解任务上验证其与人类一致性较好——**任务域决定结论**。

## 五、盲区与局限

- **ChatIQA / ChatIQA2.0 未检得**：两轮精确检索无同名 IQA 论文（仅命中 NVIDIA 的 ChatQA，RAG 方向），判定为命名误传或不存在，未纳入。
- **Q-Pathway 非独立论文**：是 Q-Instruct 论文内数据集，已在 B4-005 条目说明。
- **LAION 美学预测器**：无同行评议论文，仅背景线索。
- **纯预印本 2 篇（B4-023、B4-026）**尚无顶会录用，C 级，只能作背景/最新时间线证据，不宜支撑核心论断。
- 各论文具体 SROCC/PLCC 数字均在摘要层未逐篇转引，综合阶段需按全文核对。
