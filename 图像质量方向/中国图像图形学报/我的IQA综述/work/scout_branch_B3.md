# 分支 B3 文献检索报告：无参考/盲图像质量评价（NR-IQA/BIQA）

> 生成时间：2026-10-20
> 范围：传统 NSS 方法 → CNN 深度学习方法 → 非大模型 Transformer/自监督方法
> 边界：CLIP/多模态大模型（CLIP-IQA、Q-Align）归 B4；AIGC 专用评价归 B5；本分支仅交叉提及。

---

## 一、检索式记录（全部实跑）

| 编号 | 检索式 |
|---|---|
| Q0a | 无参考图像质量评价 深度学习（中文原词直搜） |
| Q0b | blind image quality assessment deep convolutional neural network |
| Q0c | no-reference image quality assessment transformer |
| Q1 | BRISQUE … spatial natural scene statistics Mittal / MUSIQ multi-scale image quality transformer Junjie Ke / TOPIQ top-down … CVPR 2023 |
| Q2 | NIQE completely blind / IL-NIQE integrated local … Lin Zhang / MANIQA multi-dimension attention |
| Q3 | DIIVINE distortion identification Moorthy Bovik / CORNIA unsupervised feature Ye / hyperIQA hypernetwork Su |
| Q4 | RankIQA learning from rankings Liu / Re-IQA reasoning Pramanick CVPR 2023 / UNIQUE universal … |
| Q5 | MetaIQA deep meta-learning CVPR / Hallucinated-IQA adversarial / UNIQUE toward U-NIQUe |
| Q6 | Convolutional neural networks for no-reference … Kang CVPR 2014 / NFERM edge statistics Zhai |
| Q7 | Using deep learning for blind … Kang PRL / UNIQUE in-the-wild 2021 |
| Q8 | UNIQUE U-NIQUe authentic distortions / NFERM free energy regression Zhai 2011 / self-supervised … mask autoencoder 2024 |
| Q9 | BIQI two-step framework Moorthy Bovik / UNIQUE 10082 images / Dingquan Li optimization framework |
| Q10 | "UNIQUE" "A Data Set and a Benchmark" / no-reference IQA 2025 Swin transformer generalization |

## 二、饱和判据（满足 ≥3 条）

1. 四子方向（传统 NSS / CNN / Transformer·自监督 / 2023–2026 新方法）均 ≥3 篇。
2. 2023–2026 年新方法凑齐 12 篇（见下）。
3. 最后两轮新增有效文献 < 已有总量 10%。
4. BRISQUE 引用网络中 DIIVINE、BIQI、NIQE、IL-NIQE、CORNIA、HOSA 均已回收。

## 三、正式核验文献（38 篇，按主题分组）

### A. 传统 NSS 与失真特异方法（9 篇）

| ID | 文献 | 年份/venue | 一句话定位 |
|---|---|---|---|
| B3-001 | Moorthy & Bovik, **BIQI** | 2010, IEEE SPL | 两步法（先识别失真再分类型评分）奠基短文 |
| B3-002 | Moorthy & Bovik, **DIIVINE** | 2011, IEEE TIP | 小波域 NSS 两步法早期高引代表 |
| B3-003 | Saad, Bovik, Charrier, **BLIINDS-II** | 2012, IEEE TIP | DCT 域 NSS + 贝叶斯推断 |
| B3-004 | Mittal, Moorthy, Bovik, **BRISQUE** | 2012, IEEE TIP | 空间域 MSCN NSS 最著名盲指标，必引基线 |
| B3-005 | Mittal, Soundararajan, Bovik, **NIQE** | 2013, IEEE SPL | opinion-unaware 完全盲评价代表，不接触失真样本 |
| B3-006 | L. Zhang, L. Zhang, Bovik, **IL-NIQE** | 2015, IEEE TIP | 多线索 NSS + MVG，opinion-unaware 超越同期 opinion-aware |
| B3-007 | Ye, Kumar, Kang, Doermann, **CORNIA** | 2012, ICML | 无监督字典学习，手工→深度特征过渡标志 |
| B3-008 | Xu et al., **HOSA** | 2016, IEEE TIP | codebook 晚期代表，100 词本高阶统计聚合 |
| B3-009 | Gu, Zhai et al., **NFSDM** | 2013, ICIP | 自由能+结构退化（种子"NFERM"未能直接命中后的同团队可核验替代） |

### B. CNN 深度学习方法（13 篇）

| ID | 文献 | 年份/venue | 一句话定位 |
|---|---|---|---|
| B3-010 | Kang, Ye, Li, Doermann, **CNN-NR-IQA** | 2014, CVPR | 首个浅层 CNN 端到端 NR-IQA |
| B3-011 | Kim & Lee, **BIECON** | 2017, IEEE TIP | "以 FR 行为教 NR"局部质量图 |
| B3-012 | Ma et al., **MEON** | 2017, IEEE TIP | 多任务（失真识别+质量预测）共享早期层 |
| B3-013 | Talebi & Milanfar, **NIMA** | 2018, IEEE TIP | 预测意见分数分布而非 MOS 单点 |
| B3-014 | Bosse et al., **WaDIQaM** | 2018, TPAMI | 局部质量-局部权重联合学习，NR/FR 双用 |
| B3-015 | Liu, van de Weijer, Bagdanov, **RankIQA** | 2017, ICCV | 排序学习解决小样本，首超 SOTA FR |
| B3-016 | Lin & Wang, **Hallucinated-IQA** | 2018, CVPR | GAN 幻觉参考 + 对抗学习 |
| B3-017 | Zhang, Ma et al., **DB-CNN** | 2020, IEEE TCSVT | 合成/真实失真双分支双线性池化 |
| B3-018 | Su et al., **hyperIQA** | 2020, CVPR | 超网络自适应生成质量回归规则 |
| B3-019 | Zhu, Li et al., **MetaIQA** | 2020, CVPR | 元学习解决跨失真泛化 |
| B3-020 | Zhang, Ma, Zhai, Yang, **Uncertainty-Aware BIQA** | 2021, IEEE TIP | pairwise fidelity loss + 不确定性正则 |
| B3-021 | Hosu et al., **KonIQ-10k / KonCept512** | 2020, IEEE TIP | 1 万张真实失真库 + InceptionResNet 基线 |
| B3-022 | Sun, Min, Ma, Zhai, **StairIQA** | 2022, IEEE TIP | 分层特征融合 + 迭代混合库训练 |

### C. Transformer / 自监督方法（8 篇，2021–2024）

| ID | 文献 | 年份/venue | 一句话定位 |
|---|---|---|---|
| B3-023 | Golestaneh et al., **TReS** | 2022, WACV | CNN+Transformer 混合 + 相对排序 + 自一致性 |
| B3-024 | Ke et al., **MUSIQ** | 2021, ICCV | 多尺度原生分辨率 Transformer 里程碑 |
| B3-025 | Madhusudana et al., **CONTRIQUE** | 2022, CVPRW/NTIRE | 对比学习自监督质量表征 |
| B3-026 | Yang et al., **MANIQA** | 2022, CVPRW/NTIRE 冠军 | ViT + 多维注意力块（TAB/SSTB） |
| B3-027 | Zhao et al.（快手）, **Quality-aware Pre-trained** | 2023, CVPR | 质量感知对比损失预训练 |
| B3-028 | Saha, Mishra, Bovik, **Re-IQA** | 2023, CVPR | MoE 双编码器无监督表征 |
| B3-029 | Qin et al., **DEIQT** | 2023, CVPRW/NTIRE | 数据高效 Transformer + attention panel |
| B3-030 | Chen et al., **TOPIQ** | 2024, IEEE TIP | 自上而下语义引导 CFANet，NR/FR 双用强基线 |

### D. 2023–2026 最新 NR-IQA（8 篇，本分支重点）

| ID | 文献 | 年份/venue | 一句话定位 |
|---|---|---|---|
| B3-031 | Zhang et al., **CGIQA-6k** | 2023, arXiv/TIP 系列 | in-the-wild CGI 6000 图主观库 + 深度 NR 模型 |
| B3-032 | Li et al., **CSIQA** | 2024, ICML | 全局质量上下文对比 + 质量感知掩码注意力 |
| B3-033 | Pan et al., **SGIQA** | 2024, IEEE T-Broadcasting | Swin Transformer + 语义显式引导 |
| B3-034 | Yang, Lei, Li, **Swin+NSS** | 2024, Sensors | Transformer 融合传统 NSS 分支 |
| B3-035 | Ni et al., **MDFS** | 2025, IEEE TMM | 多尺度深度特征 MVG，opinion-unaware |
| B3-036 | Ramesh et al., **DGIQA** | 2025, arXiv | 深度引导 Transformer-CNN 桥接，跨库泛化 |
| B3-037 | Zhang, **Cross-IQA** | 2024, arXiv | ViT 合成图像重建 pretext 无监督 |
| B3-038 | Fu et al., **Swin-MIQT** | 2025, IEEE Access | 移位窗口多尺度空间池化原生分辨率 |

## 四、作者/归属勘误汇总（重要）

1. **CONTRIQUE 第一作者不是 Junjie Ke**，而是 **Pavan C. Madhusudana**（UT Austin LIVE，YouTube 团队）；Junjie Ke 是 MUSIQ 第一作者。两者作者列表不同。
2. **TOPIQ 不是"Ke 等 2023 CVPR"**：第一作者为 **Chaofeng Chen**（NTU），基于 CFANet，arXiv:2308.03060，正式刊于 **IEEE TIP 2024**。
3. **Re-IQA（CVPR 2023）第一作者不是 Pramanick**，而是 **Avinab Saha**（UT Austin LIVE）；Pramanick 属 CLIP-IQA 方向（归 B4）。
4. **Hallucinated-IQA 第一作者不是"Ren 等"**，而是 **Kwan-Yee Lin**（北京大学，CVPR 2018）。
5. **RankIQA 年份为 ICCV 2017**（非 2019）。
6. **NFERM（Zhai 等）**未能直接命中确切论文；同团队可核验替代为 NFSDM（Gu, Zhai, ICIP 2013）。
7. **UNIQUE** 经五轮不同写法 scholar 检索均未直接命中标题，按灰区原则暂列 pending。

## 五、待确认（pending）

- NFERM（Zhai 等）— 需 web 精读复核
- UNIQUE（Weixia Zhang 等，疑似 IEEE TIP 2021）— 需 web 精读复核
- Dingquan Li 等的优化框架工作 — 未命中
- VisualQuality-R1（RL-to-rank MLLM 推理式评分）— 按边界归 B4 交叉参考
- Q-MAE / 掩码自监督 NR-IQA — 未命中

## 六、盲区声明

1. 本轮全部结论停留在摘要层；SROCC 等性能数字仅 CONIQ-10k（B3-021）、DEIQT（B3-029）、Swin+NSS（B3-034）摘要层明确给出，其余未补写。
2. arXiv-only 两篇（DGIQA、Cross-IQA）source_quality=C，仅作背景线索，正式综述前需确认是否已被会议/期刊接收。
3. 未做 web.fetch 全文精读，方法细节（骨干网络超参、训练集规模、库间方差）以摘要为准。
4. B4（CLIP/MLLM）与 B5（AIGC）方法未展开，仅交叉标注。
