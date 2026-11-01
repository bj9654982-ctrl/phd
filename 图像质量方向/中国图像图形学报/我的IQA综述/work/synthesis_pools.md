[SYNTHESIS_HANDOFF]
read_gate: pass
gate_status: clear
ready_for_review: yes

# 四池交接包（与 synthesis_content.md 开头一致，独立存档）

## [CLAIM_POOL]
- IQA 三十年主线沿"参考信息可用性（FR→RR→NR）"与"建模范式（手工感知/NSS→CNN→Transformer→视觉基础模型）"两轴交叉推进；范式切换不是替代上一代，而是不断扩展可评图像的边界（合成失真→真实 UGC→AI 生成）。
- SSIM（Wang et al.，2004）以结构保真替代像素误差成为 FR 指标的范式原点；FSIM、GMSD、VSI、HaarPSI 是同范式下"更高效/更鲁棒"的工程改良而非新机制。
- 深度特征作为感知度量（Zhang et al.，2018，LPIPS）把 FR 指标从手工特征拉入学习特征；DISTS（Ding et al.，2022）显式分离结构与纹理，是对"特征距离无差别"的修正。
- NR-IQA 的共识路线是：opinion-unaware 的 NSS 方法（NIQE/IL-NIQE）证明不接触失真样本也能评分；CNN 时代用大规模 MOS 把精度推高；Transformer/自监督（CONTRIQUE/TOPIQ）主攻跨库泛化。
- 2023 年 CLIP-IQA（Wang et al.，2023）与 LIQE（Zhang et al.，2023）开启"视觉-语言先验做质量"的零样本路线；Q-Future 系列（Q-Instruct/Q-Align/Co-Instruct）证明指令微调后 MLLM 可达甚至超过专用 IQA 模型。
- 未微调 MLLM 做不好 IQA：系统研究表明其只会模板化描述；Q-Bench+ 在 24 个 MLLM 中仅见 GPT-4V 成对比较超单图——微调派与未微调派结论分裂。
- 传统保真指标在 AIGC 上系统性失效：ImagenHub 测得多数自动指标与人类 Spearman<0.2；立场文献指出它们度量的是"重建保真"而非"生成伪影"。
- AIGC"质量"概念本身被解构：从单维 MOS（AGIQA-1K）到质量/真实性/对应性三维（AIGCIQA2023），再到图生图、局部编辑、AI 增强 UGC 等细分场景。
- 偏好奖励模型与对齐指标构成 AIGC 时代另一套评价体系，与传统 SROCC 范式并行；免训练 MLLM 评判（VIEScore）与人类 Spearman≈0.4，接近人-人相关 0.45。
- 评价协议本身是被低估的变量：gMAD 的对抗式选样暴露"库内高分"的脆弱性，KonX 揭示分辨率变化导致主观标签漂移。
- MLLM-as-Judge 的可信度高度依赖任务域：生成任务打分仅 30–53% 且有跨模态偏见，与其在视觉-语言理解任务上的较好一致性构成明确条件边界。
- "一个分数"的假设在 AIGC 时代不成立：AGIQA-3K 同时采集感知质量与文图对齐两维 MOS，MPS 把偏好拆成四维——质量分与偏好分是两个构念。

## [CITATION_POOL]（按主题分组，作者-年份引用串）
- 主观方法/数据库：ITU-R（2019，P001）；VQEG（2003，P003）；Mantiuk et al.（2012，P004）；Ribeiro et al.（2011，P005）；Sheikh et al.（2006，P006）；Ponomarenko et al.（2015，P008）；Ghadiyaram and Bovik（2017，P013）；Hosu et al.（2020，P064）；Fang et al.（2020，P015）；Gu et al.（2020，P016）
- 传统 FR/RR：Wang and Bovik（2009，P024）；Sheikh and Bovik（2006，P026）；Wang et al.（2004，P027）；Wang et al.（2003，P028）；Zhang et al.（2011，P031）；Xue et al.（2014，P032）；Reisenhofer et al.（2018，P034）；Soundararajan and Bovik（2012，P035）
- 学习型 FR：Zhang et al.（2018，P017）；Prashnani et al.（2018，P036）；Ding et al.（2022，P037）；Ding et al.（2021，P038）；Liao et al.（2022，P039）；Severo et al.（2024，P040）
- 传统 NR/NSS：Moorthy and Bovik（2010，P044）；Mittal et al.（2012，P047）；Mittal et al.（2013，P048）；Zhang et al.（2015，P049）；Ye et al.（2012，P050）
- CNN-NR：Kang et al.（2014，P053）；Bosse et al.（2018，P057）；Liu et al.（2017，P058）；Ma et al.（2017，P055）；Su et al.（2020，P061）；Zhu et al.（2020，P062）
- Transformer/自监督 NR：Ke et al.（2021，P067）；Madhusudana et al.（2022，P068）；Yang et al.（2022，P069）；Saha et al.（2023，P071）；Chen et al.（2024，P073）
- CLIP/MLLM：Wang et al.（2023，CLIP-IQA P080）；Zhang et al.（2023，LIQE P081）；Wu et al.（2024，Q-Instruct P084）；Zhang et al.（2024，Q-Bench+ P085）；Wu et al.（2024，Q-Align P087）；Wu et al.（2024，Co-Instruct P088）；Zhu et al.（2024，Compare2Score P089）；Wu et al.（2024b，综合研究 P092）；Chen et al.（2024，MJ-Bench P094）；Pu et al.（2025，P095）；You et al.（2025，DeQA-Score P099）
- AIGC 数据库：Gu et al.（2020，P105）；Zhang et al.（2023，AGIQA-1K P106）；Li et al.（2023，AGIQA-3K P107）；Wang et al.（2023，AIGCIQA2023 P108）；Yuan et al.（2023，PKU-I2IQA P110）；Li et al.（2024，AIGIQA-20K P111）；Qian et al.（2025，EPAIQA-15K P115）；Wang et al.（2025，AU-IQA P116）
- 奖励模型/对齐：Xu et al.（2023，ImageReward P119）；Wu et al.（2023，HPS v2 P120）；Kirstain et al.（2023，Pick-a-Pic P122）；Zhang et al.（2024，MPS P123）；Hu et al.（2023，TIFA P127）；Lin et al.（2024，VQAScore P125）；Ku et al.（2024，ImagenHub P129）；Ku et al.（2024，VIEScore P131）；Ghildyal et al.（2024，P132）
- 评价协议/综述：Ma et al.（2016，gMAD P147）；Ma et al.（2017，Waterloo P011）；Wiedemann et al.（2023，KonX P018）；Lou et al.（2024，VOS P023）；Zhai and Min（2020，P135）；Bovik（2013，P146）；方玉明等（2021，P142）；鄢杰斌等（2022，P143）；程茹秋等（2022，P144）

## [TENSION_POOL]
- 微调派（Q-Instruct/Q-Align/Co-Instruct 达 SOTA）vs 未微调派（系统 prompting：模板化描述；2AFC：粗粒度强细粒度弱）——条件为"是否指令微调"与"测试粒度"，未解。
- CLIP 特征有效性：CLIP-IQA/LIQE 零样本可行 vs Onuoha et al.（2024）实证 CLIP 特征非最优、需与简单失真特征融合——任务域决定。
- 离散等级 token（Q-Align）vs 分数分布软标签（DeQA-Score）vs 等级 token 语义混淆批评（Q-Scorer，C 级）——方法演进，未解。
- 传统指标 AIGC 失效（ImagenHub Spearman<0.2）vs 免训练 MLLM 评判部分恢复（VIEScore Spearman≈0.4）——失效的是一类指标而非自动评价。
- MLLM-as-Judge：视觉-语言任务较好 vs 生成任务打分仅 30–53% 有跨模态偏见——任务域决定。
- 质量分 vs 偏好分：单维 MOS vs 双维 vs 三维 vs 四维——构念粒度递进，统一框架缺失。
- 库内高分 vs gMAD 对抗暴露——标准库区分度问题未解。
- 众包在线 vs 实验室受控主观——生态效度 vs 控制度，VOS 揭示 MOS 平均掩盖个体分歧。

## [GAP_POOL]
- RR-IQA 2023–2026 几乎无新指标（仅 RRED 谱系）——半参考路线沉寂。
- HDR/屏幕/高分辨率/CGI 新库已建，传统 LDR 指标与深度模型的迁移性缺乏横评。
- 跨库泛化多停在 KonIQ/LIVEC 间；真实部署域（特定手机、社交压缩）外部效度研究少。
- 观察者个体差异（VOS）刚提出，"一个 MOS"的替代范式未形成。
- 局部编辑图（EPAIQA-15K）与 AI 增强 UGC（AU-IQA）新场景的系统模型对比刚起步。
- 中文 AIGC/IQA 工作未定向检索；《中国图象图形学报》2022 年后无全景 IQA 综述——定位窗口。
- 视频/3D/屏幕内容 IQA 仅交叉提及，不在本综述展开。
- MLLM 打分校准（分数分布、Thurstone 保真、离散 token 语义混淆）刚起步，无公认解法。
- 美学、技术质量、文图对齐三子构念未统一到同一框架。
- C 级预印本 19 篇（多为 AIGC 数据库 arXiv），正式发表版本与数字需终稿核对。
