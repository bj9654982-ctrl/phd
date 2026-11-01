# -*- coding: utf-8 -*-
"""Fill requirement_checklist.json resolutions for R1-R13."""
import json
from pathlib import Path
p = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\.workflow\requirement_checklist.json")
data = json.loads(p.read_text(encoding="utf-8-sig"))
res = {
 "R1": "已落实。01_文献调研总报告.md 第六节主题章节按骨架设 12 个分主题：主观评价方法论与标准（一）、质量数据库谱系（二）、传统FR指标（三）、半参考与学习型FR（四）、NSS无参考（五）、CNN时代NR-IQA（六）、Transformer/自监督NR-IQA（七）、CLIP零样本与MLLM指令微调（八）、MLLM-as-Judge与美学（九）、AIGC质量数据库（十）、偏好奖励与对齐指标（十一）、评价协议与元方法（十二）；02_综述写作大纲.md 设 0-7 章完整骨架对应同一谱系。",
 "R2": "已落实。02_综述写作大纲.md 到三级标题，每节含核心论点/代表文献（P编号）/图表建议/写作提示；开头附投稿要求摘要 9 条；给出全文目标 1.6 万字的 0-7 章字数分配表；附 12 项图表清单与 7 条写作执行纪律。",
 "R3": "已落实。03 清单定稿 150 篇（区间 80-150 上限）：2023-2026 共 76 篇占 50.7%；VERIFIED 147、MINOR 3、无 UNVERIFIABLE；A=93/B=38/C=19；seminal 53 处。各分支奠基作全收录：SSIM/MS-SSIM（Wang & Bovik）、VIF（Sheikh & Bovik）、LIVE 系列、BRISQUE/NIQE（Mittal）、WaDIQaM（Bosse）、LPIPS（R. Zhang）/DISTS（K. Ding）、MUSIQ（J. Ke）、CONTRIQUE（Madhusudana）、TOPIQ（C. Chen）、方玉明团队 SPAQ/AGIQA、翟广涛团队 AGIQA 系列、LIQE（W. Zhang）、Q-Align（H. Wu）。",
 "R4": "已落实。B5 分支 32 篇 + 综合：AGIQA-1K（Z. Zhang，2023，1080 图仅感知质量）、AGIQA-3K（C. Li，2023，质量+对齐双维）、AIGCIQA2023（J. Wang，2023，三维）、AIGIQA-20K（原名 AIGCIQA-20K，C. Li，2024，2万图/42万评分，NTIRE2024 用库）、ImageReward、HPS/HPSv2、Pick-a-Pic/PickScore、Q-Align、CLIP-IQA、Q-Boost、Co-Instruct（ECCV2024）、VIEScore、MJ-Bench、ImagenHub、HEIM、TIFA、T2I-CompBench、PKU-I2IQA/PKU-AIGIQA-4K 等均收录并核验；GeoTrp/MP-IQDF/DiffusionGIQA/T2IQA-1M 经多轮检索确认不存在，列入 05 灰区死路。01 第（十）（十一）分主题、02 第 5 章专章覆盖。",
 "R5": "已落实。03 清单含国内团队工作：北邮/上海交大翟广涛团队（方玉明、张梓晨、李晨旸等，AGIQA/AIGCIQA 系列）、上海交大丁广茂/李定权方向、江西财大方玉明团队 SPAQ、武大、浙大、南科大、港中文、上交等；02 大纲第 6 章中文团队脉络；中文期刊前序综述 3 篇（方玉明等 2021《无参考图像质量评价研究进展》、鄢杰斌等 2022、程茹秋等 2022，均 DOI 核实）+ 王志明 2015 自动化学报、高敏娟等 2021 电子学报，作为差异化定位窗口写入 02 大纲。",
 "R6": "已落实。6 路子代理全部走 scholar_search 原词直搜→多视角补搜→存在性/作者/venue 三步核验，03 清单 150 行均有核验状态（VERIFIED/MINOR）、URL/DOI、read_status（abstract 层）、quality_basis（顶会/被引/分区/官方标准）；UNVERIFIABLE 与死路不入清单（05 记录 6 组灰区）；性能数字仅保留摘要层明确出现的白名单值（01 报告与示例段实际只用 <0.2、0.4、0.45、30%-53%、13%），其余统一标'摘要层，终稿需全文精读'；无编造 DOI。",
 "R7": "已落实。02 大纲开头'投稿要求摘要'9 条：综述篇幅不限、题目字数、中英文摘要四要素约500字、关键词5-8、中图法分类号、三级阿拉伯编号、图表中英对照/300dpi、投稿入口 cjig.cn、版面费400元/页（2026起）；重要勘误：参考文献实为著者-出版年制（GB/T 7714-2015 第9.2节）非顺序编码制；期刊当前非 EI 收录（官网+中科院期刊导引证据），为 Scopus/CSCD/北大核心/中国科协T1。附官方链接：撰稿要求/投稿指南/处理流程/基本情况/GB/T 7714 PDF。",
 "R8": "已落实。01_文献调研总报告.md 共 11 节：核心结论（7条claim+总体判断段）、研究范围与方法、文献多维地图（5列表）、研究视角与本文结构（双轴taxonomy）、核心逻辑图（mermaid）、主题章节（12分主题，各含本节文献索引+多段论证+对比表+小结）、争议与开放问题（7争议+4开放问题）、可研究方向（5条三要素）、局限性（4点）、综述正文示例段（5段）、参考文献（150条，按作者字母/拼音排序，每条 [原文/DOI](url) 可点击链接，脚本核验无缺链）。",
 "R9": "已落实。03_文献库_核验清单.md 与 .csv（UTF-8-SIG，Excel 可直接打开）：150 行，字段含编号 P001-P150、作者、年份、标题、venue、source_quality(A/B/C)、authority_signal、核验状态、URL/DOI、所属主题（themes）、FR/RR/NR/其他标签、is_seminal、read_status；开头含 6 条勘误与 6 分支统计。",
 "R10": "已落实。04_核心逻辑图.md（读图说明+Mermaid源码+投稿导出建议）+ work/.workflow 双份 logic_graph.mmd + .workflow/figures/logic_graph.whiteboard.xml（16节点20边，research_visuals.py status:pass，门禁通过）；PNG 未本地导出（环境无 mmdc，04 文档已注明用 Mermaid Live Editor/mermaid-cli 导出 300dpi 方法，用户允许 Mermaid 形式）。05_检索式记录.md 汇总 6 分支 132 条检索式，含死路/灰区/饱和判据/补搜轮次。",
 "R11": "已落实。01 第十节'完整文献综述参考稿'：固定提示段原文 + work/review_draft.md 5 段连续散文，visible_length=2083，check_review_draft.py status=pass（33处作者-年份引用、0编号引用、无标题/列表/表格/超链接/禁词），追溯版 review_draft_with_ids.md 带 P 编号。",
 "R12": "已落实（以飞书交付实际结果为准）。document-delivery 阶段由子代理创建飞书 docx，插入 mermaid 白板与 150 条带链接参考文献，fetch 读回经 check_lark_doc.py 校验 pass 后写 doc_handoff.json 并 accept；若 lark-cli 登录态/网络不可用，按用户预设不阻塞本地交付，最终回复据实说明。",
 "R13": "已落实。全部交付物为第三方文献综述，无任何用户个人复现工作内容（未触碰指定目录）；无编造实验数据，性能数字仅转引摘要层白名单并标注；全部 Markdown 中文撰写，IQA/FR/RR/NR/CLIP/MLLM/AIGC/SROCC/PLCC 等术语保留英文缩写。",
}
for item in data["requirements"]:
    rid = item["id"]
    if rid in res:
        item["resolution"] = res[rid]
p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
empty = [i["id"] for i in data["requirements"] if i["in_scope"]=="yes" and not i["resolution"].strip()]
print("filled; remaining empty in-scope:", empty)
