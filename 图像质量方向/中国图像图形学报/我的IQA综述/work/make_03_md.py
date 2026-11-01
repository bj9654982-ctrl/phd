# -*- coding: utf-8 -*-
"""Generate 03_文献库_核验清单.md from final_library.json."""
import json
from pathlib import Path
WORK = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\work")
PROJ = WORK.parent
d = json.loads((WORK / "final_library.json").read_text(encoding="utf-8-sig"))
ps = d["papers"]

BRANCH_TITLE = {
    "B1": "B1 主观评价方法与自然图像/UGC 质量数据库",
    "B2": "B2 传统 FR/RR 指标与学习型感知 FR",
    "B3": "B3 NR-IQA：NSS/CNN/Transformer/自监督",
    "B4": "B4 CLIP/VLM/MLLM 时代 IQA",
    "B5": "B5 AIGC 生成图像质量评价",
    "B6": "B6 综述、评价协议与中文工作",
}

def esc(s):
    return str(s).replace("|", "｜").replace("\n", " ")

lines = []
lines.append("# 03 文献库核验清单（全景式 IQA 综述）\n")
lines.append("> 检索日期：2026-10-20；检索工具：scholar_search（学术）、general_search/web.fetch（期刊要求）；")
lines.append("> 全部条目经三步核验（存在性→元数据→引述准确性）。核验状态：VERIFIED=多源一致；MINOR=一项元数据（卷期/DOI）待终稿核对；灰区/UNVERIFIABLE 一律不入清单。")
lines.append("> 来源等级：A=顶刊顶会/权威标准/高被引经典；B=正式同行评议出版物；C=arXiv 纯预印本（终稿前须替换为正式版或谨慎使用）。")
lines.append("> 阅读状态：abstract=摘要/方法层可靠信息；性能数字（SROCC/PLCC）仅在摘要明确给出时采用，全文数据待综述写作阶段精读。\n")
lines.append(f"**总数：{len(ps)} 篇**；2023—2026 年文献 {sum(1 for p in ps if str(p.get('year')) in {'2023','2024','2025','2026'})} 篇（"
             f"{sum(1 for p in ps if str(p.get('year')) in {'2023','2024','2025','2026'})/len(ps)*100:.1f}%）；"
             f"A 级 {sum(1 for p in ps if p['source_quality']=='A')} 篇，B 级 {sum(1 for p in ps if p['source_quality']=='B')} 篇，"
             f"C 级 {sum(1 for p in ps if p['source_quality']=='C')} 篇；VERIFIED {sum(1 for p in ps if p['verification']=='VERIFIED')} 篇，MINOR {sum(1 for p in ps if p['verification']=='MINOR')} 篇。\n")

# erratum section
lines.append("## 检索过程中的重要勘误\n")
lines.append("1. **作者归属勘误**：用户提示中“Ke Jie 的 MUSIQ/CONTRIQUE/TOPIQ”“Liu Dingquan 的 LIQE/Q-Align”“Zhang Lin 的 LPIPS/DISTS”经核验均不完整——"
             "MUSIQ 第一作者为 Junjie Ke（Google），CONTRIQUE 第一作者为 Pavan C. Madhusudana（UT Austin），TOPIQ 第一作者为 Chaofeng Chen（NTU，IEEE TIP 2024）；"
             "LIQE 第一作者为 Weixia Zhang（上海交大/港城大），Q-Instruct/Q-Align/Co-Instruct 第一作者为 Haoning Wu（NTU S-Lab）；"
             "LPIPS 第一作者为 Richard Zhang（UC Berkeley/Adobe），DISTS 第一作者为 Keyan Ding，FSIM 第一作者为 Lin Zhang（港理工）。")
lines.append("2. **数据库名称勘误**：用户提示中的“AIGCIQA-20K”实际名称为 **AIGIQA-20K**（Li Chunyi 等，2024，NTIRE 2024 挑战赛用库）；"
             "AGIQA-1K（1080 图，仅感知质量）、AGIQA-3K（质量+文图对齐双维）、AIGCIQA2023（质量/真实性/对应性三视角）、AIGIQA-20K（2 万图/42 万评分）为四个不同数据库。")
lines.append("3. **提示中 GeoTrp、MP-IQDF、DiffusionGIQA、T2IQA-1M 经多轮检索不存在对应名称文献**，按灰区原则不入清单。")
lines.append("4. **TAD66K 勘误**：系图像**美学**数据集（He 等，IEEE TIP 2020），非感知质量数据库，不入 IQA 清单。")
lines.append("5. **ChatIQA/ChatIQA2.0 不存在**（仅命中 NVIDIA 的 RAG 工具 ChatQA），按命名误传处理。")
lines.append("6. **Co-Instruct 为 ECCV 2024**（非 NeurIPS 2024）。\n")

cur = None
for p in ps:
    b = p["id"].split("-")[0]
    if b != cur:
        cur = b
        lines.append(f"\n## {BRANCH_TITLE[b]}\n")
        lines.append("| 编号 | 文献（作者，年） | 标题 | 出处 | 等级 | 核验 | 标签 | 链接 |")
        lines.append("|---|---|---|---|---|---|---|---|")
    ck = p.get("citation_key","")
    link = f"[原文/DOI]({p.get('url','')})" if p.get("url") else ""
    sem = "★" if str(p.get("is_seminal","")).lower() in ("yes","true","1") else ""
    tags = " / ".join([t for t in [p.get("fr_type",""), "; ".join(p.get("themes",[])[:2])] if t])
    lines.append(f"| {p['final_id']} {sem} | {esc(ck)} | {esc(p.get('title',''))} | {esc(p.get('venue',''))} | "
                 f"{p.get('source_quality','')} | {p.get('verification','')} | {esc(tags)} | {link} |")

lines.append("\n## 说明\n")
lines.append("- 标注 ★ 者为该分支奠基性工作（is_seminal）。")
lines.append("- C 级（arXiv 预印本）集中于 B4/B5 前沿：按《中国图象图形学报》“参考文献须为正式出版物、全部提供 DOI”的要求，终稿前应逐条核对其正式录用情况并替换；2023—2026 AIGC/MLLM 领域部分里程碑数据集目前仅有 arXiv 版本，替换不掉时应在正文注明“预印本”。")
lines.append("- 各条目完整字段（key_finding、methods_notes、verify_evidence、阅读状态、来源分支）见同目录 `03_文献库_核验清单.csv` 与 `final_library.json`。")

(PROJ / "03_文献库_核验清单.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote 03_文献库_核验清单.md, lines:", len(lines))
