# -*- coding: utf-8 -*-
"""Aggregate all search queries from 6 branch JSONs into 05_检索式记录.md."""
import json
from pathlib import Path
WORK = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\work")
PROJ = WORK.parent
BRANCHS = ["B1","B2","B3","B4","B5","B6"]
TITLES = {
 "B1":"B1 主观评价方法与自然图像/UGC 质量数据库",
 "B2":"B2 传统 FR/RR 指标与学习型感知 FR",
 "B3":"B3 NR-IQA：NSS/CNN/Transformer/自监督",
 "B4":"B4 CLIP/VLM/MLLM 时代 IQA",
 "B5":"B5 AIGC 生成图像质量评价",
 "B6":"B6 综述、评价协议、中文工作与期刊要求",
}
lines = ["# 05 文献检索式记录（全景式 IQA 综述）\n",
 "> 检索日期：2026-10-20；检索工具：scholar_search（学术文献，延迟加载工具经 tool_search 发现）、general_search/web.fetch（期刊要求与元数据核验）。",
 "> 每条检索式记录：语种、目的、命中数/有效数、备注。6 路分支并行、互斥分工，合并覆盖综述全骨架。",
 "> 三步核验流程：原词直搜命中 → 多视角补检 → 标题/作者/venue/年份多源一致（scholar_search 原文链接 + 出版方页面 + 后续论文引文交叉验证）；灰区（UNVERIFIABLE）与 arXiv 纯预印本分别标记，不混入正式证据。\n",
 "## 一、检索策略\n",
 "1. **第零步原词直搜**：每路先以方向原词（中英文各一轮）直搜，命中奠基综述与代表作；",
 "2. **多视角补检**：按「方法流派、数据集、协议基准、中文团队、近年（2023—2026）前沿」5 个视角轮换关键词；",
 "3. **种子滚雪球**：从已命中论文的引文与被引中反查必收经典（SSIM/VIF/BRISQUE/WaDIQaM/MUSIQ/Q-Align/AGIQA-3K 等）；",
 "4. **元数据定向核验**：对中文期刊文献与作者归属，用 general_search 补全卷期页码与 DOI；",
 "5. **饱和判据**：连续多轮新检索无新增独立条目、必收清单全命中、各视角文献闭环，即停止。\n",
 "## 二、检索式明细\n"]

total = 0
for b in BRANCHS:
    d = json.loads((WORK / f"scout_branch_{b}.json").read_text(encoding="utf-8-sig"))
    qs = d.get("queries", d.get("search_rounds", []))
    lines.append(f"\n### {TITLES[b]}（{len(qs)} 条）\n")
    for i, q in enumerate(qs, 1):
        total += 1
        if isinstance(q, str):
            qtext = q
            lang = purpose = ""; hits = useful = ""; note = ""
        else:
            qtext = q.get("query","")
            lang = q.get("lang","")
            purpose = q.get("purpose","")
            hits = q.get("hits","")
            useful = q.get("useful","")
            note = q.get("note","")
        lines.append(f"{i}. `{qtext}`")
        meta = []
        if lang: meta.append(f"语种：{lang}")
        if purpose: meta.append(f"目的：{purpose}")
        if hits != "": meta.append(f"命中：{hits}")
        if useful != "": meta.append(f"有效：{useful}")
        if meta: lines.append("   - " + "；".join(meta))
        if note: lines.append(f"   - 备注：{note}")
    # dead ends
    for de in d.get("dead_ends", []):
        lines.append(f"   - 死路/灰区：{de}")
    for pe in d.get("pending", []):
        if isinstance(pe, dict):
            lines.append(f"   - 待补：{pe.get('title','')}——{pe.get('reason','')}")
        else:
            lines.append(f"   - 待补：{pe}")

lines.append(f"\n## 三、合计\n\n检索式共 **{total}** 条（含 scholar_search 学术检索与 general_search 核验/期刊检索）。\n")
lines.append("## 四、排除与灰区记录\n")
lines.append("- **不存在的文献（多轮检索无命中，按灰区剔除）**：GeoTrp、MP-IQDF、DiffusionGIQA、T2IQA-1M（用户提示出现，scholar_search 多轮零命中）；ChatIQA/ChatIQA2.0（仅命中 NVIDIA 同名 RAG 工具）；Q-Pathway 为 Q-Instruct 论文内数据集而非独立论文。")
lines.append("- **误归类剔除**：TAD66K 为图像美学数据集（He 等，IEEE TIP 2020），非感知质量库。")
lines.append("- **robots 限制**：学报旧站模板页（cjig.ijournals.cn）禁止自动抓取，改用官网撰稿要求页 + GB/T 7714-2015 官方 PDF 核实著录规则。")
lines.append("- **预印本纪律**：19 篇 C 级 arXiv 预印本仅作前沿脉络参考，已在 03 清单标注；终稿前须核对正式录用版。")

(PROJ / "05_检索式记录.md").write_text("\n".join(lines), encoding="utf-8")
print("total queries:", total)
print("wrote 05_检索式记录.md")
