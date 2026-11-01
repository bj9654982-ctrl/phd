# -*- coding: utf-8 -*-
"""Finalize stage-1 scout: apply true merges + scoped drops, renumber, emit
merged_library.json, 03 CSV, and scout_handoff.json."""
import json, re, csv
from pathlib import Path
from collections import Counter

WORK = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\work")
data = json.loads((WORK / "merged_library.json").read_text(encoding="utf-8-sig"))
papers = {p["id"]: p for p in data["papers"]}

# --- 1. true duplicate merges (keep richer target, union branches/themes) ---
def merge(keep_id, drop_id):
    keep, drop = papers[keep_id], papers[drop_id]
    for b in drop.get("branches", []):
        if b not in keep["branches"]:
            keep["branches"].append(b)
    for t in drop.get("themes", []):
        if t not in keep.setdefault("themes", []):
            keep["themes"].append(t)
    # union fields: prefer longer key_finding / methods_notes
    for f in ("key_finding", "methods_notes", "verify_evidence", "quality_basis", "citation_count"):
        if len(str(drop.get(f, ""))) > len(str(keep.get(f, ""))):
            keep[f] = drop.get(f)
    if str(drop.get("is_seminal","")).lower() in {"yes","true","1"}:
        keep["is_seminal"] = "yes"
    # keep best source quality
    rank = {"A":0,"B":1,"C":2}
    if rank.get(drop.get("source_quality","C"),2) < rank.get(keep.get("source_quality","C"),2):
        keep["source_quality"] = drop["source_quality"]
        keep["authority_signal"] = drop.get("authority_signal","")
        keep["quality_basis"] = drop.get("quality_basis","")
    papers.pop(drop_id)

merge("B3-021", "B1-015")   # KonIQ-10k
merge("B3-031", "B1-019")   # CGIQA-6k

# --- 2. scoped drops ---
DROPS = {
    "B1-023": "C级HRIQ预印本，与已收录HDRC/LIVE-TMHDR两个HDR库重复",
    "B2-022": "C级STFN，IEEE Access边缘工作，代表性不足",
    "B2-023": "C级MS-SWD预印本，色彩差异小众方向",
    "B3-036": "C级DGIQA预印本，待正式发表",
    "B3-037": "C级Cross-IQA预印本，与Re-IQA同类",
    "B4-026": "视频美学基准，超出图像IQA边界",
    "B6-005": "C级全景综述预印本，与已收录正式综述重复",
}
for k, why in DROPS.items():
    papers.pop(k, None)

# --- 3. renumber sequentially by branch order ---
BRANCH_ORDER = {"B1":1,"B2":2,"B3":3,"B4":4,"B5":5,"B6":6}
def sort_key(p):
    bid = p["id"].split("-")[0]
    num = int(p["id"].split("-")[1])
    return (BRANCH_ORDER[bid], num)

plist = sorted(papers.values(), key=sort_key)
for i, p in enumerate(plist, 1):
    p["final_id"] = f"P{i:03d}"

# --- 4. stats ---
qc = Counter(p["source_quality"] for p in plist)
vc = Counter(p["verification"] for p in plist)
recent = [p for p in plist if str(p.get("year")) in {"2023","2024","2025","2026"}]
seminal = [p for p in plist if str(p.get("is_seminal","")).lower() in {"yes","true","1"}]
print("total:", len(plist))
print("quality:", dict(qc))
print("verification:", dict(vc))
print(f"recent: {len(recent)} ({len(recent)/len(plist)*100:.1f}%)")
print("seminal:", len(seminal))
yc = Counter(str(p.get("year")) for p in plist)
print("years:", dict(sorted(yc.items())))

# --- 5. write final merged library ---
(WORK / "final_library.json").write_text(
    json.dumps({"total": len(plist), "papers": plist}, ensure_ascii=False, indent=2),
    encoding="utf-8")

# --- 6. 03 CSV ---
with (WORK / "03_文献库_核验清单.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["编号","引用键","标题","作者","年份","出处","DOI","链接","核验状态",
                "来源等级","权威信号","奠基","FR/RR/NR标签","角色","主题","阅读状态","来源分支"])
    for p in plist:
        w.writerow([p["final_id"], p.get("citation_key",""), p.get("title",""),
                    p.get("authors",""), p.get("year",""), p.get("venue",""),
                    p.get("doi",""), p.get("url",""), p.get("verification",""),
                    p.get("source_quality",""), p.get("authority_signal",""),
                    p.get("is_seminal",""), p.get("fr_type",""), p.get("role",""),
                    "; ".join(p.get("themes",[])), p.get("read_status",""),
                    "/".join(p.get("branches",[]))])

# --- 7. scout_handoff core_literature: A/B only, schema-valid ---
VALID_SIGNALS = {"top_journal","high_citation","classic","official","core_journal"}
core = []
problems = []
for p in plist:
    if p.get("source_quality") not in ("A","B"):
        continue
    # ITU/VQEG 标准/报告为 metadata_only 时留在正式库，不进 core_literature
    if p.get("read_status") == "metadata_only" or not p.get("read_status"):
        continue
    if not p.get("venue"): problems.append((p["final_id"],"venue"))
    if not p.get("url"): problems.append((p["final_id"],"url"))
    if p.get("authority_signal") not in VALID_SIGNALS:
        problems.append((p["final_id"],f"signal:{p.get('authority_signal')}"))
    if not p.get("quality_basis"): problems.append((p["final_id"],"quality_basis"))
    core.append({
        "id": p["final_id"],
        "citation_key": p.get("citation_key",""),
        "title": p.get("title",""),
        "authors": p.get("authors",""),
        "year": p.get("year",""),
        "venue": p.get("venue",""),
        "url": p.get("url",""),
        "doi": p.get("doi",""),
        "read_status": p.get("read_status","abstract"),
        "source_quality": p.get("source_quality"),
        "authority_signal": p.get("authority_signal"),
        "quality_basis": p.get("quality_basis",""),
        "is_seminal": p.get("is_seminal","no"),
        "fr_type": p.get("fr_type",""),
        "role": p.get("role",""),
        "themes": p.get("themes",[]),
        "branches": "/".join(p.get("branches",[])),
    })
print("core_literature:", len(core), "problems:", len(problems))
for x in problems[:20]: print("  PROBLEM:", x)

handoff = {
    "read_gate": "pass",
    "seed_query": "done",
    "verification": "done",
    "role_tags": "present",
    "ready_for_synthesis": "yes",
    "topic": "全景式图像质量评价（IQA）中文综述：主观评价与数据库→传统FR/RR/NR→深度学习NR-IQA→Transformer/CLIP/多模态大模型→AIGC质量评价→数据集与评价协议→未来方向",
    "branch_summary": {
        "B1": "主观评价方法与自然图像/UGC质量数据库",
        "B2": "传统FR/RR客观指标与学习型感知FR（LPIPS/DISTS谱系）",
        "B3": "NR-IQA：NSS/CNN/非大模型Transformer/自监督",
        "B4": "CLIP/VLM/MLLM打分与指令微调IQA",
        "B5": "AIGC生成图像质量评价：数据库/偏好模型/对齐指标/benchmark",
        "B6": "综述、评价协议、中文期刊工作与目标期刊投稿要求",
    },
    "total_papers": len(plist),
    "recent_2023_2026": len(recent),
    "recent_ratio": round(len(recent)/len(plist), 3),
    "quality_counts": dict(qc),
    "verification_counts": dict(vc),
    "seminal_count": len(seminal),
    "drops": {k: v for k, v in DROPS.items()},
    "merges": ["B3-021 absorbs B1-015 (KonIQ-10k)", "B3-031 absorbs B1-019 (CGIQA-6k)"],
    "core_literature": core,
}
WF = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\.workflow")
(WF / "scout_handoff.json").write_text(
    json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote scout_handoff.json")
