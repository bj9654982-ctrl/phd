# -*- coding: utf-8 -*-
"""Merge 6 scout branch JSONs: dedupe, stats, output merged library + CSV."""
import json, re, csv, sys
from pathlib import Path

WORK = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\work")
BRANCHES = ["B1", "B2", "B3", "B4", "B5", "B6"]

def norm_title(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", t)
    return t

def norm_author_key(entry):
    """fallback dedupe key from first author surname + year."""
    a = entry.get("authors", "") or ""
    first = re.split(r"[,，;；]", a)[0].strip()
    # take last token of English name
    m = re.findall(r"[A-Za-z']+", first)
    name = m[-1].lower() if m else first
    return f"{name}{entry.get('year','')}"

merged = {}
order = []
for b in BRANCHES:
    data = json.loads((WORK / f"scout_branch_{b}.json").read_text(encoding="utf-8-sig"))
    papers = data.get("papers", [])
    print(f"{b}: {len(papers)} papers")
    for p in papers:
        key = norm_title(p.get("title", ""))
        if not key:
            continue
        if key in merged:
            ex = merged[key]
            ex.setdefault("branches", []).append(b)
            # keep better quality: A<B<C -> prefer A
            rank = {"A": 0, "B": 1, "C": 2}
            if rank.get(p.get("source_quality", "C"), 2) < rank.get(ex.get("source_quality", "C"), 2):
                p["branches"] = ex["branches"]
                merged[key] = p
            # merge themes
            for t in p.get("themes", []):
                if t not in ex.setdefault("themes", []):
                    ex["themes"].append(t)
            continue
        p["branches"] = [b]
        merged[key] = p
        order.append(key)

papers = [merged[k] for k in order]
print(f"\nMerged unique papers: {len(papers)}")

# stats
from collections import Counter
qc = Counter(p.get("source_quality", "?") for p in papers)
vc = Counter(p.get("verification", "?") for p in papers)
recent = [p for p in papers if str(p.get("year", "")) in {"2023", "2024", "2025", "2026"}]
seminal = [p for p in papers if str(p.get("is_seminal", "")).lower() in {"yes", "true", "1"}]
years = Counter(str(p.get("year", "?")) for p in papers)
print("Quality:", dict(qc))
print("Verification:", dict(vc))
print(f"Recent (2023-2026): {len(recent)} ({len(recent)/len(papers)*100:.1f}%)")
print("Seminal:", len(seminal))
print("Year dist:", dict(sorted(years.items())))

# branch contribution
bc = Counter()
for p in papers:
    for b in p["branches"]:
        bc[b] += 1
print("Branch mentions (with dupes):", dict(bc))

# dump merged
out = {"total": len(papers), "papers": papers}
(WORK / "merged_library.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

# CSV
fields = ["id", "citation_key", "title", "authors", "year", "venue", "doi", "url",
          "verification", "source_quality", "authority_signal", "is_seminal",
          "fr_type", "role", "themes", "read_status", "branches"]
with (WORK / "文献库核验清单.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["编号", "引用键", "标题", "作者", "年份", "出处", "DOI", "链接",
                "核验状态", "来源等级", "权威信号", "奠基", "FR/RR/NR标签", "角色",
                "主题", "阅读状态", "分支"])
    for i, p in enumerate(papers, 1):
        w.writerow([
            i, p.get("citation_key", ""), p.get("title", ""), p.get("authors", ""),
            p.get("year", ""), p.get("venue", ""), p.get("doi", ""), p.get("url", ""),
            p.get("verification", ""), p.get("source_quality", ""),
            p.get("authority_signal", ""), p.get("is_seminal", ""),
            p.get("fr_type", ""), p.get("role", ""),
            "; ".join(p.get("themes", [])), p.get("read_status", ""),
            "/".join(p.get("branches", []))
        ])
print("Wrote merged_library.json and 文献库核验清单.csv")
