# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
WORK = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\work")
data = json.loads((WORK / "merged_library.json").read_text(encoding="utf-8-sig"))
papers = data["papers"]

print("=== C-grade papers (", len([p for p in papers if p["source_quality"]=="C"]), ") ===")
for p in papers:
    if p["source_quality"] == "C":
        print(f"{p['id']} | {p.get('year')} | {p['title'][:90]} | {p.get('venue','')[:40]} | {'/'.join(p['branches'])}")

print("\n=== fuzzy near-duplicate check (token overlap > 0.75) ===")
def toks(t):
    return set(re.findall(r"[a-z0-9]{3,}", t.lower()))
cjk = lambda t: set(re.findall(r"[\u4e00-\u9fff]{2,}", t))
for i in range(len(papers)):
    for j in range(i+1, len(papers)):
        a, b = papers[i]["title"], papers[j]["title"]
        ta, tb = toks(a), toks(b)
        if ta and tb:
            ov = len(ta & tb) / max(1, min(len(ta), len(tb)))
            if ov > 0.75:
                print(f"[{papers[i]['id']}] vs [{papers[j]['id']}] ov={ov:.2f}")
                print(f"   A: {a[:100]}")
                print(f"   B: {b[:100]}")
        ca, cb = cjk(a), cjk(b)
        if ca and cb:
            ov = len(ca & cb) / max(1, min(len(ca), len(cb)))
            if ov > 0.7:
                print(f"[CJK {papers[i]['id']}] vs [{papers[j]['id']}] ov={ov:.2f}")
                print(f"   A: {a[:60]}  B: {b[:60]}")
