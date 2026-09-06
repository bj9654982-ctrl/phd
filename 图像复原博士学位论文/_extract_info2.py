# -*- coding: utf-8 -*-
import os
import json
import fitz  # PyMuPDF

folder = r"J:\phd\图像复原博士学位论文"
files = [f for f in os.listdir(folder) if f.endswith('.pdf')]
# 去重
seen = set()
unique_files = []
for f in files:
    base = f.lstrip('_')
    if base not in seen:
        seen.add(base)
        unique_files.append(f)

results = {}
for fname in sorted(unique_files):
    fpath = os.path.join(folder, fname)
    print(f"Reading: {fname}")
    try:
        doc = fitz.open(fpath)
        total_pages = len(doc)
        text_parts = []
        pages_to_read = min(20, total_pages)
        for i in range(pages_to_read):
            page = doc[i]
            t = page.get_text()
            if t and len(t.strip()) > 20:
                text_parts.append(t)
        full_text = "\n".join(text_parts)
        results[fname] = {
            "total_pages": total_pages,
            "text_length": len(full_text),
            "first_pages_text": full_text[:10000]
        }
        doc.close()
    except Exception as e:
        results[fname] = {"error": str(e), "total_pages": 0}

with open(os.path.join(folder, "_thesis_info2.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone. Processed {len(results)} files.")
for k, v in results.items():
    if "error" in v:
        print(f"  ERROR: {k} - {v['error']}")
    else:
        print(f"  OK: {k} ({v['total_pages']} pages, text len={v['text_length']})")
