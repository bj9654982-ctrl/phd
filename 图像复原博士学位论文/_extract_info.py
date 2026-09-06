# -*- coding: utf-8 -*-
import os
import json
from pypdf import PdfReader

folder = r"J:\phd\图像复原博士学位论文"
files = [f for f in os.listdir(folder) if f.endswith('.pdf')]
# 去重（董伟达有重复）
seen = set()
unique_files = []
for f in files:
    # 去掉下划线前缀版本
    base = f.lstrip('_')
    if base not in seen:
        seen.add(base)
        unique_files.append(f)

results = {}
for fname in sorted(unique_files):
    fpath = os.path.join(folder, fname)
    print(f"Reading: {fname}")
    try:
        reader = PdfReader(fpath)
        total_pages = len(reader.pages)
        # 读取前15页（封面、声明、摘要、目录）
        text_parts = []
        pages_to_read = min(15, total_pages)
        for i in range(pages_to_read):
            try:
                t = reader.pages[i].extract_text()
                if t:
                    text_parts.append(t)
            except Exception as e:
                pass
        full_text = "\n".join(text_parts)
        results[fname] = {
            "total_pages": total_pages,
            "first_pages_text": full_text[:8000]  # 限制长度
        }
    except Exception as e:
        results[fname] = {"error": str(e)}

# 输出
with open(r"J:\phd\图像复原博士学位论文\_thesis_info.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone. Processed {len(results)} files.")
for k, v in results.items():
    if "error" in v:
        print(f"  ERROR: {k} - {v['error']}")
    else:
        print(f"  OK: {k} ({v['total_pages']} pages)")
