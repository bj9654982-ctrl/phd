# -*- coding: utf-8 -*-
import os
import json
import re

folder = r"J:\phd\图像复原博士学位论文"
with open(os.path.join(folder, "_thesis_info2.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

# 为每篇论文提取关键信息
summaries = {}
for fname, info in data.items():
    if "error" in info:
        continue
    text = info.get("first_pages_text", "")
    if not text:
        continue
    
    # 提取中文摘要
    abstract = ""
    m = re.search(r'摘\s*要(.*?)关键词|摘\s*要(.*?)Abstract|摘\s*要(.*?)目\s*录', text, re.DOTALL)
    if m:
        abstract = m.group(1) or m.group(2) or m.group(3) or ""
        abstract = abstract.strip()[:2000]
    
    # 提取关键词
    keywords = ""
    m2 = re.search(r'关键词[：:]\s*(.*?)(?:\n|Abstract|$)', text)
    if m2:
        keywords = m2.group(1).strip()[:200]
    
    # 提取学校
    school = ""
    m3 = re.search(r'([\u4e00-\u9fa5]+大学)\s*博士学位论文', text)
    if m3:
        school = m3.group(1)
    
    # 提取专业
    major = ""
    m4 = re.search(r'学科专业[：:]\s*(.*?)\s*(?:\n|指导教师)', text)
    if m4:
        major = m4.group(1).strip()
    
    # 提取导师
    advisor = ""
    m5 = re.search(r'指导教师[：:]\s*(.*?)\s*(?:\n|答辩日期)', text)
    if m5:
        advisor = m5.group(1).strip()
    
    # 提取年份
    year = ""
    m6 = re.search(r'(20\d{2})\s*年\s*\d{1,2}\s*月', text)
    if m6:
        year = m6.group(1)
    
    summaries[fname] = {
        "school": school,
        "major": major,
        "advisor": advisor,
        "year": year,
        "keywords": keywords,
        "abstract": abstract[:1500],
        "total_pages": info["total_pages"]
    }

# 输出简洁版
for fname, s in summaries.items():
    print("="*80)
    print(f"论文: {fname}")
    print(f"  学校: {s['school']}  专业: {s['major']}  导师: {s['advisor']}  年份: {s['year']}  页数: {s['total_pages']}")
    print(f"  关键词: {s['keywords']}")
    print(f"  摘要: {s['abstract'][:500]}...")
    print()

# 保存
with open(os.path.join(folder, "_thesis_summaries.json"), "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=2)
