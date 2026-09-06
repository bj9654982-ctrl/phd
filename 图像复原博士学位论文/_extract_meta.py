import fitz
import os
import re
import json

thesis_dir = r'J:\phd\图像复原博士学位论文'

pdf_files = [f for f in os.listdir(thesis_dir) if f.endswith('.pdf') and not f.startswith('_')]

results = {}

for pdf_file in pdf_files:
    pdf_path = os.path.join(thesis_dir, pdf_file)
    
    try:
        doc = fitz.open(pdf_path)
        # 读取前15页找年份（封面、声明、摘要等）
        text = ""
        for i in range(min(15, len(doc))):
            text += doc[i].get_text()
        
        # 查找年份模式：20XX年、20XX.XX、20XX年XX月
        year_patterns = [
            r'(20\d{2})\s*年',
            r'(20\d{2})\.',
            r'(20\d{2})年\d+月',
            r'答辩日期[:：]\s*(20\d{2})',
            r'提交日期[:：]\s*(20\d{2})',
        ]
        
        years = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text[:5000])
            years.extend(matches)
        
        # 取最常见的年份
        year = ""
        if years:
            from collections import Counter
            year = Counter(years).most_common(1)[0][0]
        
        # 提取标题（通常在第一页）
        title = ""
        first_page_text = doc[0].get_text()
        # 找最长的一行作为标题
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
        if lines:
            title = max(lines, key=len)
        
        # 提取作者
        author = ""
        author_match = re.search(r'作者[:：]\s*(\S+)', text[:3000])
        if author_match:
            author = author_match.group(1)
        
        # 提取导师
        advisor = ""
        advisor_match = re.search(r'导师[:：]\s*(\S+)', text[:3000])
        if advisor_match:
            advisor = advisor_match.group(1)
        
        # 提取学校
        school = ""
        school_match = re.search(r'(\S+大学)', text[:3000])
        if school_match:
            school = school_match.group(1)
        
        results[pdf_file] = {
            'year': year,
            'title': title,
            'author': author,
            'advisor': advisor,
            'school': school,
            'total_pages': len(doc)
        }
        
        doc.close()
        print(f"{year} | {pdf_file[:40]}... | {title[:30]}...")
        
    except Exception as e:
        print(f"Error: {pdf_file} - {e}")
        results[pdf_file] = {'error': str(e)}

# 保存
output_path = os.path.join(thesis_dir, '_thesis_meta.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {output_path}")
