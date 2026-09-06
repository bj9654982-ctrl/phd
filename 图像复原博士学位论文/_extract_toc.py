import fitz
import os
import json
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'

# 获取所有PDF文件
pdf_files = [f for f in os.listdir(thesis_dir) if f.endswith('.pdf')]

results = {}

for pdf_file in pdf_files:
    pdf_path = os.path.join(thesis_dir, pdf_file)
    print(f"Processing: {pdf_file}")
    
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        
        toc_list = []
        for level, title, page in toc:
            toc_list.append({
                'level': level,
                'title': title.strip(),
                'page': page
            })
        
        # 提取前几页文本，找年份和摘要
        first_pages_text = ""
        for i in range(min(10, len(doc))):
            first_pages_text += doc[i].get_text()
        
        # 查找年份
        year_match = re.search(r'20\d{2}', first_pages_text[:5000])
        year = year_match.group() if year_match else ""
        
        total_pages = len(doc)
        results[pdf_file] = {
            'total_pages': total_pages,
            'year': year,
            'toc': toc_list
        }
        
        doc.close()
        print(f"  Pages: {total_pages}, TOC entries: {len(toc_list)}, Year: {year}")
        
    except Exception as e:
        print(f"  Error: {e}")
        results[pdf_file] = {'error': str(e)}

# 保存结果
output_path = os.path.join(thesis_dir, '_thesis_toc.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {output_path}")
