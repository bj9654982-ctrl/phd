import fitz
import os
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'

# 中文数字转阿拉伯数字
def chinese_to_arabic(chinese_str):
    digits = {'〇': '0', '一': '1', '二': '2', '三': '3', '四': '4', 
              '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'}
    result = ''
    for ch in chinese_str:
        result += digits.get(ch, ch)
    return result

pdf_files = [f for f in os.listdir(thesis_dir) if f.endswith('.pdf') and not f.startswith('_')]

# 去掉重复的
pdf_files = [f for f in pdf_files if f != '散射介质退化图像复原技术研究_董伟达.pdf']

results = {}

for pdf_file in pdf_files:
    pdf_path = os.path.join(thesis_dir, pdf_file)
    
    try:
        doc = fitz.open(pdf_path)
        
        # 读取前10页和最后5页
        text = ""
        for i in range(min(10, len(doc))):
            text += doc[i].get_text()
        for i in range(max(0, len(doc)-5), len(doc)):
            text += doc[i].get_text()
        
        # 转换全角数字为半角
        fullwidth = '０１２３４５６７８９'
        halfwidth = '0123456789'
        for fw, hw in zip(fullwidth, halfwidth):
            text = text.replace(fw, hw)
        
        years = []
        
        # 模式1: 20XX年
        matches = re.findall(r'(20\d{2})\s*年', text)
        years.extend(matches)
        
        # 模式2: 二〇XX年
        matches = re.findall(r'二〇([一二三四五六七八九〇]{2})年', text)
        for m in matches:
            years.append('20' + chinese_to_arabic(m))
        
        # 模式3: 20XX.XX
        matches = re.findall(r'(20\d{2})\.\d{1,2}', text)
        years.extend(matches)
        
        # 模式4: 答辩日期/提交日期
        matches = re.findall(r'(?:答辩|提交|完成).*?(20\d{2})', text)
        years.extend(matches)
        
        # 统计最常见的
        from collections import Counter
        year_counts = Counter(years)
        most_common = year_counts.most_common(3)
        
        year = most_common[0][0] if most_common else '未知'
        
        results[pdf_file] = {
            'year': year,
            'total_pages': len(doc),
            'candidates': most_common
        }
        
        doc.close()
        
    except Exception as e:
        results[pdf_file] = {'error': str(e)}

# 按年份排序
sorted_papers = sorted(results.items(), key=lambda x: -int(x[1]['year']) if x[1]['year'].isdigit() else 0)

print("=" * 100)
print(f"共 {len(sorted_papers)} 篇论文（去重后）")
print("=" * 100)

for i, (filename, info) in enumerate(sorted_papers, 1):
    year = info.get('year', '未知')
    pages = info.get('total_pages', 0)
    candidates = info.get('candidates', [])
    print(f"{i:2d}. [{year}] {filename[:55]}")
    print(f"    页数: {pages} | 候选年份: {candidates}")
    print()

# 保存
import json
output_path = os.path.join(thesis_dir, '_thesis_years_final.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dict(sorted_papers), f, ensure_ascii=False, indent=2)

print(f"保存到: {output_path}")
