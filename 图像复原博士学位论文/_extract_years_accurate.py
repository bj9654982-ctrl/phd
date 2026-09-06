import fitz
import os
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'

def chinese_to_arabic(chinese_str):
    digits = {'〇': '0', '一': '1', '二': '2', '三': '3', '四': '4', 
              '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'}
    result = ''
    for ch in chinese_str:
        result += digits.get(ch, ch)
    return result

pdf_files = [f for f in os.listdir(thesis_dir) if f.endswith('.pdf') and not f.startswith('_')]
pdf_files = [f for f in pdf_files if f != '散射介质退化图像复原技术研究_董伟达.pdf']

results = {}

for pdf_file in pdf_files:
    pdf_path = os.path.join(thesis_dir, pdf_file)
    
    try:
        doc = fitz.open(pdf_path)
        
        # 只读取前5页（封面、声明、摘要等）
        text = ""
        for i in range(min(5, len(doc))):
            page_text = doc[i].get_text()
            # 转换全角数字
            fullwidth = '０１２３４５６７８９'
            halfwidth = '0123456789'
            for fw, hw in zip(fullwidth, halfwidth):
                page_text = page_text.replace(fw, hw)
            text += page_text
        
        years = []
        
        # 模式1: 20XX年X月 (更精确)
        matches = re.findall(r'(20\d{2})\s*年\s*\d+\s*月', text)
        years.extend(matches)
        
        # 模式2: 二〇XX年X月
        matches = re.findall(r'二〇([一二三四五六七八九〇]{2})年\s*\d+\s*月', text)
        for m in matches:
            years.append('20' + chinese_to_arabic(m))
        
        # 模式3: 答辩日期/提交日期/完成日期 + 年份
        for keyword in ['答辩', '提交', '完成', '定稿']:
            pattern = keyword + r'.*?(20\d{2})'
            matches = re.findall(pattern, text)
            years.extend(matches)
        
        # 模式4: 封面底部的年份
        # 找页面最后几行的年份
        lines = text.split('\n')
        for line in lines[-20:]:
            matches = re.findall(r'(20\d{2})', line)
            for m in matches:
                if 2015 <= int(m) <= 2026:
                    years.append(m)
        
        # 统计最常见的
        from collections import Counter
        year_counts = Counter(years)
        most_common = year_counts.most_common(5)
        
        year = most_common[0][0] if most_common else '未知'
        
        # 也提取标题和作者
        title = ""
        author = ""
        
        # 标题通常在第一页，字体最大
        first_page_text = doc[0].get_text()
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
        # 找最长的几行作为标题候选
        long_lines = sorted(lines, key=len, reverse=True)[:5]
        title = long_lines[0] if long_lines else ""
        
        # 找作者
        for line in lines:
            if '作者' in line or '研究生' in line:
                author_match = re.search(r'[:：]\s*(\S+)', line)
                if author_match:
                    author = author_match.group(1)
                    break
        
        results[pdf_file] = {
            'year': year,
            'title': title[:60],
            'author': author,
            'total_pages': len(doc),
            'candidates': most_common
        }
        
        doc.close()
        
    except Exception as e:
        results[pdf_file] = {'error': str(e)}

# 按年份排序
def sort_key(item):
    try:
        return -int(item[1].get('year', '0'))
    except:
        return 0

sorted_papers = sorted(results.items(), key=sort_key)

print("=" * 100)
print(f"共 {len(sorted_papers)} 篇论文（去重后）- 按年份从近到远")
print("=" * 100)

for i, (filename, info) in enumerate(sorted_papers, 1):
    year = info.get('year', '未知')
    title = info.get('title', '')[:40]
    author = info.get('author', '')
    pages = info.get('total_pages', 0)
    candidates = info.get('candidates', [])
    print(f"{i:2d}. [{year}] {filename[:50]}")
    print(f"    作者: {author} | 页数: {pages}")
    print(f"    标题: {title}")
    print(f"    候选年份: {candidates[:3]}")
    print()

# 保存
import json
output_path = os.path.join(thesis_dir, '_thesis_years_accurate.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dict(sorted_papers), f, ensure_ascii=False, indent=2)

print(f"保存到: {output_path}")
