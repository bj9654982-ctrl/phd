import fitz
import os
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'

# 年份未知的论文
unknown_year_papers = [
    '基于多信息融合的高温铸坯表面缺陷在线检测方法-曹翔宇.pdf',
    '基于正则化变分模型的高光谱图像复原和异常检测研究_李兰.pdf',
    '基于自适应低秩优化和自相似学习的自然（卫星）图像恢复技术_徐旭冬.pdf',
    '基于表征学习的水下图像清晰化方法研究-张得欢.pdf',
    '深度学习驱动的图像复原：从基础模型构建到开放场景泛化-缑元彪.pdf',
    '知识驱动的图像复原方法研究-肖杰.pdf',
    '面向图像复原的自适应表征学习方法研究-汪曦.pdf',
    '面向视觉感知和语义理解的图像复原方法研究-杨子正.pdf',
]

for pdf_file in unknown_year_papers:
    pdf_path = os.path.join(thesis_dir, pdf_file)
    
    try:
        doc = fitz.open(pdf_path)
        # 读取前20页
        text = ""
        for i in range(min(20, len(doc))):
            text += doc[i].get_text()
        
        # 查找各种年份模式
        year_patterns = [
            r'(20\d{2})\s*年',
            r'(20\d{2})年\d+月',
            r'答辩日期[:：\s]*(20\d{2})',
            r'提交日期[:：\s]*(20\d{2})',
            r'分类号.*?(20\d{2})',
            r'(20\d{2})\s*年\s*\d+\s*月',
            r'二〇([一二三四五六七八九〇]{2})年',  # 中文数字年份
        ]
        
        all_years = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text[:8000])
            for m in matches:
                if isinstance(m, tuple):
                    m = m[0]
                all_years.append(m)
        
        # 也查找论文最后几页的致谢/声明中的年份
        last_text = ""
        for i in range(max(0, len(doc)-5), len(doc)):
            last_text += doc[i].get_text()
        
        for pattern in year_patterns:
            matches = re.findall(pattern, last_text)
            for m in matches:
                if isinstance(m, tuple):
                    m = m[0]
                all_years.append(m)
        
        # 统计最常见的年份
        from collections import Counter
        year_counts = Counter(all_years)
        most_common = year_counts.most_common(5)
        
        print(f"\n{pdf_file}")
        print(f"  页数: {len(doc)}")
        print(f"  候选年份: {most_common}")
        
        # 找一些上下文
        for pattern in [r'答辩日期.*?\n', r'提交日期.*?\n', r'20\d{2}年\d+月.*?\n']:
            matches = re.findall(pattern, text[:5000])
            if matches:
                print(f"  上下文: {matches[:3]}")
        
        doc.close()
        
    except Exception as e:
        print(f"Error: {pdf_file} - {e}")
