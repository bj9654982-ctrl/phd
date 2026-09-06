import fitz
import os
import json
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'

# 实验相关关键词
exp_keywords = [
    '实验', '实验设置', '实验结果', '实验环境', '数据集', '评价指标',
    '实施步骤', '算法步骤', '训练细节', '实现细节', '参数设置',
    '结果与分析', '实验与分析', '对比实验', '消融实验'
]

def is_experiment_section(title):
    """判断是否为实验相关章节"""
    for kw in exp_keywords:
        if kw in title:
            return True
    return False

def extract_experiment_sections(pdf_path, toc):
    """从PDF中提取实验相关章节的文本"""
    doc = fitz.open(pdf_path)
    results = []
    
    # 构建页码范围
    sections = []
    for i, item in enumerate(toc):
        if is_experiment_section(item['title']):
            start_page = item['page'] - 1  # 0-indexed
            # 找下一个同级或更高级章节作为结束
            end_page = len(doc)
            for j in range(i + 1, len(toc)):
                if toc[j]['level'] <= item['level']:
                    end_page = toc[j]['page'] - 1
                    break
            sections.append({
                'title': item['title'],
                'level': item['level'],
                'start_page': start_page,
                'end_page': end_page
            })
    
    # 提取文本
    for sec in sections:
        text = ""
        for p in range(sec['start_page'], min(sec['end_page'] + 1, len(doc))):
            text += doc[p].get_text()
        
        results.append({
            'title': sec['title'],
            'start_page': sec['start_page'] + 1,
            'end_page': sec['end_page'] + 1,
            'text': text[:8000]  # 限制长度
        })
    
    doc.close()
    return results

# 主流程
with open(os.path.join(thesis_dir, '_thesis_toc.json'), 'r', encoding='utf-8') as f:
    toc_data = json.load(f)

all_results = {}

for pdf_file, info in toc_data.items():
    if 'error' in info:
        continue
    
    pdf_path = os.path.join(thesis_dir, pdf_file)
    print(f"Extracting experiments from: {pdf_file}")
    
    try:
        exp_sections = extract_experiment_sections(pdf_path, info['toc'])
        all_results[pdf_file] = {
            'year': info.get('year', ''),
            'total_pages': info['total_pages'],
            'experiment_sections': exp_sections
        }
        print(f"  Found {len(exp_sections)} experiment sections")
        
    except Exception as e:
        print(f"  Error: {e}")
        all_results[pdf_file] = {'error': str(e)}

# 保存结果
output_path = os.path.join(thesis_dir, '_thesis_experiments.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {output_path}")
