import json
import os

thesis_dir = r'J:\phd\图像复原博士学位论文'

with open(os.path.join(thesis_dir, '_thesis_experiments.json'), 'r', encoding='utf-8') as f:
    exp_data = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_years_accurate.json'), 'r', encoding='utf-8') as f:
    years_data = json.load(f)

# 按年份排序
def sort_key(item):
    filename = item[0]
    year = years_data.get(filename, {}).get('year', '0')
    try:
        return -int(year)
    except:
        return 0

sorted_papers = sorted(exp_data.items(), key=sort_key)

# 输出前5篇论文的实验内容到文件
output_file = os.path.join(thesis_dir, '_first5_papers_exp.txt')

with open(output_file, 'w', encoding='utf-8') as f:
    for idx, (filename, info) in enumerate(sorted_papers[:5]):
        year = years_data.get(filename, {}).get('year', '未知')
        f.write(f"\n{'='*100}\n")
        f.write(f"第{idx+1}篇 [{year}] {filename}\n")
        f.write(f"{'='*100}\n")
        f.write(f"实验章节数: {len(info['experiment_sections'])}\n\n")
        
        for i, sec in enumerate(info['experiment_sections']):
            f.write(f"\n--- 实验章节 {i+1}: {sec['title']} (P{sec['start_page']}-{sec['end_page']}) ---\n")
            f.write(sec['text'])
            f.write("\n" + "..."*30 + "\n")

print(f"已保存前5篇论文实验内容到: {output_file}")
