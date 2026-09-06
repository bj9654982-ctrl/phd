import json
import os

thesis_dir = r'J:\phd\图像复原博士学位论文'

with open(os.path.join(thesis_dir, '_thesis_experiments.json'), 'r', encoding='utf-8') as f:
    exp_data = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_years_accurate.json'), 'r', encoding='utf-8') as f:
    years_data = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_summaries.json'), 'r', encoding='utf-8') as f:
    summaries = json.load(f)

# 按年份排序
def sort_key(item):
    filename = item[0]
    year = years_data.get(filename, {}).get('year', '0')
    try:
        return -int(year)
    except:
        return 0

sorted_papers = sorted(exp_data.items(), key=sort_key)

# 输出第一篇论文的完整实验内容
filename, info = sorted_papers[0]
print(f"论文: {filename}")
print(f"年份: {years_data.get(filename, {}).get('year', '未知')}")
print(f"实验章节数: {len(info['experiment_sections'])}")
print("=" * 100)

for i, sec in enumerate(info['experiment_sections']):
    print(f"\n--- 实验章节 {i+1}: {sec['title']} (P{sec['start_page']}-{sec['end_page']}) ---")
    print(sec['text'])
    print("..." * 20)
