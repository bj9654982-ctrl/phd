import json
import os

thesis_dir = r'J:\phd\图像复原博士学位论文'

with open(os.path.join(thesis_dir, '_thesis_analysis.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# 按年份排序
def sort_key(item):
    try:
        return -int(item[1].get('year', '0'))
    except:
        return 0

sorted_papers = sorted(data.items(), key=sort_key)

print(f"共 {len(sorted_papers)} 篇论文\n")
print("=" * 100)

for i, (filename, info) in enumerate(sorted_papers, 1):
    year = info.get('year', '未知')
    title = info.get('title', filename.replace('.pdf', ''))[:50]
    keywords = info.get('keywords', '')[:60]
    
    print(f"\n{i:2d}. [{year}] {title}")
    print(f"    文件: {filename}")
    print(f"    关键词: {keywords}")
    print(f"    实验环境 ({len(info.get('experiment_env', []))}): {info.get('experiment_env', [])[:5]}")
    print(f"    数据集 ({len(info.get('datasets', []))}): {info.get('datasets', [])[:5]}")
    print(f"    评价指标 ({len(info.get('evaluation_metrics', []))}): {info.get('evaluation_metrics', [])[:5]}")
    print(f"    方法 ({len(info.get('methods', []))}): {info.get('methods', [])[:5]}")
    print(f"    结果 ({len(info.get('results', []))}): {info.get('results', [])[:5]}")
