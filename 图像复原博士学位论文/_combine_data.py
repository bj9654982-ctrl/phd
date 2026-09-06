import json
import os

thesis_dir = r'J:\phd\图像复原博士学位论文'

# 读取多个来源的信息
with open(os.path.join(thesis_dir, '_thesis_summaries.json'), 'r', encoding='utf-8') as f:
    summaries = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_meta.json'), 'r', encoding='utf-8') as f:
    meta = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_toc.json'), 'r', encoding='utf-8') as f:
    toc_data = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_experiments.json'), 'r', encoding='utf-8') as f:
    experiments = json.load(f)

# 整合信息
thesis_list = []
for filename in summaries.keys():
    s = summaries.get(filename, {})
    m = meta.get(filename, {})
    t = toc_data.get(filename, {})
    e = experiments.get(filename, {})
    
    # 确定年份：优先用summaries中的，其次用meta中的
    year = s.get('year', '') or m.get('year', '')
    
    # 去掉重复的董伟达论文
    if filename == '散射介质退化图像复原技术研究_董伟达.pdf':
        continue
    
    thesis_list.append({
        'filename': filename,
        'year': year,
        'school': s.get('school', '') or m.get('school', ''),
        'major': s.get('major', ''),
        'advisor': s.get('advisor', '') or m.get('advisor', ''),
        'keywords': s.get('keywords', ''),
        'abstract': s.get('abstract', ''),
        'total_pages': t.get('total_pages', 0),
        'toc': t.get('toc', []),
        'experiment_sections': e.get('experiment_sections', [])
    })

# 按年份排序
def sort_key(item):
    try:
        return -int(item['year'])
    except:
        return 0

thesis_list.sort(key=sort_key)

# 打印概览
print(f"共 {len(thesis_list)} 篇论文（去重后）\n")
print("=" * 100)
for i, t in enumerate(thesis_list, 1):
    year_str = t['year'] if t['year'] else '未知'
    print(f"{i:2d}. [{year_str}] {t['filename'][:50]}")
    print(f"    学校: {t['school']} | 导师: {t['advisor']} | 页数: {t['total_pages']}")
    print(f"    关键词: {t['keywords'][:80]}")
    print(f"    实验章节数: {len(t['experiment_sections'])}")
    print()

# 保存整合后的数据
output_path = os.path.join(thesis_dir, '_thesis_combined.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(thesis_list, f, ensure_ascii=False, indent=2)

print(f"\n整合数据已保存到: {output_path}")
