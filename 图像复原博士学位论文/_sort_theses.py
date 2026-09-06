import json

with open(r'J:\phd\图像复原博士学位论文\_thesis_summaries.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

thesis_list = []
for filename, info in data.items():
    year = info.get('year', '未知')
    school = info.get('school', '')
    major = info.get('major', '')
    advisor = info.get('advisor', '')
    keywords = info.get('keywords', '')
    abstract = info.get('abstract', '')
    total_pages = info.get('total_pages', 0)
    thesis_list.append({
        'filename': filename,
        'year': year,
        'school': school,
        'major': major,
        'advisor': advisor,
        'keywords': keywords,
        'abstract_len': len(abstract) if abstract else 0,
        'total_pages': total_pages
    })

def sort_key(item):
    try:
        return -int(item['year'])
    except:
        return 0

thesis_list.sort(key=sort_key)

for i, t in enumerate(thesis_list, 1):
    print(f"{i}. [{t['year']}] {t['filename']}")
    print(f"   学校: {t['school']} | 专业: {t['major']} | 导师: {t['advisor']}")
    print(f"   关键词: {t['keywords']}")
    print(f"   页数: {t['total_pages']} | 摘要长度: {t['abstract_len']}")
    print()
