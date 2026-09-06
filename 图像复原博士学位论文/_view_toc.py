import json

with open(r'J:\phd\图像复原博士学位论文\_thesis_toc.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查看第一篇论文的TOC结构
first_key = list(data.keys())[0]
print(f'论文: {first_key}')
print(f'页数: {data[first_key]["total_pages"]}')
print('目录:')
for item in data[first_key]['toc']:
    indent = '  ' * (item['level'] - 1)
    print(f'{indent}P{item["page"]}: {item["title"]}')
