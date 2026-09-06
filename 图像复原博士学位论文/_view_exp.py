import json

with open(r'J:\phd\图像复原博士学位论文\_thesis_experiments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查看第一篇论文的实验内容
first_key = list(data.keys())[0]
print(f"论文: {first_key}")
print(f"年份: {data[first_key]['year']}")
print(f"实验章节数: {len(data[first_key]['experiment_sections'])}")
print("=" * 80)

for i, sec in enumerate(data[first_key]['experiment_sections'][:5]):  # 只看前5个
    print(f"\n--- 实验章节 {i+1}: {sec['title']} (P{sec['start_page']}-{sec['end_page']}) ---")
    print(sec['text'][:2000])
    print("...")
