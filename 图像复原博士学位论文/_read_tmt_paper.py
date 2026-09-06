import fitz
import os
import re

pdf_path = r'J:\phd\图像复原博士学位论文\Imaging_through_the_Atmosphere_TMT.pdf'

doc = fitz.open(pdf_path)

def normalize_text(text):
    fullwidth = '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
    halfwidth = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for fw, hw in zip(fullwidth, halfwidth):
        text = text.replace(fw, hw)
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('，', ', ').replace('。', '. ')
    text = text.replace('：', ': ').replace('；', '; ')
    text = text.replace('\ue5e5', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("=" * 80)
print(f"论文总页数: {len(doc)}")
print("=" * 80)

# 1. 提取目录
print("\n【目录结构】")
toc = doc.get_toc()
if toc:
    for item in toc:
        level, title, page = item
        indent = "  " * (level - 1)
        print(f"{indent}P{page}: {title}")
else:
    print("无目录，读取前几页内容...")

# 2. 读取摘要（前2页）
print("\n" + "=" * 80)
print("【摘要/引言】")
print("=" * 80)

for i in range(min(3, len(doc))):
    text = normalize_text(doc[i].get_text())
    print(f"\n--- 第 {i+1} 页 ---")
    print(text[:3000])

# 3. 读取方法部分（找核心章节）
print("\n" + "=" * 80)
print("【核心方法/实验】")
print("=" * 80)

# 搜索关键章节
keywords = ['method', 'network', 'architecture', 'experiment', 'result', 'TMT', 'turbulence']
found_pages = []
for i in range(len(doc)):
    text = normalize_text(doc[i].get_text()).lower()
    for kw in keywords:
        if kw in text:
            found_pages.append(i)
            break

# 读取中间几页（方法部分通常在中间）
mid_start = len(doc) // 4
mid_end = len(doc) // 2
for i in range(mid_start, min(mid_end, len(doc))):
    if i % 2 == 0:  # 每隔一页读
        text = normalize_text(doc[i].get_text())
        print(f"\n--- 第 {i+1} 页 ---")
        print(text[:2000])

doc.close()
