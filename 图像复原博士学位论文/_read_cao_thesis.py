import fitz
import os
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'
pdf_file = '基于多信息融合的高温铸坯表面缺陷在线检测方法-曹翔宇.pdf'
pdf_path = os.path.join(thesis_dir, pdf_file)

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

# 1. 提取完整目录
print("=" * 80)
print("论文完整目录结构")
print("=" * 80)

toc = doc.get_toc()
for item in toc:
    level, title, page = item
    indent = "  " * (level - 1)
    print(f"{indent}P{page}: {title}")

# 2. 提取摘要
print("\n" + "=" * 80)
print("摘要")
print("=" * 80)

# 摘要通常在前几页
for i in range(5, 15):
    text = normalize_text(doc[i].get_text())
    if '摘要' in text or 'Abstract' in text:
        # 找摘要内容
        idx = text.find('摘要')
        if idx >= 0:
            abstract = text[idx:idx+2000]
            print(abstract[:1500])
            break

# 3. 提取各章节核心内容
print("\n" + "=" * 80)
print("各章节核心内容概览")
print("=" * 80)

# 找关键章节的起始页
chapters = {}
for item in toc:
    level, title, page = item
    if level == 1 and ('第' in title or '章' in title):
        chapters[title] = page

print(f"\n共 {len(chapters)} 个主要章节:")
for title, page in chapters.items():
    print(f"  - {title} (第{page}页)")

doc.close()
