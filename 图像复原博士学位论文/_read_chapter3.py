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

# 读取第3章（第52-70页）
print("=" * 100)
print("第3章：基于红外与可见光融合的氧化铁皮剔除方法")
print("=" * 100)

for page_num in range(51, 70):  # 0-indexed, so 51 = page 52
    text = normalize_text(doc[page_num].get_text())
    print(f"\n--- 第 {page_num+1} 页 ---")
    print(text[:2000])

doc.close()
