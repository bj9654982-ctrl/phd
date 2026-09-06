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

# 读取所有页面
print("=" * 100)
print("TMT论文完整内容")
print("=" * 100)

for i in range(len(doc)):
    text = normalize_text(doc[i].get_text())
    print(f"\n{'='*100}")
    print(f"第 {i+1} 页")
    print('='*100)
    print(text[:4000])

doc.close()
