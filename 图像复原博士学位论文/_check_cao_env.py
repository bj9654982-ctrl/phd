import fitz
import os
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'
pdf_file = '基于多信息融合的高温铸坯表面缺陷在线检测方法-曹翔宇.pdf'
pdf_path = os.path.join(thesis_dir, pdf_file)

doc = fitz.open(pdf_path)

# 全角转半角函数
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

# 搜索所有包含实验环境相关关键词的页面
env_keywords = ['实验环境', '实验平台', '硬件环境', '软件环境', '实验设置', 
                'GPU', '显卡', 'CPU', '处理器', '内存', '服务器', '工作站',
                'PyTorch', 'TensorFlow', 'MATLAB', 'CUDA', '操作系统',
                '实验配置', '训练环境', '测试环境']

print(f"论文: {pdf_file}")
print(f"总页数: {len(doc)}")
print("=" * 100)

found_pages = []

for i in range(len(doc)):
    text = doc[i].get_text()
    text_norm = normalize_text(text)
    
    for kw in env_keywords:
        if kw in text_norm:
            found_pages.append(i)
            break

found_pages = sorted(set(found_pages))
print(f"找到 {len(found_pages)} 个包含实验环境信息的页面: {found_pages}")
print()

# 详细读取这些页面
for page_num in found_pages:
    text = doc[page_num].get_text()
    text_norm = normalize_text(text)
    
    print(f"\n{'='*80}")
    print(f"第 {page_num+1} 页")
    print(f"{'='*80}")
    
    # 只打印包含环境关键词的段落
    paragraphs = text_norm.split('. ')
    for para in paragraphs:
        if any(kw in para for kw in env_keywords):
            if len(para) > 10 and len(para) < 500:
                print(f"  - {para.strip()}.")

doc.close()
