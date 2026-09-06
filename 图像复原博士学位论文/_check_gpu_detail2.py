import fitz
import os
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'

papers_to_check = [
    '基于表征学习的水下图像清晰化方法研究-张得欢.pdf',
    '基于水下光学图像清晰化的目标检测方法研究-崔峥.pdf',
    '基于消化道疾病的共聚焦内窥镜图像分析算法研究-周经俊.pdf',
    '基于加权关键帧与FSC-McCNN的浮选过程双目相机标定-罗金.pdf',
]

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

for pdf_file in papers_to_check:
    pdf_path = os.path.join(thesis_dir, pdf_file)
    
    try:
        doc = fitz.open(pdf_path)
        
        gpu_pages = []
        for i in range(len(doc)):
            text = normalize_text(doc[i].get_text())
            if any(kw in text for kw in ['GPU', '显卡', '实验环境', '实验平台', '硬件环境', '实验设置', '训练环境']):
                gpu_pages.append(i)
        
        print(f"\n{'='*80}")
        print(f"论文: {pdf_file[:50]}")
        print(f"总页数: {len(doc)} | 找到 {len(gpu_pages)} 个相关页面")
        print(f"{'='*80}")
        
        gpu_info = []
        for page_num in gpu_pages:
            text = normalize_text(doc[page_num].get_text())
            sentences = re.split(r'[.。;；\n]', text)
            for s in sentences:
                if any(kw in s for kw in ['GPU', '显卡', 'RTX', 'GTX', 'Tesla', 'A100', '3090', '4090', '2080', '1080', 'A6000', '8000', 'V100']):
                    if len(s) > 10 and len(s) < 300:
                        gpu_info.append(s.strip())
        
        gpu_info = list(dict.fromkeys(gpu_info))
        for info in gpu_info[:8]:
            print(f"  - {info}")
        
        doc.close()
        
    except Exception as e:
        print(f"Error: {e}")
