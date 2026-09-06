import fitz
import os
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'

# 需要仔细检查的论文（可能有多卡或高端GPU）
papers_to_check = [
    '基于多信息融合的高温铸坯表面缺陷在线检测方法-曹翔宇.pdf',
    '先验知识引导的离焦模糊图估计与图像去模糊方法研究-李晓盼.pdf',
    '基于机器视觉的铜带表面缺陷智能检测研究-段伯伟.pdf',
    '散射介质退化图像复原技术研究-董伟达.pdf',
    '面向图像复原的自适应表征学习方法研究-汪曦.pdf',
    '深度学习驱动的图像复原：从基础模型构建到开放场景泛化-缑元彪.pdf',
    '面向视觉感知和语义理解的图像复原方法研究-杨子正.pdf',
    '知识驱动的图像复原方法研究-肖杰.pdf',
    '面向道路交通场景的低质图像增强及目标检测方法研究-李明鸿.pdf',
    '面向复杂场景的无监督图像去模糊方法研究_赵随意.pdf',
    '数据驱动的低质量图像鲁棒性目标识别方法_杨州.pdf',
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
        
        # 搜索包含GPU/显卡/实验环境的页面
        gpu_pages = []
        for i in range(len(doc)):
            text = normalize_text(doc[i].get_text())
            if any(kw in text for kw in ['GPU', '显卡', '实验环境', '实验平台', '硬件环境', '实验设置', '训练环境']):
                gpu_pages.append(i)
        
        print(f"\n{'='*80}")
        print(f"论文: {pdf_file[:50]}")
        print(f"总页数: {len(doc)} | 找到 {len(gpu_pages)} 个相关页面")
        print(f"{'='*80}")
        
        # 提取GPU相关信息
        gpu_info = []
        for page_num in gpu_pages:
            text = normalize_text(doc[page_num].get_text())
            # 找包含GPU信息的句子
            sentences = re.split(r'[.。;；\n]', text)
            for s in sentences:
                if any(kw in s for kw in ['GPU', '显卡', 'RTX', 'GTX', 'Tesla', 'A100', '3090', '4090', '2080', '1080']):
                    if len(s) > 10 and len(s) < 300:
                        gpu_info.append(s.strip())
        
        # 去重
        gpu_info = list(dict.fromkeys(gpu_info))
        for info in gpu_info[:8]:
            print(f"  - {info}")
        
        doc.close()
        
    except Exception as e:
        print(f"Error: {e}")
