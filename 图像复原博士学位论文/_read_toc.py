# -*- coding: utf-8 -*-
import os
import fitz

folder = r"J:\phd\图像复原博士学位论文"

# 需要读取目录的论文
targets = [
    "先验知识引导的离焦模糊图估计与图像去模糊方法研究-李晓盼.pdf",
    "基于正则化变分模型的高光谱图像复原和异常检测研究_李兰.pdf",
    "散射介质退化图像复原技术研究-董伟达.pdf",
    "面向复杂场景的无监督图像去模糊方法研究_赵随意.pdf",
    "知识驱动的图像复原方法研究-肖杰.pdf",
    "基于自适应低秩优化和自相似学习的自然（卫星）图像恢复技术_徐旭冬.pdf",
    "新型张量分解方法与理论研究及其在高维图像处理中的应用-刘运阳.pdf",
    "基于多信息融合的高温铸坯表面缺陷在线检测方法-曹翔宇.pdf",
]

for fname in targets:
    fpath = os.path.join(folder, fname)
    if not os.path.exists(fpath):
        print(f"NOT FOUND: {fname}")
        continue
    doc = fitz.open(fpath)
    print("="*80)
    print(f"论文: {fname}")
    print(f"总页数: {len(doc)}")
    # 读取目录页（通常在摘要后，第5-15页）
    toc_text = []
    for i in range(min(20, len(doc))):
        t = doc[i].get_text()
        if t and ("目  录" in t or "目录" in t or "目 录" in t):
            # 从目录页开始读
            for j in range(i, min(i+8, len(doc))):
                toc_text.append(doc[j].get_text())
            break
    if toc_text:
        full_toc = "\n".join(toc_text)
        # 提取目录行
        lines = full_toc.split("\n")
        for line in lines:
            line = line.strip()
            if line and len(line) > 3 and any(c.isdigit() for c in line[:5]):
                print(f"  {line[:100]}")
    doc.close()
    print()
