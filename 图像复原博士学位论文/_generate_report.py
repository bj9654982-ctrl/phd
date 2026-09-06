import json
import os

thesis_dir = r'J:\phd\图像复原博士学位论文'

# 读取所有数据
with open(os.path.join(thesis_dir, '_thesis_analysis.json'), 'r', encoding='utf-8') as f:
    analysis = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_summaries.json'), 'r', encoding='utf-8') as f:
    summaries = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_toc.json'), 'r', encoding='utf-8') as f:
    toc_data = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_years_accurate.json'), 'r', encoding='utf-8') as f:
    years_data = json.load(f)

# 按年份排序
def sort_key(item):
    filename = item[0]
    year = years_data.get(filename, {}).get('year', '0')
    try:
        return -int(year)
    except:
        return 0

sorted_papers = sorted(analysis.items(), key=sort_key)

# 生成Markdown文档
md_content = """# 图像复原博士学位论文复现指南

> 按年份从近到远排列，共 22 篇论文（去重后）
> 每篇论文包含：实验环境、理论方法、实施步骤、数据集、实验结果

---

## 目录

"""

# 生成目录
for i, (filename, info) in enumerate(sorted_papers, 1):
    year = info.get('year', '未知')
    title = info.get('title', filename.replace('.pdf', ''))
    # 简化标题
    short_title = title[:50]
    md_content += f"{i}. [{year}] **{short_title}**\n"

md_content += "\n---\n\n"

# 生成每篇论文的详细内容
for i, (filename, info) in enumerate(sorted_papers, 1):
    year = info.get('year', '未知')
    title = info.get('title', filename.replace('.pdf', ''))
    summary = summaries.get(filename, {})
    toc = toc_data.get(filename, {}).get('toc', [])
    
    md_content += f"## {i}. [{year}] {title}\n\n"
    
    # 基本信息
    md_content += "### 📋 基本信息\n\n"
    md_content += f"- **论文题目**: {title}\n"
    md_content += f"- **作者**: {summary.get('author', '待补充')}\n"
    md_content += f"- **学校**: {summary.get('school', '待补充')}\n"
    md_content += f"- **专业**: {summary.get('major', '待补充')}\n"
    md_content += f"- **导师**: {summary.get('advisor', '待补充')}\n"
    md_content += f"- **年份**: {year}\n"
    md_content += f"- **页数**: {info.get('total_pages', '待补充')}\n"
    md_content += f"- **关键词**: {summary.get('keywords', '待补充')}\n\n"
    
    # 摘要
    abstract = summary.get('abstract', '')
    if abstract:
        md_content += "### 📝 摘要\n\n"
        md_content += f"{abstract[:500]}...\n\n"
    
    # 实验环境
    md_content += "### 🖥️ 实验环境\n\n"
    env_list = info.get('experiment_env', [])
    if env_list:
        # 过滤掉明显不是环境的条目
        filtered_env = []
        for env in env_list:
            if any(kw in env for kw in ['PyTorch', 'TensorFlow', 'MATLAB', 'Python', 'CUDA', 'GPU', 'CPU', 'NVIDIA', 'Intel', 'AMD', 'RTX', 'GTX', 'Tesla', '内存', 'RAM', '操作系统', 'Windows', 'Linux', 'Ubuntu']):
                filtered_env.append(env)
        
        if filtered_env:
            for env in filtered_env[:15]:
                md_content += f"- {env}\n"
        else:
            md_content += "- 待从原文中详细提取\n"
    else:
        md_content += "- 待从原文中详细提取\n"
    md_content += "\n"
    
    # 理论方法
    md_content += "### 🧠 理论方法\n\n"
    methods = info.get('methods', [])
    if methods:
        # 从摘要提取主要方法
        abstract_text = summary.get('abstract', '')
        if abstract_text:
            # 提取主要创新点
            import re
            innovations = re.findall(r'[1-9][）)、.]\s*([^。\n]{10,80})', abstract_text)
            if innovations:
                md_content += "**主要创新点：**\n\n"
                for idx, inn in enumerate(innovations[:5], 1):
                    md_content += f"{idx}. {inn.strip()}\n"
                md_content += "\n"
        
        md_content += "**核心方法/模型：**\n\n"
        for method in methods[:10]:
            if len(method) > 5 and len(method) < 100:
                md_content += f"- {method}\n"
    else:
        md_content += "- 待从原文中详细提取\n"
    md_content += "\n"
    
    # 实施步骤
    md_content += "### 📋 实施步骤\n\n"
    impl_details = info.get('implementation_details', [])
    if impl_details:
        for detail in impl_details[:10]:
            if len(detail) > 5 and len(detail) < 200:
                md_content += f"- {detail}\n"
    else:
        md_content += "- 待从原文中详细提取算法流程和训练步骤\n"
    md_content += "\n"
    
    # 数据集
    md_content += "### 📊 数据集\n\n"
    datasets = info.get('datasets', [])
    if datasets:
        for ds in datasets[:15]:
            if len(ds) > 2 and len(ds) < 100:
                md_content += f"- {ds}\n"
    else:
        md_content += "- 待从原文中详细提取\n"
    md_content += "\n"
    
    # 评价指标
    metrics = info.get('evaluation_metrics', [])
    if metrics:
        md_content += "**评价指标：**\n\n"
        for metric in metrics[:10]:
            if len(metric) > 1 and len(metric) < 50:
                md_content += f"- {metric}\n"
        md_content += "\n"
    
    # 实验结果
    md_content += "### 📈 实验结果\n\n"
    results = info.get('results', [])
    if results:
        md_content += "**关键数值结果：**\n\n"
        for result in results[:15]:
            if len(result) > 2 and len(result) < 100:
                md_content += f"- {result}\n"
        md_content += "\n"
        md_content += "> ⚠️ 注：以上为自动提取的关键数值，具体完整结果请参考原文实验章节\n\n"
    else:
        md_content += "- 待从原文中详细提取\n\n"
    
    # 论文章节结构
    if toc:
        md_content += "### 📑 论文章节结构\n\n"
        md_content += "<details>\n<summary>点击展开目录</summary>\n\n"
        for item in toc[:40]:  # 限制数量
            indent = "  " * (item['level'] - 1)
            md_content += f"{indent}- P{item['page']}: {item['title']}\n"
        md_content += "\n</details>\n\n"
    
    md_content += "---\n\n"

# 保存文档
output_path = os.path.join(thesis_dir, '图像复原博士论文复现指南.md')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"文档已生成: {output_path}")
print(f"总字数: {len(md_content)}")
print(f"论文数: {len(sorted_papers)}")
