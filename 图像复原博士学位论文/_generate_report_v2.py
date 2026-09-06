import json
import os
import re

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

def clean_text(text):
    """清理文本：全角转半角，去除多余空格和特殊字符"""
    if not text:
        return ""
    
    # 全角数字和字母转半角
    fullwidth = '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
    halfwidth = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for fw, hw in zip(fullwidth, halfwidth):
        text = text.replace(fw, hw)
    
    # 全角符号
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('，', ', ').replace('。', '. ')
    text = text.replace('：', ': ').replace('；', '; ')
    text = text.replace('［', '[').replace('］', ']')
    text = text.replace('｛', '{').replace('｝', '}')
    text = text.replace('＋', '+').replace('－', '-')
    text = text.replace('＝', '=').replace('／', '/')
    
    # 去除特殊空格字符
    text = text.replace('\ue5e5', ' ')
    text = text.replace('\u3000', ' ')
    
    # 合并多个空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def is_valid_env(item):
    """判断是否为有效的实验环境条目"""
    env_keywords = ['PyTorch', 'TensorFlow', 'MATLAB', 'Matlab', 'Python', 'CUDA', 'cuDNN',
                    'GPU', 'CPU', 'NVIDIA', 'Intel', 'AMD', 'RTX', 'GTX', 'Tesla',
                    '内存', 'RAM', '操作系统', 'Windows', 'Linux', 'Ubuntu', '服务器',
                    '工作站', '实验平台', '实验环境', '硬件环境', '软件环境']
    return any(kw in item for kw in env_keywords)

def is_valid_dataset(item):
    """判断是否为有效的数据集条目"""
    # 排除明显不是数据集的
    if len(item) < 3 or len(item) > 80:
        return False
    if '...' in item or '....' in item:
        return False
    if item.startswith('(') or item.startswith(')'):
        return False
    return True

def is_valid_result(item):
    """判断是否为有效的实验结果条目"""
    if not item or len(item) < 3 or len(item) > 80:
        return False
    # 必须包含数字
    if not re.search(r'\d', item):
        return False
    if '...' in item or '....' in item:
        return False
    return True

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
> 每篇论文包含：**实验环境、理论方法、实施步骤、数据集、实验结果** 五个维度
> ⚠️ 本文档基于PDF自动提取生成，关键数值和细节建议对照原文核实

---

## 目录

"""

# 生成目录
for i, (filename, info) in enumerate(sorted_papers, 1):
    year = info.get('year', '未知')
    title = clean_text(info.get('title', filename.replace('.pdf', '')))
    short_title = title[:45]
    md_content += f"{i}. [{year}] **{short_title}**\n"

md_content += "\n---\n\n"

# 生成每篇论文的详细内容
for i, (filename, info) in enumerate(sorted_papers, 1):
    year = info.get('year', '未知')
    title = clean_text(info.get('title', filename.replace('.pdf', '')))
    summary = summaries.get(filename, {})
    toc = toc_data.get(filename, {}).get('toc', [])
    total_pages = toc_data.get(filename, {}).get('total_pages', 0)
    
    md_content += f"## {i}. [{year}] {title}\n\n"
    
    # 基本信息
    md_content += "### 基本信息\n\n"
    md_content += f"| 项目 | 内容 |\n"
    md_content += f"|------|------|\n"
    md_content += f"| 论文题目 | {title} |\n"
    md_content += f"| 作者 | {clean_text(summary.get('author', '待补充'))} |\n"
    md_content += f"| 学校 | {clean_text(summary.get('school', '待补充'))} |\n"
    md_content += f"| 专业 | {clean_text(summary.get('major', '待补充'))} |\n"
    md_content += f"| 导师 | {clean_text(summary.get('advisor', '待补充'))} |\n"
    md_content += f"| 年份 | {year} |\n"
    md_content += f"| 页数 | {total_pages} |\n"
    md_content += f"| 关键词 | {clean_text(summary.get('keywords', '待补充'))} |\n\n"
    
    # 摘要
    abstract = clean_text(summary.get('abstract', ''))
    if abstract and len(abstract) > 50:
        md_content += "### 摘要\n\n"
        md_content += f"> {abstract[:600]}...\n\n"
    
    # 实验环境
    md_content += "### 实验环境\n\n"
    env_list = [clean_text(e) for e in info.get('experiment_env', [])]
    env_list = [e for e in env_list if is_valid_env(e) and len(e) > 5]
    # 去重
    env_list = list(dict.fromkeys(env_list))
    
    if env_list:
        for env in env_list[:15]:
            md_content += f"- {env}\n"
    else:
        md_content += "- 需从原文实验章节详细提取\n"
    md_content += "\n"
    
    # 理论方法
    md_content += "### 理论方法\n\n"
    
    # 从摘要提取主要创新点
    abstract_text = summary.get('abstract', '')
    if abstract_text:
        innovations = re.findall(r'[1-9][）)、.]\s*([^。\n]{10,100})', abstract_text)
        if innovations:
            md_content += "**主要创新点：**\n\n"
            for idx, inn in enumerate(innovations[:5], 1):
                inn_clean = clean_text(inn)
                if len(inn_clean) > 10:
                    md_content += f"{idx}. {inn_clean}\n"
            md_content += "\n"
    
    methods = [clean_text(m) for m in info.get('methods', [])]
    methods = [m for m in methods if len(m) > 8 and len(m) < 120]
    methods = list(dict.fromkeys(methods))
    
    if methods:
        md_content += "**核心方法/算法：**\n\n"
        for method in methods[:12]:
            md_content += f"- {method}\n"
    else:
        md_content += "- 需从原文方法章节详细提取\n"
    md_content += "\n"
    
    # 实施步骤
    md_content += "### 实施步骤\n\n"
    impl_details = [clean_text(d) for d in info.get('implementation_details', [])]
    impl_details = [d for d in impl_details if len(d) > 8 and len(d) < 200]
    impl_details = list(dict.fromkeys(impl_details))
    
    if impl_details:
        for detail in impl_details[:12]:
            md_content += f"- {detail}\n"
    else:
        md_content += "- 需从原文算法章节提取完整流程\n"
        md_content += "- 建议重点关注：算法伪代码、训练参数、优化器设置、学习率策略\n"
    md_content += "\n"
    
    # 数据集
    md_content += "### 数据集\n\n"
    datasets = [clean_text(d) for d in info.get('datasets', [])]
    datasets = [d for d in datasets if is_valid_dataset(d)]
    datasets = list(dict.fromkeys(datasets))
    
    if datasets:
        for ds in datasets[:15]:
            md_content += f"- {ds}\n"
    else:
        md_content += "- 需从原文实验章节详细提取\n"
    md_content += "\n"
    
    # 评价指标
    metrics = [clean_text(m) for m in info.get('evaluation_metrics', [])]
    metrics = [m for m in metrics if len(m) > 2 and len(m) < 50]
    metrics = list(dict.fromkeys(metrics))
    
    if metrics:
        md_content += "**评价指标：**\n\n"
        for metric in metrics[:10]:
            md_content += f"- {metric}\n"
        md_content += "\n"
    
    # 实验结果
    md_content += "### 实验结果\n\n"
    results = [clean_text(r) for r in info.get('results', [])]
    results = [r for r in results if is_valid_result(r)]
    results = list(dict.fromkeys(results))
    
    if results:
        md_content += "**关键数值结果：**\n\n"
        for result in results[:20]:
            md_content += f"- {result}\n"
        md_content += "\n"
        md_content += "> ⚠️ 以上为自动提取的关键数值，具体完整结果（含对比方法、消融实验等）请参考原文实验章节\n\n"
    else:
        md_content += "- 需从原文实验结果章节详细提取\n\n"
    
    # 论文章节结构
    if toc:
        md_content += "### 论文章节结构\n\n"
        md_content += "<details>\n<summary>点击展开完整目录</summary>\n\n"
        for item in toc[:50]:
            indent = "  " * (item['level'] - 1)
            title_clean = clean_text(item['title'])
            md_content += f"{indent}- P{item['page']}: {title_clean}\n"
        if len(toc) > 50:
            md_content += f"- ... 共 {len(toc)} 个目录条目\n"
        md_content += "\n</details>\n\n"
    
    md_content += "---\n\n"

# 添加附录
md_content += """## 附录：复现建议

### 通用复现步骤

1. **环境搭建**：根据论文实验环境配置对应版本的依赖
2. **数据准备**：下载论文中使用的公开数据集，按照论文描述进行预处理
3. **代码复现**：
   - 传统方法：参考算法伪代码和数学公式逐步实现
   - 深度学习方法：参考网络结构图和训练细节搭建模型
4. **实验验证**：
   - 使用论文相同的评价指标
   - 在相同数据集上进行测试
   - 对比论文报告的数值结果

### 注意事项

- 不同论文使用的深度学习框架版本可能不同，注意兼容性
- 部分数据集可能需要申请权限或付费获取
- 实验结果可能因随机种子、硬件差异而略有波动
- 建议先在小数据集上验证方法正确性，再进行完整实验

### 数据集获取建议

- 常用公开数据集（GoPro、Set5、BSD100等）可从官方网站下载
- 工业检测数据集（NEU-DET等）通常需要联系作者获取
- 水下图像数据集（UIEB、EUVP等）多为公开可下载

---

*文档生成时间：2026年8月*
*基于22篇图像复原相关博士学位论文自动提取生成*
"""

# 保存文档
output_path = os.path.join(thesis_dir, '图像复原博士论文复现指南.md')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"文档已生成: {output_path}")
print(f"总字数: {len(md_content)}")
print(f"论文数: {len(sorted_papers)}")
