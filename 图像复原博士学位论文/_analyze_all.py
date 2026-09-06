import fitz
import os
import re
import json

thesis_dir = r'J:\phd\图像复原博士学位论文'

def normalize_text(text):
    """规范化文本：全角转半角，去除多余空格"""
    # 全角数字和字母转半角
    fullwidth = '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
    halfwidth = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for fw, hw in zip(fullwidth, halfwidth):
        text = text.replace(fw, hw)
    
    # 全角符号
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('，', ',').replace('。', '.')
    text = text.replace('：', ':').replace('；', ';')
    text = text.replace('［', '[').replace('］', ']')
    text = text.replace('｛', '{').replace('｝', '}')
    text = text.replace('＋', '+').replace('－', '-')
    text = text.replace('＝', '=').replace('／', '/')
    
    return text

def extract_experiment_info(pdf_path):
    """从PDF中提取实验相关信息"""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for i in range(len(doc)):
        page_text = doc[i].get_text()
        page_text = normalize_text(page_text)
        full_text += page_text + "\n"
    
    doc.close()
    
    info = {
        'experiment_env': [],
        'datasets': [],
        'evaluation_metrics': [],
        'methods': [],
        'results': [],
        'implementation_details': []
    }
    
    # 1. 提取实验环境
    env_patterns = [
        r'(?:实验环境|实验平台|硬件环境|软件环境|实验设置|实验配置)[：:].*?[\n。]',
        r'(?:CPU|GPU|显卡|处理器|内存)[：:].*?[\n。]',
        r'(?:Python|PyTorch|TensorFlow|MATLAB|Matlab)[\s\d.]*',
        r'(?:CUDA|cuDNN)[\s\d.]*',
        r'(?:Intel|AMD|NVIDIA)[^，。\n]*',
        r'(?:操作系统|OS|Windows|Linux|Ubuntu)[^，。\n]*',
    ]
    
    for pattern in env_patterns:
        matches = re.findall(pattern, full_text)
        for m in matches:
            m = m.strip()
            if len(m) > 5 and m not in info['experiment_env']:
                info['experiment_env'].append(m)
    
    # 2. 提取数据集
    dataset_patterns = [
        r'([A-Z][A-Za-z0-9_+\-]*\s*(?:数据集|数据库|dataset))',
        r'(?:数据集|数据库)[：:][^。\n]*',
        r'(\d+\s*(?:张|幅|个|条|样本)[^。\n]*(?:图像|图片|样本))',
        r'(?:训练集|测试集|验证集)[：:][^。\n]*',
    ]
    
    for pattern in dataset_patterns:
        matches = re.findall(pattern, full_text)
        for m in matches:
            if isinstance(m, tuple):
                m = m[0]
            m = m.strip()
            if len(m) > 3 and m not in info['datasets']:
                info['datasets'].append(m)
    
    # 3. 提取评价指标
    metric_patterns = [
        r'(PSNR|SSIM|MSE|MAE|RMSE|FID|LPIPS|NIQE|BRISQUE)[^，。\n]*',
        r'(准确率|精确率|召回率|精度|F1|F-?measure|mAP|IoU)[^，。\n]*',
        r'(?:评价指标|性能指标|评估指标)[：:][^。\n]*',
        r'(AUC|ROC|PR\s*曲线)[^，。\n]*',
    ]
    
    for pattern in metric_patterns:
        matches = re.findall(pattern, full_text)
        for m in matches:
            if isinstance(m, tuple):
                m = m[0]
            m = m.strip()
            if len(m) > 2 and m not in info['evaluation_metrics']:
                info['evaluation_metrics'].append(m)
    
    # 4. 提取方法/模型
    method_patterns = [
        r'(?:提出|本文|本章)(?:的|提出了)?([^。\n]{5,30}(?:方法|算法|模型|网络|框架|策略))',
        r'(?:基于)([^。\n]{5,30})(?:的|方法|算法|模型|网络)',
        r'(CNN|CNN|Transformer|U-Net|UNet|GAN|VAE|Diffusion|扩散模型|注意力机制)[^，。\n]*',
        r'(?:损失函数|loss)[：:][^。\n]*',
        r'(?:优化器|optimizer)[：:][^。\n]*',
        r'(?:学习率|learning rate|lr)[=：:][^，。\n]*',
        r'(?:batch size|批次大小)[=：:][^，。\n]*',
        r'(?:epoch|迭代次数|训练轮次)[=：:][^，。\n]*',
    ]
    
    for pattern in method_patterns:
        matches = re.findall(pattern, full_text)
        for m in matches:
            if isinstance(m, tuple):
                m = m[0]
            m = m.strip()
            if len(m) > 4 and m not in info['methods']:
                info['methods'].append(m)
    
    # 5. 提取实验结果（数值）
    result_patterns = [
        r'(PSNR|SSIM|MSE|MAE|RMSE|FID|LPIPS)[\s:：]*([\d.]+)',
        r'(准确率|精确率|召回率|精度|F1|mAP|IoU)[\s:：]*([\d.]+%?)',
        r'(?:提升|提高|增加|降低|减少)[了]?([\d.]+%?)',
    ]
    
    for pattern in result_patterns:
        matches = re.findall(pattern, full_text)
        for m in matches:
            if isinstance(m, tuple):
                m = ' '.join(m)
            m = m.strip()
            if len(m) > 3 and m not in info['results']:
                info['results'].append(m)
    
    # 6. 提取实施细节
    impl_patterns = [
        r'(?:算法|步骤|流程)[：:].*?[\n。]',
        r'(?:输入|输出)[：:][^。\n]*',
        r'(?:参数|超参数)[：:][^。\n]*',
        r'(?:训练|测试|推理)(?:过程|流程|步骤)[：:][^。\n]*',
        r'(?:消融实验|消融研究)[：:][^。\n]*',
        r'(?:对比方法|基线方法|baseline)[：:][^。\n]*',
    ]
    
    for pattern in impl_patterns:
        matches = re.findall(pattern, full_text)
        for m in matches:
            m = m.strip()
            if len(m) > 5 and m not in info['implementation_details']:
                info['implementation_details'].append(m)
    
    # 限制每个列表的长度，避免过长
    for key in info:
        info[key] = info[key][:30]
    
    return info

# 主流程
pdf_files = [f for f in os.listdir(thesis_dir) if f.endswith('.pdf') and not f.startswith('_')]
pdf_files = [f for f in pdf_files if f != '散射介质退化图像复原技术研究_董伟达.pdf']

# 读取年份数据
with open(os.path.join(thesis_dir, '_thesis_years_accurate.json'), 'r', encoding='utf-8') as f:
    years_data = json.load(f)

# 读取摘要数据
with open(os.path.join(thesis_dir, '_thesis_summaries.json'), 'r', encoding='utf-8') as f:
    summaries = json.load(f)

all_results = {}

for i, pdf_file in enumerate(pdf_files, 1):
    pdf_path = os.path.join(thesis_dir, pdf_file)
    print(f"[{i}/{len(pdf_files)}] 分析: {pdf_file[:50]}...")
    
    try:
        info = extract_experiment_info(pdf_path)
        
        # 添加年份和摘要信息
        year = years_data.get(pdf_file, {}).get('year', '未知')
        summary = summaries.get(pdf_file, {})
        
        info['year'] = year
        info['title'] = summary.get('title', pdf_file.replace('.pdf', ''))
        info['author'] = summary.get('author', '')
        info['school'] = summary.get('school', '')
        info['keywords'] = summary.get('keywords', '')
        info['abstract'] = summary.get('abstract', '')[:500]
        
        all_results[pdf_file] = info
        
        print(f"  年份: {year} | 实验环境: {len(info['experiment_env'])} | 数据集: {len(info['datasets'])} | 指标: {len(info['evaluation_metrics'])}")
        
    except Exception as e:
        print(f"  Error: {e}")
        all_results[pdf_file] = {'error': str(e)}

# 按年份排序
def sort_key(item):
    try:
        return -int(item[1].get('year', '0'))
    except:
        return 0

sorted_results = sorted(all_results.items(), key=sort_key)

# 保存结果
output_path = os.path.join(thesis_dir, '_thesis_analysis.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dict(sorted_results), f, ensure_ascii=False, indent=2)

print(f"\n分析完成！共 {len(sorted_results)} 篇论文")
print(f"结果保存到: {output_path}")
