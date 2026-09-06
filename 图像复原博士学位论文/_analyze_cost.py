import json
import os
import re

thesis_dir = r'J:\phd\图像复原博士学位论文'

with open(os.path.join(thesis_dir, '_thesis_analysis.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(os.path.join(thesis_dir, '_thesis_years_accurate.json'), 'r', encoding='utf-8') as f:
    years_data = json.load(f)

# GPU价格参考（2026年市场行情，单位：元）
gpu_prices = {
    'RTX 4090': 16000,
    'RTX 4090Ti': 20000,
    'RTX 3090': 5500,      # 二手
    'RTX 3090Ti': 7000,    # 二手
    'RTX 2080Ti': 2500,    # 二手
    'RTX 2070S': 1500,     # 二手
    'RTX 2080': 2000,      # 二手
    'Titan RTX': 8000,     # 二手
    'Tesla V100': 6000,    # 二手
    'Tesla A100': 50000,
    'RTX A6000': 15000,    # 二手
    'RTX 8000': 12000,     # 二手
    'GTX 1080Ti': 1200,    # 二手
    'GTX 1080': 800,       # 二手
    'RTX 4070Ti': 5000,
    'RTX 3080': 3000,      # 二手
    'RTX 3070': 2000,      # 二手
    'CPU only': 3000,      # 仅CPU
    'MATLAB only': 3000,   # 仅MATLAB+CPU
}

# 其他硬件成本估算（元）
base_cost = {
    'cpu': 2000,      # 普通CPU
    'ram': 1500,      # 32-64GB内存
    'storage': 1000,  # 1TB SSD
    'motherboard': 1000,
    'psu': 800,
    'case': 500,
    'cooling': 500,
}

def extract_gpu_info(env_list):
    """从环境列表中提取GPU信息"""
    gpu_model = None
    gpu_count = 1
    
    env_text = ' '.join(env_list)
    
    # 匹配GPU型号
    gpu_patterns = [
        (r'RTX\s*4090', 'RTX 4090'),
        (r'4090', 'RTX 4090'),
        (r'RTX\s*3090', 'RTX 3090'),
        (r'3090', 'RTX 3090'),
        (r'RTX\s*2080\s*Ti', 'RTX 2080Ti'),
        (r'2080\s*Ti', 'RTX 2080Ti'),
        (r'RTX\s*2080', 'RTX 2080'),
        (r'2080', 'RTX 2080'),
        (r'RTX\s*2070\s*S', 'RTX 2070S'),
        (r'2070\s*Super', 'RTX 2070S'),
        (r'Titan\s*RTX', 'Titan RTX'),
        (r'Tesla\s*V100', 'Tesla V100'),
        (r'V100', 'Tesla V100'),
        (r'Tesla\s*A100', 'Tesla A100'),
        (r'A100', 'Tesla A100'),
        (r'RTX\s*A6000', 'RTX A6000'),
        (r'A6000', 'RTX A6000'),
        (r'RTX\s*8000', 'RTX 8000'),
        (r'8000', 'RTX 8000'),
        (r'GTX\s*1080\s*Ti', 'GTX 1080Ti'),
        (r'1080\s*Ti', 'GTX 1080Ti'),
        (r'GTX\s*1080', 'GTX 1080'),
        (r'1080', 'GTX 1080'),
        (r'RTX\s*4070\s*Ti', 'RTX 4070Ti'),
        (r'4070\s*Ti', 'RTX 4070Ti'),
        (r'RTX\s*3080', 'RTX 3080'),
        (r'3080', 'RTX 3080'),
        (r'RTX\s*3070', 'RTX 3070'),
        (r'3070', 'RTX 3070'),
    ]
    
    for pattern, model in gpu_patterns:
        if re.search(pattern, env_text, re.IGNORECASE):
            gpu_model = model
            break
    
    # 匹配GPU数量
    count_patterns = [
        r'(\d+)\s*块\s*(?:NVIDIA\s*)?(?:GeForce\s*)?RTX',
        r'(\d+)\s*张\s*(?:NVIDIA\s*)?(?:GeForce\s*)?RTX',
        r'(\d+)\s*个\s*(?:NVIDIA\s*)?(?:GeForce\s*)?RTX',
        r'(\d+)\s*卡',
        r'(\d+)x\s*(?:RTX|GTX|Tesla)',
        r'(\d+)\s*\*\s*(?:RTX|GTX|Tesla)',
        r'(\d+)\s*GPU',
        r'(\d+)\s*显卡',
    ]
    
    for pattern in count_patterns:
        match = re.search(pattern, env_text)
        if match:
            count = int(match.group(1))
            if 1 <= count <= 8:
                gpu_count = count
                break
    
    # 特殊关键词判断
    if '4块' in env_text or '4卡' in env_text or '4张' in env_text:
        gpu_count = 4
    elif '2块' in env_text or '2卡' in env_text or '2张' in env_text:
        gpu_count = 2
    elif '8块' in env_text or '8卡' in env_text:
        gpu_count = 8
    
    return gpu_model, gpu_count

def estimate_cost(gpu_model, gpu_count, has_gpu):
    """估算硬件总成本"""
    if not has_gpu:
        return 3000  # 仅CPU
    
    gpu_price = gpu_prices.get(gpu_model, 5000)  # 默认5000
    gpu_total = gpu_price * gpu_count
    
    # 基础硬件成本（随GPU数量调整）
    base = sum(base_cost.values())
    if gpu_count >= 4:
        base += 3000  # 多卡需要更好的主板、电源、机箱
    elif gpu_count >= 2:
        base += 1000
    
    return gpu_total + base

# 分析每篇论文
results = []

for filename, info in data.items():
    year = years_data.get(filename, {}).get('year', '未知')
    title = info.get('title', filename.replace('.pdf', ''))
    env_list = info.get('experiment_env', [])
    
    # 判断是否有GPU
    has_gpu = any('GPU' in env or '显卡' in env or 'RTX' in env or 'GTX' in env or 'Tesla' in env for env in env_list)
    has_matlab = any('MATLAB' in env or 'Matlab' in env for env in env_list)
    has_pytorch = any('PyTorch' in env for env in env_list)
    
    gpu_model, gpu_count = extract_gpu_info(env_list)
    
    # 估算成本
    if has_gpu and gpu_model:
        cost = estimate_cost(gpu_model, gpu_count, True)
        gpu_desc = f"{gpu_count} × {gpu_model}"
    elif has_gpu:
        cost = 8000  # 未知GPU，估算
        gpu_desc = "GPU（型号未明确）"
    elif has_matlab:
        cost = 5000  # MATLAB + 普通CPU
        gpu_desc = "仅CPU + MATLAB"
    else:
        cost = 4000
        gpu_desc = "CPU为主"
    
    results.append({
        'filename': filename,
        'title': title[:50],
        'year': year,
        'gpu_desc': gpu_desc,
        'gpu_model': gpu_model,
        'gpu_count': gpu_count,
        'has_gpu': has_gpu,
        'estimated_cost': cost,
        'env_count': len(env_list),
    })

# 按成本排序
results.sort(key=lambda x: x['estimated_cost'])

print("=" * 100)
print("22篇论文实验环境成本估算（从低到高）")
print("=" * 100)
print(f"{'排名':<4} {'年份':<6} {'GPU配置':<25} {'估算成本(元)':<12} {'论文标题'}")
print("-" * 100)

for i, r in enumerate(results, 1):
    print(f"{i:<4} {r['year']:<6} {r['gpu_desc']:<25} {r['estimated_cost']:<12,} {r['title']}")

print("\n" + "=" * 100)
print(f"最便宜的3篇论文：")
print("=" * 100)
for i, r in enumerate(results[:3], 1):
    print(f"\n第{i}名：{r['title']}")
    print(f"  年份：{r['year']}")
    print(f"  GPU配置：{r['gpu_desc']}")
    print(f"  估算硬件成本：约 {r['estimated_cost']:,} 元")

# 保存结果
output_path = os.path.join(thesis_dir, '_thesis_cost_analysis.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n详细数据已保存到: {output_path}")
