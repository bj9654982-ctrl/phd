import torch
from collections import defaultdict

# 加载预训练模型
model_path = r"J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\TMT-20260807T140132Z-1-001\TMT\TMT_dynamic.pth"
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

print("=== 网络结构分析 ===\n")

# 按前缀分组
groups = defaultdict(list)
for name, param in state_dict.items():
    prefix = name.split('.')[0]
    groups[prefix].append((name, param.shape))

print("【顶层模块】")
for prefix in sorted(groups.keys()):
    print(f"  {prefix}: {len(groups[prefix])} 个参数")

print("\n=== 编码器结构 ===")
# 分析各层
for level in ['l1', 'l2', 'l3', 'l4']:
    blocks = set()
    channels = None
    for name, shape in state_dict.items():
        if f'encode_{level}' in name:
            block_id = name.split('.')[1]
            blocks.add(block_id)
            if channels is None and 'norm1.LN.weight' in name:
                channels = shape[0]
    if blocks:
        print(f"  encode_{level}: {len(blocks)} 个block, 通道数={channels}")

print("\n=== 解码器结构 ===")
for level in ['l1', 'l2', 'l3', 'l4']:
    blocks = set()
    channels = None
    for name, shape in state_dict.items():
        if f'decode_{level}' in name:
            block_id = name.split('.')[1]
            blocks.add(block_id)
            if channels is None and 'norm1.LN.weight' in name:
                channels = shape[0]
    if blocks:
        print(f"  decode_{level}: {len(blocks)} 个block, 通道数={channels}")

print("\n=== 下采样/上采样层 ===")
for name, shape in state_dict.items():
    if 'down' in name or 'up' in name:
        if 'weight' in name and len(shape) == 5:
            print(f"  {name}: {list(shape)}")

print(f"\n总参数量: {sum(p.numel() for p in state_dict.values())/1e6:.2f} M")
