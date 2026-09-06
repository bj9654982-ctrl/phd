import torch

# 加载预训练模型
model_path = r"J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\TMT-20260807T140132Z-1-001\TMT\TMT_dynamic.pth"
state_dict = torch.load(model_path, map_location='cpu')

print("=== 模型参数列表 ===")
print(f"总共有 {len(state_dict)} 个参数\n")

# 打印前30个参数的名称和形状
for i, (name, param) in enumerate(state_dict.items()):
    if i < 50:
        print(f"{i:3d}. {name:60s} 形状: {list(param.shape)}")

print(f"\n... 共 {len(state_dict)} 个参数")

# 统计参数量
total_params = sum(p.numel() for p in state_dict.values())
print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f} M)")
