"""
第8课：模型保存与加载
========================
训练好的模型需要保存下来，用于推理、部署或继续训练。

两种保存方式：
1. state_dict（推荐）：只保存模型参数，体积小，灵活
2. 整个模型：保存完整模型对象，体积大，依赖类定义

还需要保存：
- 优化器状态（用于继续训练）
- 训练轮数（epoch）
- 最佳指标（用于早停判断）

运行方式：
    conda activate torch-gpu
    python 08_save_load.py
"""

import torch
import torch.nn as nn
import os

print("=" * 60)
print("1. 定义一个简单模型")
print("=" * 60)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleModel()
print(f"模型:\n{model}")

# 随机输入，看输出
x = torch.randn(1, 10)
with torch.no_grad():
    original_output = model(x)
print(f"\n原始模型输出: {original_output}")

print("\n" + "=" * 60)
print("2. 方式一：保存 state_dict（推荐）")
print("=" * 60)

# state_dict 是一个字典，包含模型所有可学习参数
print("模型的 state_dict 键:")
for key, value in model.state_dict().items():
    print(f"  {key}: shape={value.shape}")

# 保存
save_path = 'model_state_dict.pth'
torch.save(model.state_dict(), save_path)
print(f"\n已保存 state_dict 到: {save_path}")
print(f"文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")

# 加载（需要先创建模型实例，再加载参数）
model_loaded = SimpleModel()
model_loaded.load_state_dict(torch.load(save_path, weights_only=True))
model_loaded.eval()

# 验证输出一致
with torch.no_grad():
    loaded_output = model_loaded(x)
print(f"加载后模型输出: {loaded_output}")
print(f"输出是否一致: {torch.allclose(original_output, loaded_output)}")

print("\n" + "=" * 60)
print("3. 方式二：保存整个模型（不推荐）")
print("=" * 60)

save_path2 = 'model_full.pth'
torch.save(model, save_path2)
print(f"已保存完整模型到: {save_path2}")
print(f"文件大小: {os.path.getsize(save_path2) / 1024:.1f} KB")

# 加载（不需要先创建模型实例，但需要类定义在同一文件/可导入）
model_loaded2 = torch.load(save_path2, weights_only=False)
model_loaded2.eval()
with torch.no_grad():
    output2 = model_loaded2(x)
print(f"加载后输出: {output2}")
print(f"输出是否一致: {torch.allclose(original_output, output2)}")

print("""
注意：保存整个模型的缺点：
1. 文件更大
2. 依赖模型类的定义路径，重构代码后可能加载失败
3. 不够灵活
推荐始终使用 state_dict 方式
""")

print("\n" + "=" * 60)
print("4. 保存训练检查点（Checkpoint）—— 继续训练用")
print("=" * 60)

# 实际训练中，需要保存更多信息以便断点续训
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 模拟训练几步
for _ in range(5):
    output = model(x)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

checkpoint = {
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_loss': 0.123,
    'train_losses': [0.5, 0.4, 0.3, 0.2, 0.123],
}

checkpoint_path = 'checkpoint_epoch5.pth'
torch.save(checkpoint, checkpoint_path)
print(f"检查点已保存到: {checkpoint_path}")

# 加载检查点，继续训练
model_new = SimpleModel()
optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.001)
scheduler_new = torch.optim.lr_scheduler.StepLR(optimizer_new, step_size=10, gamma=0.5)

checkpoint_loaded = torch.load(checkpoint_path, weights_only=False)
model_new.load_state_dict(checkpoint_loaded['model_state_dict'])
optimizer_new.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
scheduler_new.load_state_dict(checkpoint_loaded['scheduler_state_dict'])
start_epoch = checkpoint_loaded['epoch']
best_loss = checkpoint_loaded['best_loss']

print(f"从 epoch {start_epoch} 继续训练，best_loss={best_loss}")
print(f"优化器学习率: {optimizer_new.param_groups[0]['lr']}")

print("\n" + "=" * 60)
print("5. 保存最佳模型（Best Model）")
print("=" * 60)

# 训练时跟踪验证集指标，保存最佳的
best_acc = 0.0
best_model_path = 'best_model.pth'

# 模拟训练过程
for epoch in range(10):
    # 模拟验证准确率递增
    val_acc = 0.5 + epoch * 0.05 + torch.randn(1).item() * 0.02
    val_acc = min(val_acc, 0.99)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch+1}: 新最佳准确率 {val_acc:.4f}，模型已保存")
    else:
        print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}，未超过最佳 {best_acc:.4f}")

print(f"\n最佳模型保存在: {best_model_path}, 准确率: {best_acc:.4f}")

print("\n" + "=" * 60)
print("6. 设备相关：CPU/GPU 加载")
print("=" * 60)

# 保存时在 GPU，加载时在 CPU
model_gpu = SimpleModel()
if torch.cuda.is_available():
    model_gpu = model_gpu.cuda()
    torch.save(model_gpu.state_dict(), 'model_gpu.pth')
    print("模型在 GPU 上训练并保存")

    # 加载到 CPU（map_location 参数）
    model_cpu = SimpleModel()
    state_dict = torch.load('model_gpu.pth', map_location='cpu', weights_only=True)
    model_cpu.load_state_dict(state_dict)
    print("已用 map_location='cpu' 加载到 CPU")

    # 加载到 GPU
    model_gpu2 = SimpleModel().cuda()
    state_dict2 = torch.load('model_gpu.pth', map_location='cuda:0', weights_only=True)
    model_gpu2.load_state_dict(state_dict2)
    print("已用 map_location='cuda:0' 加载到 GPU")

print("\n" + "=" * 60)
print("7. 最佳实践总结")
print("=" * 60)
print("""
1. 始终用 state_dict 保存模型参数，不要保存整个模型对象

2. 训练检查点保存：
   - model_state_dict
   - optimizer_state_dict
   - scheduler_state_dict
   - epoch
   - best_metric
   - 其他需要恢复的状态

3. 推理时只需要保存和加载 state_dict

4. 加载时用 map_location 处理设备差异：
   - GPU训练 -> CPU推理：map_location='cpu'
   - CPU训练 -> GPU推理：map_location='cuda:0' 后 .cuda()

5. 文件命名规范：
   - best_model.pth：最佳模型
   - checkpoint_epochXX.pth：训练检查点
   - model_final.pth：最终模型

6. 加载后调用 model.eval() 设置为评估模式
""")

# 清理临时文件
for f in ['model_state_dict.pth', 'model_full.pth', 'checkpoint_epoch5.pth',
          'best_model.pth', 'model_gpu.pth']:
    if os.path.exists(f):
        os.remove(f)
print("\n临时文件已清理")

print("\n" + "=" * 60)
print("第8课完成！模型保存加载是工程化的关键。")
print("下一课：迁移学习 —— 站在巨人的肩膀上")
print("=" * 60)
