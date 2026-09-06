"""
知识点9：TMT数据集构建
========================

【为什么要学这个？】
深度学习的效果很大程度上取决于数据。
了解TMT的数据集是怎么构建的，参数怎么设置，
可以帮助你理解为什么TMT效果好，也可以指导你自己做实验。

【本教程你将学会】
1. 静态场景数据集（来源、数量、划分）
2. 动态场景数据集（来源、数量、划分）
3. 湍流参数设置（强度、核尺寸、距离、时间相关性）
4. 数据增强策略
5. 训练/测试划分
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'J:\phd\Imaging through the Atmosphere using Turbulence Mitigation Transformer复现\08_数据集'
os.makedirs(output_dir, exist_ok=True)


def demo_static_dataset():
    """
    演示1：静态场景数据集
    """
    print("=" * 60)
    print("演示1：静态场景数据集")
    print("=" * 60)
    
    print("""
【什么是静态场景？】
- 场景是静止的，没有运动物体
- 只有湍流在变化
- 比如：建筑物、山脉、树木、风景照...

【数据来源】
Place数据集（MIT的场景识别数据集）
  - 包含大量不同场景的图片
  - 室内、室外、自然、城市...
  - 多样性很好

【数据规模】
- 总序列数：9,017 个序列
- 每序列帧数：50 帧
- 总帧数：9,017 × 50 = 450,850 帧

【训练/测试划分】
- 训练集：7,499 个序列（约83%）
- 测试集：1,518 个序列（约17%）

【为什么用静态场景？】
1. 数据容易获取（现成的图像数据集就行）
2. 控制变量：只有湍流变化，没有运动干扰
3. 可以评估纯湍流复原的效果
4. 是最基础的测试场景

【怎么生成序列？】
1. 取一张清晰图像（来自Place）
2. 用湍流模拟器生成50帧不同的湍流退化
3. 这50帧就构成一个序列
4. 每帧的湍流都不一样，但底层场景一样
    """)
    
    # 可视化静态数据集的规模
    print("\n【数据集规模可视化】")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    categories = ['训练集', '测试集']
    counts = [7499, 1518]
    colors = ['#4CAF50', '#FF9800']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{count:,} 序列\n({count/9017*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('序列数量', fontsize=14)
    ax.set_title('静态场景数据集划分（共9,017序列，每序列50帧）', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 9000)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '14_静态数据集划分.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 14_静态数据集划分.png")
    plt.close()


def demo_dynamic_dataset():
    """
    演示2：动态场景数据集
    """
    print("\n" + "=" * 60)
    print("演示2：动态场景数据集")
    print("=" * 60)
    
    print("""
【什么是动态场景？】
- 场景中有运动物体
- 不仅有湍流变化，还有物体运动
- 比如：行人、车辆、动物、运动的人...

【为什么需要动态场景？】
- 真实世界很多场景都是动态的
- 静态场景训练的模型，在动态场景上可能效果差
- 动态场景更难，也更有挑战性

【数据来源】
两个来源：
1. SVW（Sports Videos in the Wild）
   - 体育视频数据集
   - 包含各种运动场景
   - 有很多人体运动

2. TSRWGAN生成的视频
   - 用GAN生成的视频
   - 补充更多样的动态场景
   - 增加数据多样性

【数据规模】
- 总视频数：4,684 个视频
- 总帧数：1,979,564 帧
- 平均每个视频：约422帧

【训练/测试划分】
- 训练集：3,500 个视频（约75%）
- 测试集：1,184 个视频（约25%）

【和静态场景的区别】
  静态场景：一张图生成50帧，只有湍流变化
  动态场景：视频本身就有运动，再加湍流变化

【为什么更难？】
1. 运动和湍流混在一起，更难分离
2. 运动模糊和湍流模糊叠加
3. 需要同时处理运动和湍流
4. 时间相关性更复杂

【TMT的优势】
因为TMT用了TCJA，能处理更多帧，
所以在动态场景上也能很好地利用时间信息。
    """)
    
    # 对比静态和动态数据集
    print("\n【静态 vs 动态数据集对比】")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(2)
    width = 0.35
    
    static_counts = [7499, 1518]
    dynamic_counts = [3500, 1184]
    
    bars1 = ax.bar(x - width/2, static_counts, width, label='静态场景', color='#4CAF50', edgecolor='black')
    bars2 = ax.bar(x + width/2, dynamic_counts, width, label='动态场景', color='#2196F3', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['训练集', '测试集'], fontsize=14)
    ax.set_ylabel('序列/视频数量', fontsize=14)
    ax.set_title('静态场景 vs 动态场景数据集规模对比', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '15_静态vs动态数据集对比.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 15_静态vs动态数据集对比.png")
    plt.close()


def demo_turbulence_parameters():
    """
    演示3：湍流参数设置
    """
    print("\n" + "=" * 60)
    print("演示3：湍流参数设置")
    print("=" * 60)
    
    print("""
【为什么参数设置很重要？】
- 参数决定了训练数据的分布
- 参数设置得好，网络泛化能力强
- 参数设置不好，网络过拟合或欠拟合
- 要覆盖各种可能的湍流情况

【主要参数】

1. 湍流强度（D/r0）
   - D: 孔径直径
   - r0: Fried参数（相干长度）
   - D/r0越大，湍流越强
   - 范围：0.001 ~ 0.2（弱到强）

2. 核尺寸（PSF大小）
   - 静态场景：固定33×33
   - 动态场景：9 ~ 33 随机采样
   - 核越大，模糊越严重

3. 距离（成像距离）
   - 范围：50 ~ 2000 米
   - 距离越远，湍流越强
   - 增加数据多样性

4. 时间相关性（α）
   - 范围：0.2 ~ 0.95
   - α大：变化慢，强相关
   - α小：变化快，弱相关
   - 随机采样，覆盖各种情况

【湍流强度的三档】
  弱湍流：D/r0 小
  中湍流：D/r0 中等
  强湍流：D/r0 大

静态场景的比例：
  弱:中:强 = 20% : 40% : 40%

动态场景的比例：
  弱:中:强 = 33% : 33% : 33%

【为什么这么设置？】
- 强湍流更难，所以多一些
- 但也不能全是强湍流，弱的也要学
- 动态场景平均分配，因为更复杂
- 各种参数都覆盖，网络才能泛化
    """)
    
    # 可视化参数分布
    print("\n【参数分布可视化】")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 湍流强度分布（静态场景）
    ax = axes[0, 0]
    intensities = ['弱湍流', '中湍流', '强湍流']
    static_ratio = [20, 40, 40]
    dynamic_ratio = [33, 33, 33]
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, static_ratio, width, label='静态场景', color='#4CAF50', edgecolor='black')
    ax.bar(x + width/2, dynamic_ratio, width, label='动态场景', color='#2196F3', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(intensities, fontsize=12)
    ax.set_ylabel('比例 (%)', fontsize=12)
    ax.set_title('湍流强度分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. 核尺寸范围
    ax = axes[0, 1]
    kernel_sizes = ['静态场景', '动态场景']
    kernel_min = [33, 9]
    kernel_max = [33, 33]
    
    ax.barh(kernel_sizes, [33, 33], left=[0, 0], color='lightgray', edgecolor='black', alpha=0.5, label='范围')
    ax.barh(kernel_sizes, [33, 24], left=[0, 9], color='#FF9800', edgecolor='black', label='实际范围')
    
    ax.set_xlabel('核尺寸 (像素)', fontsize=12)
    ax.set_title('核尺寸范围 (9~33)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # 3. 距离范围
    ax = axes[1, 0]
    ax.barh(['成像距离'], [2000], left=[50], color='#9C27B0', edgecolor='black', height=0.3)
    ax.set_xlabel('距离 (米)', fontsize=12)
    ax.set_title('成像距离范围 (50~2000m)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2200)
    ax.grid(axis='x', alpha=0.3)
    ax.text(1000, 0, '50 ~ 2000 米', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # 4. 时间相关性
    ax = axes[1, 1]
    ax.barh(['时间相关性 α'], [0.75], left=[0.2], color='#E91E63', edgecolor='black', height=0.3)
    ax.set_xlabel('α 值', fontsize=12)
    ax.set_title('时间相关性范围 (0.2~0.95)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    ax.text(0.575, 0, '0.2 ~ 0.95', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    plt.suptitle('TMT数据集参数设置', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '16_湍流参数设置.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 16_湍流参数设置.png")
    plt.close()


def demo_data_augmentation():
    """
    演示4：数据增强策略
    """
    print("\n" + "=" * 60)
    print("演示4：数据增强策略")
    print("=" * 60)
    
    print("""
【为什么需要数据增强？】
- 增加数据多样性
- 防止过拟合
- 提高网络泛化能力
- 用有限的数据获得更好的效果

【TMT用了哪些数据增强？】

1. 随机Gamma校正
   - 随机改变图像的亮度/对比度
   - Gamma值随机采样
   - 模拟不同光照条件

2. 随机饱和度
   - 随机改变颜色饱和度
   - 增加颜色多样性
   - 对彩色图像有效

3. 随机裁剪
   - 从大图中随机裁剪小patch
   - 增加空间多样性
   - 训练时用patch，测试时用全图

4. 随机翻转
   - 水平翻转
   - 垂直翻转
   - 简单但有效

5. 随机旋转
   - 90度、180度、270度旋转
   - 增加旋转不变性

6. 加高斯噪声
   - 噪声水平：β ~ uniform(0, 0.0004)
   - 模拟传感器噪声
   - 提高鲁棒性

【注意：数据增强是对清晰图像做的】
- 先对清晰图像做增强
- 然后再用湍流模拟器退化
- 这样GT和输入是对应的
    """)
    
    # 可视化数据增强
    print("\n【数据增强类型汇总】")
    
    augments = [
        ('随机Gamma校正', '亮度/对比度变化', '#FF5722'),
        ('随机饱和度', '颜色饱和度变化', '#FF9800'),
        ('随机裁剪', '空间位置变化', '#FFC107'),
        ('随机翻转', '水平/垂直翻转', '#4CAF50'),
        ('随机旋转', '90°/180°/270°', '#2196F3'),
        ('高斯噪声', '传感器噪声', '#9C27B0'),
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    y_pos = np.arange(len(augments))
    names = [a[0] for a in augments]
    descs = [a[1] for a in augments]
    colors = [a[2] for a in augments]
    
    bars = ax.barh(y_pos, [1]*len(augments), color=colors, edgecolor='black', height=0.6)
    
    for i, (name, desc, color) in enumerate(augments):
        ax.text(0.5, i, f'{name}\n({desc})', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel('数据增强方法', fontsize=14)
    ax.set_title('TMT数据增强策略', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '17_数据增强策略.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 17_数据增强策略.png")
    plt.close()


def demo_training_settings():
    """
    演示5：训练设置
    """
    print("\n" + "=" * 60)
    print("演示5：训练设置")
    print("=" * 60)
    
    print("""
【分阶段训练】

TMT是分两阶段训练的：

阶段1：训练去倾斜模块（Tilt-Removal）
  - 迭代次数：400K 次
  - Batch size：4
  - 损失：多尺度Charbonnier损失
  - 目标：学习去除几何畸变（倾斜）

阶段2：训练去模糊模块（Blur-Removal）
  - 固定去倾斜模块的参数
  - 迭代次数：600K 次
  - Batch size：4
  - 损失：Charbonnier损失
  - 目标：学习去除模糊

【为什么分阶段训练？】
1. 训练更稳定：每个阶段目标明确
2. 收敛更快：先学简单的，再学难的
3. 效果更好：分而治之，每个阶段都能做好
4. 显存友好：不需要同时训两个大网络

【优化器】
- 优化器：Adam
- β1 = 0.9, β2 = 0.999
- 学习率调度：余弦退火（Cosine Annealing）
- 初始学习率：2e-4
- 最终学习率：1e-6

【学习率调度】
余弦退火：
  - 学习率从高到低，按余弦曲线下降
  - 前期学习快，后期精细调整
  - 比固定学习率效果好

【输入帧数】
- 训练时：12帧（或20帧）
- 测试时：可以用更多帧
- 更多帧 = 更多信息 = 更好效果
- TMT的优势：TCJA显存小，可以用更多帧
    """)
    
    # 可视化训练流程
    print("\n【训练流程可视化】")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # 阶段1
    ax.barh(['阶段1：去倾斜模块'], [400], left=[0], color='#4CAF50', edgecolor='black', height=0.5, label='阶段1')
    ax.text(200, 0, '400K 迭代\nBatch size=4\n多尺度Charbonnier损失', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 阶段2
    ax.barh(['阶段2：去模糊模块'], [600], left=[400], color='#2196F3', edgecolor='black', height=0.5, label='阶段2')
    ax.text(700, 0, '600K 迭代\nBatch size=4\nCharbonnier损失\n（固定去倾斜模块）', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax.set_xlabel('训练迭代次数 (K)', fontsize=14)
    ax.set_title('TMT分阶段训练流程（总计1000K迭代）', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1050)
    ax.legend(fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # 箭头
    ax.annotate('', xy=(400, 0.3), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=(1000, -0.3), xytext=(400, -0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '18_训练流程.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 18_训练流程.png")
    plt.close()


def demo_evaluation_metrics():
    """
    演示6：评估指标
    """
    print("\n" + "=" * 60)
    print("演示6：评估指标")
    print("=" * 60)
    
    print("""
【常用评估指标】

图像复原任务常用的评估指标：

1. PSNR（峰值信噪比）
   - 单位：dB
   - 越高越好
   - 最常用的指标
   - 基于像素级的MSE

2. SSIM（结构相似性）
   - 范围：0~1
   - 越高越好
   - 考虑亮度、对比度、结构
   - 更符合人眼感知

3. LPIPS（学习感知图像块相似度）
   - 基于深度特征
   - 越低越好
   - 更符合人眼感知
   - 最近几年越来越常用

【TMT用了哪些指标？】
- PSNR：主要指标
- SSIM：主要指标
- LPIPS：辅助指标
- 运行时间：速度指标
- 参数量：模型大小

【TMT的实验结果（静态场景，12帧）】
  PSNR: 27.73 dB
  SSIM: 0.8341
  参数量: 26.04M
  速度: 1.52秒/帧（540×960）

【更多帧的效果】
  4帧:  PSNR ≈ 26.x dB
  8帧:  PSNR ≈ 27.x dB
  12帧: PSNR = 27.73 dB
  20帧: PSNR = 28.44 dB
  
  → 帧越多，效果越好！
  → 这就是TMT的优势：能处理更多帧
    """)
    
    # 可视化不同帧数的效果
    print("\n【不同帧数的PSNR对比】")
    
    frames = [4, 8, 12, 20]
    psnr = [26.5, 27.2, 27.73, 28.44]  # 近似值
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(frames, psnr, 'o-', linewidth=3, markersize=12, color='#E91E63')
    
    for f, p in zip(frames, psnr):
        ax.text(f, p + 0.1, f'{p:.2f} dB', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('输入帧数', fontsize=14)
    ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_title('不同输入帧数的复原效果（静态场景）', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(frames)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '19_不同帧数PSNR对比.png'), dpi=150, bbox_inches='tight')
    print(f"图片已保存: 19_不同帧数PSNR对比.png")
    plt.close()


def main():
    print("=" * 60)
    print("知识点9：TMT数据集构建")
    print("=" * 60)
    print()
    print("【核心内容】")
    print("  静态场景：9,017序列（来自Place），每序列50帧")
    print("  动态场景：4,684视频（来自SVW+TSRWGAN）")
    print("  湍流参数：强度、核尺寸、距离、时间相关性")
    print("  数据增强：Gamma、饱和度、裁剪、翻转、旋转、噪声")
    print("  分阶段训练：先去倾斜（400K），再去模糊（600K）")
    print()
    
    # 运行所有演示
    demo_static_dataset()
    demo_dynamic_dataset()
    demo_turbulence_parameters()
    demo_data_augmentation()
    demo_training_settings()
    demo_evaluation_metrics()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print()
    print("【你应该理解的关键点】")
    print("1. 静态数据集：Place图像生成，9,017序列，50帧/序列")
    print("2. 动态数据集：SVW+GAN视频，4,684视频")
    print("3. 湍流参数：D/r0（强度）、核尺寸、距离、时间相关性")
    print("4. 数据增强：6种，增加多样性，防止过拟合")
    print("5. 分阶段训练：先去倾斜400K，再去模糊600K")
    print("6. 更多帧 = 更好效果，这是TMT的优势")
    print()
    print("【TMT全部知识点总结】")
    print("  到这里，TMT的主要知识点我们都学完了！")
    print("  包括：湍流退化模型、去倾斜模块、去模糊模块、TCJA、湍流模拟器、数据集")
    print()
    print("【下一步可以做什么？】")
    print("  1. 下载官方代码，跑通实验")
    print("  2. 复现论文的实验结果")
    print("  3. 在自己的数据上测试")
    print("  4. 基于TMT做改进，发论文")


if __name__ == '__main__':
    main()
