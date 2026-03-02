import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式（KDD风格）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

# 数据 - Train Ratio从15%到1%（展示衰退趋势）
train_ratios = [15, 10, 5, 1]  # x轴：从高到低
x_labels = ['15%', '10%', '5%', '1%']

# AUROC数据（按15%, 10%, 5%, 1%顺序）
auroc_ggad = [0.5411, 0.5306, 0.5087, 0.5125]
auroc_rho = [0.5849, 0.5763, 0.5098, 0.5137]
auroc_vecgad = [0.6702, 0.6714, 0.6496, 0.5876]

# AUPRC数据（按15%, 10%, 5%, 1%顺序）
auprc_ggad = [0.2463, 0.2392, 0.2295, 0.2259]
auprc_rho = [0.2999, 0.2683, 0.2493, 0.2299]
auprc_vecgad = [0.3222, 0.3207, 0.303, 0.2647]

# KDD风格的颜色和标记
colors = ['#E24A33', '#348ABD', '#228B22']  # 红、蓝、绿
markers = ['o', 's', '^']  # 圆、方、三角
linestyles = ['-', '--', '-.']

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

# 绘制 AUROC 子图
ax1 = axes[0]
ax1.set_box_aspect(0.4)
ax1.plot(range(len(train_ratios)), auroc_ggad, 
         color=colors[0], marker=markers[0], 
         linestyle=linestyles[0], label='GGAD',
         markeredgecolor='white', markeredgewidth=1.5)
ax1.plot(range(len(train_ratios)), auroc_rho, 
         color=colors[1], marker=markers[1], 
         linestyle=linestyles[1], label='RHO',
         markeredgecolor='white', markeredgewidth=1.5)
ax1.plot(range(len(train_ratios)), auroc_vecgad, 
         color=colors[2], marker=markers[2], 
         linestyle=linestyles[2], label='VecGAD',
         markeredgecolor='white', markeredgewidth=1.5)

ax1.set_xlabel('Train Ratio')
ax1.set_ylabel('AUROC')
ax1.set_xticks(range(len(train_ratios)))
ax1.set_xticklabels(x_labels)
ax1.set_ylim([0.48, 0.72])
ax1.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.25, 0.02, '(a) AUROC', ha='center', fontsize=16)

# 绘制 AUPRC 子图
ax2 = axes[1]
ax2.set_box_aspect(0.4)
ax2.plot(range(len(train_ratios)), auprc_ggad, 
         color=colors[0], marker=markers[0], 
         linestyle=linestyles[0], label='GGAD',
         markeredgecolor='white', markeredgewidth=1.5)
ax2.plot(range(len(train_ratios)), auprc_rho, 
         color=colors[1], marker=markers[1], 
         linestyle=linestyles[1], label='RHO',
         markeredgecolor='white', markeredgewidth=1.5)
ax2.plot(range(len(train_ratios)), auprc_vecgad, 
         color=colors[2], marker=markers[2], 
         linestyle=linestyles[2], label='VecGAD',
         markeredgecolor='white', markeredgewidth=1.5)

ax2.set_xlabel('Train Ratio')
ax2.set_ylabel('AUPRC')
ax2.set_xticks(range(len(train_ratios)))
ax2.set_xticklabels(x_labels)
ax2.set_ylim([0.20, 0.35])
ax2.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.75, 0.02, '(b) AUPRC', ha='center', fontsize=16)

# 调整布局 - 紧凑排列
plt.tight_layout()
plt.subplots_adjust(bottom=0.22, wspace=0.28)

# 保存为PDF和PNG
plt.savefig('train_ratio_performance_decay.pdf', format='pdf', bbox_inches='tight', pad_inches=0.03)
plt.savefig('train_ratio_performance_decay.png', format='png', bbox_inches='tight', pad_inches=0.03)

print("图表已保存为 train_ratio_performance_decay.pdf")
plt.show()