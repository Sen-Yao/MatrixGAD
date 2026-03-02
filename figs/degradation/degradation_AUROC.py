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

# KDD风格的颜色和标记
colors = ['#E24A33', '#348ABD', '#228B22']  # 红、蓝、绿
markers = ['o', 's', '^']  # 圆、方、三角
linestyles = ['-', '--', '-.']

# 创建图形（单图）
fig, ax = plt.subplots(figsize=(5.5, 3.5))

# 绘制 AUROC 图
ax.set_box_aspect(0.6)
ax.plot(range(len(train_ratios)), auroc_ggad, 
         color=colors[0], marker=markers[0], 
         linestyle=linestyles[0], label='GGAD',
         markeredgecolor='white', markeredgewidth=1.5)
ax.plot(range(len(train_ratios)), auroc_rho, 
         color=colors[1], marker=markers[1], 
         linestyle=linestyles[1], label='RHO',
         markeredgecolor='white', markeredgewidth=1.5)
ax.plot(range(len(train_ratios)), auroc_vecgad, 
         color=colors[2], marker=markers[2], 
         linestyle=linestyles[2], label='VecGAD',
         markeredgecolor='white', markeredgewidth=1.5)

ax.set_xlabel('Train Ratio')
ax.set_ylabel('AUROC')
ax.set_xticks(range(len(train_ratios)))
ax.set_xticklabels(x_labels)
ax.set_ylim([0.48, 0.72])
ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.5, 0.02, '(a) AUROC', ha='center', fontsize=16)

# 调整布局 - 左右对称留白
plt.tight_layout()
plt.subplots_adjust(left=0.12, right=0.88, bottom=0.22)  # 右侧留出与左侧对称的空白

# 保存为PDF和PNG
# 保存为PDF和PNG（移除bbox_inches='tight'）
plt.savefig('train_ratio_auroc.pdf', format='pdf', pad_inches=0.03)
plt.savefig('train_ratio_auroc.png', format='png', pad_inches=0.03)

print("图表已保存为 train_ratio_auroc.pdf")
plt.show()