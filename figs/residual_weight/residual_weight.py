import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式（KDD风格）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'lines.markersize': 7,
})

# 数据
alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

data = {
    'Amazon': {
        'AUROC': [0.7846, 0.7383, 0.9103, 0.9341, 0.9391, 0.9297, 0.9159, 0.9131, 0.9266, 0.9247, 0.9238],
        'AUPRC': [0.4091, 0.3128, 0.7382, 0.8054, 0.8064, 0.8014, 0.7936, 0.7924, 0.7997, 0.7976, 0.798]
    },
    'Elliptic': {
        'AUROC': [0.495, 0.4833, 0.4571, 0.437, 0.5682, 0.6842, 0.7627, 0.774, 0.7753, 0.7584, 0.749],
        'AUPRC': [0.0875, 0.0853, 0.0812, 0.0788, 0.1204, 0.209, 0.2813, 0.2761, 0.2746, 0.2357, 0.2199]
    },
    'Tolokers': {
        'AUROC': [0.5699, 0.6145, 0.6312, 0.6496, 0.6325, 0.6236, 0.6411, 0.6509, 0.6511, 0.6386, 0.616],
        'AUPRC': [0.268, 0.2896, 0.2951, 0.303, 0.2958, 0.2851, 0.2944, 0.3019, 0.304, 0.3024, 0.3001]
    }
}

# KDD风格的颜色和标记
colors = ['#E24A33', '#348ABD', '#8EBA42']  # 红、蓝、绿
markers = ['o', 's', '^']  # 圆、方、三角
linestyles = ['-', '-', '-']

# 创建图形 (两个子图: AUROC 和 AUPRC)
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

# 绘制 AUROC 子图
ax1 = axes[0]
ax1.set_box_aspect(0.4)
for idx, (dataset, values) in enumerate(data.items()):
    ax1.plot(alpha_values, values['AUROC'], 
             color=colors[idx], marker=markers[idx], 
             linestyle=linestyles[idx], label=dataset,
             markeredgecolor='white', markeredgewidth=1.2)

ax1.set_xlabel(r'Residual Weight $\alpha$')
ax1.set_ylabel('AUROC')
ax1.set_xticks(alpha_values)
ax1.set_xticklabels(x_labels)
ax1.set_xlim([-0.05, 1.05])
ax1.set_ylim([0.35, 1.0])
ax1.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.25, 0.02, '(a) AUROC', ha='center', fontsize=16)

# 绘制 AUPRC 子图
ax2 = axes[1]
ax2.set_box_aspect(0.4)
for idx, (dataset, values) in enumerate(data.items()):
    ax2.plot(alpha_values, values['AUPRC'], 
             color=colors[idx], marker=markers[idx], 
             linestyle=linestyles[idx], label=dataset,
             markeredgecolor='white', markeredgewidth=1.2)

ax2.set_xlabel(r'Residual Weight $\alpha$')
ax2.set_ylabel('AUPRC')
ax2.set_xticks(alpha_values)
ax2.set_xticklabels(x_labels)
ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([0.0, 0.9])
ax2.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.75, 0.02, '(b) AUPRC', ha='center', fontsize=16)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.20)

# 保存为PDF和PNG
plt.savefig('residual_weight.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('residual_weight.png', format='png', bbox_inches='tight', pad_inches=0.05)

print("图表已保存为 residual_weight_analysis.pdf 和 residual_weight_analysis.png")
plt.show()