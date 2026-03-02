import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式（KDD风格）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

# 数据
propagation_steps = [2, 4, 6, 8, 10, 12, 14, 16, 18]
x_labels = ['2', '4', '6', '8', '10', '12', '14', '16', '18']

data = {
    'Amazon': {
        'AUROC': [0.8916, 0.9353, 0.9154, 0.8153, 0.6795, 0.5849, 0.5168, 0.4331, 0.3956],
        'AUPRC': [0.7891, 0.803, 0.7543, 0.4305, 0.1815, 0.1138, 0.0751, 0.0603, 0.0549]
    },
    'Elliptic': {
        'AUROC': [0.6584, 0.7316, 0.7551, 0.7212, 0.6629, 0.6746, 0.6668, 0.615, 0.6195],
        'AUPRC': [0.138, 0.2129, 0.2744, 0.2649, 0.1761, 0.2004, 0.2022, 0.1749, 0.1544]
    },
    'Tolokers': {
        'AUROC': [0.5874, 0.5735, 0.5373, 0.5336, 0.5324, 0.5348, 0.5354, 0.5317, 0.5132],
        'AUPRC': [0.2711, 0.2677, 0.2403, 0.2363, 0.2363, 0.2331, 0.2345, 0.2363, 0.2256]
    }
}

# KDD风格的颜色和标记
colors = ['#E24A33', '#348ABD', '#8EBA42']  # 红、蓝、绿
markers = ['o', 's', '^']  # 圆、方、三角
linestyles = ['-', '-', '-']

# 创建图形 (两个子图: AUROC 和 AUPRC)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

# 绘制 AUROC 子图
ax1 = axes[0]
ax1.set_box_aspect(0.4)
for idx, (dataset, values) in enumerate(data.items()):
    ax1.plot(range(len(propagation_steps)), values['AUROC'], 
             color=colors[idx], marker=markers[idx], 
             linestyle=linestyles[idx], label=dataset,
             markeredgecolor='white', markeredgewidth=1.5)

ax1.set_xlabel('Propagation Steps ($K$)')
ax1.set_ylabel('AUROC')
ax1.set_xticks(range(len(propagation_steps)))
ax1.set_xticklabels(x_labels)
ax1.set_ylim([0.3, 1.0])
ax1.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.25, 0.02, '(a) AUROC', ha='center', fontsize=16)

# 绘制 AUPRC 子图
ax2 = axes[1]
ax2.set_box_aspect(0.4)
for idx, (dataset, values) in enumerate(data.items()):
    ax2.plot(range(len(propagation_steps)), values['AUPRC'], 
             color=colors[idx], marker=markers[idx], 
             linestyle=linestyles[idx], label=dataset,
             markeredgecolor='white', markeredgewidth=1.5)

ax2.set_xlabel('Propagation Steps ($K$)')
ax2.set_ylabel('AUPRC')
ax2.set_xticks(range(len(propagation_steps)))
ax2.set_xticklabels(x_labels)
ax2.set_ylim([0.0, 0.9])
ax2.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.75, 0.02, '(b) AUPRC', ha='center', fontsize=16)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# 保存为PDF
plt.savefig('propagation_steps.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('propagation_steps.png', format='png', bbox_inches='tight', pad_inches=0.05)

print("图表已保存为 propagation_steps_analysis.pdf")
plt.show()