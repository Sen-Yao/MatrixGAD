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
    'legend.fontsize': 13,  # 稍微调大一点图例字号
    'figure.dpi': 300,  
    'savefig.dpi': 300,  
    'axes.linewidth': 1.2,  
    'lines.linewidth': 2,  
    'lines.markersize': 8,  
})  
  
# 数据  
train_ratios = [15, 10, 5, 2, 1]    
x_labels = ['15%', '10%', '5%', '2%', '1%']    
    
data = {    
    'Amazon': {    
        'AUROC': [0.9323, 0.9388, 0.9341, 0.9396, 0.9162],    
        'AUPRC': [0.7971, 0.8039, 0.8054, 0.8169, 0.7914]    
    },    
    'Elliptic': {    
        'AUROC': [0.7617, 0.7644, 0.7627, 0.734, 0.5567],    
        'AUPRC': [0.331, 0.3114, 0.2813, 0.2203, 0.1135]    
    },    
    'Tolokers': {    
        'AUROC': [0.6702, 0.6714, 0.6496, 0.6258, 0.5876],    
        'AUPRC': [0.3222, 0.3207, 0.303, 0.294, 0.2647]    
    }    
}  
  
# KDD风格的颜色和标记  
colors = ['#E24A33', '#348ABD', '#8EBA42']  
markers = ['o', 's', '^']  
linestyles = ['-', '-', '-']  
  
# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 4)) # 高度稍微增加一点给图例留空间
  
# 绘制 AUROC 子图  
ax1 = axes[0]  
ax1.set_box_aspect(0.4)  
lines = [] # 用于存储绘图对象以生成统一图例
for idx, (dataset, values) in enumerate(data.items()):  
    line, = ax1.plot(range(len(train_ratios)), values['AUROC'],   
             color=colors[idx], marker=markers[idx],   
             linestyle=linestyles[idx], label=dataset,  
             markeredgecolor='white', markeredgewidth=1.5)  
    lines.append(line)
  
ax1.set_xlabel('Training Ratio')  
ax1.set_ylabel('AUROC')  
ax1.set_xticks(range(len(train_ratios)))  
ax1.set_xticklabels(x_labels)  
ax1.set_ylim([0.3, 1.0])  
ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)  
  
# 绘制 AUPRC 子图  
ax2 = axes[1]  
ax2.set_box_aspect(0.4)  
for idx, (dataset, values) in enumerate(data.items()):  
    ax2.plot(range(len(train_ratios)), values['AUPRC'],   
             color=colors[idx], marker=markers[idx],   
             linestyle=linestyles[idx], label=dataset,  
             markeredgecolor='white', markeredgewidth=1.5)  
  
ax2.set_xlabel('Training Ratio')  
ax2.set_ylabel('AUPRC')  
ax2.set_xticks(range(len(train_ratios)))  
ax2.set_xticklabels(x_labels)  
ax2.set_ylim([0.0, 1.0]) # 统一刻度上限让视觉更平衡
ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)  
  
# --- 统一图例设置 ---
# ncol=3 表示水平排列三列，bbox_to_anchor 用于精确定位
fig.legend(handles=lines, labels=data.keys(), 
           loc='upper center', ncol=3, 
           frameon=True, edgecolor='black', fancybox=False,
           bbox_to_anchor=(0.5, 0.98)) 

# 设置子标题 (a) 和 (b)
fig.text(0.28, 0.05, '(a) AUROC', ha='center', fontsize=16)  
fig.text(0.76, 0.05, '(b) AUPRC', ha='center', fontsize=16)  
  
# 调整布局：使用 subplots_adjust 给顶部图例和底部标题留出空间
plt.tight_layout()  
plt.subplots_adjust(top=0.82, bottom=0.22)  
  
# 保存
plt.savefig('training_ratio.pdf', format='pdf', bbox_inches='tight')  
plt.savefig('training_ratio.png', format='png', bbox_inches='tight')  
plt.show()