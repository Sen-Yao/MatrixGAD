"""
基于 δ (delta) 的异常检测评估

验证 ISP 特征的实际检测能力：
1. 计算每个 δ_k 的 AUC/AP
2. 计算组合策略的 AUC/AP
3. 可视化正常/异常节点的 δ 分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

from utils import load_mat, load_dgraph, normalize_adj, preprocess_features, nagphormer_tokenization
import scipy.sparse as sp


def compute_deltas(node_tokens):
    """
    计算 δ 矩阵
    
    Args:
        node_tokens: [N, pp_k+1, D]
    
    Returns:
        deltas: [N, pp_k] where δ_k = ||token_{k+1} - token_k||
    """
    # δ_k = ||token_{k+1} - token_k||
    deltas = torch.norm(node_tokens[:, 1:] - node_tokens[:, :-1], p=2, dim=2)
    return deltas


def evaluate_delta_detection(deltas, labels, reverse=False):
    """
    基于 δ 进行异常检测评估
    
    Args:
        deltas: [N, pp_k]
        labels: [N] (0=正常, 1=异常)
        reverse: 是否反向检测（用于正常δ更高的数据集，如Amazon）
    
    Returns:
        results: 包含各种策略的 AUC/AP 结果
    """
    results = {}
    labels = np.array(labels)
    
    num_deltas = deltas.shape[1]
    
    # 先计算 δ 比值，判断数据集类型
    normal_mask = labels == 0
    abnormal_mask = labels == 1
    
    normal_mean = deltas[normal_mask].mean(dim=0)
    abnormal_mean = deltas[abnormal_mask].mean(dim=0)
    ratio = abnormal_mean / (normal_mean + 1e-8)
    
    print("\n--- 数据集特征分析 ---")
    print(f"正常节点 δ 均值: {normal_mean.numpy()}")
    print(f"异常节点 δ 均值: {abnormal_mean.numpy()}")
    print(f"δ 比值 (异常/正常): {ratio.numpy()}")
    
    avg_ratio = ratio.mean().item()
    if avg_ratio > 1:
        print(f"平均比值 {avg_ratio:.2f} > 1: 异常节点 δ 更高，使用正向检测")
    else:
        print(f"平均比值 {avg_ratio:.2f} < 1: 正常节点 δ 更高，建议使用反向检测")
    
    # 单特征检测（每个 δ_k 单独评估）
    print("\n--- 单特征检测 (每个 δ_k 单独) ---")
    print(f"{'特征':<12} {'AUC(正向)':>10} {'AP(正向)':>10} {'AUC(反向)':>10} {'AP(反向)':>10}")
    print("-" * 56)
    
    for k in range(num_deltas):
        score = deltas[:, k].numpy()
        
        # 处理可能的异常值
        if np.isnan(score).any() or np.isinf(score).any():
            print(f"δ_{k:<10} 包含异常值，跳过")
            continue
        
        # 正向检测：δ 越大越异常
        auc_forward = roc_auc_score(labels, score)
        ap_forward = average_precision_score(labels, score)
        
        # 反向检测：δ 越小越异常
        auc_reverse = roc_auc_score(labels, -score)
        ap_reverse = average_precision_score(labels, -score)
        
        results[f'delta_{k}'] = {
            'AUC_forward': auc_forward, 'AP_forward': ap_forward,
            'AUC_reverse': auc_reverse, 'AP_reverse': ap_reverse
        }
        
        # 选择最佳方向
        best_auc = max(auc_forward, auc_reverse)
        best_dir = "正向" if auc_forward >= auc_reverse else "反向"
        
        print(f"δ_{k:<10} {auc_forward:>10.4f} {ap_forward:>10.4f} {auc_reverse:>10.4f} {ap_reverse:>10.4f}  *{best_dir}")
    
    # 组合策略
    print("\n--- 组合策略检测 ---")
    print(f"{'策略':<15} {'AUC(正向)':>10} {'AP(正向)':>10} {'AUC(反向)':>10} {'AP(反向)':>10}")
    print("-" * 60)
    
    delta_np = deltas.numpy()
    
    # 求和
    sum_score = delta_np.sum(axis=1)
    auc_sum_f = roc_auc_score(labels, sum_score)
    ap_sum_f = average_precision_score(labels, sum_score)
    auc_sum_r = roc_auc_score(labels, -sum_score)
    ap_sum_r = average_precision_score(labels, -sum_score)
    results['sum'] = {'AUC_forward': auc_sum_f, 'AP_forward': ap_sum_f, 'AUC_reverse': auc_sum_r, 'AP_reverse': ap_sum_r}
    print(f"{'Σδ (求和)':<15} {auc_sum_f:>10.4f} {ap_sum_f:>10.4f} {auc_sum_r:>10.4f} {ap_sum_r:>10.4f}")
    
    # 最大值
    max_score = delta_np.max(axis=1)
    auc_max_f = roc_auc_score(labels, max_score)
    ap_max_f = average_precision_score(labels, max_score)
    auc_max_r = roc_auc_score(labels, -max_score)
    ap_max_r = average_precision_score(labels, -max_score)
    results['max'] = {'AUC_forward': auc_max_f, 'AP_forward': ap_max_f, 'AUC_reverse': auc_max_r, 'AP_reverse': ap_max_r}
    print(f"{'max(δ)':<15} {auc_max_f:>10.4f} {ap_max_f:>10.4f} {auc_max_r:>10.4f} {ap_max_r:>10.4f}")
    
    # 均值
    mean_score = delta_np.mean(axis=1)
    auc_mean_f = roc_auc_score(labels, mean_score)
    ap_mean_f = average_precision_score(labels, mean_score)
    auc_mean_r = roc_auc_score(labels, -mean_score)
    ap_mean_r = average_precision_score(labels, -mean_score)
    results['mean'] = {'AUC_forward': auc_mean_f, 'AP_forward': ap_mean_f, 'AUC_reverse': auc_mean_r, 'AP_reverse': ap_mean_r}
    print(f"{'mean(δ)':<15} {auc_mean_f:>10.4f} {ap_mean_f:>10.4f} {auc_mean_r:>10.4f} {ap_mean_r:>10.4f}")
    
    # 最后一个 δ
    last_score = delta_np[:, -1]
    auc_last_f = roc_auc_score(labels, last_score)
    ap_last_f = average_precision_score(labels, last_score)
    auc_last_r = roc_auc_score(labels, -last_score)
    ap_last_r = average_precision_score(labels, -last_score)
    results['last'] = {'AUC_forward': auc_last_f, 'AP_forward': ap_last_f, 'AUC_reverse': auc_last_r, 'AP_reverse': ap_last_r}
    print(f"{'δ_last':<15} {auc_last_f:>10.4f} {ap_last_f:>10.4f} {auc_last_r:>10.4f} {ap_last_r:>10.4f}")
    
    # 加权组合
    weights = np.arange(1, num_deltas + 1)
    weighted_score = (delta_np * weights).sum(axis=1) / weights.sum()
    auc_weighted_f = roc_auc_score(labels, weighted_score)
    ap_weighted_f = average_precision_score(labels, weighted_score)
    auc_weighted_r = roc_auc_score(labels, -weighted_score)
    ap_weighted_r = average_precision_score(labels, -weighted_score)
    results['weighted'] = {'AUC_forward': auc_weighted_f, 'AP_forward': ap_weighted_f, 'AUC_reverse': auc_weighted_r, 'AP_reverse': ap_weighted_r}
    print(f"{'weighted':<15} {auc_weighted_f:>10.4f} {ap_weighted_f:>10.4f} {auc_weighted_r:>10.4f} {ap_weighted_r:>10.4f}")
    
    # 找出最佳策略和方向
    best_result = None
    best_auc = 0
    best_strategy = None
    best_direction = None
    
    for key, val in results.items():
        if val['AUC_forward'] > best_auc:
            best_auc = val['AUC_forward']
            best_strategy = key
            best_direction = '正向'
        if val['AUC_reverse'] > best_auc:
            best_auc = val['AUC_reverse']
            best_strategy = key
            best_direction = '反向'
    
    print(f"\n最佳策略: {best_strategy} ({best_direction}) AUC={best_auc:.4f}")
    
    return results


def visualize_delta_distribution(deltas, labels, alpha, save_path):
    """
    可视化正常 vs 异常节点的 δ 分布
    
    Args:
        deltas: [N, pp_k]
        labels: [N]
        alpha: 传播参数
        save_path: 保存路径
    """
    num_deltas = deltas.shape[1]
    
    fig, axes = plt.subplots(2, (num_deltas + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = np.array(labels)
    delta_np = deltas.numpy()
    
    for k in range(num_deltas):
        ax = axes[k]
        
        # 正常节点
        normal_deltas = delta_np[labels == 0, k]
        # 异常节点
        abnormal_deltas = delta_np[labels == 1, k]
        
        # 限制范围（去除极端值，用 P99）
        p99 = np.percentile(np.concatenate([normal_deltas, abnormal_deltas]), 99)
        normal_deltas_clipped = np.clip(normal_deltas, 0, p99)
        abnormal_deltas_clipped = np.clip(abnormal_deltas, 0, p99)
        
        sns.histplot(normal_deltas_clipped, ax=ax, label='Normal', color='blue', alpha=0.5, stat='density')
        sns.histplot(abnormal_deltas_clipped, ax=ax, label='Abnormal', color='red', alpha=0.5, stat='density')
        
        ax.set_title(f'δ_{k} Distribution (α={alpha})')
        ax.set_xlabel(f'δ_{k}')
        ax.set_ylabel('Density')
        ax.legend()
    
    # 隐藏多余的子图
    for k in range(num_deltas, len(axes)):
        axes[k].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"分布图保存至: {save_path}")


def visualize_roc_curve(deltas, labels, save_path):
    """
    可视化 ROC 曲线
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = np.array(labels)
    delta_np = deltas.numpy()
    
    # 选择几个代表性的策略画 ROC 曲线
    strategies = {
        'δ_0': delta_np[:, 0],
        'δ_5': delta_np[:, -1],
        'Σδ': delta_np.sum(axis=1),
        'max(δ)': delta_np.max(axis=1),
    }
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for (name, score), color in zip(strategies.items(), colors):
        fpr, tpr, _ = roc_curve(labels, score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Delta-based Anomaly Detection')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线保存至: {save_path}")


def run_delta_detection_experiment(dataset='tolokers', pp_k=6, alpha=0.2, device=0,
                                   train_rate=0.05, seed=42, save_dir='figs/delta_detection'):
    """
    运行完整的 δ 异常检测实验
    """
    import argparse
    import random
    
    # 创建 args 对象
    args = argparse.Namespace(
        dataset=dataset,
        pp_k=pp_k,
        progregate_alpha=alpha,
        device=device,
        train_rate=train_rate,
        seed=seed,
        data_split_seed=seed,
        sample_rate=0.15
    )
    
    print("="*60)
    print(f"δ 异常检测实验 - 数据集: {dataset}")
    print(f"参数: pp_k={pp_k}, alpha={alpha}")
    print("="*60)
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 加载数据
    if dataset == 'dgraph':
        adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, _, _, normal_for_train_idx, normal_for_generation_idx = load_dgraph(
            train_rate=train_rate, val_rate=0.1, args=args
        )
    else:
        adj, features, labels, all_idx, idx_train, idx_val, \
        idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(
            dataset, train_rate, 0.1, args=args
        )
        
        if dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
            features, _ = preprocess_features(features)
        else:
            features = features.todense()
        
        adj = normalize_adj(adj)
        adj = (adj + sp.eye(adj.shape[0])).todense()
        features = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj)
    
    print(f"节点数: {features.shape[0]}, 特征维度: {features.shape[1]}")
    
    # 执行 NAGphormer tokenization
    print("\n执行 NAGphormer Tokenization...")
    node_tokens = nagphormer_tokenization(features, adj, args)
    print(f"Token shape: {node_tokens.shape}")  # [N, pp_k+1, D]
    
    # 计算 δ
    print("\n计算 δ...")
    deltas = compute_deltas(node_tokens)
    print(f"δ shape: {deltas.shape}")  # [N, pp_k]
    
    # 准备标签
    labels_array = np.squeeze(np.array(ano_label))
    
    # 评估检测性能
    print("\n" + "="*60)
    print("异常检测性能评估")
    print("="*60)
    
    results = evaluate_delta_detection(deltas, labels_array)
    
    # 可视化
    os.makedirs(save_dir, exist_ok=True)
    
    # 分布图
    visualize_delta_distribution(
        deltas, labels_array, alpha,
        save_path=os.path.join(save_dir, f'{dataset}_delta_distribution.pdf')
    )
    
    # ROC 曲线
    visualize_roc_curve(
        deltas, labels_array,
        save_path=os.path.join(save_dir, f'{dataset}_roc_curves.pdf')
    )
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Delta-based Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='tolokers',
                        choices=['BlogCatalog', 'Flickr', 'ACM', 'Coris', 'Amazon', 'reddit', 'dgraph', 'tolokers', 'photo', 'elliptic', 't_finance'])
    parser.add_argument('--pp_k', type=int, default=6, help='Number of propagation hops')
    parser.add_argument('--alpha', type=float, default=0.2, help='Restart probability')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='figs/delta_detection')
    
    args = parser.parse_args()
    
    run_delta_detection_experiment(
        dataset=args.dataset,
        pp_k=args.pp_k,
        alpha=args.alpha,
        device=args.device,
        train_rate=args.train_rate,
        seed=args.seed,
        save_dir=args.save_dir
    )