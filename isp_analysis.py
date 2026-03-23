"""
ISP (Iterative Similarity Progression) Analysis Module

验证猜想：
1. 在 NAGphormer tokenization 过程中，随着传播次数增加，n-hop 的 token 会逐渐达到某个收敛向量
2. 对于异常节点，此收敛过程的曲线与正常节点存在不同

ISP 定义：某一 hop 的 token 到收敛向量的 L2 范数
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from scipy import stats
from tqdm import tqdm

from utils import load_mat, load_dgraph, normalize_adj, preprocess_features, nagphormer_tokenization
import scipy.sparse as sp


def print_detailed_stats(isp_matrix, label=""):
    """
    打印 ISP 的详细统计信息，包括百分位数
    
    Args:
        isp_matrix: ISP 矩阵, shape [N, num_deltas]
        label: 标签名称
    """
    num_deltas = isp_matrix.shape[1]
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    
    # 打印表头
    header = f"{'δ_k':<6}"
    for p in percentiles:
        if p == 0:
            header += f"{'Min':>10}"
        elif p == 100:
            header += f"{'Max':>10}"
        else:
            header += f"{f'P{p}':>10}"
    header += f"{'Mean':>10} {'Std':>10}"
    print(header)
    print("-" * (6 + 10 * len(percentiles) + 20))
    
    # 打印每个 δ_k 的统计
    for k in range(num_deltas):
        values = isp_matrix[:, k].numpy()
        row = f"δ_{k:<4}"
        for p in percentiles:
            val = np.percentile(values, p)
            row += f"{val:>10.4f}"
        row += f"{values.mean():>10.4f} {values.std():>10.4f}"
        print(row)


def compute_isp(tokens, mode='delta'):
    """
    计算单个节点的 ISP (Iterative Similarity Progression)
    
    Args:
        tokens: 节点的 token 序列, shape [pp_k+1, D]
        mode: 计算模式
            - 'delta': 相邻 hop 之间的距离变化 δ_k = ||token_{k+1} - token_k||
            - 'to_final': 到最终 hop 的距离（原始定义，有偏）
    
    Returns:
        isp_values: ISP 值序列
            - mode='delta': shape [pp_k], 表示每个 hop 的变化量
            - mode='to_final': shape [pp_k+1], 表示到收敛向量的距离
    """
    num_hops = tokens.shape[0]
    
    if mode == 'delta':
        # 计算相邻 hop 之间的 L2 距离
        # δ_k = ||token_{k+1} - token_k||
        isp_values = torch.norm(tokens[1:] - tokens[:-1], p=2, dim=1)
        return isp_values
    elif mode == 'to_final':
        # 原始定义：到最终 hop 的距离
        converged_vector = tokens[-1]
        isp_values = torch.norm(tokens - converged_vector.unsqueeze(0), p=2, dim=1)
        return isp_values
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_isp_batch(node_tokens, labels=None, normal_indices=None, abnormal_indices=None, mode='delta'):
    """
    批量计算所有节点的 ISP
    
    Args:
        node_tokens: 所有节点的 token 序列, shape [N, pp_k+1, D]
        labels: 节点标签 (0=正常, 1=异常), shape [N]
        normal_indices: 正常节点索引
        abnormal_indices: 异常节点索引
        mode: ISP 计算模式
            - 'delta': 相邻 hop 之间的变化量 δ_k = ||token_{k+1} - token_k|| (推荐)
            - 'to_final': 到最终 hop 的距离（有偏，不推荐）
    
    Returns:
        isp_matrix: ISP 矩阵, shape [N, pp_k] (delta模式) 或 [N, pp_k+1] (to_final模式)
        normal_isp: 正常节点的 ISP
        abnormal_isp: 异常节点的 ISP
    """
    N, num_hops, D = node_tokens.shape
    
    # 计算所有节点的 ISP
    if mode == 'delta':
        isp_matrix = torch.zeros(N, num_hops - 1)  # delta模式只有 pp_k 个值
    else:
        isp_matrix = torch.zeros(N, num_hops)
    
    for i in range(N):
        isp_matrix[i] = compute_isp(node_tokens[i], mode=mode)
    
    # 分别提取正常和异常节点的 ISP
    if labels is not None:
        normal_mask = (labels == 0)
        abnormal_mask = (labels == 1)
        normal_isp = isp_matrix[normal_mask]
        abnormal_isp = isp_matrix[abnormal_mask]
    elif normal_indices is not None and abnormal_indices is not None:
        normal_isp = isp_matrix[normal_indices]
        abnormal_isp = isp_matrix[abnormal_indices]
    else:
        normal_isp = None
        abnormal_isp = None
    
    return isp_matrix, normal_isp, abnormal_isp


def analyze_convergence(node_tokens, labels, verbose=True, mode='delta'):
    """
    分析 token 收敛性
    
    Args:
        node_tokens: 所有节点的 token 序列, shape [N, pp_k+1, D]
        labels: 节点标签, shape [N]
        verbose: 是否打印详细信息
        mode: ISP 计算模式
            - 'delta': 相邻 hop 之间的变化量 δ_k = ||token_{k+1} - token_k|| (推荐)
            - 'to_final': 到最终 hop 的距离（有偏，不推荐）
    
    Returns:
        results: 包含分析结果的字典
    """
    N, num_tokens, D = node_tokens.shape  # num_tokens = pp_k + 1
    
    # 计算 ISP (使用 delta 模式)
    isp_matrix, normal_isp, abnormal_isp = compute_isp_batch(node_tokens, labels, mode=mode)
    
    num_isp_values = isp_matrix.shape[1]  # delta模式为 pp_k，to_final模式为 pp_k+1
    
    results = {
        'isp_matrix': isp_matrix,
        'normal_isp': normal_isp,
        'abnormal_isp': abnormal_isp,
        'num_tokens': num_tokens,
        'num_isp_values': num_isp_values,
        'num_nodes': N,
        'num_normal': normal_isp.shape[0] if normal_isp is not None else 0,
        'num_abnormal': abnormal_isp.shape[0] if abnormal_isp is not None else 0,
        'mode': mode,
    }
    
    # 计算统计量
    # 正常节点的 ISP 统计
    if normal_isp is not None and normal_isp.shape[0] > 0:
        normal_isp_mean = normal_isp.mean(dim=0)
        normal_isp_std = normal_isp.std(dim=0)
        results['normal_isp_mean'] = normal_isp_mean
        results['normal_isp_std'] = normal_isp_std
    
    # 异常节点的 ISP 统计
    if abnormal_isp is not None and abnormal_isp.shape[0] > 0:
        abnormal_isp_mean = abnormal_isp.mean(dim=0)
        abnormal_isp_std = abnormal_isp.std(dim=0)
        results['abnormal_isp_mean'] = abnormal_isp_mean
        results['abnormal_isp_std'] = abnormal_isp_std
    
    # 验证猜想1：ISP (delta) 是否单调递减
    # delta ISP 递减意味着: δ_{k+1} < δ_k，即变化量越来越小（收敛）
    if num_isp_values > 1:
        isp_diff = isp_matrix[:, 1:] - isp_matrix[:, :-1]  # 负值表示递减
        results['isp_diff'] = isp_diff
        results['convergence_ratio'] = (isp_diff < 0).float().mean().item()  # 递减比例
        
        # 更细粒度的收敛分析：从第 k hop 到 k+1 hop 的递减比例
        convergence_per_hop = []
        for k in range(num_isp_values - 1):
            ratio = (isp_matrix[:, k+1] < isp_matrix[:, k]).float().mean().item()
            convergence_per_hop.append(ratio)
        results['convergence_per_hop'] = convergence_per_hop
    else:
        results['convergence_ratio'] = float('nan')
        results['convergence_per_hop'] = []
    
    if verbose:
        print("\n" + "="*60)
        print("ISP 收敛性分析结果 (mode=delta)")
        print("="*60)
        print(f"总节点数: {N}")
        print(f"正常节点数: {results['num_normal']}")
        print(f"异常节点数: {results['num_abnormal']}")
        print(f"Token 数量 (pp_k+1): {num_tokens}")
        print(f"ISP 值数量 (δ_k): {num_isp_values}")
        
        print(f"\n--- 猜想1验证：δ_k 收敛性 ---")
        print(f"ISP 定义: δ_k = ||token_{{k+1}} - token_k|| (相邻 hop 间的变化量)")
        print(f"收敛 = δ_{{k+1}} < δ_k (变化量递减)")
        if not np.isnan(results['convergence_ratio']):
            print(f"总体收敛比例 (δ递减): {results['convergence_ratio']:.4f}")
            print(f"各 hop 的收敛比例:")
            for k, ratio in enumerate(results['convergence_per_hop']):
                print(f"  δ_{k} → δ_{k+1}: {ratio:.4f}")
        
        # 打印详细的 ISP 分布统计
        print(f"\n--- ISP 详细分布统计 ---")
        
        if normal_isp is not None:
            print(f"\n[正常节点] ISP 分布:")
            print_detailed_stats(normal_isp, "正常节点")
        
        if abnormal_isp is not None:
            print(f"\n[异常节点] ISP 分布:")
            print_detailed_stats(abnormal_isp, "异常节点")
        
        # 打印正常 vs 异常节点的对比
        if normal_isp is not None and abnormal_isp is not None:
            print(f"\n[正常 vs 异常] ISP 均值对比:")
            print(f"{'δ_k':<8} {'正常均值':>12} {'异常均值':>12} {'比值(异常/正常)':>15}")
            print("-" * 55)
            for k in range(num_isp_values):
                n_mean = results['normal_isp_mean'][k].item()
                a_mean = results['abnormal_isp_mean'][k].item()
                ratio = a_mean / (n_mean + 1e-8)
                print(f"δ_{k:<6} {n_mean:>12.4f} {a_mean:>12.4f} {ratio:>15.4f}")
        
        # 猜想2验证：正常/异常节点 ISP 差异
        if normal_isp is not None and abnormal_isp is not None:
            print(f"\n--- 猜想2验证：正常/异常节点 δ 差异 ---")
            
            # 使用 t-test 检验差异显著性
            for k in range(num_isp_values):
                t_stat, p_value = stats.ttest_ind(
                    normal_isp[:, k].numpy(), 
                    abnormal_isp[:, k].numpy()
                )
                print(f"  δ_{k}: t={t_stat:.4f}, p={p_value:.4e}", end="")
                if p_value < 0.05:
                    print(" *显著差异")
                else:
                    print()
            
            # 计算总变化量差异 (第一个δ vs 最后一个δ)
            if num_isp_values > 1:
                # 总变化量比例: δ_0 / (δ_0 + δ_1 + ... + δ_{k-1})
                normal_total_change = normal_isp.sum(dim=1)
                abnormal_total_change = abnormal_isp.sum(dim=1)
                
                # 第一个变化量占比
                normal_first_ratio = normal_isp[:, 0] / (normal_total_change + 1e-8)
                abnormal_first_ratio = abnormal_isp[:, 0] / (abnormal_total_change + 1e-8)
                
                results['normal_first_ratio'] = normal_first_ratio
                results['abnormal_first_ratio'] = abnormal_first_ratio
                
                print(f"\n第一个变化量占比 δ_0 / Σδ:")
                print(f"  正常节点: {normal_first_ratio.mean():.4f} ± {normal_first_ratio.std():.4f}")
                print(f"  异常节点: {abnormal_first_ratio.mean():.4f} ± {abnormal_first_ratio.std():.4f}")
                
                t_stat, p_value = stats.ttest_ind(
                    normal_first_ratio.numpy(),
                    abnormal_first_ratio.numpy()
                )
                print(f"  t-test: t={t_stat:.4f}, p={p_value:.4e}")
    
    return results


def visualize_isp_curves(results, save_dir='figs/isp_analysis', dataset_name='dataset'):
    """
    可视化 ISP 曲线
    
    Args:
        results: analyze_convergence 的返回结果
        save_dir: 保存目录
        dataset_name: 数据集名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_isp_values = results['num_isp_values']
    isp_indices = list(range(num_isp_values))
    
    # 图1：正常/异常节点的 ISP 曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：ISP 曲线
    ax1 = axes[0]
    
    if 'normal_isp_mean' in results:
        normal_mean = results['normal_isp_mean'].numpy()
        normal_std = results['normal_isp_std'].numpy()
        ax1.plot(isp_indices, normal_mean, 'b-o', label='Normal Nodes', linewidth=2, markersize=8)
        ax1.fill_between(isp_indices, 
                         normal_mean - normal_std, 
                         normal_mean + normal_std, 
                         alpha=0.2, color='blue')
    
    if 'abnormal_isp_mean' in results:
        abnormal_mean = results['abnormal_isp_mean'].numpy()
        abnormal_std = results['abnormal_isp_std'].numpy()
        ax1.plot(isp_indices, abnormal_mean, 'r-s', label='Abnormal Nodes', linewidth=2, markersize=8)
        ax1.fill_between(isp_indices, 
                         abnormal_mean - abnormal_std, 
                         abnormal_mean + abnormal_std, 
                         alpha=0.2, color='red')
    
    ax1.set_xlabel('Delta Index (k)', fontsize=12)
    ax1.set_ylabel('δ_k = ||token_{k+1} - token_k||', fontsize=12)
    ax1.set_title(f'ISP Delta Curves - {dataset_name}', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(isp_indices)
    
    # 右图：收敛比例
    ax2 = axes[1]
    convergence_per_hop = results.get('convergence_per_hop', [])
    if convergence_per_hop:
        bars = ax2.bar(range(len(convergence_per_hop)), convergence_per_hop, 
                       color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
        ax2.set_xlabel('Hop Transition (k → k+1)', fontsize=12)
        ax2.set_ylabel('Proportion of Converging Nodes', fontsize=12)
        ax2.set_title('Convergence Ratio per Hop', fontsize=14)
        ax2.set_ylim([0, 1.1])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{dataset_name}_isp_curves.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ISP 曲线图保存至: {save_path}")
    
    # 图2：收敛速度分布
    if 'normal_convergence_rate' in results and 'abnormal_convergence_rate' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        normal_rates = results['normal_convergence_rate'].numpy()
        abnormal_rates = results['abnormal_convergence_rate'].numpy()
        
        # 绘制直方图
        bins = np.linspace(0, 1, 30)
        ax.hist(normal_rates, bins=bins, alpha=0.5, label='Normal Nodes', color='blue', density=True)
        ax.hist(abnormal_rates, bins=bins, alpha=0.5, label='Abnormal Nodes', color='red', density=True)
        
        # 添加均值线
        ax.axvline(normal_rates.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Normal Mean: {normal_rates.mean():.3f}')
        ax.axvline(abnormal_rates.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Abnormal Mean: {abnormal_rates.mean():.3f}')
        
        ax.set_xlabel('Convergence Rate', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Convergence Rate Distribution - {dataset_name}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        save_path2 = os.path.join(save_dir, f'{dataset_name}_convergence_rate.pdf')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.savefig(save_path2.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"收敛速度分布图保存至: {save_path2}")
    
    # 图3：ISP 热力图（抽样节点）
    isp_matrix = results['isp_matrix']
    sample_size = min(100, isp_matrix.shape[0])
    sample_indices = np.random.choice(isp_matrix.shape[0], sample_size, replace=False)
    sample_isp = isp_matrix[sample_indices].numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sample_isp, aspect='auto', cmap='viridis')
    ax.set_xlabel('Hop Index', fontsize=12)
    ax.set_ylabel('Node Index (sampled)', fontsize=12)
    ax.set_title(f'ISP Heatmap - {dataset_name}', fontsize=14)
    plt.colorbar(im, ax=ax, label='ISP Value')
    
    save_path3 = os.path.join(save_dir, f'{dataset_name}_isp_heatmap.pdf')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.savefig(save_path3.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ISP 热力图保存至: {save_path3}")


def run_isp_experiment(dataset='BlogCatalog', pp_k=6, alpha=0.2, device=0, 
                       train_rate=0.05, seed=42, save_dir='figs/isp_analysis'):
    """
    运行完整的 ISP 分析实验
    
    Args:
        dataset: 数据集名称
        pp_k: 传播层数
        alpha: 传播时的 restart 概率
        device: GPU 设备
        train_rate: 训练集比例
        seed: 随机种子
        save_dir: 结果保存目录
    """
    import argparse
    
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
    print(f"ISP 分析实验 - 数据集: {dataset}")
    print(f"参数: pp_k={pp_k}, alpha={alpha}")
    print("="*60)
    
    # 设置随机种子
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 加载数据
    if dataset == 'dgraph':
        adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, _, _, normal_for_train_idx, normal_for_generation_idx = load_dgraph(
            train_rate=train_rate, val_rate=0.1, args=args
        )
        features = features
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
    
    # 分析收敛性
    labels_array = np.squeeze(np.array(ano_label))
    results = analyze_convergence(node_tokens, labels_array, verbose=True)
    
    # 可视化
    visualize_isp_curves(results, save_dir=save_dir, dataset_name=dataset)
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)
    
    return results


def quick_isp_check(node_tokens, labels=None, num_samples=5):
    """
    快速检查 ISP 值，用于调试
    
    Args:
        node_tokens: [N, pp_k+1, D]
        labels: [N]
        num_samples: 打印的样本数
    """
    N, num_hops, D = node_tokens.shape
    
    print(f"\n快速 ISP 检查:")
    print(f"Token shape: {node_tokens.shape}")
    print(f"前 {num_samples} 个节点的 ISP 值:")
    
    for i in range(min(num_samples, N)):
        isp = compute_isp(node_tokens[i])
        label_str = ""
        if labels is not None:
            label_str = f" [Label={labels[i]}]"
        print(f"  Node {i}{label_str}: ISP = {isp.numpy()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ISP Analysis for NAGphormer Tokenization')
    parser.add_argument('--dataset', type=str, default='BlogCatalog', 
                        choices=['BlogCatalog', 'Flickr', 'ACM', 'Coris', 'Amazon', 'reddit', 'dgraph', 'tolokers', 'photo', 'elliptic', 't_finance'])
    parser.add_argument('--pp_k', type=int, default=6, help='Number of propagation hops')
    parser.add_argument('--alpha', type=float, default=0.2, help='Restart probability')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='figs/isp_analysis')
    
    args = parser.parse_args()
    
    run_isp_experiment(
        dataset=args.dataset,
        pp_k=args.pp_k,
        alpha=args.alpha,
        device=args.device,
        train_rate=args.train_rate,
        seed=args.seed,
        save_dir=args.save_dir
    )