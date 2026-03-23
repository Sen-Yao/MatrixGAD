"""
Delta向量分析：验证向量形式Delta的异常检测能力

核心问题：
1. Delta向量（不取L2范数）是否具有异常检测能力？
2. 向量形式 vs L2范数形式，哪种更有效？
3. Delta向量与原始token的信息冗余度如何？
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils import load_mat, load_dgraph, normalize_adj, preprocess_features, nagphormer_tokenization
import scipy.sparse as sp


def compute_delta_vectors(node_tokens):
    """
    计算Delta向量（不取L2范数）
    
    Args:
        node_tokens: [N, K+1, D] - 多hop token序列
    
    Returns:
        delta_vectors: [N, K, D] - 相邻hop的变化向量
    """
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]
    return delta_vectors


def compute_delta_norms(node_tokens):
    """
    计算Delta的L2范数（当前方法）
    
    Args:
        node_tokens: [N, K+1, D]
    
    Returns:
        delta_norms: [N, K] - 每个hop的delta范数
    """
    delta_vectors = node_tokens[:, 1:] - node_tokens[:, :-1]
    delta_norms = torch.norm(delta_vectors, p=2, dim=-1)
    return delta_norms


def evaluate_delta_vector_classification(delta_vectors, labels, method='flatten'):
    """
    评估Delta向量的分类能力
    
    Args:
        delta_vectors: [N, K, D]
        labels: [N]
        method: 如何处理向量
            - 'flatten': 展平为 [N, K*D]
            - 'mean': 对hop维度取平均 [N, D]
            - 'sum': 对hop维度求和 [N, D]
            - 'last': 只用最后一个hop [N, D]
            - 'max': 对hop维度取最大 [N, D]
    
    Returns:
        results: {method: {'AUC': ..., 'AP': ...}}
    """
    labels = np.array(labels)
    
    if method == 'flatten':
        # 展平所有hop和特征维度
        features = delta_vectors.reshape(delta_vectors.shape[0], -1).numpy()
    elif method == 'mean':
        features = delta_vectors.mean(dim=1).numpy()
    elif method == 'sum':
        features = delta_vectors.sum(dim=1).numpy()
    elif method == 'last':
        features = delta_vectors[:, -1, :].numpy()
    elif method == 'max':
        features = delta_vectors.abs().max(dim=1)[0].numpy()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 使用简单分类器评估
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(features, labels)
        probas = clf.predict_proba(features)[:, 1]
        
        auc = roc_auc_score(labels, probas)
        ap = average_precision_score(labels, probas)
    except Exception as e:
        print(f"Classification failed: {e}")
        auc, ap = 0.5, 0.5
    
    return {'AUC': auc, 'AP': ap}


def evaluate_delta_norm_classification(delta_norms, labels):
    """
    评估Delta范数（当前方法）的分类能力
    """
    labels = np.array(labels)
    features = delta_norms.numpy()
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(features, labels)
        probas = clf.predict_proba(features)[:, 1]
        
        auc = roc_auc_score(labels, probas)
        ap = average_precision_score(labels, probas)
    except Exception as e:
        print(f"Classification failed: {e}")
        auc, ap = 0.5, 0.5
    
    return {'AUC': auc, 'AP': ap}


def analyze_feature_importance(delta_vectors, labels, top_k=10):
    """
    分析哪些特征维度的Delta对异常检测最重要
    """
    labels = np.array(labels)
    
    # 计算每个特征维度在每个hop的区分能力
    num_hops = delta_vectors.shape[1]
    num_features = delta_vectors.shape[2]
    
    feature_aucs = np.zeros((num_hops, num_features))
    
    for k in range(num_hops):
        for d in range(num_features):
            feature_values = delta_vectors[:, k, d].numpy()
            
            try:
                auc = roc_auc_score(labels, feature_values)
                # 如果AUC < 0.5，反向
                if auc < 0.5:
                    auc = 1 - auc
                feature_aucs[k, d] = auc
            except:
                feature_aucs[k, d] = 0.5
    
    return feature_aucs


def compute_redundancy_with_original(node_tokens, delta_vectors):
    """
    计算Delta向量与原始token的冗余度
    
    Returns:
        correlation: Delta向量与原始token的相关性矩阵
    """
    # 使用最后一个hop进行分析
    original_last = node_tokens[:, -1, :].numpy()  # [N, D]
    delta_last = delta_vectors[:, -1, :].numpy()   # [N, D]
    
    # 计算每个特征维度的相关性
    correlations = np.array([
        np.corrcoef(original_last[:, d], delta_last[:, d])[0, 1]
        if not (np.isnan(original_last[:, d]).any() or np.isnan(delta_last[:, d]).any())
        else 0.0
        for d in range(original_last.shape[1])
    ])
    
    return correlations


def visualize_delta_vector_distribution(delta_vectors, labels, save_path, num_features_to_show=5):
    """
    可视化Delta向量的分布
    """
    labels = np.array(labels)
    num_hops = delta_vectors.shape[1]
    
    # 选择方差最大的几个特征维度
    feature_var = delta_vectors.var(dim=0).mean(dim=0).numpy()  # [D]
    top_features = np.argsort(feature_var)[-num_features_to_show:]
    
    fig, axes = plt.subplots(num_features_to_show, num_hops, figsize=(4*num_hops, 3*num_features_to_show))
    
    delta_np = delta_vectors.numpy()
    
    for fi, feat_idx in enumerate(top_features):
        for k in range(num_hops):
            ax = axes[fi, k] if num_features_to_show > 1 else axes[k]
            
            normal_vals = delta_np[labels == 0, k, feat_idx]
            abnormal_vals = delta_np[labels == 1, k, feat_idx]
            
            # Clip extreme values
            p1, p99 = np.percentile(np.concatenate([normal_vals, abnormal_vals]), [1, 99])
            normal_vals = np.clip(normal_vals, p1, p99)
            abnormal_vals = np.clip(abnormal_vals, p1, p99)
            
            sns.histplot(normal_vals, ax=ax, label='Normal', color='blue', alpha=0.5, stat='density')
            sns.histplot(abnormal_vals, ax=ax, label='Abnormal', color='red', alpha=0.5, stat='density')
            
            ax.set_title(f'Feature {feat_idx}, Hop {k}')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {save_path}")


def run_delta_vector_analysis(dataset='tolokers', pp_k=6, alpha=0.2, device=0,
                              train_rate=0.05, seed=42, save_dir='figs/delta_vector'):
    """
    运行完整的Delta向量分析
    """
    import argparse
    import random
    
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
    
    print("="*70)
    print(f"Delta向量分析 - 数据集: {dataset}")
    print(f"参数: pp_k={pp_k}, alpha={alpha}")
    print("="*70)
    
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
    
    print(f"\n数据集信息:")
    print(f"  节点数: {features.shape[0]}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  异常节点数: {sum(np.array(ano_label) == 1)}")
    print(f"  异常率: {sum(np.array(ano_label) == 1) / len(ano_label) * 100:.2f}%")
    
    # Tokenization
    print("\n执行 NAGphormer Tokenization...")
    node_tokens = nagphormer_tokenization(features, adj, args)
    print(f"Token shape: {node_tokens.shape}")
    
    # 计算Delta向量和Delta范数
    print("\n计算Delta向量...")
    delta_vectors = compute_delta_vectors(node_tokens)
    print(f"Delta向量 shape: {delta_vectors.shape}")  # [N, K, D]
    
    print("计算Delta范数...")
    delta_norms = compute_delta_norms(node_tokens)
    print(f"Delta范数 shape: {delta_norms.shape}")  # [N, K]
    
    # 准备标签
    labels_array = np.squeeze(np.array(ano_label))
    
    # ========== 实验1：Delta向量分类能力评估 ==========
    print("\n" + "="*70)
    print("实验1：Delta向量分类能力评估")
    print("="*70)
    
    vector_methods = ['flatten', 'mean', 'sum', 'last', 'max']
    vector_results = {}
    
    print(f"\n{'方法':<15} {'AUC':>10} {'AP':>10}")
    print("-"*40)
    
    for method in vector_methods:
        result = evaluate_delta_vector_classification(delta_vectors, labels_array, method)
        vector_results[method] = result
        print(f"{method:<15} {result['AUC']:>10.4f} {result['AP']:>10.4f}")
    
    # ========== 实验2：Delta范数分类能力评估（基线） ==========
    print("\n" + "="*70)
    print("实验2：Delta范数分类能力评估（当前方法基线）")
    print("="*70)
    
    norm_result = evaluate_delta_norm_classification(delta_norms, labels_array)
    print(f"\n{'Delta范数':<15} {norm_result['AUC']:>10.4f} {norm_result['AP']:>10.4f}")
    
    # ========== 实验3：单hop Delta向量评估 ==========
    print("\n" + "="*70)
    print("实验3：各hop Delta向量单独评估")
    print("="*70)
    
    print(f"\n{'Hop':<10} {'向量AUC':>12} {'向量AP':>12} {'范数AUC':>12} {'范数AP':>12}")
    print("-"*60)
    
    for k in range(delta_vectors.shape[1]):
        # 单hop向量
        vec_result = evaluate_delta_vector_classification(
            delta_vectors[:, k:k+1, :], labels_array, 'last'
        )
        # 单hop范数
        norm_auc = roc_auc_score(labels_array, delta_norms[:, k].numpy())
        norm_ap = average_precision_score(labels_array, delta_norms[:, k].numpy())
        
        print(f"Hop {k:<5} {vec_result['AUC']:>12.4f} {vec_result['AP']:>12.4f} {norm_auc:>12.4f} {norm_ap:>12.4f}")
    
    # ========== 实验4：信息冗余度分析 ==========
    print("\n" + "="*70)
    print("实验4：Delta向量与原始Token的信息冗余度")
    print("="*70)
    
    correlations = compute_redundancy_with_original(node_tokens, delta_vectors)
    print(f"\nDelta向量与原始Token各特征维度的相关性:")
    print(f"  平均相关性: {np.nanmean(correlations):.4f}")
    print(f"  最大相关性: {np.nanmax(correlations):.4f}")
    print(f"  最小相关性: {np.nanmin(correlations):.4f}")
    print(f"  相关性标准差: {np.nanstd(correlations):.4f}")
    
    # ========== 实验5：特征重要性分析 ==========
    print("\n" + "="*70)
    print("实验5：特征维度重要性分析")
    print("="*70)
    
    feature_aucs = analyze_feature_importance(delta_vectors, labels_array)
    
    print(f"\n各hop最有区分度的特征维度Top5:")
    for k in range(min(3, feature_aucs.shape[0])):
        top_features = np.argsort(feature_aucs[k])[-5:][::-1]
        top_aucs = feature_aucs[k, top_features]
        print(f"  Hop {k}: 特征{top_features}, AUC={top_aucs}")
    
    # ========== 保存可视化 ==========
    os.makedirs(save_dir, exist_ok=True)
    
    # 分布可视化
    visualize_delta_vector_distribution(
        delta_vectors, labels_array,
        save_path=os.path.join(save_dir, f'{dataset}_delta_vector_distribution.pdf')
    )
    
    # 特征重要性热图
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(feature_aucs, ax=ax, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                xticklabels=10, yticklabels=range(feature_aucs.shape[0]))
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Hop')
    ax.set_title(f'Feature Importance Heatmap (AUC) - {dataset}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset}_feature_importance.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{dataset}_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n特征重要性热图保存至: {os.path.join(save_dir, f'{dataset}_feature_importance.pdf')}")
    
    # ========== 总结 ==========
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    best_vector_method = max(vector_results.keys(), key=lambda x: vector_results[x]['AUC'])
    best_vector_auc = vector_results[best_vector_method]['AUC']
    
    print(f"\nDelta向量最佳方法: {best_vector_method}")
    print(f"  AUC: {best_vector_auc:.4f}")
    print(f"  AP: {vector_results[best_vector_method]['AP']:.4f}")
    
    print(f"\nDelta范数（当前方法）:")
    print(f"  AUC: {norm_result['AUC']:.4f}")
    print(f"  AP: {norm_result['AP']:.4f}")
    
    improvement = best_vector_auc - norm_result['AUC']
    if improvement > 0:
        print(f"\n✅ Delta向量比Delta范数提升: +{improvement:.4f}")
    else:
        print(f"\n❌ Delta向量比Delta范数下降: {improvement:.4f}")
    
    print(f"\n信息冗余度: 平均相关性 {np.nanmean(correlations):.4f}")
    if abs(np.nanmean(correlations)) < 0.3:
        print("  → 低冗余，Delta向量与原始Token互补性强")
    elif abs(np.nanmean(correlations)) < 0.6:
        print("  → 中等冗余，Delta向量有额外信息")
    else:
        print("  → 高冗余，Delta向量信息可能大部分被原始Token覆盖")
    
    return {
        'vector_results': vector_results,
        'norm_result': norm_result,
        'correlations': correlations,
        'feature_aucs': feature_aucs
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Delta Vector Analysis')
    parser.add_argument('--dataset', type=str, default='tolokers',
                        choices=['BlogCatalog', 'Flickr', 'ACM', 'Coris', 'Amazon', 
                                'reddit', 'dgraph', 'tolokers', 'photo', 'elliptic', 't_finance'])
    parser.add_argument('--pp_k', type=int, default=6, help='Number of propagation hops')
    parser.add_argument('--alpha', type=float, default=0.2, help='Restart probability')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='figs/delta_vector')
    
    args = parser.parse_args()
    
    run_delta_vector_analysis(
        dataset=args.dataset,
        pp_k=args.pp_k,
        alpha=args.alpha,
        device=args.device,
        train_rate=args.train_rate,
        seed=args.seed,
        save_dir=args.save_dir
    )