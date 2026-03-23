"""
批量Delta向量分析：在多个数据集上验证Delta向量的异常检测能力
"""

import torch
import numpy as np
import os
from delta_vector_analysis import run_delta_vector_analysis

# 数据集列表
DATASETS = ['t_finance', 'Amazon', 'tolokers', 'photo', 'elliptic', 'reddit']

def run_batch_analysis():
    """批量运行所有数据集的分析"""
    results = {}
    
    for dataset in DATASETS:
        print(f"\n{'='*70}")
        print(f"正在处理数据集: {dataset}")
        print(f"{'='*70}")
        
        try:
            result = run_delta_vector_analysis(
                dataset=dataset,
                pp_k=6,
                alpha=0.2,
                device=0,
                train_rate=0.05,
                seed=42,
                save_dir=f'figs/delta_vector'
            )
            results[dataset] = result
        except Exception as e:
            print(f"数据集 {dataset} 分析失败: {e}")
            results[dataset] = None
    
    # 汇总结果
    print("\n" + "="*70)
    print("汇总结果")
    print("="*70)
    
    print(f"\n{'数据集':<15} {'向量AUC':>12} {'范数AUC':>12} {'提升':>12} {'相关性':>12}")
    print("-"*65)
    
    for dataset in DATASETS:
        if results[dataset] is not None:
            vector_results = results[dataset]['vector_results']
            norm_result = results[dataset]['norm_result']
            correlations = results[dataset]['correlations']
            
            # 找最佳向量方法
            best_method = max(vector_results.keys(), key=lambda x: vector_results[x]['AUC'])
            best_vector_auc = vector_results[best_method]['AUC']
            norm_auc = norm_result['AUC']
            improvement = best_vector_auc - norm_auc
            avg_corr = np.nanmean(correlations)
            
            print(f"{dataset:<15} {best_vector_auc:>12.4f} {norm_auc:>12.4f} {improvement:>+12.4f} {avg_corr:>12.4f}")
        else:
            print(f"{dataset:<15} {'FAILED':>12}")
    
    # 保存汇总结果到文件
    with open('docs/diary/2026-03-23-delta-vector-summary.md', 'w') as f:
        f.write("# Delta向量批量验证结果汇总\n\n")
        f.write(f"**验证日期**: 2026-03-23\n\n")
        f.write("## 结果对比表\n\n")
        f.write("| 数据集 | 向量AUC | 范数AUC | 提升 | 相关性 |\n")
        f.write("|--------|---------|---------|------|--------|\n")
        
        for dataset in DATASETS:
            if results[dataset] is not None:
                vector_results = results[dataset]['vector_results']
                norm_result = results[dataset]['norm_result']
                correlations = results[dataset]['correlations']
                
                best_method = max(vector_results.keys(), key=lambda x: vector_results[x]['AUC'])
                best_vector_auc = vector_results[best_method]['AUC']
                norm_auc = norm_result['AUC']
                improvement = best_vector_auc - norm_auc
                avg_corr = np.nanmean(correlations)
                
                f.write(f"| {dataset} | {best_vector_auc:.4f} | {norm_auc:.4f} | {improvement:+.4f} | {avg_corr:.4f} |\n")
            else:
                f.write(f"| {dataset} | FAILED | - | - | - |\n")
        
        f.write("\n## 详细结果\n\n")
        for dataset in DATASETS:
            if results[dataset] is not None:
                f.write(f"### {dataset}\n\n")
                vector_results = results[dataset]['vector_results']
                norm_result = results[dataset]['norm_result']
                
                f.write("**向量方法对比:**\n\n")
                f.write("| 方法 | AUC | AP |\n")
                f.write("|------|-----|----|\n")
                for method, res in vector_results.items():
                    f.write(f"| {method} | {res['AUC']:.4f} | {res['AP']:.4f} |\n")
                
                f.write(f"\n**范数基线:** AUC={norm_result['AUC']:.4f}, AP={norm_result['AP']:.4f}\n\n")
    
    print("\n汇总结果已保存到: docs/diary/2026-03-23-delta-vector-summary.md")
    
    return results


if __name__ == "__main__":
    run_batch_analysis()