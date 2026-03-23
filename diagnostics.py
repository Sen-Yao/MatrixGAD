"""
Training Diagnostics Module - Monitor key metrics during model training
"""

import torch
import numpy as np
import time


class DiagnosticCache:
    """Cache for diagnostic computation to ensure we use the same batch of nodes every time"""
    def __init__(self):
        self.cached_batch = None  # (concated_input_features, labels, batch_global_indices)
        self.cached_batch_indices = None  # global indices of the cached batch nodes
        self.is_initialized = False


# ============================================================
# MFMGAD Diagnostic Functions
# ============================================================

class MFMGADDiagnosticCache:
    """Cache for MFMGAD diagnostic computation"""
    def __init__(self):
        self.cached_batch = None
        self.cached_batch_indices = None
        self.is_initialized = False


def compute_mfmgad_diagnostics(model, data_loader, ano_label, idx_test, device, args, 
                               normal_for_train_idx=None, cache=None):
    """
    Compute diagnostic metrics for MFMGAD during evaluation
    
    MFMGAD特有的关键指标：
    1. 频域Token质量分析
    2. 注意力权重正交性
    3. 一致性预测误差
    4. 可疑度分数分布
    5. 嵌入空间分离度
    
    Args:
        model: MFMGAD模型
        data_loader: 数据加载器
        ano_label: 异常标签
        idx_test: 测试集索引
        device: 设备
        args: 参数配置
        normal_for_train_idx: 训练用的正常节点索引
        cache: 诊断缓存
    
    Returns:
        diagnostics: 诊断指标字典
    """
    total_start = time.time()
    timing = {}
    
    # 构建索引映射
    idx_test_set = set(idx_test)
    idx_test_np = np.array(idx_test)
    sorted_indices = np.argsort(idx_test_np)
    idx_test_sorted = idx_test_np[sorted_indices]
    
    model.eval()
    
    # Step 1: 获取批次数据
    t0 = time.time()
    if cache is not None and cache.is_initialized:
        concated_input_features = cache.cached_batch[0].to(device)
        batch_global_indices = cache.cached_batch[2].to(device) if cache.cached_batch[2] is not None else None
    else:
        first_item = next(iter(data_loader))
        concated_input_features = first_item[0].to(device)
        batch_global_indices = first_item[2].to(device) if len(first_item) > 2 else None
        
        if cache is not None:
            cache.cached_batch = (first_item[0], first_item[1] if len(first_item) > 1 else None, 
                                  first_item[2] if len(first_item) > 2 else None)
            cache.cached_batch_indices = batch_global_indices.cpu() if batch_global_indices is not None else None
            cache.is_initialized = True
    timing['get_batch'] = time.time() - t0
    
    # Step 2: 模型前向传播获取中间变量
    t0 = time.time()
    with torch.no_grad():
        # 调用模型，获取详细信息
        result = model(concated_input_features, None, None, None, False, args)
        
        # MFMGAD 推理模式返回: 
        # (embeddings, logits, anomaly_scores, consistency_errors, suspicion_scores, attn_weights)
        embeddings = result[0].squeeze(0)  # [num_nodes, hidden_dim]
        logits = result[1].squeeze(0) if result[1] is not None else None  # [num_nodes, 1] or [1, 1]
        anomaly_scores = result[2]  # [num_nodes]
        consistency_errors = result[3]  # [num_nodes] - 推理模式下可能为0
        suspicion_scores = result[4]  # [num_nodes] - 推理模式下可能为0
        attn_weights = result[5]  # [num_nodes, num_prompts, num_hops]
        
        # 获取频域token用于计算额外的诊断指标
        try:
            frequency_tokens, _ = model.extract_frequency_tokens(concated_input_features)
        except:
            frequency_tokens = None
        
        # 推理模式下手动计算 consistency_errors 和 suspicion_scores
        # 如果它们是0，我们使用重构误差作为替代
        if frequency_tokens is not None and model is not None:
            # 计算重构误差作为 consistency 的替代指标
            try:
                # 使用模型的 token_decoder 计算重构
                embeddings_for_recon = embeddings.unsqueeze(0) if embeddings.dim() == 2 else embeddings
                reconstructed = model.token_decoder(embeddings_for_recon).squeeze(0)
                target = frequency_tokens.view(frequency_tokens.size(0), -1)
                recon_error = (reconstructed - target).norm(dim=1)
                # 归一化
                if recon_error.max() > recon_error.min():
                    consistency_errors = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + 1e-8)
                else:
                    consistency_errors = torch.zeros_like(recon_error)
            except:
                pass
            
            # 计算距离中心的距离作为 suspicion 的替代指标
            try:
                # 使用注意力权重计算主导 Prompt
                if attn_weights is not None and attn_weights.size(0) > 0:
                    attn_sum = attn_weights.sum(dim=-1)  # [num_nodes, num_prompts]
                    dominant_prompts = torch.argmax(attn_sum, dim=-1)  # [num_nodes]
                    
                    # 计算每个节点的特征向量到全局均值的距离
                    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    global_center = embeddings_norm.mean(dim=0, keepdim=True)
                    distances = torch.norm(embeddings_norm - global_center, dim=1)
                    
                    # 归一化
                    if distances.max() > distances.min():
                        suspicion_scores = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
                    else:
                        suspicion_scores = torch.zeros_like(distances)
            except:
                pass
    timing['model_forward'] = time.time() - t0
    
    if embeddings.size(0) == 0:
        return {}
    
    # Step 3: 分离正常/异常节点
    t0 = time.time()
    idx_test_dev = torch.tensor(idx_test, device=device)
    
    # 获取测试集掩码
    if batch_global_indices is not None:
        test_mask = torch.isin(batch_global_indices, idx_test_dev)
        test_embeddings = embeddings[test_mask]
        test_logits = logits[test_mask] if logits is not None and logits.dim() > 1 else logits
        test_anomaly_scores = anomaly_scores[test_mask] if anomaly_scores is not None else None
        test_consistency_errors = consistency_errors[test_mask] if consistency_errors is not None else None
        test_suspicion_scores = suspicion_scores[test_mask] if suspicion_scores is not None else None
        test_attn_weights = attn_weights[test_mask] if attn_weights is not None else None
        test_indices_cpu = batch_global_indices[test_mask].cpu().numpy()
    else:
        test_embeddings = embeddings
        test_logits = logits
        test_anomaly_scores = anomaly_scores
        test_consistency_errors = consistency_errors
        test_suspicion_scores = suspicion_scores
        test_attn_weights = attn_weights
        test_indices_cpu = np.arange(len(idx_test))
    
    # 获取测试标签
    valid_mask = np.isin(test_indices_cpu, idx_test)
    valid_test_indices = test_indices_cpu[valid_mask]
    
    if len(valid_test_indices) > 0:
        positions_in_sorted = np.searchsorted(idx_test_sorted, valid_test_indices)
        test_labels = ano_label[sorted_indices[positions_in_sorted]]
        
        # 更新测试集数据
        test_embeddings = test_embeddings[torch.tensor(valid_mask, device=device)]
        if test_logits is not None and test_logits.dim() > 1:
            test_logits = test_logits[torch.tensor(valid_mask, device=device)]
        if test_anomaly_scores is not None:
            test_anomaly_scores = test_anomaly_scores[torch.tensor(valid_mask, device=device)]
        if test_consistency_errors is not None:
            test_consistency_errors = test_consistency_errors[torch.tensor(valid_mask, device=device)]
        if test_suspicion_scores is not None:
            test_suspicion_scores = test_suspicion_scores[torch.tensor(valid_mask, device=device)]
        if test_attn_weights is not None:
            test_attn_weights = test_attn_weights[torch.tensor(valid_mask, device=device)]
    else:
        test_labels = np.array([])
    
    norm_mask_test = (test_labels == 0) if len(test_labels) > 0 else np.array([])
    abnorm_mask_test = (test_labels == 1) if len(test_labels) > 0 else np.array([])
    
    # 分离正常/异常嵌入
    norm_embs = test_embeddings[torch.tensor(norm_mask_test, dtype=torch.bool, device=device)] if len(norm_mask_test) > 0 else torch.tensor([], device=device)
    abnorm_embs = test_embeddings[torch.tensor(abnorm_mask_test, dtype=torch.bool, device=device)] if len(abnorm_mask_test) > 0 else torch.tensor([], device=device)
    
    # 分离正常/异常注意力权重
    norm_attn = test_attn_weights[torch.tensor(norm_mask_test, dtype=torch.bool, device=device)] if test_attn_weights is not None and len(norm_mask_test) > 0 else None
    abnorm_attn = test_attn_weights[torch.tensor(abnorm_mask_test, dtype=torch.bool, device=device)] if test_attn_weights is not None and len(abnorm_mask_test) > 0 else None
    timing['prepare_data'] = time.time() - t0
    
    # Step 4: 计算诊断指标
    t0 = time.time()
    diagnostics = {}
    
    # ============ 1. 嵌入空间分析 ============
    if norm_embs.size(0) > 1:
        # 正常节点嵌入的余弦相似度（检测嵌入坍塌）
        num_samples = min(norm_embs.size(0), 500)
        sampled_norm_embs = torch.nn.functional.normalize(norm_embs[:num_samples], p=2, dim=1)
        cos_sim_matrix = torch.mm(sampled_norm_embs, sampled_norm_embs.t())
        mask = torch.eye(num_samples, dtype=torch.bool, device=device).flatten()
        diagnostics['norm_avg_cos_sim'] = cos_sim_matrix.flatten()[~mask].mean().item()
        diagnostics['norm_cos_sim_std'] = cos_sim_matrix.flatten()[~mask].std().item()
    else:
        diagnostics['norm_avg_cos_sim'] = float('nan')
        diagnostics['norm_cos_sim_std'] = float('nan')
    
    # 正常/异常节点中心距离
    if norm_embs.size(0) > 0 and abnorm_embs.size(0) > 0:
        norm_center = norm_embs.mean(dim=0)
        abnorm_center = abnorm_embs.mean(dim=0)
        diagnostics['center_dist'] = torch.norm(norm_center - abnorm_center, p=2).item()
        diagnostics['norm_intra_dist'] = torch.norm(norm_embs - norm_center, p=2, dim=1).mean().item()
        diagnostics['abnorm_intra_dist'] = torch.norm(abnorm_embs - abnorm_center, p=2, dim=1).mean().item() if abnorm_embs.size(0) > 1 else 0.0
        diagnostics['separation_ratio'] = diagnostics['center_dist'] / (diagnostics['norm_intra_dist'] + 1e-8)
    else:
        diagnostics['center_dist'] = float('nan')
        diagnostics['norm_intra_dist'] = float('nan')
        diagnostics['abnorm_intra_dist'] = float('nan')
        diagnostics['separation_ratio'] = float('nan')
    
    # ============ 2. 注意力权重分析 ============
    if test_attn_weights is not None and test_attn_weights.size(0) > 0:
        # 整体注意力统计
        diagnostics['attn_mean'] = test_attn_weights.mean().item()
        diagnostics['attn_std'] = test_attn_weights.std().item()
        diagnostics['attn_max'] = test_attn_weights.max().item()
        diagnostics['attn_min'] = test_attn_weights.min().item()
        
        # 各Prompt的注意力分布
        attn_per_prompt = test_attn_weights.sum(dim=-1)  # [num_nodes, num_prompts]
        diagnostics['attn_entropy_per_prompt'] = _compute_attention_entropy(attn_per_prompt)
        
        # 正交性检查（不同Prompt之间的注意力向量余弦相似度）
        ortho_loss = _compute_orthogonality(test_attn_weights)
        diagnostics['attn_orthogonality'] = ortho_loss
        
        # 正常/异常节点的注意力差异
        if norm_attn is not None and norm_attn.size(0) > 0:
            diagnostics['norm_attn_mean'] = norm_attn.mean().item()
            diagnostics['norm_attn_std'] = norm_attn.std().item()
        else:
            diagnostics['norm_attn_mean'] = float('nan')
            diagnostics['norm_attn_std'] = float('nan')
        
        if abnorm_attn is not None and abnorm_attn.size(0) > 0:
            diagnostics['abnorm_attn_mean'] = abnorm_attn.mean().item()
            diagnostics['abnorm_attn_std'] = abnorm_attn.std().item()
        else:
            diagnostics['abnorm_attn_mean'] = float('nan')
            diagnostics['abnorm_attn_std'] = float('nan')
    else:
        diagnostics['attn_mean'] = float('nan')
        diagnostics['attn_std'] = float('nan')
        diagnostics['attn_max'] = float('nan')
        diagnostics['attn_min'] = float('nan')
        diagnostics['attn_entropy_per_prompt'] = float('nan')
        diagnostics['attn_orthogonality'] = float('nan')
        diagnostics['norm_attn_mean'] = float('nan')
        diagnostics['norm_attn_std'] = float('nan')
        diagnostics['abnorm_attn_mean'] = float('nan')
        diagnostics['abnorm_attn_std'] = float('nan')
    
    # ============ 3. 一致性误差分析 ============
    if test_consistency_errors is not None and test_consistency_errors.size(0) > 0:
        diagnostics['consistency_error_mean'] = test_consistency_errors.mean().item()
        diagnostics['consistency_error_std'] = test_consistency_errors.std().item()
        diagnostics['consistency_error_max'] = test_consistency_errors.max().item()
        diagnostics['consistency_error_min'] = test_consistency_errors.min().item()
        
        # 正常/异常节点的一致性误差差异
        norm_consistency = test_consistency_errors[torch.tensor(norm_mask_test, dtype=torch.bool, device=device)] if len(norm_mask_test) > 0 else None
        abnorm_consistency = test_consistency_errors[torch.tensor(abnorm_mask_test, dtype=torch.bool, device=device)] if len(abnorm_mask_test) > 0 else None
        
        if norm_consistency is not None and norm_consistency.size(0) > 0:
            diagnostics['norm_consistency_mean'] = norm_consistency.mean().item()
            diagnostics['norm_consistency_std'] = norm_consistency.std().item()
        else:
            diagnostics['norm_consistency_mean'] = float('nan')
            diagnostics['norm_consistency_std'] = float('nan')
        
        if abnorm_consistency is not None and abnorm_consistency.size(0) > 0:
            diagnostics['abnorm_consistency_mean'] = abnorm_consistency.mean().item()
            diagnostics['abnorm_consistency_std'] = abnorm_consistency.std().item()
        else:
            diagnostics['abnorm_consistency_mean'] = float('nan')
            diagnostics['abnorm_consistency_std'] = float('nan')
    else:
        diagnostics['consistency_error_mean'] = float('nan')
        diagnostics['consistency_error_std'] = float('nan')
        diagnostics['consistency_error_max'] = float('nan')
        diagnostics['consistency_error_min'] = float('nan')
        diagnostics['norm_consistency_mean'] = float('nan')
        diagnostics['norm_consistency_std'] = float('nan')
        diagnostics['abnorm_consistency_mean'] = float('nan')
        diagnostics['abnorm_consistency_std'] = float('nan')
    
    # ============ 4. 可疑度分数分析 ============
    if test_suspicion_scores is not None and test_suspicion_scores.size(0) > 0:
        diagnostics['suspicion_mean'] = test_suspicion_scores.mean().item()
        diagnostics['suspicion_std'] = test_suspicion_scores.std().item()
        diagnostics['suspicion_max'] = test_suspicion_scores.max().item()
        diagnostics['suspicion_min'] = test_suspicion_scores.min().item()
        
        # 正常/异常节点的可疑度差异
        norm_suspicion = test_suspicion_scores[torch.tensor(norm_mask_test, dtype=torch.bool, device=device)] if len(norm_mask_test) > 0 else None
        abnorm_suspicion = test_suspicion_scores[torch.tensor(abnorm_mask_test, dtype=torch.bool, device=device)] if len(abnorm_mask_test) > 0 else None
        
        if norm_suspicion is not None and norm_suspicion.size(0) > 0:
            diagnostics['norm_suspicion_mean'] = norm_suspicion.mean().item()
            diagnostics['norm_suspicion_std'] = norm_suspicion.std().item()
        else:
            diagnostics['norm_suspicion_mean'] = float('nan')
            diagnostics['norm_suspicion_std'] = float('nan')
        
        if abnorm_suspicion is not None and abnorm_suspicion.size(0) > 0:
            diagnostics['abnorm_suspicion_mean'] = abnorm_suspicion.mean().item()
            diagnostics['abnorm_suspicion_std'] = abnorm_suspicion.std().item()
            
            # 可疑度分离度（正常应该低，异常应该高）
            if diagnostics['norm_suspicion_mean'] is not float('nan'):
                diagnostics['suspicion_separation'] = diagnostics['abnorm_suspicion_mean'] - diagnostics['norm_suspicion_mean']
        else:
            diagnostics['abnorm_suspicion_mean'] = float('nan')
            diagnostics['abnorm_suspicion_std'] = float('nan')
            diagnostics['suspicion_separation'] = float('nan')
    else:
        diagnostics['suspicion_mean'] = float('nan')
        diagnostics['suspicion_std'] = float('nan')
        diagnostics['suspicion_max'] = float('nan')
        diagnostics['suspicion_min'] = float('nan')
        diagnostics['norm_suspicion_mean'] = float('nan')
        diagnostics['norm_suspicion_std'] = float('nan')
        diagnostics['abnorm_suspicion_mean'] = float('nan')
        diagnostics['abnorm_suspicion_std'] = float('nan')
        diagnostics['suspicion_separation'] = float('nan')
    
    # ============ 5. 异常得分分析 ============
    if test_anomaly_scores is not None and test_anomaly_scores.size(0) > 0:
        diagnostics['anomaly_score_mean'] = test_anomaly_scores.mean().item()
        diagnostics['anomaly_score_std'] = test_anomaly_scores.std().item()
        
        # 正常/异常节点的异常得分差异
        norm_scores = test_anomaly_scores[torch.tensor(norm_mask_test, dtype=torch.bool, device=device)] if len(norm_mask_test) > 0 else None
        abnorm_scores = test_anomaly_scores[torch.tensor(abnorm_mask_test, dtype=torch.bool, device=device)] if len(abnorm_mask_test) > 0 else None
        
        if norm_scores is not None and norm_scores.size(0) > 0:
            diagnostics['norm_anomaly_score_mean'] = norm_scores.mean().item()
            diagnostics['norm_anomaly_score_std'] = norm_scores.std().item()
        else:
            diagnostics['norm_anomaly_score_mean'] = float('nan')
            diagnostics['norm_anomaly_score_std'] = float('nan')
        
        if abnorm_scores is not None and abnorm_scores.size(0) > 0:
            diagnostics['abnorm_anomaly_score_mean'] = abnorm_scores.mean().item()
            diagnostics['abnorm_anomaly_score_std'] = abnorm_scores.std().item()
            
            if diagnostics['norm_anomaly_score_mean'] is not float('nan'):
                diagnostics['anomaly_score_margin'] = diagnostics['abnorm_anomaly_score_mean'] - diagnostics['norm_anomaly_score_mean']
        else:
            diagnostics['abnorm_anomaly_score_mean'] = float('nan')
            diagnostics['abnorm_anomaly_score_std'] = float('nan')
            diagnostics['anomaly_score_margin'] = float('nan')
    else:
        diagnostics['anomaly_score_mean'] = float('nan')
        diagnostics['anomaly_score_std'] = float('nan')
        diagnostics['norm_anomaly_score_mean'] = float('nan')
        diagnostics['norm_anomaly_score_std'] = float('nan')
        diagnostics['abnorm_anomaly_score_mean'] = float('nan')
        diagnostics['abnorm_anomaly_score_std'] = float('nan')
        diagnostics['anomaly_score_margin'] = float('nan')
    
    # ============ 6. 样本统计 ============
    diagnostics['num_normal'] = norm_embs.size(0) if norm_embs.size(0) > 0 else 0
    diagnostics['num_abnormal'] = abnorm_embs.size(0) if abnorm_embs.size(0) > 0 else 0
    
    timing['compute_metrics'] = time.time() - t0
    timing['total'] = time.time() - total_start
    diagnostics['timing'] = timing
    
    return diagnostics


def _compute_attention_entropy(attn_per_prompt):
    """
    计算注意力的熵（衡量Prompt选择的多样性）
    熵越高，说明Prompt选择越分散；熵越低，说明越集中在某些Prompt
    
    Args:
        attn_per_prompt: [num_nodes, num_prompts] 每个节点的各Prompt注意力总和
    
    Returns:
        avg_entropy: 平均熵
    """
    if attn_per_prompt.size(0) == 0:
        return float('nan')
    
    # 归一化为概率分布
    prob = torch.nn.functional.softmax(attn_per_prompt, dim=-1)
    
    # 计算熵: -sum(p * log(p))
    log_prob = torch.log(prob + 1e-10)
    entropy = -torch.sum(prob * log_prob, dim=-1)
    
    return entropy.mean().item()


def _compute_orthogonality(attn_weights):
    """
    计算不同Prompt之间注意力向量的正交性
    正交性越低（接近0），说明不同Prompt学到的特征越不重叠
    
    Args:
        attn_weights: [num_nodes, num_prompts, num_hops]
    
    Returns:
        ortho_score: 正交性得分（非对角线余弦相似度的均值）
    """
    if attn_weights.size(0) == 0:
        return float('nan')
    
    num_prompts = attn_weights.size(1)
    
    # 归一化
    norms = torch.norm(attn_weights, dim=-1, keepdim=True) + 1e-8
    normalized = attn_weights / norms
    
    # 平均所有节点的注意力模式
    avg_pattern = normalized.mean(dim=0)  # [num_prompts, num_hops]
    
    # 计算Prompt之间的余弦相似度
    cos_sim = torch.mm(avg_pattern, avg_pattern.t())
    
    # 排除对角线
    mask = ~torch.eye(num_prompts, dtype=torch.bool, device=attn_weights.device)
    ortho_score = cos_sim[mask].abs().mean().item()
    
    return ortho_score


def print_mfmgad_diagnostics(diagnostics, epoch, current_lr=None, losses=None, dynamic_weights=None):
    """
    打印MFMGAD的诊断信息
    
    Args:
        diagnostics: 诊断指标字典
        epoch: 当前epoch
        current_lr: 当前学习率
        losses: 损失字典 {'rec': ..., 'ortho': ..., 'consistency': ..., 'contrast': ...}
        dynamic_weights: 动态权重字典
    """
    d = diagnostics
    
    # 打印时间信息
    if 'timing' in d:
        t = d['timing']
        print(f"[MFMGAD-DiagTime@E{epoch}] Total={t.get('total', 0):.3f}s | "
              f"GetBatch={t.get('get_batch', 0):.3f}s, "
              f"ModelFwd={t.get('model_forward', 0):.3f}s, "
              f"Prepare={t.get('prepare_data', 0):.3f}s, "
              f"Metrics={t.get('compute_metrics', 0):.3f}s")
    
    # 打印损失信息
    if current_lr is not None and losses is not None and dynamic_weights is not None:
        w_rec = dynamic_weights.get('rec_loss_weight', 1.0)
        w_ortho = dynamic_weights.get('ortho_loss_weight', 0.1)
        w_consistency = dynamic_weights.get('consistency_weight', 1.0)
        w_contrast = dynamic_weights.get('contrast_weight', 1.0)
        
        weighted_rec = losses.get('rec', 0.0) * w_rec
        weighted_ortho = losses.get('ortho', 0.0) * w_ortho
        weighted_consistency = losses.get('consistency', 0.0) * w_consistency
        weighted_contrast = losses.get('contrast', 0.0) * w_contrast
        
        print(f"[MFMGAD-Diag@E{epoch}] lr={current_lr:.2e} | "
              f"Loss(w): Rec={_format_value(weighted_rec)}({_format_weight(w_rec)}x), "
              f"Ortho={_format_value(weighted_ortho)}({_format_weight(w_ortho)}x), "
              f"Cons={_format_value(weighted_consistency)}({_format_weight(w_consistency)}x), "
              f"Contrast={_format_value(weighted_contrast)}({_format_weight(w_contrast)}x)")
    
    # 打印嵌入空间分析
    print(f"  Embed: norm_cos_sim={d.get('norm_avg_cos_sim', float('nan')):.3f}±{d.get('norm_cos_sim_std', 0):.3f}, "
          f"center_dist={d.get('center_dist', float('nan')):.3f}, "
          f"sep_ratio={d.get('separation_ratio', float('nan')):.3f}")
    
    # 打印注意力权重分析
    if not np.isnan(d.get('attn_mean', float('nan'))):
        print(f"  Attn: mean={d['attn_mean']:.4f}, std={d['attn_std']:.4f}, "
              f"ortho={d['attn_orthogonality']:.4f}, "
              f"entropy={d['attn_entropy_per_prompt']:.4f}")
        
        # 打印正常/异常节点的注意力差异
        if not np.isnan(d.get('norm_attn_mean', float('nan'))):
            print(f"        norm={d['norm_attn_mean']:.4f}±{d['norm_attn_std']:.4f}, "
                  f"abnorm={d['abnorm_attn_mean']:.4f}±{d['abnorm_attn_std']:.4f}")
    
    # 打印一致性误差分析
    if not np.isnan(d.get('consistency_error_mean', float('nan'))):
        print(f"  Consistency: mean={d['consistency_error_mean']:.4f}±{d['consistency_error_std']:.4f}, "
              f"range=[{d['consistency_error_min']:.4f}, {d['consistency_error_max']:.4f}]")
        if not np.isnan(d.get('norm_consistency_mean', float('nan'))):
            print(f"               norm={d['norm_consistency_mean']:.4f}±{d['norm_consistency_std']:.4f}, "
                  f"abnorm={d['abnorm_consistency_mean']:.4f}±{d['abnorm_consistency_std']:.4f}")
    
    # 打印可疑度分数分析
    if not np.isnan(d.get('suspicion_mean', float('nan'))):
        print(f"  Suspicion: mean={d['suspicion_mean']:.4f}±{d['suspicion_std']:.4f}, "
              f"sep={d.get('suspicion_separation', float('nan')):.4f}")
        if not np.isnan(d.get('norm_suspicion_mean', float('nan'))):
            print(f"              norm={d['norm_suspicion_mean']:.4f}±{d['norm_suspicion_std']:.4f}, "
                  f"abnorm={d['abnorm_suspicion_mean']:.4f}±{d['abnorm_suspicion_std']:.4f}")
    
    # 打印异常得分分析
    if not np.isnan(d.get('anomaly_score_mean', float('nan'))):
        print(f"  AnomalyScore: mean={d['anomaly_score_mean']:.4f}±{d['anomaly_score_std']:.4f}, "
              f"margin={d.get('anomaly_score_margin', float('nan')):.4f}")
        if not np.isnan(d.get('norm_anomaly_score_mean', float('nan'))):
            print(f"                 norm={d['norm_anomaly_score_mean']:.4f}±{d['norm_anomaly_score_std']:.4f}, "
                  f"abnorm={d['abnorm_anomaly_score_mean']:.4f}±{d['abnorm_anomaly_score_std']:.4f}")
    
    # 打印样本统计
    print(f"  Samples: normal={d.get('num_normal', 0)}, abnormal={d.get('num_abnormal', 0)}")


# ============================================================
# PromptGAD Diagnostic Functions (Original)
# ============================================================

def compute_diagnostics(model, data_loader, ano_label, idx_test, device, args, normal_for_train_idx=None, cache=None):
    """
    Compute diagnostic metrics during evaluation
    
    FULL GPU ACCELERATION: Keep everything on GPU!
    
    If cache is provided, use the same batch of nodes every time for faster computation
    while still tracking how metrics change with training.
    """
    total_start = time.time()
    timing = {}
    
    # PRE-OPTIMIZATION: Build index map ONCE at the beginning
    idx_to_pos = {idx: pos for pos, idx in enumerate(idx_test)}
    idx_test_set = set(idx_test)
    
    # Create numpy array for vectorized operations
    idx_test_np = np.array(idx_test)
    # Create a searchsorted-based lookup (much faster than dictionary)
    sorted_indices = np.argsort(idx_test_np)
    idx_test_sorted = idx_test_np[sorted_indices]
    
    model.eval()
    
    # Step 1: Get batch
    t0 = time.time()
    if cache is not None and cache.is_initialized:
        # Use cached batch
        concated_input_features = cache.cached_batch[0].to(device)
        batch_global_indices = cache.cached_batch[2].to(device) if cache.cached_batch[2] is not None else None
    else:
        # Get first batch from data_loader
        first_item = next(iter(data_loader))
        concated_input_features = first_item[0].to(device)
        batch_global_indices = first_item[2].to(device) if len(first_item) > 2 else None
        
        # Cache the batch if cache is provided
        if cache is not None:
            cache.cached_batch = (first_item[0], first_item[1] if len(first_item) > 1 else None, first_item[2] if len(first_item) > 2 else None)
            cache.cached_batch_indices = batch_global_indices.cpu() if batch_global_indices is not None else None
            cache.is_initialized = True
    timing['get_batch'] = time.time() - t0
    
    # First pass: collect embeddings and attention weights from the single batch
    t0 = time.time()
    with torch.no_grad():
        # 检查模型是否支持返回注意力权重
        try:
            # 尝试获取注意力权重
            emb, _, logits, _, _, _, _, _, _, _, _, _, prompt_attn_weights = model(
                concated_input_features, None, None, None, False, args, return_attn_weights=True
            )
        except:
            # 如果不支持，使用普通方式
            emb, _, logits, _, _, _, _, _, _, _, _, _ = model(
                concated_input_features, None, None, None, False, args
            )
            prompt_attn_weights = None
        
        concatenated_embs = emb.squeeze(0)
        concatenated_logits = logits.squeeze(0) if logits is not None else None
        concatenated_global_indices = batch_global_indices
    timing['first_model_forward'] = time.time() - t0
    
    if concatenated_embs.size(0) == 0:
        return {}
    
    # Second pass: collect pseudo-anomalies from the same batch
    t0 = time.time()
    all_norm_embs = []
    all_outlier_embs = []
    
    if normal_for_train_idx is not None and len(normal_for_train_idx) > 0:
        # Ensure normal_for_train_idx is on GPU
        normal_for_train_idx_dev = normal_for_train_idx.to(device) if normal_for_train_idx.device != device else normal_for_train_idx
        
        with torch.no_grad():
            if concatenated_global_indices is not None:
                # Find normal nodes in this batch (on GPU)
                is_known_normal_mask = torch.isin(concatenated_global_indices, normal_for_train_idx_dev)
                local_normal_for_train_idx = torch.nonzero(is_known_normal_mask, as_tuple=False).squeeze(-1)
                
                if len(local_normal_for_train_idx) > 0:
                    # Get pseudo-anomalies
                    emb_train, _, _, outlier_emb, _, _, _, _, _, _, _, _ = model(
                        concated_input_features, None, None, local_normal_for_train_idx, True, args
                    )
                    
                    # Extract normal and outlier embeddings (keep on GPU)
                    num_normals = len(local_normal_for_train_idx)
                    batch_norm_embs = emb_train.squeeze(0)[:num_normals]
                    all_norm_embs.append(batch_norm_embs)
                    
                    if outlier_emb is not None and outlier_emb.size(0) > 0:
                        all_outlier_embs.append(outlier_emb)
    timing['second_model_forward'] = time.time() - t0
    
    # Get test set embeddings and logits (on GPU) - note: we might not have test nodes in our single batch
    t0 = time.time()
    
    idx_test_dev = torch.tensor(idx_test, device=device)
    test_mask = torch.isin(concatenated_global_indices, idx_test_dev) if concatenated_global_indices is not None else torch.ones(len(concatenated_embs), dtype=torch.bool, device=device)
    test_embs = concatenated_embs[test_mask]
    test_logits = concatenated_logits[test_mask] if concatenated_logits is not None else None
    
    # Get test labels (on CPU for numpy operations)
    test_labels = ano_label
    if concatenated_global_indices is not None and test_mask.any():
        # Move to CPU for numpy index lookup
        test_indices_cpu = concatenated_global_indices[test_mask].cpu().numpy()
        
        # Find valid indices that are actually in idx_test
        valid_mask = np.isin(test_indices_cpu, idx_test)
        valid_test_indices = test_indices_cpu[valid_mask]
        
        if len(valid_test_indices) > 0:
            # OPTIMIZATION: Use searchsorted for O(log n) lookups instead of list comprehension
            # Find positions in sorted array
            positions_in_sorted = np.searchsorted(idx_test_sorted, valid_test_indices)
            # Map back to original positions
            test_indices_in_array = sorted_indices[positions_in_sorted]
            test_labels = ano_label[test_indices_in_array]
            # Filter test_embs and test_logits to only include valid test nodes
            test_embs = test_embs[torch.tensor(valid_mask, device=device)]
            if test_logits is not None:
                test_logits = test_logits[torch.tensor(valid_mask, device=device)]
        else:
            test_labels = np.array([])
            test_embs = torch.tensor([], device=device)
            test_logits = torch.tensor([], device=device) if test_logits is not None else None
    else:
        test_labels = np.array([])
        test_embs = torch.tensor([], device=device)
        test_logits = torch.tensor([], device=device) if test_logits is not None else None
    
    norm_mask_test = (test_labels == 0) if len(test_labels) > 0 else np.array([])
    abnorm_mask_test = (test_labels == 1) if len(test_labels) > 0 else np.array([])
    
    # Move masks back to GPU for embedding indexing
    norm_embs = test_embs[torch.tensor(norm_mask_test, dtype=torch.bool, device=device)] if len(norm_mask_test) > 0 else torch.tensor([], device=device)
    abnorm_embs = test_embs[torch.tensor(abnorm_mask_test, dtype=torch.bool, device=device)] if len(abnorm_mask_test) > 0 else torch.tensor([], device=device)
    
    # Get logits for normal and abnormal nodes
    norm_logits = test_logits[torch.tensor(norm_mask_test, dtype=torch.bool, device=device)] if (test_logits is not None and len(norm_mask_test) > 0) else torch.tensor([], device=device)
    abnorm_logits = test_logits[torch.tensor(abnorm_mask_test, dtype=torch.bool, device=device)] if (test_logits is not None and len(abnorm_mask_test) > 0) else torch.tensor([], device=device)
    
    # Use model-generated pseudo-anomalies if available, otherwise fallback
    if len(all_norm_embs) > 0 and len(all_outlier_embs) > 0:
        consistent_norm_embs = torch.cat(all_norm_embs, dim=0)
        outlier_embs = torch.cat(all_outlier_embs, dim=0)
    else:
        # Fallback: use normal embeddings if we have them, otherwise use all embeddings
        consistent_norm_embs = norm_embs if norm_embs.size(0) > 0 else concatenated_embs
        outlier_embs = consistent_norm_embs
    timing['prepare_embeddings'] = time.time() - t0
    
    # Compute metrics
    t0 = time.time()
    diagnostics = {}
    
    # Logit metrics - compute from actual logits
    if norm_logits.size(0) > 0 and abnorm_logits.size(0) > 0:
        diagnostics['norm_logits_mean'] = norm_logits.mean().item()
        diagnostics['abnorm_logits_mean'] = abnorm_logits.mean().item()
        diagnostics['logit_margin'] = diagnostics['abnorm_logits_mean'] - diagnostics['norm_logits_mean']
        
        # Compute std for all test logits
        all_test_logits = torch.cat([norm_logits, abnorm_logits], dim=0)
        diagnostics['logit_std'] = all_test_logits.std().item() if all_test_logits.size(0) > 1 else 0.0
    else:
        diagnostics['norm_logits_mean'] = float('nan')
        diagnostics['abnorm_logits_mean'] = float('nan')
        diagnostics['logit_margin'] = float('nan')
        diagnostics['logit_std'] = float('nan')
    
    # Outlier logits metrics - compute from outlier embeddings if we have a way to get logits
    # For now, set to nan since we don't have outlier logits from this batch
    diagnostics['outlier_logits_mean'] = float('nan')
    diagnostics['outlier_logits_std'] = float('nan')
    diagnostics['outlier_logits_max'] = float('nan')
    diagnostics['outlier_logits_min'] = float('nan')
    diagnostics['outlier_embs'] = outlier_embs.cpu() if outlier_embs is not None else None
    
    # Embedding collapse
    if norm_embs.size(0) > 1:
        num_samples = min(norm_embs.size(0), 1000)
        sampled_norm_embs = torch.nn.functional.normalize(norm_embs[:num_samples], p=2, dim=1)
        cos_sim_matrix = torch.mm(sampled_norm_embs, sampled_norm_embs.t())
        mask = torch.eye(num_samples, dtype=torch.bool, device=device).flatten()
        diagnostics['avg_cos_sim'] = cos_sim_matrix.flatten()[~mask].mean().item()
        diagnostics['cos_sim_std'] = cos_sim_matrix.flatten()[~mask].std().item()
    else:
        diagnostics['avg_cos_sim'] = float('nan')
        diagnostics['cos_sim_std'] = float('nan')
    
    # Euclidean separation
    if norm_embs.size(0) > 0 and abnorm_embs.size(0) > 0:
        norm_center = norm_embs.mean(dim=0)
        abnorm_center = abnorm_embs.mean(dim=0)
        diagnostics['center_dist'] = torch.norm(norm_center - abnorm_center, p=2).item()
        diagnostics['norm_intra_dist'] = torch.norm(norm_embs - norm_center, p=2, dim=1).mean().item()
        diagnostics['abnorm_intra_dist'] = torch.norm(abnorm_embs - abnorm_center, p=2, dim=1).mean().item() if abnorm_embs.size(0) > 1 else 0.0
        diagnostics['separation_ratio'] = diagnostics['center_dist'] / (diagnostics['norm_intra_dist'] + 1e-8)
    else:
        diagnostics['center_dist'] = float('nan')
        diagnostics['norm_intra_dist'] = float('nan')
        diagnostics['abnorm_intra_dist'] = float('nan')
        diagnostics['separation_ratio'] = float('nan')
    
    # Triangular Geometry with MODEL-GENERATED PSEUDO-ANOMALIES!
    if outlier_embs is not None and outlier_embs.size(0) > 0 and consistent_norm_embs.size(0) > 0 and abnorm_embs.size(0) > 0:
        norm_center = consistent_norm_embs.mean(dim=0)
        abnorm_center = abnorm_embs.mean(dim=0)
        outlier_center = outlier_embs.mean(dim=0)
        
        diagnostics['dist_norm_outlier'] = torch.norm(norm_center - outlier_center, p=2).item()
        diagnostics['dist_norm_abnorm'] = torch.norm(norm_center - abnorm_center, p=2).item()
        diagnostics['dist_outlier_abnorm'] = torch.norm(outlier_center - abnorm_center, p=2).item()
        diagnostics['outlier_intra_dist'] = torch.norm(outlier_embs - outlier_center, p=2, dim=1).mean().item()
        
        dir_norm_to_outlier = torch.nn.functional.normalize(outlier_center - norm_center, p=2, dim=0)
        dir_norm_to_abnorm = torch.nn.functional.normalize(abnorm_center - norm_center, p=2, dim=0)
        diagnostics['cos_sim_directions'] = torch.dot(dir_norm_to_outlier, dir_norm_to_abnorm).item()
        diagnostics['angle_degrees'] = np.degrees(np.arccos(np.clip(diagnostics['cos_sim_directions'], -1.0, 1.0)))
        diagnostics['outlier_separation_ratio'] = diagnostics['dist_norm_outlier'] / (diagnostics['norm_intra_dist'] + 1e-8)
        diagnostics['outlier_closer_to_abnorm'] = diagnostics['dist_outlier_abnorm'] < diagnostics['dist_norm_outlier']
    else:
        diagnostics['dist_norm_outlier'] = float('nan')
        diagnostics['dist_norm_abnorm'] = float('nan')
        diagnostics['dist_outlier_abnorm'] = float('nan')
        diagnostics['outlier_intra_dist'] = float('nan')
        diagnostics['cos_sim_directions'] = float('nan')
        diagnostics['angle_degrees'] = float('nan')
        diagnostics['outlier_separation_ratio'] = float('nan')
        diagnostics['outlier_closer_to_abnorm'] = False
    
    # Sample statistics
    diagnostics['num_normal'] = norm_embs.size(0) if norm_embs.size(0) > 0 else 0
    diagnostics['num_abnormal'] = abnorm_embs.size(0) if abnorm_embs.size(0) > 0 else 0
    diagnostics['num_outlier'] = outlier_embs.size(0) if outlier_embs is not None else 0
    
    # 保存注意力权重用于打印
    diagnostics['prompt_attn_weights'] = prompt_attn_weights
    
    # Pseudo-anomaly quality metrics
    if outlier_embs is not None and outlier_embs.size(0) > 0:
        norm_embs_normalized = torch.nn.functional.normalize(consistent_norm_embs, p=2, dim=1)
        outlier_embs_normalized = torch.nn.functional.normalize(outlier_embs, p=2, dim=1)
        
        num_normal_sample = min(1000, norm_embs_normalized.size(0))
        num_outlier_sample = min(1000, outlier_embs_normalized.size(0))
        
        sampled_norm_embs = norm_embs_normalized[torch.randperm(norm_embs_normalized.size(0), device=device)[:num_normal_sample]] if norm_embs_normalized.size(0) >= num_normal_sample else norm_embs_normalized
        sampled_outlier_embs = outlier_embs_normalized[torch.randperm(outlier_embs_normalized.size(0), device=device)[:num_outlier_sample]] if outlier_embs_normalized.size(0) >= num_outlier_sample else outlier_embs_normalized
        
        cos_sim_matrix = torch.mm(sampled_norm_embs, sampled_outlier_embs.t())
        diagnostics['pseudo_anomaly_difficulty_coeff'] = cos_sim_matrix.mean().item()
        
        if abnorm_embs.size(0) > 0:
            abnorm_embs_normalized = torch.nn.functional.normalize(abnorm_embs, p=2, dim=1)
            num_abnorm_sample = min(1000, abnorm_embs_normalized.size(0))
            sampled_abnorm_embs = abnorm_embs_normalized[torch.randperm(abnorm_embs_normalized.size(0), device=device)[:num_abnorm_sample]] if abnorm_embs_normalized.size(0) >= num_abnorm_sample else abnorm_embs_normalized
            
            cos_sim_matrix_true = torch.mm(sampled_outlier_embs, sampled_abnorm_embs.t())
            diagnostics['pseudo_anomaly_authenticity_score'] = cos_sim_matrix_true.mean().item()
        else:
            diagnostics['pseudo_anomaly_authenticity_score'] = float('nan')
    else:
        diagnostics['pseudo_anomaly_difficulty_coeff'] = float('nan')
        diagnostics['pseudo_anomaly_authenticity_score'] = float('nan')
    
    timing['compute_metrics'] = time.time() - t0
    timing['total'] = time.time() - total_start
    
    # Save timing info to diagnostics
    diagnostics['timing'] = timing
    
    return diagnostics


def _format_value(value, precision=4, threshold=0.001):
    import math
    if math.isnan(value) or math.isinf(value):
        return f"{value}"
    
    abs_val = abs(value)
    if abs_val == 0:
        return f"{value:.{precision}f}"
    elif abs_val < threshold:
        return f"{value:.2e}"
    else:
        return f"{value:.{precision}f}"


def _format_weight(weight):
    import math
    if math.isnan(weight) or math.isinf(weight):
        return f"{weight}"
    
    abs_val = abs(weight)
    if abs_val == 0:
        return "0"
    elif abs_val < 0.01:
        return f"{weight:.1e}"
    elif abs_val < 1:
        return f"{weight:.2f}"
    else:
        return f"{weight:.1f}"


def print_diagnostics(diagnostics, epoch, current_lr=None, losses=None, dynamic_weights=None, ortho_loss_weight=None):
    d = diagnostics
    
    # Print timing info first
    if 'timing' in d:
        t = d['timing']
        print(f"[DiagTime@E{epoch}] Total={t.get('total', 0):.3f}s | "
              f"GetBatch={t.get('get_batch', 0):.3f}s, "
              f"Model1={t.get('first_model_forward', 0):.3f}s, "
              f"Model2={t.get('second_model_forward', 0):.3f}s, "
              f"Prepare={t.get('prepare_embeddings', 0):.3f}s, "
              f"Metrics={t.get('compute_metrics', 0):.3f}s")
    
    if current_lr is not None and losses is not None and dynamic_weights is not None:
        w_bce = dynamic_weights.get('bce_loss_weight', 1.0)
        w_rec = dynamic_weights.get('rec_loss_weight', 1.0)
        w_ring = dynamic_weights.get('ring_loss_weight', 1.0)
        w_ortho = dynamic_weights.get('ortho_loss_weight', 0.1)
        w_uni = dynamic_weights.get('uniformity_loss_weight', 0.1)
        weighted_bce = losses['bce'] * w_bce
        weighted_rec = losses['rec'] * w_rec
        weighted_ring = losses['ring'] * w_ring
        weighted_ortho = losses.get('ortho', 0.0) * w_ortho
        weighted_uni = losses.get('uniformity', 0.0) * w_uni
        print(f"[Diag@E{epoch}] lr={current_lr:.2e} | "
              f"Loss(w): BCE={_format_value(weighted_bce)}({_format_weight(w_bce)}x), "
              f"Rec={_format_value(weighted_rec)}({_format_weight(w_rec)}x), "
              f"Ring={_format_value(weighted_ring)}({_format_weight(w_ring)}x), "
              f"Ortho={_format_value(weighted_ortho)}({_format_weight(w_ortho)}x), "
              f"Uni={_format_value(weighted_uni)}({_format_weight(w_uni)}x)")
    
    print(f"  Logit: norm={d['norm_logits_mean']:.3f}, abnorm={d['abnorm_logits_mean']:.3f}, margin={d['logit_margin']:.3f}, std={d['logit_std']:.3f}")
    
    print(f"  CosSim: avg={d['avg_cos_sim']:.3f}, std={d['cos_sim_std']:.3f}")
    
    print(f"  Sep: center_dist={d['center_dist']:.3f}, norm_intra={d['norm_intra_dist']:.3f}, abnorm_intra={d['abnorm_intra_dist']:.3f}, ratio={d['separation_ratio']:.3f}")
    
    if not np.isnan(d.get('dist_norm_outlier', float('nan'))):
        closer_str = "YES" if d['outlier_closer_to_abnorm'] else "NO"
        print(f"  TriGeo: N→O={d['dist_norm_outlier']:.3f}, N→A={d['dist_norm_abnorm']:.3f}, O→A={d['dist_outlier_abnorm']:.3f} | "
              f"angle={d['angle_degrees']:.1f}°, cos_sim={d['cos_sim_directions']:.3f} | outlier→abnorm closer: {closer_str}")
    
    if not np.isnan(d.get('pseudo_anomaly_difficulty_coeff', float('nan'))):
        print(f"  PseudoAnomaly: 伪异常质量评估:")
        print(f"    伪异常难度系数: {d['pseudo_anomaly_difficulty_coeff']:.4f}")
        if not np.isnan(d.get('pseudo_anomaly_authenticity_score', float('nan'))):
            print(f"    伪异常真实性分数: {d['pseudo_anomaly_authenticity_score']:.4f}")
    
    # 打印注意力权重矩阵
    # if 'prompt_attn_weights' in d and d['prompt_attn_weights'] is not None:
        # print_attention_weights(d['prompt_attn_weights'], epoch)


def print_attention_weights(attn_weights, epoch):
    """
    打印注意力权重矩阵，形状为 [batch_size, num_prompts, num_hops]
    """
    print(f"\n=== 注意力权重矩阵 (Epoch {epoch}) ===")
    
    # 将 tensor 移动到 CPU
    attn_weights_cpu = attn_weights.cpu().numpy()
    
    batch_size, num_prompts, num_hops = attn_weights_cpu.shape
    
    print(f"形状: batch={batch_size}, num_prompts={num_prompts}, num_hops={num_hops}")
    print(f"\n前3个样本的注意力权重矩阵:")
    
    # 只打印前几个样本以避免输出过多
    num_samples_to_print = min(3, batch_size)
    
    for sample_idx in range(num_samples_to_print):
        print(f"\n--- 样本 {sample_idx} ---")
        print(f"{'Prompt':>8}", end="")
        for hop_idx in range(num_hops):
            print(f"  Hop{hop_idx:>3}", end="")
        print()
        
        for prompt_idx in range(num_prompts):
            print(f"P{prompt_idx:>5}:", end="")
            for hop_idx in range(num_hops):
                weight = attn_weights_cpu[sample_idx, prompt_idx, hop_idx]
                # 格式化输出
                if abs(weight) < 0.001:
                    print(f" {weight:+.1e}", end="")
                else:
                    print(f" {weight:+.4f}", end="")
            print()
    
    # 打印统计信息
    print(f"\n--- 统计信息 ---")
    print(f"平均注意力权重: {np.mean(attn_weights_cpu):.6f}")
    print(f"注意力权重标准差: {np.std(attn_weights_cpu):.6f}")
    print(f"最大注意力权重: {np.max(attn_weights_cpu):.6f}")
    print(f"最小注意力权重: {np.min(attn_weights_cpu):.6f}")
