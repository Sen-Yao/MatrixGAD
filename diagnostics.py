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
    
    # First pass: collect embeddings from the single batch
    t0 = time.time()
    with torch.no_grad():
        emb, _, _, _, _, _, _, _, _, _, _, _ = model(
            concated_input_features, None, None, None, False, args
        )
        concatenated_embs = emb.squeeze(0)
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
    
    # Get test set embeddings (on GPU) - note: we might not have test nodes in our single batch
    t0 = time.time()
    
    idx_test_dev = torch.tensor(idx_test, device=device)
    test_mask = torch.isin(concatenated_global_indices, idx_test_dev) if concatenated_global_indices is not None else torch.ones(len(concatenated_embs), dtype=torch.bool, device=device)
    test_embs = concatenated_embs[test_mask]
    
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
            # Filter test_embs to only include valid test nodes
            test_embs = test_embs[torch.tensor(valid_mask, device=device)]
        else:
            test_labels = np.array([])
            test_embs = torch.tensor([], device=device)
    else:
        test_labels = np.array([])
        test_embs = torch.tensor([], device=device)
    
    norm_mask_test = (test_labels == 0) if len(test_labels) > 0 else np.array([])
    abnorm_mask_test = (test_labels == 1) if len(test_labels) > 0 else np.array([])
    
    # Move masks back to GPU for embedding indexing
    norm_embs = test_embs[torch.tensor(norm_mask_test, dtype=torch.bool, device=device)] if len(norm_mask_test) > 0 else torch.tensor([], device=device)
    abnorm_embs = test_embs[torch.tensor(abnorm_mask_test, dtype=torch.bool, device=device)] if len(abnorm_mask_test) > 0 else torch.tensor([], device=device)
    
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
    
    # Logit metrics (set to nan for simplicity)
    diagnostics['logit_margin'] = float('nan')
    diagnostics['logit_std'] = float('nan')
    diagnostics['norm_logits_mean'] = float('nan')
    diagnostics['abnorm_logits_mean'] = float('nan')
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
