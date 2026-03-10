"""
Training Diagnostics Module - Monitor key metrics during model training
"""

import torch
import numpy as np


def compute_diagnostics(model, data_loader, ano_label, idx_test, device, args, normal_for_train_idx=None):
    """
    Compute diagnostic metrics during evaluation
    
    Args:
        model: The model being trained
        data_loader: Test data loader
        ano_label: Anomaly labels array
        idx_test: Test set indices
        device: Computing device
        args: Training arguments
        normal_for_train_idx: Indices of normal nodes for generating pseudo-anomalies
        
    Returns:
        dict: Dictionary containing diagnostic metrics
    """
    model.eval()
    
    all_batched_logits = []
    all_batched_embs = []
    all_outlier_logits = []
    all_outlier_embs = []
    all_reconstruction_errors = []  # Store reconstruction error vectors
    
    with torch.no_grad():
        for item in data_loader:
            concated_input_features = item[0].to(device)
            labels = item[1].to(device)
            
            # Get model outputs without pseudo-anomalies (for test set evaluation)
            emb, emb_combine, logits, _, _, _, _, _, rec_error = model(
                concated_input_features, None, None, None, False, args
            )
            
            all_batched_logits.append(logits.squeeze(0))
            all_batched_embs.append(emb.squeeze(0))
            
            # Store reconstruction error vectors if available
            if rec_error is not None:
                all_reconstruction_errors.append(rec_error.cpu())
            
            # Get pseudo-anomaly logits if normal_for_train_idx is provided
            if normal_for_train_idx is not None:
                # Create a local index for pseudo-anomaly generation
                # We use the first few nodes in the batch as normal samples to generate pseudo-anomalies
                batch_size = concated_input_features.size(1)
                # IMPORTANT: Need enough nodes so that int(num_pseudo * sample_rate) > 0
                # sample_rate is typically 0.15, so we need at least 7 nodes (7 * 0.15 = 1.05)
                num_pseudo = min(1000, max(100, int(10 / args.sample_rate)))  # Ensure enough for sample_rate
                num_pseudo = min(num_pseudo, batch_size)
                
                if num_pseudo > 0:
                    local_normal_idx = torch.arange(num_pseudo, device=device)
                    
                    # Get model outputs with pseudo-anomaly generation
                    emb_pseudo, emb_combine_pseudo, logits_pseudo, outlier_emb, _, _, _, _, _ = model(
                        concated_input_features, None, None, local_normal_idx, True, args
                    )
                    
                    # Extract outlier logits and embeddings
                    # outlier_emb is generated from the model, use it directly
                    if outlier_emb is not None and outlier_emb.size(0) > 0:
                        # logits_pseudo has shape [1, num_pseudo + len(local_normal_idx), 1]
                        # The last num_outliers logits correspond to pseudo-anomalies
                        num_outliers = outlier_emb.size(0)
                        pseudo_logits = logits_pseudo.squeeze(0)[-num_outliers:].squeeze(-1)
                        all_outlier_logits.append(pseudo_logits.cpu())
                        all_outlier_embs.append(outlier_emb.cpu())
    
    # Concatenate all batched results
    concatenated_logits = torch.cat(all_batched_logits, dim=0)
    concatenated_embs = torch.cat(all_batched_embs, dim=0)
    
    # Concatenate reconstruction errors if available
    if len(all_reconstruction_errors) > 0:
        concatenated_rec_errors = torch.cat(all_reconstruction_errors, dim=0)
    else:
        concatenated_rec_errors = None
    
    # Concatenate outlier results if available
    if len(all_outlier_logits) > 0:
        outlier_logits = torch.cat(all_outlier_logits, dim=0)
        outlier_embs = torch.cat(all_outlier_embs, dim=0)
    else:
        outlier_logits = None
        outlier_embs = None
    
    # ==========================================
    # Diagnostic Probes
    # ==========================================
    
    # Split test set into "true normal" vs "true abnormal"
    test_labels = ano_label[idx_test]
    norm_mask = (test_labels == 0)
    abnorm_mask = (test_labels == 1)
    
    logits_tensor = concatenated_logits.cpu().squeeze()
    embs_tensor = concatenated_embs.cpu()
    
    # Extract corresponding Logits and Embeddings
    norm_logits = logits_tensor[norm_mask]
    abnorm_logits = logits_tensor[abnorm_mask]
    norm_embs = embs_tensor[norm_mask]
    abnorm_embs = embs_tensor[abnorm_mask]
    
    diagnostics = {}
    
    # ==========================================
    # Probe 1: Logit Analysis (including pseudo-anomaly)
    # ==========================================
    if len(abnorm_logits) > 0 and len(norm_logits) > 0:
        logit_margin = (abnorm_logits.mean() - norm_logits.mean()).item()
        logit_std = logits_tensor.std().item()
        norm_logits_mean = norm_logits.mean().item()
        abnorm_logits_mean = abnorm_logits.mean().item()
    else:
        logit_margin = float('nan')
        logit_std = float('nan')
        norm_logits_mean = float('nan')
        abnorm_logits_mean = float('nan')
    
    diagnostics['logit_margin'] = logit_margin
    diagnostics['logit_std'] = logit_std
    diagnostics['norm_logits_mean'] = norm_logits_mean
    diagnostics['abnorm_logits_mean'] = abnorm_logits_mean
    
    # Add pseudo-anomaly logit statistics
    if outlier_logits is not None and len(outlier_logits) > 0:
        diagnostics['outlier_logits_mean'] = outlier_logits.mean().item()
        diagnostics['outlier_logits_std'] = outlier_logits.std().item()
        diagnostics['outlier_logits_max'] = outlier_logits.max().item()
        diagnostics['outlier_logits_min'] = outlier_logits.min().item()
        diagnostics['outlier_embs'] = outlier_embs
    else:
        diagnostics['outlier_logits_mean'] = float('nan')
        diagnostics['outlier_logits_std'] = float('nan')
        diagnostics['outlier_logits_max'] = float('nan')
        diagnostics['outlier_logits_min'] = float('nan')
        diagnostics['outlier_embs'] = None
    
    # ==========================================
    # Probe 2: Embedding Collapse (Cosine Similarity)
    # ==========================================
    if norm_embs.size(0) > 1:
        num_samples = min(norm_embs.size(0), 1000)
        sampled_norm_embs = torch.nn.functional.normalize(norm_embs[:num_samples], p=2, dim=1)
        cos_sim_matrix = torch.mm(sampled_norm_embs, sampled_norm_embs.t())
        mask = torch.eye(num_samples, dtype=torch.bool).flatten()
        avg_cos_sim = cos_sim_matrix.flatten()[~mask].mean().item()
        cos_sim_std = cos_sim_matrix.flatten()[~mask].std().item()
    else:
        avg_cos_sim = float('nan')
        cos_sim_std = float('nan')
    
    diagnostics['avg_cos_sim'] = avg_cos_sim
    diagnostics['cos_sim_std'] = cos_sim_std
    
    # ==========================================
    # Probe 3: Euclidean Separation
    # ==========================================
    if norm_embs.size(0) > 0 and abnorm_embs.size(0) > 0:
        norm_center = norm_embs.mean(dim=0)
        abnorm_center = abnorm_embs.mean(dim=0)
        center_dist = torch.norm(norm_center - abnorm_center, p=2).item()
        
        norm_intra_dist = torch.norm(norm_embs - norm_center, p=2, dim=1).mean().item()
        if abnorm_embs.size(0) > 1:
            abnorm_intra_dist = torch.norm(abnorm_embs - abnorm_center, p=2, dim=1).mean().item()
        else:
            abnorm_intra_dist = 0.0
        
        separation_ratio = center_dist / (norm_intra_dist + 1e-8)
    else:
        center_dist = float('nan')
        norm_intra_dist = float('nan')
        abnorm_intra_dist = float('nan')
        separation_ratio = float('nan')
    
    diagnostics['center_dist'] = center_dist
    diagnostics['norm_intra_dist'] = norm_intra_dist
    diagnostics['abnorm_intra_dist'] = abnorm_intra_dist
    diagnostics['separation_ratio'] = separation_ratio
    
    # ==========================================
    # Probe 4: Triangular Geometry (Normal, Pseudo-Anomaly, True Anomaly)
    # ==========================================
    if outlier_embs is not None and outlier_embs.size(0) > 0 and norm_embs.size(0) > 0 and abnorm_embs.size(0) > 0:
        # Compute centers for each class
        norm_center = norm_embs.mean(dim=0)
        abnorm_center = abnorm_embs.mean(dim=0)
        outlier_center = outlier_embs.mean(dim=0)
        
        # Distance between centers
        dist_norm_outlier = torch.norm(norm_center - outlier_center, p=2).item()
        dist_norm_abnorm = torch.norm(norm_center - abnorm_center, p=2).item()
        dist_outlier_abnorm = torch.norm(outlier_center - abnorm_center, p=2).item()
        
        # Intra-class distances
        outlier_intra_dist = torch.norm(outlier_embs - outlier_center, p=2, dim=1).mean().item()
        
        # Cosine similarity between center directions
        # Direction from normal to outlier
        dir_norm_to_outlier = torch.nn.functional.normalize(outlier_center - norm_center, p=2, dim=0)
        # Direction from normal to true anomaly
        dir_norm_to_abnorm = torch.nn.functional.normalize(abnorm_center - norm_center, p=2, dim=0)
        # Cosine similarity between these two directions
        cos_sim_directions = torch.dot(dir_norm_to_outlier, dir_norm_to_abnorm).item()
        
        # Angle between the two directions (in degrees)
        angle_degrees = np.degrees(np.arccos(np.clip(cos_sim_directions, -1.0, 1.0)))
        
        # Separation ratios
        # How well does pseudo-anomaly separate from normal relative to true anomaly?
        outlier_separation_ratio = dist_norm_outlier / (norm_intra_dist + 1e-8)
        
        # Is pseudo-anomaly closer to true anomaly than to normal?
        outlier_closer_to_abnorm = dist_outlier_abnorm < dist_norm_outlier
        
        diagnostics['dist_norm_outlier'] = dist_norm_outlier
        diagnostics['dist_norm_abnorm'] = dist_norm_abnorm
        diagnostics['dist_outlier_abnorm'] = dist_outlier_abnorm
        diagnostics['outlier_intra_dist'] = outlier_intra_dist
        diagnostics['cos_sim_directions'] = cos_sim_directions
        diagnostics['angle_degrees'] = angle_degrees
        diagnostics['outlier_separation_ratio'] = outlier_separation_ratio
        diagnostics['outlier_closer_to_abnorm'] = outlier_closer_to_abnorm
    else:
        diagnostics['dist_norm_outlier'] = float('nan')
        diagnostics['dist_norm_abnorm'] = float('nan')
        diagnostics['dist_outlier_abnorm'] = float('nan')
        diagnostics['outlier_intra_dist'] = float('nan')
        diagnostics['cos_sim_directions'] = float('nan')
        diagnostics['angle_degrees'] = float('nan')
        diagnostics['outlier_separation_ratio'] = float('nan')
        diagnostics['outlier_closer_to_abnorm'] = False
    
    # ==========================================
    # Probe 5: Reconstruction Error Vector Magnitude Distribution
    # ==========================================
    if concatenated_rec_errors is not None:
        # 计算所有节点的重构误差向量模长 ||R_i||
        rec_error_magnitudes = torch.norm(concatenated_rec_errors, p=2, dim=1)
        
        # 分别计算正常点和异常点的模长
        norm_rec_magnitudes = rec_error_magnitudes[norm_mask]
        abnorm_rec_magnitudes = rec_error_magnitudes[abnorm_mask]
        
        # 正常点模长统计
        if len(norm_rec_magnitudes) > 0:
            diagnostics['norm_rec_mag_mean'] = norm_rec_magnitudes.mean().item()
            diagnostics['norm_rec_mag_std'] = norm_rec_magnitudes.std().item()
            diagnostics['norm_rec_mag_min'] = norm_rec_magnitudes.min().item()
            diagnostics['norm_rec_mag_max'] = norm_rec_magnitudes.max().item()
            diagnostics['norm_rec_mag_median'] = norm_rec_magnitudes.median().item()
            # 计算分位数
            diagnostics['norm_rec_mag_q25'] = torch.quantile(norm_rec_magnitudes, 0.25).item()
            diagnostics['norm_rec_mag_q75'] = torch.quantile(norm_rec_magnitudes, 0.75).item()
        else:
            diagnostics['norm_rec_mag_mean'] = float('nan')
            diagnostics['norm_rec_mag_std'] = float('nan')
            diagnostics['norm_rec_mag_min'] = float('nan')
            diagnostics['norm_rec_mag_max'] = float('nan')
            diagnostics['norm_rec_mag_median'] = float('nan')
            diagnostics['norm_rec_mag_q25'] = float('nan')
            diagnostics['norm_rec_mag_q75'] = float('nan')
        
        # 异常点模长统计
        if len(abnorm_rec_magnitudes) > 0:
            diagnostics['abnorm_rec_mag_mean'] = abnorm_rec_magnitudes.mean().item()
            diagnostics['abnorm_rec_mag_std'] = abnorm_rec_magnitudes.std().item()
            diagnostics['abnorm_rec_mag_min'] = abnorm_rec_magnitudes.min().item()
            diagnostics['abnorm_rec_mag_max'] = abnorm_rec_magnitudes.max().item()
            diagnostics['abnorm_rec_mag_median'] = abnorm_rec_magnitudes.median().item()
            # 计算分位数
            diagnostics['abnorm_rec_mag_q25'] = torch.quantile(abnorm_rec_magnitudes, 0.25).item()
            diagnostics['abnorm_rec_mag_q75'] = torch.quantile(abnorm_rec_magnitudes, 0.75).item()
        else:
            diagnostics['abnorm_rec_mag_mean'] = float('nan')
            diagnostics['abnorm_rec_mag_std'] = float('nan')
            diagnostics['abnorm_rec_mag_min'] = float('nan')
            diagnostics['abnorm_rec_mag_max'] = float('nan')
            diagnostics['abnorm_rec_mag_median'] = float('nan')
            diagnostics['abnorm_rec_mag_q25'] = float('nan')
            diagnostics['abnorm_rec_mag_q75'] = float('nan')
        
        # 计算模长差异的统计显著性
        if len(norm_rec_magnitudes) > 0 and len(abnorm_rec_magnitudes) > 0:
            # 模长差异（异常点模长 - 正常点模长）
            mag_diff = diagnostics['abnorm_rec_mag_mean'] - diagnostics['norm_rec_mag_mean']
            diagnostics['rec_mag_diff'] = mag_diff
            # 模长比率（异常点模长 / 正常点模长）
            diagnostics['rec_mag_ratio'] = diagnostics['abnorm_rec_mag_mean'] / (diagnostics['norm_rec_mag_mean'] + 1e-8)
        else:
            diagnostics['rec_mag_diff'] = float('nan')
            diagnostics['rec_mag_ratio'] = float('nan')
        
        # 存储原始模长数据（用于可视化）
        diagnostics['norm_rec_magnitudes'] = norm_rec_magnitudes.numpy()
        diagnostics['abnorm_rec_magnitudes'] = abnorm_rec_magnitudes.numpy()
    else:
        diagnostics['norm_rec_mag_mean'] = float('nan')
        diagnostics['norm_rec_mag_std'] = float('nan')
        diagnostics['norm_rec_mag_min'] = float('nan')
        diagnostics['norm_rec_mag_max'] = float('nan')
        diagnostics['norm_rec_mag_median'] = float('nan')
        diagnostics['norm_rec_mag_q25'] = float('nan')
        diagnostics['norm_rec_mag_q75'] = float('nan')
        diagnostics['abnorm_rec_mag_mean'] = float('nan')
        diagnostics['abnorm_rec_mag_std'] = float('nan')
        diagnostics['abnorm_rec_mag_min'] = float('nan')
        diagnostics['abnorm_rec_mag_max'] = float('nan')
        diagnostics['abnorm_rec_mag_median'] = float('nan')
        diagnostics['abnorm_rec_mag_q25'] = float('nan')
        diagnostics['abnorm_rec_mag_q75'] = float('nan')
        diagnostics['rec_mag_diff'] = float('nan')
        diagnostics['rec_mag_ratio'] = float('nan')
        diagnostics['norm_rec_magnitudes'] = None
        diagnostics['abnorm_rec_magnitudes'] = None
    
    # ==========================================
    # Sample Statistics
    # ==========================================
    diagnostics['num_normal'] = norm_embs.size(0) if norm_embs.size(0) > 0 else 0
    diagnostics['num_abnormal'] = abnorm_embs.size(0) if abnorm_embs.size(0) > 0 else 0
    diagnostics['num_outlier'] = outlier_embs.size(0) if outlier_embs is not None else 0
    
    return diagnostics


def print_diagnostics(diagnostics, epoch, current_lr=None, losses=None, dynamic_weights=None, ortho_loss_weight=None):
    """
    Print diagnostic info in compact format
    
    Args:
        diagnostics: Dictionary of diagnostic metrics
        epoch: Current epoch
        current_lr: Current learning rate
        losses: Dict with 'bce', 'rec', 'ring', 'ortho' raw losses
        dynamic_weights: Dict with loss weights (including 'ortho_loss_weight')
        ortho_loss_weight: (deprecated) Weight for orthogonal loss, now read from dynamic_weights
    """
    d = diagnostics
    
    # Line 1: Training status (lr, losses)
    if current_lr is not None and losses is not None and dynamic_weights is not None:
        w_bce = dynamic_weights.get('bce_loss_weight', 1.0)
        w_rec = dynamic_weights.get('rec_loss_weight', 1.0)
        w_ring = dynamic_weights.get('ring_loss_weight', 1.0)
        w_ortho = dynamic_weights.get('ortho_loss_weight', 0.1)
        weighted_bce = losses['bce'] * w_bce
        weighted_rec = losses['rec'] * w_rec
        weighted_ring = losses['ring'] * w_ring
        weighted_ortho = losses.get('ortho', 0.0) * w_ortho
        print(f"[Diag@E{epoch}] lr={current_lr:.2e} | "
              f"Loss(w): BCE={weighted_bce:.4f}({w_bce:.1f}x), Rec={weighted_rec:.4f}({w_rec:.1f}x), Ring={weighted_ring:.4f}({w_ring:.1f}x), Ortho={weighted_ortho:.4f}({w_ortho:.1f}x)")
    
    # Line 2: Logit analysis (including pseudo-anomaly/outlier)
    outlier_logit_str = ""
    if not np.isnan(d.get('outlier_logits_mean', float('nan'))):
        outlier_logit_str = f", outlier={d['outlier_logits_mean']:.3f}(±{d['outlier_logits_std']:.3f})"
    print(f"  Logit: norm={d['norm_logits_mean']:.3f}, abnorm={d['abnorm_logits_mean']:.3f}{outlier_logit_str}, margin={d['logit_margin']:.3f}, std={d['logit_std']:.3f}")
    
    # Line 3: CosSim analysis
    print(f"  CosSim: avg={d['avg_cos_sim']:.3f}, std={d['cos_sim_std']:.3f}")
    
    # Line 4: Separation metrics (normal vs abnormal)
    print(f"  Sep: center_dist={d['center_dist']:.3f}, norm_intra={d['norm_intra_dist']:.3f}, abnorm_intra={d['abnorm_intra_dist']:.3f}, ratio={d['separation_ratio']:.3f}")
    
    # Line 5: Triangular geometry (normal, pseudo-anomaly, true anomaly)
    if not np.isnan(d.get('dist_norm_outlier', float('nan'))):
        closer_str = "YES" if d['outlier_closer_to_abnorm'] else "NO"
        print(f"  TriGeo: N→O={d['dist_norm_outlier']:.3f}, N→A={d['dist_norm_abnorm']:.3f}, O→A={d['dist_outlier_abnorm']:.3f} | "
              f"angle={d['angle_degrees']:.1f}°, cos_sim={d['cos_sim_directions']:.3f} | outlier→abnorm closer: {closer_str}")
    
    # Line 6: Reconstruction Error Vector Magnitude Distribution
    if not np.isnan(d.get('norm_rec_mag_mean', float('nan'))):
        print(f"  RecMag: ||R_i|| 分布:")
        print(f"    正常点: mean={d['norm_rec_mag_mean']:.4f}(±{d['norm_rec_mag_std']:.4f}), "
              f"median={d['norm_rec_mag_median']:.4f}, [{d['norm_rec_mag_q25']:.4f}, {d['norm_rec_mag_q75']:.4f}], "
              f"min={d['norm_rec_mag_min']:.4f}, max={d['norm_rec_mag_max']:.4f}")
        if not np.isnan(d.get('abnorm_rec_mag_mean', float('nan'))):
            print(f"    异常点: mean={d['abnorm_rec_mag_mean']:.4f}(±{d['abnorm_rec_mag_std']:.4f}), "
                  f"median={d['abnorm_rec_mag_median']:.4f}, [{d['abnorm_rec_mag_q25']:.4f}, {d['abnorm_rec_mag_q75']:.4f}], "
                  f"min={d['abnorm_rec_mag_min']:.4f}, max={d['abnorm_rec_mag_max']:.4f}")
            print(f"    差异: diff={d['rec_mag_diff']:.4f}, ratio={d['rec_mag_ratio']:.4f}x")
