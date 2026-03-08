"""
Training Diagnostics Module - Monitor key metrics during model training
"""

import torch
import numpy as np


def compute_diagnostics(model, data_loader, ano_label, idx_test, device, args):
    """
    Compute diagnostic metrics during evaluation
    
    Args:
        model: The model being trained
        data_loader: Test data loader
        ano_label: Anomaly labels array
        idx_test: Test set indices
        device: Computing device
        args: Training arguments
        
    Returns:
        dict: Dictionary containing diagnostic metrics
    """
    model.eval()
    
    all_batched_logits = []
    all_batched_embs = []
    
    with torch.no_grad():
        for item in data_loader:
            concated_input_features = item[0].to(device)
            labels = item[1].to(device)
            
            # Get model outputs
            emb, emb_combine, logits, _, _, _, _, _ = model(
                concated_input_features, None, None, None, False, args
            )
            
            all_batched_logits.append(logits.squeeze(0))
            all_batched_embs.append(emb.squeeze(0))
    
    # Concatenate all batched results
    concatenated_logits = torch.cat(all_batched_logits, dim=0)
    concatenated_embs = torch.cat(all_batched_embs, dim=0)
    
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
    # Probe 1: Logit Analysis
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
    # Sample Statistics
    # ==========================================
    diagnostics['num_normal'] = norm_embs.size(0) if norm_embs.size(0) > 0 else 0
    diagnostics['num_abnormal'] = abnorm_embs.size(0) if abnorm_embs.size(0) > 0 else 0
    
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
    
    # Line 2: Logit & CosSim analysis
    print(f"  Logit: norm={d['norm_logits_mean']:.3f}, abnorm={d['abnorm_logits_mean']:.3f}, margin={d['logit_margin']:.3f}, std={d['logit_std']:.3f} | "
          f"CosSim: avg={d['avg_cos_sim']:.3f}, std={d['cos_sim_std']:.3f}")
    
    # Line 3: Separation metrics
    print(f"  Sep: center_dist={d['center_dist']:.3f}, norm_intra={d['norm_intra_dist']:.3f}, abnorm_intra={d['abnorm_intra_dist']:.3f}, ratio={d['separation_ratio']:.3f}")
