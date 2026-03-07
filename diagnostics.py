"""
诊断指标计算模块

该模块封装了模型训练过程中的诊断指标计算逻辑，包括：
- Logit分析（预测置信度与翻转诊断）
- Embedding分析（表征塌缩监控、流形分离度）
- 诊断结果的记录和输出
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List


@dataclass
class DiagnosticMetrics:
    """诊断指标数据类"""
    # 基础指标（无默认值，必须放在前面）
    auc: float
    ap: float
    
    # Logit分析指标（无默认值）
    logit_margin: float  # 异常与正常节点的logit均值差
    logit_std: float     # logit的标准差
    
    # Embedding分析指标（无默认值）
    emb_cos_sim: float      # 正常节点embedding的平均余弦相似度（检测塌缩）
    emb_center_dist: float   # 正常与异常节点中心的欧氏距离
    
    # 分类器平均Logit值（有默认值，放在后面）
    logit_N_real: Optional[float] = None    # 真正常节点的平均logit
    logit_A_pseudo: Optional[float] = None   # 伪异常节点的平均logit
    logit_A_real: Optional[float] = None     # 真异常节点的平均logit
    
    # 特征中心距离指标（有默认值）
    dist_Nreal_Areal: Optional[float] = None    # 真正常与真异常的特征中心距离
    dist_Nreal_Apseudo: Optional[float] = None  # 真正常与伪异常的特征中心距离
    dist_Areal_Apseudo: Optional[float] = None  # 真异常与伪异常的特征中心距离
    
    # Loss指标（可选）
    weighted_bce_loss: Optional[float] = None
    weighted_uniformity_loss: Optional[float] = None
    weighted_rec_loss: Optional[float] = None


class DiagnosticCalculator:
    """诊断指标计算器"""
    
    def __init__(self, device: torch.device):
        """
        Args:
            device: 计算设备
        """
        self.device = device
    
    def compute_logits_analysis(
        self, 
        logits: torch.Tensor, 
        labels: np.ndarray,
        outlier_logits: Optional[torch.Tensor] = None
    ) -> Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
        """
        计算Logit分析指标
        
        Args:
            logits: 模型输出的logits张量 [N]（测试集节点的logits）
            labels: 真实标签数组 [N], 0表示正常, 1表示异常
            outlier_logits: 伪异常节点的logits张量 [M]（可选）
            
        Returns:
            logit_margin: 异常与正常节点的logit均值差
            logit_std: logit的标准差
            logit_N_real: 真正常节点的平均logit
            logit_A_pseudo: 伪异常节点的平均logit
            logit_A_real: 真异常节点的平均logit
        """
        # 确保logits在CPU上
        logits_cpu = logits.cpu().squeeze()
        
        # 划分正常和异常节点
        norm_mask = (labels == 0)
        abnorm_mask = (labels == 1)
        
        norm_logits = logits_cpu[norm_mask]  # N_real 的 logits
        abnorm_logits = logits_cpu[abnorm_mask]  # A_real 的 logits
        
        # 计算指标
        logit_margin = (abnorm_logits.mean() - norm_logits.mean()).item()
        logit_std = logits_cpu.std().item()
        
        # 计算三类节点的平均 Logit 值
        logit_N_real = norm_logits.mean().item() if len(norm_logits) > 0 else None
        logit_A_real = abnorm_logits.mean().item() if len(abnorm_logits) > 0 else None
        
        # 计算伪异常节点的平均 Logit 值
        logit_A_pseudo = None
        if outlier_logits is not None and outlier_logits.numel() > 0:
            outlier_logits_cpu = outlier_logits.cpu().squeeze()
            logit_A_pseudo = outlier_logits_cpu.mean().item()
        
        return logit_margin, logit_std, logit_N_real, logit_A_pseudo, logit_A_real
    
    def compute_embedding_analysis(
        self, 
        embeddings: torch.Tensor, 
        labels: np.ndarray,
        max_samples: int = 1000
    ) -> Tuple[float, float]:
        """
        计算Embedding分析指标
        
        Args:
            embeddings: 节点embedding张量 [N, D]
            labels: 真实标签数组 [N], 0表示正常, 1表示异常
            max_samples: 计算余弦相似度时的最大采样数（防止OOM）
            
        Returns:
            emb_cos_sim: 正常节点embedding的平均余弦相似度
            emb_center_dist: 正常与异常节点中心的欧氏距离
        """
        # 确保embeddings在CPU上
        embs_cpu = embeddings.cpu()
        
        # 划分正常和异常节点
        norm_mask = (labels == 0)
        abnorm_mask = (labels == 1)
        
        norm_embs = embs_cpu[norm_mask]
        abnorm_embs = embs_cpu[abnorm_mask]
        
        # 探针2：表征塌缩监控（计算正常节点间的平均余弦相似度）
        num_samples = min(norm_embs.size(0), max_samples)
        sampled_norm_embs = torch.nn.functional.normalize(
            norm_embs[:num_samples], p=2, dim=1
        )
        cos_sim_matrix = torch.mm(sampled_norm_embs, sampled_norm_embs.t())
        # 排除对角线（自身相似度）
        mask = torch.eye(num_samples, dtype=torch.bool).flatten()
        avg_cos_sim = cos_sim_matrix.flatten()[~mask].mean().item()
        
        # 探针3：真实流形分离度（正常与异常中心的欧氏距离）
        norm_center = norm_embs.mean(dim=0)
        abnorm_center = abnorm_embs.mean(dim=0)
        center_dist = torch.norm(norm_center - abnorm_center, p=2).item()
        
        return avg_cos_sim, center_dist
    
    def compute_center_distances(
        self,
        embeddings: torch.Tensor,
        labels: np.ndarray,
        outlier_emb: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        计算特征中心距离指标
        
        Args:
            embeddings: 测试集节点的embedding张量 [N, D]
            labels: 测试集标签数组 [N], 0表示正常, 1表示异常
            outlier_emb: 伪异常节点的embedding张量 [M, D]（可选）
            
        Returns:
            dist_Nreal_Areal: 真正常与真异常的特征中心距离
            dist_Nreal_Apseudo: 真正常与伪异常的特征中心距离
            dist_Areal_Apseudo: 真异常与伪异常的特征中心距离
        """
        # 确保embeddings在CPU上
        embs_cpu = embeddings.cpu()
        
        # 划分真正常和真异常节点
        norm_mask = (labels == 0)
        abnorm_mask = (labels == 1)
        
        norm_embs = embs_cpu[norm_mask]  # 真正常节点
        abnorm_embs = embs_cpu[abnorm_mask]  # 真异常节点
        
        # 计算真正常和真异常的特征中心
        norm_center = norm_embs.mean(dim=0)  # N_real 的中心
        abnorm_center = abnorm_embs.mean(dim=0)  # A_real 的中心
        
        # Dist(N_real, A_real): 真正常与真异常的特征中心距离
        dist_Nreal_Areal = torch.norm(norm_center - abnorm_center, p=2).item()
        
        # 如果没有提供伪异常embedding，则只返回第一个距离
        if outlier_emb is None or outlier_emb.size(0) == 0:
            return dist_Nreal_Areal, None, None
        
        # 计算伪异常的特征中心
        pseudo_center = outlier_emb.cpu().mean(dim=0)  # A_pseudo 的中心
        
        # Dist(N_real, A_pseudo): 真正常与伪异常的特征中心距离
        dist_Nreal_Apseudo = torch.norm(norm_center - pseudo_center, p=2).item()
        
        # Dist(A_real, A_pseudo): 真异常与伪异常的特征中心距离
        dist_Areal_Apseudo = torch.norm(abnorm_center - pseudo_center, p=2).item()
        
        return dist_Nreal_Areal, dist_Nreal_Apseudo, dist_Areal_Apseudo
    
    def compute_all_metrics(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: np.ndarray,
        test_indices: np.ndarray,
        outlier_emb: Optional[torch.Tensor] = None,
        outlier_logits: Optional[torch.Tensor] = None,
        bce_loss: Optional[float] = None,
        uniformity_loss: Optional[float] = None,
        rec_loss: Optional[float] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> DiagnosticMetrics:
        """
        计算所有诊断指标
        
        Args:
            logits: 模型输出的logits张量 [N]（测试集节点的logits）
            embeddings: 节点embedding张量 [N, D]
            labels: 完整的标签数组（用于获取测试集标签）
            test_indices: 测试集索引
            outlier_emb: 伪异常节点的embedding张量 [M, D]（可选）
            outlier_logits: 伪异常节点的logits张量 [M]（可选）
            bce_loss: 评估阶段的BCE loss
            uniformity_loss: 评估阶段的uniformity loss
            rec_loss: 评估阶段的重建loss
            loss_weights: 损失权重字典，用于计算加权loss
            
        Returns:
            DiagnosticMetrics: 包含所有诊断指标的数据对象
        """
        # 获取测试集标签
        test_labels = labels[test_indices]
        
        # 计算AUC和AP
        logits_np = np.squeeze(logits.cpu().detach().numpy())
        auc = roc_auc_score(test_labels, logits_np)
        ap = average_precision_score(test_labels, logits_np, average='macro', pos_label=1)
        
        # 计算Logit分析指标（包括三类节点的平均Logit值）
        logit_margin, logit_std, logit_N_real, logit_A_pseudo, logit_A_real = self.compute_logits_analysis(
            logits, test_labels, outlier_logits
        )
        
        # 计算Embedding分析指标
        emb_cos_sim, emb_center_dist = self.compute_embedding_analysis(embeddings, test_labels)
        
        # 计算特征中心距离指标
        dist_Nreal_Areal, dist_Nreal_Apseudo, dist_Areal_Apseudo = self.compute_center_distances(
            embeddings, test_labels, outlier_emb
        )
        
        # 计算加权loss（如果提供了loss值和权重）
        weighted_bce_loss = None
        weighted_uniformity_loss = None
        weighted_rec_loss = None
        
        if bce_loss is not None and loss_weights is not None:
            weighted_bce_loss = loss_weights.get('bce_loss_weight', 1.0) * bce_loss
            weighted_uniformity_loss = loss_weights.get('uniformity_loss_weight', 1.0) * uniformity_loss
            weighted_rec_loss = loss_weights.get('rec_loss_weight', 1.0) * rec_loss
        
        return DiagnosticMetrics(
            auc=auc,
            ap=ap,
            logit_margin=logit_margin,
            logit_std=logit_std,
            logit_N_real=logit_N_real,
            logit_A_pseudo=logit_A_pseudo,
            logit_A_real=logit_A_real,
            emb_cos_sim=emb_cos_sim,
            emb_center_dist=emb_center_dist,
            dist_Nreal_Areal=dist_Nreal_Areal,
            dist_Nreal_Apseudo=dist_Nreal_Apseudo,
            dist_Areal_Apseudo=dist_Areal_Apseudo,
            weighted_bce_loss=weighted_bce_loss,
            weighted_uniformity_loss=weighted_uniformity_loss,
            weighted_rec_loss=weighted_rec_loss
        )


def log_diagnostics(
    metrics: DiagnosticMetrics,
    epoch: int,
    current_lr: float,
    use_wandb: bool = True
) -> Dict[str, Any]:
    """
    记录和输出诊断指标
    
    Args:
        metrics: 诊断指标数据对象
        epoch: 当前epoch
        current_lr: 当前学习率
        use_wandb: 是否使用wandb记录
        
    Returns:
        用于wandb.log的字典
    """
    # 构建wandb日志字典
    log_dict = {
        "AUC": metrics.auc,
        "AP": metrics.ap,
        "Diag/Logit_Margin": metrics.logit_margin,
        "Diag/Logit_Std": metrics.logit_std,
        "Diag/Emb_Cos_Sim": metrics.emb_cos_sim,
        "Diag/Emb_Center_Dist": metrics.emb_center_dist
    }
    
    # 添加特征中心距离指标（如果有）
    if metrics.dist_Nreal_Areal is not None:
        log_dict["Diag/Dist_Nreal_Areal"] = metrics.dist_Nreal_Areal
    if metrics.dist_Nreal_Apseudo is not None:
        log_dict["Diag/Dist_Nreal_Apseudo"] = metrics.dist_Nreal_Apseudo
    if metrics.dist_Areal_Apseudo is not None:
        log_dict["Diag/Dist_Areal_Apseudo"] = metrics.dist_Areal_Apseudo
    
    # 添加分类器平均Logit值指标（如果有）
    if metrics.logit_N_real is not None:
        log_dict["Diag/Logit_N_real"] = metrics.logit_N_real
    if metrics.logit_A_pseudo is not None:
        log_dict["Diag/Logit_A_pseudo"] = metrics.logit_A_pseudo
    if metrics.logit_A_real is not None:
        log_dict["Diag/Logit_A_real"] = metrics.logit_A_real
    
    # 添加loss指标（如果有）
    if metrics.weighted_bce_loss is not None:
        log_dict["Diag/Weighted_BCE_Loss"] = metrics.weighted_bce_loss
    if metrics.weighted_uniformity_loss is not None:
        log_dict["Diag/Weighted_Uniformity_Loss"] = metrics.weighted_uniformity_loss
    if metrics.weighted_rec_loss is not None:
        log_dict["Diag/Weighted_Rec_Loss"] = metrics.weighted_rec_loss
    
    # 打印诊断结果
    print(f"\n[Epoch {epoch}] Diagnostic Metrics:")
    print(f"  AUC: {metrics.auc:.4f} | AP: {metrics.ap:.4f} | LR: {current_lr:.6f}")
    print(f"  Logit_Margin: {metrics.logit_margin:.4f} | Logit_Std: {metrics.logit_std:.4f}")
    
    # 打印分类器平均Logit值
    if metrics.logit_N_real is not None or metrics.logit_A_pseudo is not None or metrics.logit_A_real is not None:
        logit_parts = []
        if metrics.logit_N_real is not None:
            logit_parts.append(f"N_real: {metrics.logit_N_real:.4f}")
        if metrics.logit_A_pseudo is not None:
            logit_parts.append(f"A_pseudo: {metrics.logit_A_pseudo:.4f}")
        if metrics.logit_A_real is not None:
            logit_parts.append(f"A_real: {metrics.logit_A_real:.4f}")
        print(f"  Avg Logit | {' | '.join(logit_parts)}")
    
    print(f"  Emb_Cos_Sim: {metrics.emb_cos_sim:.4f} | Emb_Center_Dist: {metrics.emb_center_dist:.4f}")
    
    # 打印特征中心距离指标
    if metrics.dist_Nreal_Areal is not None:
        print(f"  Dist(N_real, A_real): {metrics.dist_Nreal_Areal:.4f}", end="")
        if metrics.dist_Nreal_Apseudo is not None:
            print(f" | Dist(N_real, A_pseudo): {metrics.dist_Nreal_Apseudo:.4f}", end="")
        if metrics.dist_Areal_Apseudo is not None:
            print(f" | Dist(A_real, A_pseudo): {metrics.dist_Areal_Apseudo:.4f}")
        else:
            print()  # 换行
    
    if metrics.weighted_bce_loss is not None:
        print(f"  Weighted_BCE_Loss: {metrics.weighted_bce_loss:.4f} | "
              f"Weighted_Uniformity_Loss: {metrics.weighted_uniformity_loss:.4f}")
        print(f"  Weighted_Rec_Loss: {metrics.weighted_rec_loss:.4f}")
    
    # 记录到wandb
    if use_wandb:
        import wandb
        wandb.log(log_dict, step=epoch)
    
    return log_dict


class TrainTimeDiagnosticsCollector:
    """
    训练时诊断数据收集器
    
    在训练过程中收集诊断所需的数据，避免eval时重复计算
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置收集器"""
        self.all_logits = []
        self.all_embeddings = []
        self.all_labels = []
        self.total_rec_loss = 0.0
        self.total_uniformity_loss = 0.0
        self.total_bce_loss = 0.0
        self.num_batches = 0
    
    def collect_batch(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        rec_loss: float = 0.0,
        uniformity_loss: float = 0.0,
        bce_loss: float = 0.0
    ):
        """
        收集一个batch的诊断数据
        
        Args:
            logits: 当前batch的logits
            embeddings: 当前batch的embeddings
            labels: 当前batch的标签
            rec_loss: 当前batch的重建loss
            uniformity_loss: 当前batch的uniformity loss
            bce_loss: 当前batch的BCE loss
        """
        self.all_logits.append(logits.detach().cpu())
        self.all_embeddings.append(embeddings.detach().cpu())
        self.all_labels.append(labels.detach().cpu())
        self.total_rec_loss += rec_loss
        self.total_uniformity_loss += uniformity_loss
        self.total_bce_loss += bce_loss
        self.num_batches += 1
    
    def get_aggregated_data(self) -> Dict[str, Any]:
        """
        获取聚合后的诊断数据
        
        Returns:
            包含所有聚合数据的字典
        """
        if self.num_batches == 0:
            return None
        
        return {
            'logits': torch.cat(self.all_logits, dim=0),
            'embeddings': torch.cat(self.all_embeddings, dim=0),
            'labels': torch.cat(self.all_labels, dim=0),
            'avg_rec_loss': self.total_rec_loss / self.num_batches,
            'avg_uniformity_loss': self.total_uniformity_loss / self.num_batches,
            'avg_bce_loss': self.total_bce_loss / self.num_batches
        }
    
    def get_avg_losses(self) -> Dict[str, float]:
        """
        获取平均loss值
        
        Returns:
            包含平均loss的字典
        """
        if self.num_batches == 0:
            return {'rec': 0.0, 'uniformity': 0.0, 'bce': 0.0}
        
        return {
            'rec': self.total_rec_loss / self.num_batches,
            'uniformity': self.total_uniformity_loss / self.num_batches,
            'bce': self.total_bce_loss / self.num_batches
        }
