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
    # 基础指标
    auc: float
    ap: float
    
    # Logit分析指标
    logit_margin: float  # 异常与正常节点的logit均值差
    logit_std: float     # logit的标准差
    
    # Embedding分析指标
    emb_cos_sim: float      # 正常节点embedding的平均余弦相似度（检测塌缩）
    emb_center_dist: float   # 正常与异常节点中心的欧氏距离
    
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
        labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算Logit分析指标
        
        Args:
            logits: 模型输出的logits张量 [N]
            labels: 真实标签数组 [N], 0表示正常, 1表示异常
            
        Returns:
            logit_margin: 异常与正常节点的logit均值差
            logit_std: logit的标准差
        """
        # 确保logits在CPU上
        logits_cpu = logits.cpu().squeeze()
        
        # 划分正常和异常节点
        norm_mask = (labels == 0)
        abnorm_mask = (labels == 1)
        
        norm_logits = logits_cpu[norm_mask]
        abnorm_logits = logits_cpu[abnorm_mask]
        
        # 计算指标
        logit_margin = (abnorm_logits.mean() - norm_logits.mean()).item()
        logit_std = logits_cpu.std().item()
        
        return logit_margin, logit_std
    
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
    
    def compute_all_metrics(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: np.ndarray,
        test_indices: np.ndarray,
        bce_loss: Optional[float] = None,
        uniformity_loss: Optional[float] = None,
        rec_loss: Optional[float] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> DiagnosticMetrics:
        """
        计算所有诊断指标
        
        Args:
            logits: 模型输出的logits张量 [N]
            embeddings: 节点embedding张量 [N, D]
            labels: 完整的标签数组（用于获取测试集标签）
            test_indices: 测试集索引
            eval_bce_loss: 评估阶段的BCE loss
            eval_uniformity_loss: 评估阶段的uniformity loss
            eval_rec_loss: 评估阶段的重建loss
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
        
        # 计算Logit分析指标
        logit_margin, logit_std = self.compute_logits_analysis(logits, test_labels)
        
        # 计算Embedding分析指标
        emb_cos_sim, emb_center_dist = self.compute_embedding_analysis(embeddings, test_labels)
        
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
            emb_cos_sim=emb_cos_sim,
            emb_center_dist=emb_center_dist,
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
    print(f"  Emb_Cos_Sim: {metrics.emb_cos_sim:.4f} | Emb_Center_Dist: {metrics.emb_center_dist:.4f}")
    
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
