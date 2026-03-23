import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FeedForwardNetwork(nn.Module):
    """前馈神经网络层"""
    
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1

        self.linear_q = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.head_dim, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        batch_size = q.size(0)

        # 线性投影并重塑为多头形式
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.head_dim)

        # 转置以适应注意力计算: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        # 缩放点积注意力
        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias

        attention_weights = torch.softmax(x, dim=3)
        x = self.att_dropout(attention_weights)
        x = x.matmul(v)

        # 重塑输出
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.head_dim)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x, attention_weights


class EncoderLayer(nn.Module):
    """Transformer 编码器层"""
    
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        # 自注意力层
        y = self.self_attention_norm(x)
        y, attention_weights = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        
        # 前馈网络层
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x, attention_weights


class ConsistencyPredictor(nn.Module):
    """
    一致性预测器：用于预测被 mask 的频域 token
    
    输入：可见的频域 token + 正常基线
    输出：对被 mask token 的预测
    """
    
    def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
        super(ConsistencyPredictor, self).__init__()
        
        self.cross_attention = MultiHeadAttention(hidden_dim, dropout, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, visible_tokens, baseline_tokens, visible_mask):
        """
        Args:
            visible_tokens: 可见的频域 token [batch_size, num_prompts, hidden_dim]
            baseline_tokens: 正常基线 token [batch_size, num_prompts, hidden_dim]
            visible_mask: 可见性掩码 [batch_size, num_prompts], True 表示可见
        
        Returns:
            predicted_tokens: 对所有位置的预测（包括可见和 mask） [batch_size, num_prompts, hidden_dim]
        """
        batch_size, num_prompts, hidden_dim = visible_tokens.shape
        
        # 使用交叉注意力：baseline 作为 query，visible tokens 作为 key/value
        # 这确保预测朝着"正常模式"的方向
        y = self.layer_norm1(baseline_tokens)
        y, _ = self.cross_attention(y, visible_tokens, visible_tokens)
        
        # 残差连接
        predicted = baseline_tokens + y
        
        # MLP 预测
        y = self.layer_norm2(predicted)
        predicted = predicted + self.predictor(y)
        
        return predicted


class MFMGAD(nn.Module):
    """
    Masked Frequency Modeling for Graph Anomaly Detection
    
    核心思想：
    1. 使用可学习的频域滤波器 (Prompt) 提取节点的多频域视角特征
    2. 通过 Masked Frequency Prediction 学习跨频一致性
    3. 双分支学习：一致性学习 + Mining 对比学习
    4. 软标签机制避免假阳性
    
    模块架构：
    - Module 1: Prompt Tokenizer (频域视角提取)
    - Module 2: Masked Frequency Prediction (一致性学习)
    - Module 3: Transformer Encoder (表示学习)
    - Module 4: Dual-branch Learning (双分支学习)
    """
    
    def __init__(self, input_dim, hidden_dim, activation, args):
        super(MFMGAD, self).__init__()
        
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
        self.args = args
        self.input_dim = input_dim
        self.num_prompts = getattr(args, 'num_prompts', 8)
        
        # ==================== Module 1: Prompt Tokenizer ====================
        # 可学习的频域滤波器 (Prompt)，每个 Prompt 关注不同的频域成分
        self.prompts = nn.Parameter(torch.randn(1, self.num_prompts, input_dim))
        
        # 符号注意力的投影层
        self.sign_query = nn.Linear(input_dim, input_dim)
        self.sign_key = nn.Linear(input_dim, input_dim)
        
        # LayerNorm
        self.prompt_layer_norm = nn.LayerNorm(input_dim)
        
        # ==================== Module 2: Masked Frequency Prediction ====================
        # 一致性预测器
        self.consistency_predictor = ConsistencyPredictor(
            input_dim, 
            num_heads=getattr(args, 'GT_num_heads', 2),
            dropout=getattr(args, 'GT_dropout', 0.4)
        )
        
        # 正常基线（运行时动态计算，不作为模型参数）
        self.register_buffer('prompt_baselines', torch.zeros(self.num_prompts, input_dim))
        self.register_buffer('baseline_initialized', torch.tensor(False))
        
        # ==================== Module 3: Transformer Encoder ====================
        encoder_layers = [
            EncoderLayer(input_dim, args.GT_ffn_dim, args.GT_dropout, 
                        args.GT_attention_dropout, args.GT_num_heads)
            for _ in range(args.GT_num_layers)
        ]
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.final_layer_norm = nn.LayerNorm(input_dim)
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # ==================== Module 4: Dual-branch Learning ====================
        # 分类器网络（用于对比学习）
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1, bias=False)
        )
        
        # Token 解码器（用于重构任务）
        self.token_decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.num_prompts * input_dim)
        )
        
        # 重构误差投影层
        self.recon_error_proj = nn.Sequential(
            nn.Linear(self.num_prompts * input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
        self.to(self.device)

    # ==================== Module 1: Prompt Tokenizer ====================
    
    def extract_frequency_tokens(self, node_tokens, base_temp=None):
        """
        使用频域滤波器 (Prompt) 提取节点的多频域视角特征
        
        每个 Prompt 相当于一个频域滤波器，输出节点在该频域视角下的表示
        
        Args:
            node_tokens: 节点特征序列 [batch_size, num_hops, input_dim]
            base_temp: 温度参数
        
        Returns:
            frequency_tokens: 频域 token 矩阵 [batch_size, num_prompts, input_dim]
            attn_weights: 注意力权重 [batch_size, num_prompts, num_hops]
        """
        if base_temp is None:
            base_temp = self.args.tokenizer_temp
            
        batch_size = node_tokens.size(0)
        num_hops = node_tokens.size(1)

        # 扩展频域滤波器以匹配 batch size
        queries = self.prompts.expand(batch_size, -1, -1)  # [batch_size, num_prompts, input_dim]
        keys = node_tokens
        values = node_tokens

        # 计算幅值注意力分数
        score_mag = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.input_dim)
        
        # 计算方向注意力分数
        score_sign = torch.matmul(self.sign_query(queries), self.sign_key(keys).transpose(-1, -2))
        
        # 合并注意力的幅值和方向
        magnitude = F.softmax(score_mag / base_temp, dim=-1)
        sign = torch.tanh(score_sign / base_temp)
        
        attn_weights = magnitude * sign
        frequency_tokens = torch.matmul(attn_weights, values)
        frequency_tokens = self.prompt_layer_norm(frequency_tokens)

        return frequency_tokens, attn_weights

    # ==================== Module 2: Masked Frequency Prediction ====================
    
    def apply_frequency_mask(self, frequency_tokens, mask_ratio=None):
        """
        随机 mask 若干频域 token（向量化实现）
        
        Args:
            frequency_tokens: 频域 token [batch_size, num_prompts, input_dim]
            mask_ratio: mask 比例
        
        Returns:
            masked_tokens: mask 后的 token（被 mask 位置置零）
            mask: 布尔掩码，True 表示被 mask
            visible_mask: 布尔掩码，True 表示可见
        """
        if mask_ratio is None:
            mask_ratio = getattr(self.args, 'mask_ratio', 0.25)
            
        batch_size, num_prompts, _ = frequency_tokens.shape
        device = frequency_tokens.device
        
        # 向量化生成 mask：生成随机分数，选择每行最小的 num_mask 个位置
        num_mask = max(1, int(num_prompts * mask_ratio))
        
        # 生成随机分数 [batch_size, num_prompts]
        rand_scores = torch.rand(batch_size, num_prompts, device=device)
        
        # 获取每行最小的 num_mask 个位置的索引
        _, mask_indices = torch.topk(rand_scores, num_mask, dim=1, largest=False)
        
        # 创建 mask
        mask = torch.zeros(batch_size, num_prompts, dtype=torch.bool, device=device)
        mask.scatter_(1, mask_indices, True)
        
        visible_mask = ~mask
        
        # 创建 masked tokens（被 mask 位置置零）
        masked_tokens = frequency_tokens * visible_mask.unsqueeze(-1).float()
        
        return masked_tokens, mask, visible_mask
    
    def update_baselines(self, frequency_tokens, normal_idx):
        """
        更新正常基线：基于已标注正常节点计算每个 Prompt 的基线向量
        
        Args:
            frequency_tokens: 频域 token [num_nodes, num_prompts, input_dim]
            normal_idx: 正常节点索引
        """
        with torch.no_grad():
            normal_tokens = frequency_tokens[normal_idx]  # [num_normal, num_prompts, input_dim]
            
            # 计算每个 Prompt 的正常基线（均值）
            new_baselines = normal_tokens.mean(dim=0)  # [num_prompts, input_dim]
            
            # 使用移动平均更新
            momentum = 0.9
            if self.baseline_initialized:
                self.prompt_baselines = momentum * self.prompt_baselines + (1 - momentum) * new_baselines
            else:
                self.prompt_baselines = new_baselines
                self.baseline_initialized = torch.tensor(True, device=self.device)
    
    def compute_consistency_loss(self, frequency_tokens, mask, predicted_tokens):
        """
        计算跨频一致性损失（只在被 mask 的位置）
        
        正常节点：预测误差小 → 一致性分数高
        异常节点：预测误差大 → 一致性分数低
        
        Args:
            frequency_tokens: 原始频域 token [batch_size, num_prompts, input_dim]
            mask: 布尔掩码，True 表示被 mask
            predicted_tokens: 预测的频域 token [batch_size, num_prompts, input_dim]
        
        Returns:
            consistency_loss: 一致性损失
            prediction_errors: 每个样本的预测误差（用于后续可疑度计算）
        """
        # 只计算被 mask 位置的误差
        mask_expanded = mask.unsqueeze(-1).expand_as(frequency_tokens)
        
        # 预测误差
        prediction_errors = (predicted_tokens - frequency_tokens) ** 2
        masked_errors = prediction_errors * mask_expanded.float()
        
        # 每个样本的总误差
        sample_errors = masked_errors.sum(dim=(1, 2)) / (mask.sum(dim=1, keepdim=True).float() * self.input_dim + 1e-8)
        
        # 平均一致性损失
        consistency_loss = masked_errors.sum() / (mask.sum().float() * self.input_dim + 1e-8)
        
        return consistency_loss, sample_errors

    # ==================== Module 3: Transformer Encoder ====================
    
    def encode_with_cls_token(self, tokens):
        """
        使用 CLS Token 编码特征序列
        
        Args:
            tokens: 特征序列 [batch_size, num_prompts, input_dim]
        
        Returns:
            cls_output: CLS Token 的编码结果 [1, batch_size, input_dim]
        """
        batch_size = tokens.size(0)
        
        # 拼接 CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # 通过 Transformer 编码器层
        for layer in self.encoder_layers:
            tokens, _ = layer(tokens)
        
        # 应用最终 LayerNorm
        tokens = self.final_layer_norm(tokens)
        
        # 提取 CLS Token 作为输出
        cls_output = tokens[:, 0, :].unsqueeze(0)
        return cls_output

    # ==================== Module 4: Dual-branch Learning ====================
    
    def compute_dominant_prompts(self, attn_weights):
        """
        计算每个节点的主导频域滤波器（注意力权重和最大的 Prompt）
        
        Args:
            attn_weights: 注意力权重 [batch_size, num_prompts, num_hops]
        
        Returns:
            dominant_prompts: 主导 Prompt 索引 [batch_size]
        """
        attn_sum = attn_weights.sum(dim=-1)  # [batch_size, num_prompts]
        dominant_prompts = torch.argmax(attn_sum, dim=-1)
        return dominant_prompts
    
    def compute_suspicion_scores(self, embeddings, dominant_prompts, consistency_errors, normal_idx):
        """
        计算可疑度分数（向量化实现）
        
        suspicious = distance_to_baseline + λ × consistency_error
        
        Args:
            embeddings: 节点嵌入 [1, batch_size, input_dim]
            dominant_prompts: 主导 Prompt 索引 [batch_size]
            consistency_errors: 一致性预测误差 [batch_size]
            normal_idx: 正常节点索引（局部批次索引）
        
        Returns:
            suspicion_scores: 可疑度分数 [batch_size]
        """
        batch_size = embeddings.size(1)
        lambda_consistency = getattr(self.args, 'suspicion_lambda', 0.5)
        
        # 边界检查：确保 normal_idx 在有效范围内
        if normal_idx is None or len(normal_idx) == 0:
            return torch.zeros(batch_size, device=self.device), torch.zeros(self.num_prompts, self.input_dim, device=self.device)
        
        # 确保 normal_idx 不超过 batch_size
        normal_idx = normal_idx[normal_idx < batch_size]
        if len(normal_idx) == 0:
            return torch.zeros(batch_size, device=self.device), torch.zeros(self.num_prompts, self.input_dim, device=self.device)
        
        # 归一化嵌入
        embeddings_norm = F.normalize(embeddings[0], p=2, dim=1)  # [batch_size, input_dim]
        
        # 向量化计算每个 Prompt 的正常中心
        # 使用 scatter_reduce 进行高效聚合
        prompt_centers = torch.zeros(self.num_prompts, self.input_dim, device=self.device)
        normal_dominant = dominant_prompts[normal_idx]  # [num_normal]
        normal_embeddings = embeddings_norm[normal_idx]  # [num_normal, input_dim]
        
        # 使用 one-hot 编码进行向量化聚合
        one_hot = F.one_hot(normal_dominant, num_classes=self.num_prompts).float()  # [num_normal, num_prompts]
        counts = one_hot.sum(dim=0, keepdim=True).t()  # [num_prompts, 1]
        
        # 计算加权和
        weighted_sum = torch.matmul(one_hot.t(), normal_embeddings)  # [num_prompts, input_dim]
        
        # 避免除零
        counts = counts.clamp(min=1)
        prompt_centers = weighted_sum / counts
        
        # 向量化计算距离
        # 扩展维度进行广播: [batch_size, 1, input_dim] - [1, num_prompts, input_dim]
        emb_expanded = embeddings_norm.unsqueeze(1)  # [batch_size, 1, input_dim]
        centers_expanded = prompt_centers.unsqueeze(0)  # [1, num_prompts, input_dim]
        
        # 计算所有节点到所有中心的距离
        all_distances = torch.norm(emb_expanded - centers_expanded, dim=2)  # [batch_size, num_prompts]
        
        # 根据 dominant_prompts 选择对应的距离
        distances = all_distances[torch.arange(batch_size, device=self.device), dominant_prompts]
        
        # 归一化距离和一致性误差
        d_min, d_max = distances.min(), distances.max()
        distances_norm = (distances - d_min) / (d_max - d_min + 1e-8)
        
        c_min, c_max = consistency_errors.min(), consistency_errors.max()
        consistency_norm = (consistency_errors - c_min) / (c_max - c_min + 1e-8)
        
        # 综合可疑度
        suspicion_scores = distances_norm + lambda_consistency * consistency_norm
        
        return suspicion_scores, prompt_centers
    
    def compute_soft_labels(self, suspicion_scores, threshold=None, temperature=None):
        """
        基于可疑度计算软标签
        
        soft_label = sigmoid((suspicion - threshold) / temperature)
        
        Args:
            suspicion_scores: 可疑度分数 [num_nodes]
            threshold: 阈值
            temperature: 温度参数
        
        Returns:
            soft_labels: 软标签 [num_nodes]
        """
        if threshold is None:
            threshold = getattr(self.args, 'mining_threshold_base', 0.5)
        if temperature is None:
            temperature = getattr(self.args, 'mining_temperature', 1.0)
        
        soft_labels = torch.sigmoid((suspicion_scores - threshold) / temperature)
        return soft_labels
    
    def compute_contrastive_loss(self, normal_embeddings, unknown_embeddings, soft_labels):
        """
        计算软标签对比学习损失
        
        Args:
            normal_embeddings: 正常节点嵌入 [num_normal, input_dim]
            unknown_embeddings: 未知节点嵌入 [num_unknown, input_dim]
            soft_labels: 未知节点的软标签 [num_unknown]
        
        Returns:
            contrastive_loss: 对比学习损失
        """
        # 正样本：正常节点自身（自对比）
        # 负样本：可疑节点（根据软标签加权）
        
        # 确保 soft_labels 是 1D
        soft_labels = soft_labels.view(-1)
        
        # 计算正常节点的中心
        normal_center = normal_embeddings.mean(dim=0, keepdim=True)  # [1, input_dim]
        normal_center = F.normalize(normal_center, p=2, dim=1)
        
        # 计算未知节点到正常中心的相似度
        unknown_norm = F.normalize(unknown_embeddings, p=2, dim=1)
        similarities = torch.mm(unknown_norm, normal_center.t()).squeeze(-1)  # [num_unknown]
        
        # 确保 similarities 是 1D
        similarities = similarities.view(-1)
        
        # 转换为 logits
        logits = similarities * 10  # 温度缩放
        
        # 使用软标签计算 BCE 损失
        # soft_label 接近 1 表示可疑（应该远离正常中心，即 similarity 应该低）
        # soft_label 接近 0 表示正常（应该接近正常中心，即 similarity 应该高）
        contrastive_loss = F.binary_cross_entropy_with_logits(
            logits, 1 - soft_labels, reduction='mean'
        )
        
        return contrastive_loss

    # ==================== 正交损失 ====================
    
    def compute_orthogonal_loss(self, attn_weights):
        """
        计算正交损失，确保不同频域滤波器学习到不同的特征
        
        正交性约束：对于任意两个不同的 Prompt p_i, p_j: cos(O_i, O_j) ≈ 0
        
        Args:
            attn_weights: 注意力权重 [batch_size, num_prompts, num_hops]
        
        Returns:
            ortho_loss: 正交损失值
        """
        # 归一化注意力权重
        norms = torch.norm(attn_weights, dim=-1, keepdim=True) + 1e-8
        normalized_weights = attn_weights / norms

        # 计算不同 Prompt 之间的余弦相似度
        cos_sim = torch.matmul(normalized_weights, normalized_weights.transpose(-1, -2))

        # 排除对角线，计算非对角线元素的绝对值均值
        mask = ~torch.eye(self.num_prompts, dtype=torch.bool, device=attn_weights.device)
        ortho_loss = cos_sim[:, mask].abs().mean()

        return ortho_loss

    # ==================== 前向传播 ====================
    
    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False, return_details=False):
        """
        前向传播
        
        Args:
            input_tokens: 输入节点特征 [num_nodes, pp_k+1, input_dim]
            adj: 邻接矩阵（未使用）
            _: 占位参数
            normal_for_train_idx: 训练用的正常节点索引
            train_flag: 是否训练模式
            args: 参数配置
            sparse: 是否使用稀疏格式
            return_details: 是否返回详细信息（用于调试）
        
        Returns:
            训练模式:
                embeddings: 节点嵌入 [1, num_nodes, input_dim]
                logits: 分类 logits
                recon_loss: 重构损失
                ortho_loss: 正交损失
                consistency_loss: 一致性损失
                contrast_loss: 对比学习损失
                suspicion_scores: 可疑度分数
                attention_weights: 注意力权重
            推理模式:
                embeddings: 节点嵌入
                logits: 分类 logits
                anomaly_scores: 异常得分
                consistency_errors: 一致性误差
                suspicion_scores: 可疑度分数
        """
        num_nodes = input_tokens.size(0)
        
        # ==================== Module 1: Prompt Tokenizer ====================
        # 提取频域 token
        frequency_tokens, attn_weights = self.extract_frequency_tokens(input_tokens)
        
        # 计算正交损失
        ortho_loss = self.compute_orthogonal_loss(attn_weights)
        
        # ==================== 更新正常基线 ====================
        if train_flag and normal_for_train_idx is not None and len(normal_for_train_idx) > 0:
            self.update_baselines(frequency_tokens, normal_for_train_idx)
        
        # ==================== Module 2: Masked Frequency Prediction ====================
        if train_flag:
            # 训练模式：应用 mask 并预测
            masked_tokens, mask, visible_mask = self.apply_frequency_mask(frequency_tokens)
            
            # 扩展基线以匹配 batch
            batch_size = frequency_tokens.size(0)
            baseline_expanded = self.prompt_baselines.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 一致性预测
            predicted_tokens = self.consistency_predictor(masked_tokens, baseline_expanded, visible_mask)
            
            # 计算一致性损失（只在正常节点上）
            consistency_loss, consistency_errors = self.compute_consistency_loss(
                frequency_tokens, mask, predicted_tokens
            )
            
            # 使用预测后的完整 token 序列进行编码
            # 对于可见位置使用原始 token，对于 mask 位置使用预测 token
            encoded_tokens = torch.where(
                visible_mask.unsqueeze(-1).expand_as(frequency_tokens),
                frequency_tokens,
                predicted_tokens
            )
        else:
            # 推理模式：不应用 mask，直接使用原始 token
            encoded_tokens = frequency_tokens
            consistency_loss = torch.tensor(0.0, device=self.device)
            consistency_errors = torch.zeros(num_nodes, device=self.device)
            mask = None
        
        # ==================== Module 3: Transformer Encoder ====================
        embeddings = self.encode_with_cls_token(encoded_tokens)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # ==================== 重构损失 ====================
        reconstructed_features = self.token_decoder(embeddings).squeeze(0)
        target = frequency_tokens.view(-1, self.num_prompts * self.input_dim)
        recon_loss = F.mse_loss(reconstructed_features, target)
        
        # ==================== Module 4: Dual-branch Learning ====================
        contrast_loss = torch.tensor(0.0, device=self.device)
        batch_size = frequency_tokens.size(0)
        suspicion_scores = torch.zeros(batch_size, device=self.device)
        
        if train_flag and normal_for_train_idx is not None and len(normal_for_train_idx) > 0:
            # 计算主导 Prompt
            dominant_prompts = self.compute_dominant_prompts(attn_weights)
            
            # 一致性误差需要是 batch_size 维度
            # consistency_errors 目前是每个样本的误差，尺寸已经是 [batch_size]
            
            # 计算可疑度 - 使用局部索引
            suspicion_scores, prompt_centers = self.compute_suspicion_scores(
                embeddings, dominant_prompts, consistency_errors, normal_for_train_idx
            )
            
            # 计算软标签
            soft_labels = self.compute_soft_labels(suspicion_scores)
            
            # 分离正常节点和未知节点（向量化操作）
            # 使用 torch 操作替代 Python set/list
            all_indices = torch.arange(batch_size, device=self.device)
            is_normal_mask = torch.isin(all_indices, normal_for_train_idx)
            local_unknown_indices_tensor = all_indices[~is_normal_mask]
            
            if len(local_unknown_indices_tensor) > 0:
                normal_embeddings = embeddings[0, normal_for_train_idx, :]
                unknown_embeddings = embeddings[0, local_unknown_indices_tensor, :]
                # 确保 soft_labels 是 1D 并正确索引
                unknown_soft_labels = soft_labels.view(-1)[local_unknown_indices_tensor].float()
                
                # 计算对比学习损失
                contrast_loss = self.compute_contrastive_loss(
                    normal_embeddings, unknown_embeddings, unknown_soft_labels
                )
            
            # 分类器：只使用正常节点，避免显存爆炸
            # 对比学习已经通过 contrast_loss 实现了正常/异常的分离
            logits = self.classifier(embeddings[:, normal_for_train_idx, :])
        else:
            # 推理模式
            # 计算主导 Prompt
            dominant_prompts = self.compute_dominant_prompts(attn_weights)
            
            # 推理模式下，使用简化的分类器调用
            # 只取 CLS token 进行分类，避免显存爆炸
            logits = self.classifier(embeddings)  # embeddings: [1, batch_size, input_dim]
        
        # ==================== 异常得分计算（推理时）====================
        anomaly_scores = None
        if not train_flag:
            # anomaly_score = α × consistency_error + β × suspicion
            alpha = getattr(args, 'consistency_weight', 1.0)
            beta = getattr(args, 'suspicion_weight', 1.0)
            
            # 一致性误差：重构误差作为替代
            recon_error = self.recon_error_proj(reconstructed_features - target)
            consistency_error_per_node = recon_error.norm(dim=1)
            
            # 归一化
            consistency_norm = (consistency_error_per_node - consistency_error_per_node.min()) / \
                              (consistency_error_per_node.max() - consistency_error_per_node.min() + 1e-8)
            suspicion_norm = (suspicion_scores - suspicion_scores.min()) / \
                            (suspicion_scores.max() - suspicion_scores.min() + 1e-8)
            
            anomaly_scores = alpha * consistency_norm + beta * suspicion_norm
        
        # 返回结果
        if train_flag:
            if return_details:
                return {
                    'embeddings': embeddings,
                    'logits': logits,
                    'recon_loss': recon_loss,
                    'ortho_loss': ortho_loss,
                    'consistency_loss': consistency_loss,
                    'contrast_loss': contrast_loss,
                    'suspicion_scores': suspicion_scores,
                    'attention_weights': attn_weights,
                    'frequency_tokens': frequency_tokens,
                    'predicted_tokens': predicted_tokens if hasattr(self, 'predicted_tokens') else None,
                    'mask': mask
                }
            else:
                return (embeddings, logits, recon_loss, ortho_loss, consistency_loss, 
                        contrast_loss, suspicion_scores, attn_weights)
        else:
            if return_details:
                return {
                    'embeddings': embeddings,
                    'logits': logits,
                    'anomaly_scores': anomaly_scores,
                    'consistency_errors': consistency_errors,
                    'suspicion_scores': suspicion_scores,
                    'attention_weights': attn_weights,
                    'frequency_tokens': frequency_tokens
                }
            else:
                return (embeddings, logits, anomaly_scores, consistency_errors, 
                        suspicion_scores, attn_weights)