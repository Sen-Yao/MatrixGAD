import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class PromptGAD(nn.Module):
    """
    Prompt-based Graph Anomaly Detection 模型
    
    核心思想：使用可学习的 Prompt Token 提取图节点的多视角特征，
    通过对比学习区分正常节点与异常节点。
    """
    
    def __init__(self, input_dim, hidden_dim, activation, args):
        super(PromptGAD, self).__init__()
        
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
        self.args = args
        self.input_dim = input_dim
        self.num_prompts = getattr(args, 'num_prompts', 8)

        # 分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1, bias=False)
        )

        # Graph Transformer 编码器
        encoder_layers = [
            EncoderLayer(input_dim, args.GT_ffn_dim, args.GT_dropout, 
                        args.GT_attention_dropout, args.GT_num_heads)
            for _ in range(args.GT_num_layers)
        ]
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.final_layer_norm = nn.LayerNorm(input_dim)

        # 可学习的 Prompt Token
        self.prompts = nn.Parameter(torch.randn(1, self.num_prompts, input_dim))

        # 符号注意力的投影层
        self.sign_query = nn.Linear(input_dim, input_dim)
        self.sign_key = nn.Linear(input_dim, input_dim)

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

        # LayerNorm
        self.prompt_layer_norm = nn.LayerNorm(input_dim)

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        self.to(self.device)

    def extract_prompt_features(self, node_tokens, base_temp=None, high_temp=None, temp_per_sample=None):
        """
        使用 Prompt Token 提取节点特征的多视角表示
        
        Args:
            node_tokens: 节点特征序列 [batch_size, num_hops, input_dim]
            base_temp: 基础温度参数
            high_temp: 对部分 token 使用的高温参数（用于伪异常生成）
            temp_per_sample: 每个样本的温度倍数 [batch_size]
        
        Returns:
            prompt_features: 提取的多视角特征 [batch_size, num_prompts, input_dim]
            attn_weights: 注意力权重 [batch_size, num_prompts, num_hops]
        """
        if base_temp is None:
            base_temp = self.args.tokenizer_temp
            
        batch_size = node_tokens.size(0)
        num_hops = node_tokens.size(1)

        # 扩展 Prompt 以匹配 batch size
        queries = self.prompts.expand(batch_size, -1, -1)  # [batch_size, num_prompts, input_dim]
        keys = node_tokens
        values = node_tokens

        # 计算幅值注意力分数
        score_mag = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.input_dim)
        
        # 计算方向注意力分数
        score_sign = torch.matmul(self.sign_query(queries), self.sign_key(keys).transpose(-1, -2))
        
        # 处理温度参数
        if high_temp is not None or temp_per_sample is not None:
            # 部分升温模式：前 num_hops//2 个 token 使用高温
            partial_idx = (num_hops - 1) // 2
            
            if temp_per_sample is not None:
                # 每个样本使用不同温度
                temp_multiplier = temp_per_sample.view(batch_size, 1, 1)
                temp_mask = torch.ones(batch_size, self.num_prompts, num_hops, device=node_tokens.device)
                temp_mask = temp_mask * base_temp * temp_multiplier
                
                if partial_idx > 0 and high_temp is not None:
                    high_temp_values = high_temp * temp_multiplier
                    temp_mask[:, :, 1:partial_idx+1] = high_temp_values.expand(-1, self.num_prompts, partial_idx)
            else:
                # 所有样本使用相同温度
                temp_mask = torch.ones(batch_size, self.num_prompts, num_hops, device=node_tokens.device)
                temp_mask = temp_mask * base_temp
                if partial_idx > 0 and high_temp is not None:
                    temp_mask[:, :, 1:partial_idx+1] = high_temp
            
            magnitude = F.softmax(score_mag / temp_mask, dim=-1)
            sign = torch.tanh(score_sign / temp_mask)
        else:
            # 标准模式：所有 token 使用相同温度
            magnitude = F.softmax(score_mag / base_temp, dim=-1)
            sign = torch.tanh(score_sign / base_temp)

        # 合并注意力的幅值和方向
        attn_weights = magnitude * sign
        prompt_features = torch.matmul(attn_weights, values)
        prompt_features = self.prompt_layer_norm(prompt_features)

        return prompt_features, attn_weights

    def compute_orthogonal_loss(self, attn_weights):
        """
        计算正交损失，确保不同 Prompt 学习到不同的特征
        
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

    def _compute_dominant_prompts_and_centers(self, embeddings, normal_idx, attn_weights):
        """
        计算每个节点的主导 Prompt 和每个 Prompt 的中心
        
        Args:
            embeddings: 节点嵌入 [1, num_nodes, input_dim]
            normal_idx: 正常节点索引
            attn_weights: 注意力权重 [num_nodes, num_prompts, num_hops]
        
        Returns:
            dominant_all: 所有节点的主导 Prompt 索引
            dominant_normal: 正常节点的主导 Prompt 索引
            prompt_centers: 每个 Prompt 的中心向量
            normal_embeddings_norm: 归一化的正常节点嵌入
        """
        # 提取正常节点嵌入
        normal_embeddings = embeddings[0, normal_idx, :]
        normal_embeddings_norm = F.normalize(normal_embeddings, p=2, dim=1)
        
        # 计算主导 Prompt（注意力权重和最大的 Prompt）
        attn_sum_all = attn_weights.sum(dim=-1)
        dominant_all = torch.argmax(attn_sum_all, dim=-1)
        
        attn_sum_normal = attn_weights[normal_idx, :, :].sum(dim=-1)
        dominant_normal = torch.argmax(attn_sum_normal, dim=-1)
        
        # 计算每个 Prompt 的中心（基于正常节点）
        prompt_centers = torch.zeros(self.num_prompts, self.input_dim, device=embeddings.device)
        for p in range(self.num_prompts):
            mask = (dominant_normal == p)
            if mask.sum() >= 1:
                prompt_centers[p] = normal_embeddings_norm[mask].mean(dim=0).detach()
        
        return dominant_all, dominant_normal, prompt_centers, normal_embeddings_norm

    def _generate_pseudo_anomalies(self, node_tokens, normal_idx, embeddings, attn_weights, dominant_all):
        """
        生成伪异常样本
        
        Args:
            node_tokens: 节点特征序列
            normal_idx: 用于生成伪异常的正常节点索引
            embeddings: 节点嵌入
            attn_weights: 注意力权重
            dominant_all: 所有节点的主导 Prompt 索引
        
        Returns:
            pseudo_anomaly_embeddings: 伪异常节点嵌入
        """
        sample_rate = self.args.sample_rate
        num_samples = int(len(normal_idx) * sample_rate)
        
        # 随机选择正常节点
        perm = torch.randperm(normal_idx.size(0), device=normal_idx.device)
        selected_idx = normal_idx[perm[:num_samples]]
        selected_tokens = node_tokens[selected_idx, :, :]
        selected_dominant = dominant_all[selected_idx]
        
        # 获取温度参数
        base_temp = getattr(self.args, 'tokenizer_temp', 1.0)
        hallucination_ratio = getattr(self.args, 'tokenizer_hallucination_ratio', 2.0)
        distance_scale = getattr(self.args, 'hallucination_temp_distance_scale', 0.0)
        
        if distance_scale > 0:
            # 动态温度模式
            pseudo_anomaly_embeddings = self._generate_pseudo_anomalies_dynamic_temp(
                selected_tokens, selected_dominant, embeddings, selected_idx,
                base_temp, hallucination_ratio, distance_scale
            )
        else:
            # 固定温度模式
            pseudo_anomaly_embeddings = self._generate_pseudo_anomalies_fixed_temp(
                selected_tokens, selected_dominant,
                base_temp, hallucination_ratio
            )
        
        return pseudo_anomaly_embeddings, selected_idx

    def _generate_pseudo_anomalies_fixed_temp(self, tokens, dominant_prompts, base_temp, hallucination_ratio):
        """固定温度模式生成伪异常"""
        high_temp = base_temp * hallucination_ratio
        
        # 正常温度处理
        normal_features, _ = self.extract_prompt_features(tokens, base_temp)
        normal_features = normal_features.detach()
        
        # 高温处理（部分 token）
        hallucinated_features, _ = self.extract_prompt_features(tokens, base_temp, high_temp=high_temp)
        hallucinated_features = hallucinated_features.detach()
        
        # 创建掩码：只对主导 Prompt 使用高温特征
        batch_size, num_prompts, _ = normal_features.shape
        mask = torch.zeros(batch_size, num_prompts, dtype=torch.bool, device=tokens.device)
        mask[torch.arange(batch_size), dominant_prompts] = True
        
        # 混合特征
        mixed_features = torch.where(mask.unsqueeze(-1), hallucinated_features, normal_features)
        
        # 编码并归一化
        embeddings = self.encode_with_cls_token(mixed_features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.squeeze(0)

    def _generate_pseudo_anomalies_dynamic_temp(self, tokens, dominant_prompts, embeddings, selected_idx,
                                                 base_temp, hallucination_ratio, distance_scale):
        """动态温度模式生成伪异常"""
        # 计算节点到主导 Prompt 中心的距离
        _, _, prompt_centers, _ = self._compute_dominant_prompts_and_centers(
            embeddings, selected_idx, 
            torch.zeros(embeddings.size(1), self.num_prompts, 1, device=embeddings.device)
        )
        
        node_embeddings = F.normalize(embeddings[0, selected_idx, :], p=2, dim=1)
        centers = prompt_centers[dominant_prompts]
        distances = torch.norm(node_embeddings - centers, dim=1)
        
        # 归一化距离
        max_distance = distances.max() if distances.numel() > 0 else 1.0
        normalized_distances = distances / (max_distance + 1e-8)
        
        # 计算动态温度倍数
        dynamic_ratios = 1.0 + (hallucination_ratio - 1.0) * (1.0 + distance_scale * normalized_distances)
        
        # 正常温度处理
        normal_features, _ = self.extract_prompt_features(tokens, base_temp)
        normal_features = normal_features.detach()
        
        # 动态高温处理
        high_temp = hallucination_ratio * base_temp
        hallucinated_features, _ = self.extract_prompt_features(
            tokens, base_temp, high_temp=high_temp, temp_per_sample=dynamic_ratios
        )
        hallucinated_features = hallucinated_features.detach()
        
        # 创建掩码并混合
        batch_size, num_prompts, _ = normal_features.shape
        mask = torch.zeros(batch_size, num_prompts, dtype=torch.bool, device=tokens.device)
        mask[torch.arange(batch_size), dominant_prompts] = True
        
        mixed_features = torch.where(mask.unsqueeze(-1), hallucinated_features, normal_features)
        
        # 编码并归一化
        embeddings = self.encode_with_cls_token(mixed_features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.squeeze(0)

    def compute_recon_loss(self, prompt_features, reconstructed_features, normal_embeddings, normal_idx):
        """
        计算重构损失
        
        Args:
            prompt_features: Prompt 提取的特征 [num_nodes, num_prompts, input_dim]
            reconstructed_features: 重构的特征 [num_nodes, num_prompts * input_dim]
            normal_embeddings: 正常节点嵌入
            normal_idx: 正常节点索引
        
        Returns:
            recon_loss: 重构损失值
        """
        target = prompt_features.view(-1, self.num_prompts * self.input_dim)
        recon_loss = F.mse_loss(reconstructed_features, target)
        return recon_loss

    def compute_uniformity_loss(self, normal_embeddings, dominant_prompts, prompt_centers):
        """
        计算 Prompt-aware 均匀性损失
        
        Args:
            normal_embeddings: 归一化的正常节点嵌入 [num_normal, input_dim]
            dominant_prompts: 正常节点的主导 Prompt 索引 [num_normal]
            prompt_centers: Prompt 中心向量 [num_prompts, input_dim]
        
        Returns:
            uniformity_loss: 均匀性损失
        """
        tau = self.args.GNA_temp
        lambda_inter = getattr(self.args, 'lambda_inter', 0.1)
        
        # 模式内聚合损失
        intra_loss = self._compute_intra_pattern_loss(normal_embeddings, dominant_prompts, tau)
        
        # 模式间分散损失
        inter_loss = self._compute_inter_pattern_loss(normal_embeddings, prompt_centers, tau)
        
        return intra_loss + lambda_inter * inter_loss

    def _compute_intra_pattern_loss(self, embeddings, dominant_prompts, tau):
        """计算同模式节点的聚合损失"""
        num_nodes = embeddings.size(0)
        device = embeddings.device
        
        intra_loss = torch.tensor(0.0, device=device)
        valid_count = 0
        
        for p in range(self.num_prompts):
            mask = (dominant_prompts == p)
            pattern_nodes = torch.where(mask)[0]
            
            if pattern_nodes.size(0) < 2:
                continue
            
            pattern_embeddings = embeddings[pattern_nodes, :]
            similarity = torch.mm(pattern_embeddings, pattern_embeddings.t()) / tau
            
            # 排除对角线
            diag_mask = torch.eye(pattern_nodes.size(0), device=device, dtype=torch.bool)
            similarity = similarity.masked_fill(diag_mask, float('-inf'))
            
            intra_loss = intra_loss + torch.logsumexp(similarity, dim=1).mean()
            valid_count += 1
        
        if valid_count > 0:
            intra_loss = intra_loss / valid_count
        
        return intra_loss

    def _compute_inter_pattern_loss(self, embeddings, prompt_centers, tau):
        """计算不同模式间的分散损失"""
        device = embeddings.device
        
        # 找有效中心
        valid_mask = (prompt_centers.norm(dim=1) > 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if valid_indices.size(0) < 2:
            return torch.tensor(0.0, device=device)
        
        valid_centers = F.normalize(prompt_centers[valid_indices, :], p=2, dim=1)
        similarity = torch.mm(valid_centers, valid_centers.t()) / tau
        
        # 取上三角（排除对角线）
        upper_mask = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
        
        return torch.exp(similarity[upper_mask]).mean()

    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False, return_attn_weights=False):
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
            return_attn_weights: 是否返回注意力权重
        
        Returns:
            embeddings: 节点嵌入
            combined_embeddings: 组合嵌入（训练时）
            logits: 分类 logits
            pseudo_anomaly_embeddings: 伪异常嵌入
            noised_embeddings: 噪声嵌入（未使用）
            recon_loss: 重构损失
            ring_loss: 环形损失（未使用）
            ortho_loss: 正交损失
            recon_error: 重构误差向量
            uniformity_loss: 均匀性损失
            original_prompt_features: 原始 Prompt 特征
            reconstructed_features: 重构的特征
            attn_weights: 注意力权重（可选）
        """
        # 提取 Prompt 特征
        prompt_features, attn_weights = self.extract_prompt_features(
            input_tokens, getattr(self.args, 'tokenizer_temp', 1.0)
        )

        # 计算正交损失
        ortho_loss = self.compute_orthogonal_loss(attn_weights)

        # 使用 CLS Token 编码
        embeddings = self.encode_with_cls_token(prompt_features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # 初始化输出变量
        combined_embeddings = None
        pseudo_anomaly_embeddings = None
        recon_error = None
        original_prompt_features = prompt_features
        reconstructed_features = self.token_decoder(embeddings).squeeze(0)

        uniformity_loss = torch.tensor(0.0, device=embeddings.device)
        recon_loss = torch.tensor(0.0, device=embeddings.device)
        ring_loss = torch.tensor(0.0, device=embeddings.device)

        if train_flag:
            # 训练模式：生成伪异常并计算损失
            dominant_all, dominant_normal, prompt_centers, normal_embeddings = \
                self._compute_dominant_prompts_and_centers(embeddings, normal_for_train_idx, attn_weights)
            
            # 生成伪异常
            pseudo_anomaly_embeddings, selected_idx = self._generate_pseudo_anomalies(
                input_tokens, normal_for_train_idx, embeddings, attn_weights, dominant_all
            )
            
            # 计算重构损失
            recon_loss = self.compute_recon_loss(
                prompt_features, reconstructed_features, embeddings, selected_idx
            )
            
            # 计算均匀性损失
            uniformity_loss = self.compute_uniformity_loss(
                normal_embeddings, dominant_normal, prompt_centers
            )
            
            # 组合正常节点和伪异常嵌入
            normal_embeddings = embeddings[:, normal_for_train_idx, :]
            combined_embeddings = torch.cat([normal_embeddings, pseudo_anomaly_embeddings.unsqueeze(0)], dim=1)
            combined_embeddings = F.normalize(combined_embeddings, p=2, dim=-1)
            
            classifier_input = combined_embeddings
        else:
            # 推理模式：计算重构误差
            target = prompt_features.view(-1, self.num_prompts * self.input_dim)
            recon_error = self.recon_error_proj(reconstructed_features - target)
            classifier_input = embeddings

        # 分类
        logits = self.classifier(classifier_input)
        embeddings = embeddings.clone()

        if return_attn_weights:
            return (embeddings, combined_embeddings, logits, pseudo_anomaly_embeddings, None,
                    recon_loss, ring_loss, ortho_loss, recon_error, uniformity_loss,
                    original_prompt_features, reconstructed_features, attn_weights)
        else:
            return (embeddings, combined_embeddings, logits, pseudo_anomaly_embeddings, None,
                    recon_loss, ring_loss, ortho_loss, recon_error, uniformity_loss,
                    original_prompt_features, reconstructed_features)