import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math

from check_gpu_memory import print_gpu_memory_usage, print_tensor_memory, clear_gpu_memory

from playground import *

class FeedForwardNetwork(nn.Module):
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
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = 1

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias


        # 这里的 x 就是经过 softmax 归一化的注意力权重
        attention_weights = torch.softmax(x, dim=3)

        x = self.att_dropout(attention_weights) # Dropout 应用于注意力权重

        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)

        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None):


        y = self.self_attention_norm(x)
        y, attention_weights = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        ## 实现的是transformer 和 FFN的LayerNorm 以及相关操作
        
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x, attention_weights

class MatrixGAD(nn.Module):
    def __init__(self, n_in, n_h, activation, args):
        super(MatrixGAD, self).__init__()
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
        self.args = args
        self.n_in = n_in  # 保存输入维度用于重构
        # --- 1. 保留判别器与投影层 ---
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False) # 输出 logits
        self.act = nn.ReLU()
        # LayerNorm 用于控制 Logit 爆炸
        self.ln1 = nn.LayerNorm(int(n_h / 2))
        self.ln2 = nn.LayerNorm(int(n_h / 4))
        # --- 2. Tokenizer 定义 ---
        # T0: 属性特征专用
        self.id_projection = nn.Linear(n_in, args.embedding_dim)
        # T1, T3, T4, T5: 残差/拓扑特征
        self.res_projection = nn.Linear(n_in, args.embedding_dim)
        # T2: 度数特征 (假设输入是标量)
        self.deg_encoder = nn.Sequential(
            nn.Linear(1, args.embedding_dim),
            nn.GELU(),
            nn.Linear(args.embedding_dim, args.embedding_dim)
        )
        # --- 3. Transformer Encoder ---
        self.token_length = 4
        encoders = [EncoderLayer(args.embedding_dim, args.GT_ffn_dim, args.GT_dropout, args.GT_attention_dropout, args.GT_num_heads)
                    for _ in range(args.GT_num_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.embedding_dim)
        # Embeddings
        self.type_embedding = nn.Parameter(torch.zeros(1, self.token_length + 1, args.embedding_dim))
        nn.init.xavier_uniform_(self.type_embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embedding_dim))
        
        # --- 4. 重构模块 (参考 GGADFormer 设计) ---
        # Token Decoder: 将 CLS embedding 解码回 token 空间
        # 输出维度: token_length * n_in (T0, T1, T2, T3)
        # 注意: T2 是度数特征(标量)，所以实际输出是 3*n_in + 1
        self.token_decoder = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, 3 * n_in + 1)  # T0, T1, T3 各 n_in 维，T2 是 1 维度数
        )
        
        # Reconstruction Projection: 将重构误差投影到 embedding 维度
        self.reconstruction_proj = nn.Sequential(
            nn.Linear(3 * n_in + 1, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, args.embedding_dim)
        )
        
        # 重构损失函数
        self.recon_loss_fn = nn.MSELoss()
        
        self.to(self.device)
    def compute_infoNCE_uniformity_loss(self, emb, normal_idx, args):
        """
        计算InfoNCE均匀性损失，推开不同正常节点在嵌入空间中的距离
        Args:
            emb: [1, N, embedding_dim] - 节点的嵌入表征
            normal_idx: 正常节点索引（可以是全局索引或局部batch索引）
            args: 包含GNA_temp等超参数的配置
        Returns:
            uniformity_loss: InfoNCE均匀性损失
        """
        # 提取正常节点的嵌入: [num_normal, embedding_dim]
        normal_emb = emb[0, normal_idx, :]  # [num_normal, embedding_dim]
        num_normal = normal_emb.size(0)
        
        # 如果正常节点数量少于2，无法计算InfoNCE损失
        if num_normal < 2:
            return torch.tensor(0.0, device=emb.device)
        
        # L2 归一化，便于计算余弦相似度
        normal_emb_norm = F.normalize(normal_emb, p=2, dim=1)  # [num_normal, embedding_dim]
        
        # 计算所有节点对之间的余弦相似度矩阵
        # similarity_matrix[i,j] = cos_sim(node_i, node_j)
        similarity_matrix = torch.mm(normal_emb_norm, normal_emb_norm.t())  # [num_normal, num_normal]
        
        # 应用温度参数
        similarity_matrix = similarity_matrix / args.GNA_temp
        
        # 创建掩码，排除对角线元素（自己与自己的相似度）
        mask = torch.eye(num_normal, device=emb.device, dtype=torch.bool)
        
        # 并行计算InfoNCE损失
        # 对于每个锚点i，我们希望它与其他所有节点的相似度都尽可能小
        # 使用掩码将对角线元素设为极小值，这样就不会影响logsumexp计算
        similarity_matrix_masked = similarity_matrix.masked_fill(mask, float('-inf'))

        # 并行计算所有节点的logsumexp值
        # 对每一行计算logsumexp，得到每个节点与其他节点的相似度之和
        log_sum_exp_values = torch.logsumexp(similarity_matrix_masked, dim=1)  # [num_normal]

        # 平均化损失
        uniformity_loss = log_sum_exp_values.mean() - math.log(num_normal - 1)
        
        return uniformity_loss
    
    def compute_uniformity_loss_from_emb(self, normal_emb, args):
        """
        直接从正常节点嵌入计算InfoNCE均匀性损失（用于eval阶段）
        Args:
            normal_emb: [num_normal, embedding_dim] - 正常节点的嵌入
            args: 包含GNA_temp等超参数的配置
        Returns:
            uniformity_loss: InfoNCE均匀性损失
        """
        num_normal = normal_emb.size(0)
        
        # 如果正常节点数量少于2，无法计算InfoNCE损失
        if num_normal < 2:
            return torch.tensor(0.0, device=normal_emb.device)
        
        # L2 归一化，便于计算余弦相似度
        normal_emb_norm = F.normalize(normal_emb, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(normal_emb_norm, normal_emb_norm.t())
        
        # 应用温度参数
        similarity_matrix = similarity_matrix / args.GNA_temp
        
        # 创建掩码，排除对角线元素
        mask = torch.eye(num_normal, device=normal_emb.device, dtype=torch.bool)
        similarity_matrix_masked = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # 计算logsumexp
        log_sum_exp_values = torch.logsumexp(similarity_matrix_masked, dim=1)
        
        # 平均化损失
        uniformity_loss = log_sum_exp_values.mean() - math.log(num_normal - 1)
        
        return uniformity_loss
    
    def compute_rec_loss(self, input_tokens, reconstructed_tokens, normal_for_generation_emb, reencoded_emb, normal_for_generation_idx):
        """
        计算 Token 空间和 Embedding 空间的重构损失 (参考 GGADFormer 设计)
        Args:
            input_tokens: 原始采样的 Token 序列 [N, 4, D]
            reconstructed_tokens: 经过解码器重构的 Token 序列 [N, 3*n_in + 1]
            normal_for_generation_emb: 第一次编码的正常节点嵌入 [1, num_normal, D]
            reencoded_emb: 将重构 Token 序列进行二次编码的嵌入结果 [num_normal, D]
            normal_for_generation_idx: 用于生成的正常节点索引
        Returns:
            loss_rec: 重构损失值
        """
        # Token 空间重构损失
        # 需要将 input_tokens 转换为与 reconstructed_tokens 相同的形状
        # input_tokens: [N, 4, D] -> T0, T1, T2, T3
        # T2 是度数特征 (标量)，其他是 n_in 维
        # 重构目标: [T0, T1, T3] 展平 + T2 (度数) = 3*n_in + 1
        
        # 提取并展平 T0, T1, T3
        t0 = input_tokens[:, 0, :]  # [N, n_in]
        t1 = input_tokens[:, 1, :]  # [N, n_in]
        t2 = input_tokens[:, 2, 0].unsqueeze(1)  # [N, 1] - 度数特征
        t3 = input_tokens[:, 3, :]  # [N, n_in]
        
        # 拼接: [T0, T1, T2, T3] -> [N, 3*n_in + 1]
        target_tokens = torch.cat([t0, t1, t2, t3], dim=1)  # [N, 3*n_in + 1]
        
        # Token 重构损失
        token_rec_loss = self.recon_loss_fn(reconstructed_tokens, target_tokens)
        
        # Embedding 空间重构损失
        # 计算原始嵌入与重构后嵌入的距离
        emb_rec_loss = torch.mean(torch.norm(normal_for_generation_emb.squeeze(0) - reencoded_emb, dim=-1))
        
        # 组合损失
        lambda_rec_tok = getattr(self.args, 'lambda_rec_tok', 1.0)
        lambda_rec_emb = getattr(self.args, 'lambda_rec_emb', 1.0)
        loss_rec = lambda_rec_tok * token_rec_loss + lambda_rec_emb * emb_rec_loss
        
        return loss_rec

    def TransformerEncoder(self, tokens):
        """
        输入: tokens [N, 4, D_in]
        输出: emb [1, N, D_emb]
        """
        # 分别投影
        t0 = self.id_projection(tokens[:, 0:1, :])
        t1 = self.res_projection(tokens[:, 1:2, :])
        # 注意：T2 度数特征处理，需要确保输入维度正确
        t2_input = tokens[:, 2:3, 0:1].reshape(-1, 1) # 取标量
        t2 = self.deg_encoder(t2_input).unsqueeze(1)
        t3 = self.res_projection(tokens[:, 3:4, :])
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        
        # 拼接 [CLS, T0, T1, T2, T3]
        emb = torch.cat([cls_tokens, t0, t1, t2, t3], dim=1)
        emb = emb + self.type_embedding
        for layer in self.layers:
            emb, _ = layer(emb)
        
        emb = self.final_ln(emb)
        final_h = emb[:, 0, :] # 取 CLS token
        return final_h.unsqueeze(0) # [1, N, D]
    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False):
        """
        核心逻辑修改：结合重构损失设计 (参考 GGADFormer)
        - 重构学习: 对 T0~T3 进行 token 空间和 embedding 空间的双重重构
        - Context-Ego Mismatching: 生成 outlier embeddings
        """
        # 1. 编码所有节点 (用于测试或无监督特征提取)
        # input_tokens: [N, 4, D]
        emb = self.TransformerEncoder(input_tokens) # [1, N, D]
        
        # 初始化返回变量
        logits = None
        emb_combine = None
        labels = None
        outlier_emb = None
        loss_rec = torch.tensor(0.0, device=self.device)
        loss_uniformity = torch.tensor(0.0, device=self.device)
        
        if train_flag:
            normal_tokens = input_tokens[normal_for_train_idx]
            
            # ========== 重构学习 (参考 GGADFormer 设计) ==========
            # 1. 解码: 将 embedding 解码回 token 空间
            reconstructed_tokens = self.token_decoder(emb.squeeze(0))  # [N, 3*n_in + 1]
            
            # 2. 计算重构误差
            # 提取目标 tokens (与 decoder 输出格式对应)
            t0 = input_tokens[:, 0, :]  # [N, n_in]
            t1 = input_tokens[:, 1, :]  # [N, n_in]
            t2 = input_tokens[:, 2, 0].unsqueeze(1)  # [N, 1] - 度数特征
            t3 = input_tokens[:, 3, :]  # [N, n_in]
            target_tokens = torch.cat([t0, t1, t2, t3], dim=1)  # [N, 3*n_in + 1]
            
            # 重构误差
            reconstruction_error = reconstructed_tokens - target_tokens  # [N, 3*n_in + 1]
            
            # 3. 将重构误差投影到 embedding 维度
            reconstruction_error_proj = self.reconstruction_proj(reconstruction_error)  # [N, embedding_dim]
            
            # 4. 采样正常节点用于生成 outlier
            sample_rate = getattr(args, 'sample_rate', 0.5)
            num_samples = max(1, int(len(normal_for_train_idx) * sample_rate))
            sample_indices = torch.randperm(len(normal_for_train_idx), device=self.device)[:num_samples]
            normal_for_generation_idx = normal_for_train_idx[sample_indices]
            
            normal_for_generation_emb = emb[:, normal_for_generation_idx, :]  # [1, num_samples, D]
            
            # 5. 使用重构误差生成 outlier embedding
            outlier_beta = getattr(args, 'outlier_beta', 1.0)
            outlier_emb = normal_for_generation_emb.squeeze(0) + outlier_beta * reconstruction_error_proj[normal_for_generation_idx]
            
            # 6. 将重构后的 tokens 重新编码为 embedding (用于 embedding 空间重构损失)
            # 需要将 reconstructed_tokens 转换回 [N, 4, D] 格式
            N = reconstructed_tokens.size(0)
            
            rec_t0 = reconstructed_tokens[:, :self.n_in].unsqueeze(1)  # [N, 1, n_in]
            rec_t1 = reconstructed_tokens[:, self.n_in:2*self.n_in].unsqueeze(1)  # [N, 1, n_in]
            rec_t2 = reconstructed_tokens[:, 2*self.n_in:2*self.n_in+1].unsqueeze(1)  # [N, 1, 1]
            rec_t3 = reconstructed_tokens[:, 2*self.n_in+1:].unsqueeze(1)  # [N, 1, n_in]
            
            # 拼接重构后的 tokens
            reconstructed_tokens_vector = torch.zeros(N, 4, self.n_in, device=self.device)
            reconstructed_tokens_vector[:, 0, :] = rec_t0.squeeze(1)
            reconstructed_tokens_vector[:, 1, :] = rec_t1.squeeze(1)
            reconstructed_tokens_vector[:, 2, 0] = rec_t2.squeeze()  # 度数特征
            reconstructed_tokens_vector[:, 3, :] = rec_t3.squeeze(1)
            
            # 重新编码 (detach 以避免影响梯度)
            reencoded_emb = self.TransformerEncoder(reconstructed_tokens_vector)[:, normal_for_generation_idx, :].detach().squeeze(0)
            
            # 7. 计算重构损失
            loss_rec = self.compute_rec_loss(
                input_tokens, reconstructed_tokens, 
                normal_for_generation_emb, reencoded_emb, 
                normal_for_generation_idx
            )
            
            # ========== Context-Ego Mismatching (保留原有逻辑) ==========
            perm_idx = torch.randperm(normal_tokens.size(0), device=self.device)
            mismatched_tokens = normal_tokens.clone()
            mismatched_tokens[:, 1:, :] = normal_tokens[perm_idx, 1:, :]
            mismatched_outlier_emb = self.TransformerEncoder(mismatched_tokens).squeeze(0)
            
            # 合并两种 outlier embedding
            normal_emb = emb[:, normal_for_train_idx, :].squeeze(0)
            outlier_emb = torch.cat([outlier_emb, mismatched_outlier_emb], dim=0)  # 直接覆盖 outlier_emb
            
            # InfoNCE 均匀性损失
            uniformity_loss = self.compute_infoNCE_uniformity_loss(emb, normal_for_train_idx, args)
            loss_uniformity = uniformity_loss
            
            emb_combine = torch.cat((normal_emb, outlier_emb), dim=0)
            
            # 保留 LayerNorm (控制 Logit 爆炸)
            f_1 = self.act(self.ln1(self.fc1(emb_combine)))
            f_2 = self.act(self.ln2(self.fc2(f_1)))
            logits = self.fc3(f_2)
        else:
            # 测试模式：对所有节点进行预测
            f_1 = self.act(self.fc1(emb.squeeze(0)))
            f_2 = self.act(self.fc2(f_1))
            logits = self.fc3(f_2) # [N, 1]
            labels = None
            emb_combine = emb.squeeze(0)
            
            # eval 模式下，如果提供了 normal_for_train_idx，计算 uniformity_loss 和 rec_loss
            if normal_for_train_idx is not None and len(normal_for_train_idx) > 1:
                loss_uniformity = self.compute_infoNCE_uniformity_loss(emb, normal_for_train_idx, args)
                
                # 计算重构损失
                # 1. 解码: 将 embedding 解码回 token 空间
                reconstructed_tokens = self.token_decoder(emb.squeeze(0))  # [N, 3*n_in + 1]
                
                # 2. 提取目标 tokens
                t0 = input_tokens[:, 0, :]  # [N, n_in]
                t1 = input_tokens[:, 1, :]  # [N, n_in]
                t2 = input_tokens[:, 2, 0].unsqueeze(1)  # [N, 1] - 度数特征
                t3 = input_tokens[:, 3, :]  # [N, n_in]
                target_tokens = torch.cat([t0, t1, t2, t3], dim=1)  # [N, 3*n_in + 1]
                
                # 3. 计算 Token 空间重构损失
                token_rec_loss = self.recon_loss_fn(reconstructed_tokens, target_tokens)
                
                # 4. 计算 Embedding 空间重构损失
                # 将重构后的 tokens 重新编码
                N = reconstructed_tokens.size(0)
                rec_t0 = reconstructed_tokens[:, :self.n_in].unsqueeze(1)
                rec_t1 = reconstructed_tokens[:, self.n_in:2*self.n_in].unsqueeze(1)
                rec_t2 = reconstructed_tokens[:, 2*self.n_in:2*self.n_in+1].unsqueeze(1)
                rec_t3 = reconstructed_tokens[:, 2*self.n_in+1:].unsqueeze(1)
                
                reconstructed_tokens_vector = torch.zeros(N, 4, self.n_in, device=self.device)
                reconstructed_tokens_vector[:, 0, :] = rec_t0.squeeze(1)
                reconstructed_tokens_vector[:, 1, :] = rec_t1.squeeze(1)
                reconstructed_tokens_vector[:, 2, 0] = rec_t2.squeeze()
                reconstructed_tokens_vector[:, 3, :] = rec_t3.squeeze(1)
                
                # 重新编码
                with torch.no_grad():
                    reencoded_emb = self.TransformerEncoder(reconstructed_tokens_vector).squeeze(0)
                
                # 计算 embedding 重构损失
                emb_rec_loss = torch.mean(torch.norm(emb.squeeze(0) - reencoded_emb, dim=-1))
                
                # 组合损失
                lambda_rec_tok = getattr(args, 'lambda_rec_tok', 1.0)
                lambda_rec_emb = getattr(args, 'lambda_rec_emb', 1.0)
                loss_rec = lambda_rec_tok * token_rec_loss + lambda_rec_emb * emb_rec_loss
        
        # 返回接口保持一致性
        return emb, emb_combine, logits, outlier_emb, None, loss_rec, loss_uniformity
