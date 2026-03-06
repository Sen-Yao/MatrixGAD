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
        self.token_length = 6
        encoders = [EncoderLayer(args.embedding_dim, args.GT_ffn_dim, args.GT_dropout, args.GT_attention_dropout, args.GT_num_heads)
                    for _ in range(args.GT_num_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.embedding_dim)
        # Embeddings
        self.type_embedding = nn.Parameter(torch.zeros(1, self.token_length + 1, args.embedding_dim))
        nn.init.xavier_uniform_(self.type_embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embedding_dim))
        # --- 4. 移除不再需要的重构模块 ---
        # 删除了 token_decoder, reconstruction_proj 等
        self.to(self.device)
    def compute_infoNCE_uniformity_loss(self, emb, normal_for_train_idx, args):
        """
        计算InfoNCE均匀性损失，推开不同正常节点在嵌入空间中的距离
        Args:
            emb: [1, N, embedding_dim] - 所有节点的嵌入表征
            normal_for_train_idx: 训练时使用的正常节点索引
            args: 包含GNA_temp等超参数的配置
        Returns:
            uniformity_loss: InfoNCE均匀性损失
        """
        # 提取正常节点的嵌入: [num_normal, embedding_dim]
        normal_emb = emb[0, normal_for_train_idx, :]  # [num_normal, embedding_dim]
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

    def TransformerEncoder(self, tokens):
        """
        输入: tokens [N, 6, D_in]
        输出: emb [1, N, D_emb]
        """
        # 分别投影
        t0 = self.id_projection(tokens[:, 0:1, :])
        t1 = self.res_projection(tokens[:, 1:2, :])
        # 注意：T2 度数特征处理，需要确保输入维度正确
        t2_input = tokens[:, 2:3, 0:1].reshape(-1, 1) # 取标量
        t2 = self.deg_encoder(t2_input).unsqueeze(1)
        t3 = self.res_projection(tokens[:, 3:4, :])
        t4 = self.res_projection(tokens[:, 4:5, :])
        t5 = self.res_projection(tokens[:, 5:6, :])
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        
        # 拼接 [CLS, T0, T1, T2, T3, T4, T5]
        emb = torch.cat([cls_tokens, t0, t1, t2, t3, t4, t5], dim=1)
        emb = emb + self.type_embedding
        for layer in self.layers:
            emb, _ = layer(emb)
        
        emb = self.final_ln(emb)
        final_h = emb[:, 0, :] # 取 CLS token
        return final_h.unsqueeze(0) # [1, N, D]
    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False):
        """
        核心逻辑修改：Idea 3 - Context-Ego Mismatching
        """
        # 1. 编码所有节点 (用于测试或无监督特征提取)
        # input_tokens: [N, 6, D]
        emb = self.TransformerEncoder(input_tokens) # [1, N, D]
        # 初始化返回变量
        logits = None
        emb_combine = None
        labels = None
        # 占位符，保持接口兼容
        outlier_emb = None
        loss_rec = torch.tensor(0.0, device=self.device)
        loss_uniformity = torch.tensor(0.0, device=self.device)
        if train_flag:
            normal_tokens = input_tokens[normal_for_train_idx]
            
            # 改进 A：保留单次错位，但让位移量是 Batch 级别的随机数 
            # 意义：不增加架构复杂度的前提下，避免模型记住 shift=1 的固定相差距离
            # 让每个节点的拓扑特征随机匹配给其他节点，打破批次内的一致性错位偏移
            perm_idx = torch.randperm(normal_tokens.size(0), device=self.device)
            # rolled_idx = torch.roll(torch.arange(normal_tokens.size(0)), shifts=shift).to(self.device)
            mismatched_tokens = normal_tokens.clone()
            mismatched_tokens[:, 1:, :] = normal_tokens[perm_idx, 1:, :]
            outlier_emb = self.TransformerEncoder(mismatched_tokens).squeeze(0)
            normal_emb = emb[:, normal_for_train_idx, :].squeeze(0) # 直接复用已编码的特征省显存
            # 改进 B：激活你写好的 InfoNCE 均匀性损失 (核心解药)
            # 强制要求正常节点之间保持一定的夹角，拒绝 Cos_Sim 走向 1.0!
            uniformity_loss = self.compute_infoNCE_uniformity_loss(emb, normal_for_train_idx, args)
            # 将 uniformity_loss 传递给损失接口
            loss_uniformity = uniformity_loss
            loss_rec = torch.tensor(0.0, device=self.device)
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
            # 测试时不需要 labels 和 emb_combine
            labels = None
            emb_combine = emb.squeeze(0)
        # 返回接口保持一致性
        # 注意：外部训练循环需自行计算 BCELoss(logits, labels)
        return emb, emb_combine, logits, outlier_emb, None, loss_rec, loss_uniformity
