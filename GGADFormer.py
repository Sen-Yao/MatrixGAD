import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

from check_gpu_memory import print_gpu_memory_usage, print_tensor_memory, clear_gpu_memory

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

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits



class GGADFormer(nn.Module):
    def __init__(self, n_in, n_h, activation, args):
        super(GGADFormer, self).__init__()

        # 设置设备
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
        self.args = args

        # 设置批次大小
        self.batchsize = getattr(args, 'batchsize', None)
        
        self.gcn1 = GCN(args.embedding_dim, args.embedding_dim, activation)
        self.gcn2 = GCN(args.embedding_dim, args.embedding_dim, activation)

        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.act = nn.ReLU()

        self.n_in = n_in

        # Graph Transformer
        encoders = [EncoderLayer(args.embedding_dim, args.GT_ffn_dim, args.GT_dropout, args.GT_attention_dropout, args.GT_num_heads)
                    for _ in range(args.GT_num_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.embedding_dim)
        self.read_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.token_projection = nn.Linear(self.n_in, args.embedding_dim)

        self.token_decoder = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, (args.pp_k+1) * self.n_in)
        )

        # 重构损失函数
        self.recon_loss_fn = nn.MSELoss()

        # 投影层：将重构误差从2*n_in维度投影到embedding_dim维度
        self.reconstruction_proj = nn.Sequential(
            nn.Linear((args.pp_k+1) * n_in, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, args.embedding_dim)
        )

        # 将模型移动到指定设备
        self.to(self.device)
    
    def TransformerEncoder(self, tokens):
        """
        Inputs:
            - tokens: 输入节点的 tokens 序列，形状 [batch_size, pp_k+1, feature_dim]
        Outputs:
            - emb: 输入节点的编码结果，形状 [1, batch_size, embedding_dim]
        """
        emb = self.token_projection(tokens)
        for i, l in enumerate(self.layers):
            emb, current_attention_weights = self.layers[i](emb)
            if i == len(self.layers) - 1: # 拿到最后一层的注意力
                attention_weights = current_attention_weights
                # 聚合多头注意力
                agg_attention_weights = torch.mean(attention_weights, dim=1)
                # agg_attention_weights: [N, args.pp_k+1, args.pp_k+1]
        emb = self.final_ln(emb)

        # attention_scores: [N, args.pp_k+1], 表示每个节点的自身特征 (0-hop) 对每个后续 hop 的注意力分数
        attention_scores = agg_attention_weights[:, 0, :]

        # 基于 attention_scores 进行池化，得到最终编码结果
        # emb: [1, N, embedding_dim]
        emb = torch.bmm(attention_scores.unsqueeze(1), emb).squeeze(1).unsqueeze(0)

        return emb

    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False):

        # input_tokens: (N, args.pp_k+1, d)
        emb = self.TransformerEncoder(input_tokens)

        # 生成全局中心点
        h_mean = torch.mean(emb, dim=1, keepdim=True)

        outlier_emb = None
        emb_combine = None
        noised_normal_for_generation_emb = None

        gna_loss = torch.tensor(0.0, device=emb.device)
        proj_loss = torch.tensor(0.0, device=emb.device)
        uniformity_loss = torch.tensor(0.0, device=emb.device)
        loss_ring = torch.tensor(0.0, device=emb.device)
        con_loss = torch.tensor(0.0, device=emb.device)
        loss_rec = torch.tensor(0.0, device=emb.device)
        if train_flag:
            # start_time = time.time()
            # 高效重排
            perm = torch.randperm(normal_for_train_idx.size(0), device=normal_for_train_idx.device)
            normal_for_train_idx = normal_for_train_idx[perm]
            # print(f"time for shuffle:{time.time() - start_time}")
            normal_for_generation_idx = normal_for_train_idx[: int(len(normal_for_train_idx) * args.sample_rate)]            
            normal_for_generation_emb = emb[:, normal_for_generation_idx, :]
            # print(f"time for normal_for_generation_emb:{time.time() - start_time}")
            # Noise
            noise = torch.randn(normal_for_generation_emb.size(), device=self.device) * args.var + args.mean
            noised_normal_for_generation_emb = normal_for_generation_emb + noise
            # print(f"time for noise:{time.time() - start_time}")

            # 重构学习
            reconstructed_tokens = self.token_decoder(emb).squeeze(0)  # [num_nodes, (args.pp_k+1)*n_in]
            reconstruction_error = reconstructed_tokens - input_tokens.view(-1, (args.pp_k+1) * self.n_in)
            # Project reconstruction error to embedding dimension
            reconstruction_error_proj = self.reconstruction_proj(reconstruction_error[normal_for_generation_idx, :])

            # Ablation study:
            if args.ablation_random_dir:
                # 计算原始扰动向量的模长 (Magnitude)
                # dim=1 表示计算每个样本向量的范数，keepdim=True 保持形状为 (Batch, 1) 以便广播
                norms = torch.norm(reconstruction_error_proj, p=2, dim=1, keepdim=True)
                
                # 生成同维度的随机向量 (Random Direction)
                # 从标准正态分布采样
                random_vec = torch.randn_like(reconstruction_error_proj)
                
                # 将随机向量归一化为单位向量 (Unit Vector)
                random_dir = torch.nn.functional.normalize(random_vec, p=2, dim=1)
                
                # 赋予随机方向以原始模长
                reconstruction_error_proj = norms * random_dir

            outlier_emb = normal_for_generation_emb + args.outlier_beta * reconstruction_error_proj
            outlier_emb = outlier_emb.squeeze(0)

            # 中心点对齐损失，鼓励离群点距离全局中心的距离保持在一个 ring 内
            # 计算离群点嵌入与全局中心的距离
            outlier_to_center_dist = torch.norm(outlier_emb - h_mean.squeeze(0), p=2, dim=1)
            # 只有超过 confidence_margin 的距离才会产生损失
            ring_out_range_loss = torch.relu(args.ring_R_min - outlier_to_center_dist)
            ring_in_range_loss = torch.relu(outlier_to_center_dist - args.ring_R_max)

            loss_ring = torch.mean(ring_out_range_loss + ring_in_range_loss)
            # 将重构后的 tokens 再编码为 embedding
            reconstructed_tokens_vector = torch.reshape(reconstructed_tokens, (-1, args.pp_k+1, self.n_in))
            reencoded_emb = self.TransformerEncoder(reconstructed_tokens_vector)[:, normal_for_generation_idx, :].detach().squeeze(0)
            loss_rec = self.compute_rec_loss(input_tokens, reconstructed_tokens, normal_for_generation_emb, reencoded_emb, normal_for_generation_idx)

            emb_combine = torch.cat((emb[:, normal_for_train_idx, :], torch.unsqueeze(outlier_emb, 0)), 1)

            f_1 = self.fc1(emb_combine)
        else:
            f_1 = self.fc1(emb)
        f_1 = self.act(f_1)
        f_2 = self.fc2(f_1)
        f_2 = self.act(f_2)
        logits = self.fc3(f_2)
        emb = emb.clone()

        # gna_loss = torch.tensor(0.0, device=emb.device)
        return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring

    def compute_rec_loss(self, input_tokens, reconstructed_tokens, normal_for_generation_emb, reencoded_emb, normal_for_generation_idx):
        """
        计算 Token 空间和 Embedding 空间的重构损失
        Args:
            input_tokens: 原始采样的 Token 序列
            reconstructed_tokens： 经过解码器重构的 Token 序列
            emb: 第一次编码的嵌入结果
            reencoded_emb: 将重构 Token 序列进行二次编码的嵌入结果
        Returns:
            loss_rec: 重构损失值
        """
        token_rec_loss = self.recon_loss_fn(reconstructed_tokens, input_tokens.view(-1, (self.args.pp_k+1) * self.n_in))
        # 计算距离
        emb_rec_loss = torch.mean(torch.norm(normal_for_generation_emb.squeeze(0) - reencoded_emb, dim=-1))  # [N]
        loss_rec = self.args.lambda_rec_tok * token_rec_loss + self.args.lambda_rec_emb * emb_rec_loss
        return loss_rec


    # InfoNCE uniformity loss - 推开不同正常节点间的距离
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
        uniformity_loss = log_sum_exp_values.mean()
        
        return uniformity_loss