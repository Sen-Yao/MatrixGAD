import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math

from check_gpu_memory import print_gpu_memory_usage, print_tensor_memory, clear_gpu_memory

from playground import check_token_collapse

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



class PromptGAD(nn.Module):
    def __init__(self, n_in, n_h, activation, args):
        super(PromptGAD, self).__init__()

        # 设置设备
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
        self.args = args

        # 设置批次大小
        self.batchsize = getattr(args, 'batchsize', None)
        
        self.gcn1 = GCN(n_in, n_in, activation)
        self.gcn2 = GCN(n_in, n_in, activation)

        # 使用 n_in 替代 n_h 作为线性层的输入维度
        self.fc1 = nn.Linear(n_in, int(n_in / 2), bias=False)
        self.fc2 = nn.Linear(int(n_in / 2), int(n_in / 4), bias=False)
        self.fc3 = nn.Linear(int(n_in / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_in, n_in, bias=False)
        self.act = nn.ReLU()

        self.n_in = n_in

        # Graph Transformer
        encoders = [EncoderLayer(self.n_in, args.GT_ffn_dim, args.GT_dropout, args.GT_attention_dropout, args.GT_num_heads)
                    for _ in range(args.GT_num_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(self.n_in)
        self.read_out = nn.Linear(self.n_in, self.n_in)

        # 可学习的离散 Prompt Token (频域视角)
        # M 个 prompt，每个维度为 n_in (保持与输入token相同的维度)
        self.num_prompts = getattr(args, 'num_prompts', 8)
        self.prompts = nn.Parameter(torch.randn(1, self.num_prompts, self.n_in))

        # Signed Attention 的投影层，用于计算方向（正负号）
        self.sign_q = nn.Linear(self.n_in, self.n_in)
        self.sign_k = nn.Linear(self.n_in, self.n_in)

        # Token 解码器：从 embedding 重构 prompt_tokens
        # 输出维度: M * n_in (M 个 prompt tokens，每个维度为 n_in)
        self.token_decoder = nn.Sequential(
            nn.Linear(self.n_in, self.n_in),
            nn.ReLU(),
            nn.Linear(self.n_in, self.num_prompts * self.n_in)
        )

        # 重构损失函数
        self.recon_loss_fn = nn.MSELoss()

        # 投影层：将重构误差从 M*n_in 维度投影到 n_in 维度
        self.reconstruction_proj = nn.Sequential(
            nn.Linear(self.num_prompts * self.n_in, self.n_in),
            nn.ReLU(),
            nn.Linear(self.n_in, self.n_in)
        )

        # LayerNorm for prompt_tokens
        self.prompt_tokens_ln = nn.LayerNorm(self.n_in)

        # 可学习的 CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.n_in))

        # 将模型移动到指定设备
        self.to(self.device)

    def tokenizer(self, raw_tokens, tokenizer_temp=None, partial_temp=None):
        """
        使用可学习的 Prompt Token 对原始 tokens 进行交叉注意力查询，
        提取 M 种不同的"频域视角"特征。

        Args:
            raw_tokens: 原 NAGphormer 特征 [Batch, num_hops, d_model]
                        注意：这里的 d_model 是 n_in (输入特征维度)
            tokenizer_temp: 温度参数，默认为 self.args.tokenizer_temp
            partial_temp: 如果不为 None，则只对前 pp_k // 2 个 token 使用此温度，
                         后面的 token 使用 tokenizer_temp。这样可以实现部分升温。

        Returns:
            prompt_tokens: 提取的新视角 Token [Batch, M, n_in]
            attn_weights: 注意力权重 [Batch, M, num_hops]，用于计算正交损失
        """
        # 如果没有传入 tokenizer_temp，则使用默认值
        if tokenizer_temp is None:
            tokenizer_temp = self.args.tokenizer_temp
            
        B = raw_tokens.size(0)
        M = self.num_prompts
        d_model = self.n_in  # 使用 n_in 而不是 args.embedding_dim
        num_hops = raw_tokens.size(1)  # num_hops = pp_k + 1

        # 扩展 Prompt 匹配 Batch Size
        Q = self.prompts.expand(B, -1, -1)  # [B, M, n_in]
        K = raw_tokens  # [B, num_hops, n_in] - 直接使用输入 tokens，不进行投影
        V = raw_tokens  # [B, num_hops, n_in] - 直接使用输入 tokens，不进行投影

        # 1. 计算重要性 (Magnitude) - 传统的 Softmax
        # score: [B, M, num_hops]
        score_mag = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_model)
        
        # 2. 计算方向/突变 (Sign) - 打破低通滤波诅咒的关键！
        # range: [-1, 1]
        score_sign = torch.matmul(self.sign_q(Q), self.sign_k(K).transpose(-1, -2))
        
        # ========== 原来的逻辑 (先注释保留) ==========
        # magnitude = F.softmax(score_mag / tokenizer_temp, dim=-1)
        # sign = torch.tanh(score_sign / tokenizer_temp)
        # ===========================================
        
        # ========== 新逻辑：支持部分升温 ==========
        if partial_temp is not None:
            # 只对前 pp_k // 2 个 token 使用高温，后面的使用正常温度
            # 注意：num_hops = pp_k + 1，所以 pp_k = num_hops - 1
            pp_k = num_hops - 1
            partial_idx = pp_k // 2  # 前 pp_k // 2 个 token（不包括 0-hop？或者包括？按照任务描述是前 pp_k // 2 个）
            
            # 创建温度掩码：前 partial_idx + 1 个位置（包括 0-hop）使用 partial_temp，后面的使用 tokenizer_temp
            # 或者按照任务描述：仅升温前 pp_k // 2 个 token（可能是指 hop 1 到 hop pp_k//2）
            # 这里按照任务描述：前 pp_k // 2 个 token（从 1-hop 开始算）
            temp_mask = torch.ones(B, M, num_hops, device=raw_tokens.device) * tokenizer_temp
            
            # 前 partial_idx 个 hop（1-hop 到 partial_idx-hop）使用高温
            # 注意：索引 0 是 0-hop，索引 1 是 1-hop，...，索引 pp_k 是 pp_k-hop
            if partial_idx > 0:
                temp_mask[:, :, 1:partial_idx+1] = partial_temp
            
            # 应用不同的温度
            magnitude = F.softmax(score_mag / temp_mask, dim=-1)
            sign = torch.tanh(score_sign / temp_mask)
        else:
            # 原来的逻辑：所有 token 使用相同温度
            magnitude = F.softmax(score_mag / tokenizer_temp, dim=-1)
            sign = torch.tanh(score_sign / tokenizer_temp)
        # ===========================================

        # 3. 合成动态滤波器权重
        attn_weights = magnitude * sign  # [B, M, num_hops]

        # 4. 提取出全新视角的 Token!
        # prompt_tokens 相当于自适应地提取不同频域视角
        prompt_tokens = torch.matmul(attn_weights, V)  # [B, M, n_in]

        # 对 prompt_tokens 应用 LayerNorm
        prompt_tokens = self.prompt_tokens_ln(prompt_tokens)

        # 我们把 attn_weights 一起返回，为了算正交 Loss
        return prompt_tokens, attn_weights

    def compute_filter_orthogonal_loss(self, attn_weights):
        """
        计算正交损失，惩罚不同 Prompt 学出相同的 hop 组合权重

        Args:
            attn_weights: [B, M, num_hops]

        Returns:
            ortho_loss: 正交损失值
        """
        B, M, hops = attn_weights.shape

        # 归一化每个 Prompt 的滤波器权重
        norms = torch.norm(attn_weights, dim=-1, keepdim=True) + 1e-8
        normalized_weights = attn_weights / norms

        # 计算两两 Prompt 权重的余弦相似度 [B, M, M]
        cos_sim = torch.matmul(normalized_weights, normalized_weights.transpose(-1, -2))

        # 扣除对角线 (自己和自己必然是 1)，只算非对角线的相似度绝对值
        mask = ~torch.eye(M, dtype=torch.bool, device=attn_weights.device)
        ortho_loss = cos_sim[:, mask].abs().mean()

        return ortho_loss
    
    def TransformerEncoder(self, tokens):
        """
        Inputs:
            - tokens: 输入节点的 tokens 序列，形状 [batch_size, pp_k+1, n_in]
        Outputs:
            - emb: 输入节点的编码结果，形状 [1, batch_size, n_in]
        """

        emb = tokens  # 直接使用输入tokens，不进行投影
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
        # emb: [1, N, n_in]
        emb = torch.bmm(attention_scores.unsqueeze(1), emb).squeeze(1).unsqueeze(0)

        return emb

    def TransformerEncoderWithTokens(self, tokens):
        """
        处理已经投影过的 tokens（包含原始 tokens 和 prompt tokens 的组合）

        Inputs:
            - tokens: 已经投影过的 tokens 序列，形状 [batch_size, pp_k+1 + M, n_in]
        Outputs:
            - emb: 输入节点的编码结果，形状 [1, batch_size, n_in]
        """
        for i, l in enumerate(self.layers):
            tokens, current_attention_weights = self.layers[i](tokens)
            if i == len(self.layers) - 1:  # 拿到最后一层的注意力
                attention_weights = current_attention_weights
                # 聚合多头注意力
                agg_attention_weights = torch.mean(attention_weights, dim=1)
                # agg_attention_weights: [N, pp_k+1 + M, pp_k+1 + M]
        emb = self.final_ln(tokens)

        # attention_scores: [N, pp_k+1 + M]
        # 我们关注前 pp_k+1 个位置（原始 tokens）的第一个位置（0-hop）对所有位置的注意力
        attention_scores = agg_attention_weights[:, 0, :]

        # 基于 attention_scores 进行池化，得到最终编码结果
        # emb: [1, N, n_in]
        emb = torch.bmm(attention_scores.unsqueeze(1), emb).squeeze(1).unsqueeze(0)

        return emb

    def TransformerEncoderWithPromptTokens(self, tokens):
        """
        处理只有 prompt_tokens 的情况（不包含原始 tokens）

        Inputs:
            - tokens: 已经投影过的 prompt_tokens 序列，形状 [batch_size, M, n_in]
        Outputs:
            - emb: 输入节点的编码结果，形状 [1, batch_size, n_in]
        """
        for i, l in enumerate(self.layers):
            tokens, current_attention_weights = self.layers[i](tokens)
            if i == len(self.layers) - 1:  # 拿到最后一层的注意力
                attention_weights = current_attention_weights
                # 聚合多头注意力
                agg_attention_weights = torch.mean(attention_weights, dim=1)
                # agg_attention_weights: [N, M, M]
        emb = self.final_ln(tokens)

        # 由于只有 M 个 prompt_tokens，没有特定的 0-hop 位置
        # 我们对所有位置的注意力进行平均池化，得到统一的注意力分数
        # attention_scores: [N, M]，对每一行进行平均
        attention_scores = torch.mean(agg_attention_weights, dim=1)

        # 基于 attention_scores 进行池化，得到最终编码结果
        # emb: [1, N, n_in]
        emb = torch.bmm(attention_scores.unsqueeze(1), emb).squeeze(1).unsqueeze(0)

        return emb

    def TransformerEncoderWithCLS(self, tokens):
        """
        使用 CLS token 处理 tokens，CLS token 经过多层更新后直接作为输出

        Inputs:
            - tokens: 已经投影过的 prompt_tokens 序列，形状 [batch_size, M, n_in]
        Outputs:
            - emb: CLS token 的编码结果，形状 [1, batch_size, n_in]
        """
        batch_size = tokens.size(0)
        
        # 将 CLS token 扩展到 batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, n_in]
        
        # 拼接 CLS token 和 prompt_tokens: [batch_size, 1+M, n_in]
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # 经过所有 Transformer 层
        for i, l in enumerate(self.layers):
            tokens, _ = self.layers[i](tokens)
        
        # 应用 final LayerNorm
        emb = self.final_ln(tokens)
        
        # 直接取 CLS token（第一个位置）作为输出
        cls_output = emb[:, 0, :]  # [batch_size, n_in]
        
        # 调整形状为 [1, batch_size, n_in]
        return cls_output.unsqueeze(0)

    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False, return_attn_weights=False):
        """
        Args:
            return_attn_weights: 如果为 True，则额外返回注意力权重用于诊断
        """

        # input_tokens: (N, args.pp_k+1, d)

        # 使用 tokenizer 提取 M 种不同的"频域视角"特征
        # prompt_tokens: [N, M, n_in]
        # prompt_attn_weights: [N, M, pp_k+1]
        prompt_tokens, prompt_attn_weights = self.tokenizer(input_tokens, getattr(self.args, 'tokenizer_temp', 1.0))

        # 计算正交损失
        ortho_loss = self.compute_filter_orthogonal_loss(prompt_attn_weights)

        # 使用 CLS token 进行编码
        # 新视角 tokens: [N, M, n_in]
        # CLS token 与 prompt_tokens 一起经过 Transformer 层更新
        emb = self.TransformerEncoderWithCLS(prompt_tokens)
        emb = F.normalize(emb, p=2, dim=-1)
        # 生成全局中心点
        h_mean = torch.mean(emb, dim=1, keepdim=True)

        outlier_emb = None
        emb_combine = None
        noised_normal_for_generation_emb = None
        reconstruction_error_proj = None  # 重构误差向量 R_i
        
        # 初始化用于诊断的 token 存储
        original_prompt_tokens = None  # 原始 tokenizer 输出的 tokens
        reconstructed_tokens = None  # 重构后的 tokens

        gna_loss = torch.tensor(0.0, device=emb.device)
        proj_loss = torch.tensor(0.0, device=emb.device)
        uniformity_loss = torch.tensor(0.0, device=emb.device)
        loss_ring = torch.tensor(0.0, device=emb.device)
        con_loss = torch.tensor(0.0, device=emb.device)
        loss_rec = torch.tensor(0.0, device=emb.device)
        
        # 在训练或非训练模式下都计算重构 tokens（用于诊断）
        # reconstructed_tokens: [num_nodes, M * n_in]
        reconstructed_tokens = self.token_decoder(emb).squeeze(0)
        original_prompt_tokens = prompt_tokens  # 保存原始的 prompt_tokens
        
        if train_flag:
            # start_time = time.time()
            # 高效重排
            perm = torch.randperm(normal_for_train_idx.size(0), device=normal_for_train_idx.device)
            normal_for_train_idx = normal_for_train_idx[perm]
            # print(f"time for shuffle:{time.time() - start_time}")
            normal_for_generation_idx = normal_for_train_idx[: int(len(normal_for_train_idx) * args.sample_rate)]            
            normal_for_generation_emb = emb[:, normal_for_generation_idx, :]

            # ==================== 新的伪异常生成逻辑 ====================
            # 被选取用于生成伪异常的那些正常节点，在tokenizer的过程中，随机mask掉hallucination_prompt_ratio倍数的prompt使用高温生成
            # 其余prompt使用正常温度，然后这些伪异常prompt token正常过transformer编码器，其结果作为伪异常

            # 获取用于生成伪异常的正常节点的原始token序列
            batch_normal_tokens_for_generation = input_tokens[normal_for_generation_idx, :, :]  # [num_normal_gen, num_hops+1, d]

            # 计算增温后的tokenizer温度
            hallucinated_temp = getattr(self.args, 'tokenizer_temp', 1.0) * getattr(self.args, 'tokenizer_hallucination_ratio', 2.0)
            normal_temp = getattr(self.args, 'tokenizer_temp', 1.0)

            # ========== 原来的逻辑 (先注释保留) ==========
            # # 使用正常温度和高温分别处理tokens，并添加detach()防止梯度回传到tokenizer和prompts
            # normal_prompt_tokens, _ = self.tokenizer(batch_normal_tokens_for_generation, normal_temp)
            # normal_prompt_tokens = normal_prompt_tokens.detach()
            # hallucinated_prompt_tokens, _ = self.tokenizer(batch_normal_tokens_for_generation, hallucinated_temp)
            # hallucinated_prompt_tokens = hallucinated_prompt_tokens.detach()
            # 
            # # 创建随机mask，决定哪些prompt使用高温
            # B, M, d_model = normal_prompt_tokens.shape
            # mask = torch.rand(B, M, device=batch_normal_tokens_for_generation.device) < getattr(self.args, 'hallucination_prompt_ratio', 0.2)
            # 
            # # 根据mask混合正常和高温处理的prompt
            # mixed_prompt_tokens = torch.where(mask.unsqueeze(-1), hallucinated_prompt_tokens, normal_prompt_tokens)
            # ===========================================
            
            # ========== 新逻辑：仅对前 pp_k // 2 个 token 进行升温 ==========
            # 使用正常温度处理所有tokens
            normal_prompt_tokens, _ = self.tokenizer(batch_normal_tokens_for_generation, normal_temp)
            normal_prompt_tokens = normal_prompt_tokens.detach()
            
            # 使用部分升温：前 pp_k // 2 个 token 使用高温，后面的使用正常温度
            hallucinated_prompt_tokens, _ = self.tokenizer(batch_normal_tokens_for_generation, normal_temp, partial_temp=hallucinated_temp)
            hallucinated_prompt_tokens = hallucinated_prompt_tokens.detach()
            
            # 创建随机mask，决定哪些prompt使用部分升温
            B, M, d_model = normal_prompt_tokens.shape
            mask = torch.rand(B, M, device=batch_normal_tokens_for_generation.device) < getattr(self.args, 'hallucination_prompt_ratio', 0.2)
            
            # 根据mask混合正常和部分升温处理的prompt
            mixed_prompt_tokens = torch.where(mask.unsqueeze(-1), hallucinated_prompt_tokens, normal_prompt_tokens)
            # ===========================================

            # 使用CLS token处理混合后的prompt_tokens作为伪异常
            hallucinated_emb = self.TransformerEncoderWithCLS(mixed_prompt_tokens)
            hallucinated_emb = F.normalize(hallucinated_emb, p=2, dim=-1)

            # 将混合处理后的嵌入作为伪异常样本
            outlier_emb = hallucinated_emb.squeeze(0)  # [num_normal_gen, n_in]

            # ===========================================================

            # 计算重构损失：目标是 prompt_tokens
            loss_rec = self.compute_rec_loss(prompt_tokens, reconstructed_tokens, normal_for_generation_emb, normal_for_generation_idx)

            # 使用原始正常节点嵌入和生成的伪异常嵌入进行对比学习
            emb_combine = torch.cat((emb[:, normal_for_train_idx, :], torch.unsqueeze(outlier_emb, 0)), 1)
            emb_combine = F.normalize(emb_combine, p=2, dim=-1)
            # 计算 InfoNCE 均匀性损失，只计算正常节点间的排斥力
            uniformity_loss = self.compute_infoNCE_uniformity_loss(emb, normal_for_train_idx, args)

            f_1 = self.fc1(emb_combine)
        else:
            # 在非训练模式下也计算重构误差向量（用于诊断）
            reconstructed_tokens = self.token_decoder(emb).squeeze(0)
            target_tokens = prompt_tokens.view(-1, self.num_prompts * self.n_in)
            reconstruction_error = reconstructed_tokens - target_tokens
            # Project reconstruction error to n_in dimension for all nodes
            reconstruction_error_proj = self.reconstruction_proj(reconstruction_error)
            
           
            f_1 = self.fc1(emb)
        f_1 = self.act(f_1)
        f_2 = self.fc2(f_1)
        f_2 = self.act(f_2)
        logits = self.fc3(f_2)
        emb = emb.clone()

        # 返回正交损失、重构误差向量、均匀性损失以及用于诊断的原始和重构 tokens
        if return_attn_weights:
            return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring, ortho_loss, reconstruction_error_proj, uniformity_loss, original_prompt_tokens, reconstructed_tokens, prompt_attn_weights
        else:
            return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring, ortho_loss, reconstruction_error_proj, uniformity_loss, original_prompt_tokens, reconstructed_tokens

    def compute_rec_loss(self, prompt_tokens, reconstructed_tokens, normal_for_generation_emb, normal_for_generation_idx):
        """
        计算 Token 空间的重构损失
        重构目标是 prompt_tokens (Prompt Token 提取的频域视角)
        使用 MSE 损失：基于重构前后的绝对值差异
        
        Args:
            prompt_tokens: Prompt Token 提取的新视角 Token [N, M, n_in]
            reconstructed_tokens: 经过解码器重构的 Token 序列 [N, M * n_in]
            normal_for_generation_emb: 正常节点的嵌入
            normal_for_generation_idx: 用于生成异常的正常节点索引
        Returns:
            loss_rec: 重构损失值
        """
        # 计算重构损失：目标是 prompt_tokens
        # prompt_tokens: [N, M, n_in] -> flatten: [N, M * n_in]
        target_tokens = prompt_tokens.view(-1, self.num_prompts * self.n_in)
        
        # MSE 损失：基于绝对值差异
        # 计算每个样本的均方误差并平均
        token_rec_loss = F.mse_loss(reconstructed_tokens, target_tokens)
        
        return token_rec_loss


    # InfoNCE uniformity loss - 推开不同正常节点间的距离
    def compute_infoNCE_uniformity_loss(self, emb, normal_for_train_idx, args):
        """
        计算InfoNCE均匀性损失，推开不同正常节点在嵌入空间中的距离
        Args:
            emb: [1, N, n_in] - 所有节点的嵌入表征
            normal_for_train_idx: 训练时使用的正常节点索引
            args: 包含GNA_temp等超参数的配置
        Returns:
            uniformity_loss: InfoNCE均匀性损失
        """
        # 提取正常节点的嵌入: [num_normal, n_in]
        normal_emb = emb[0, normal_for_train_idx, :]  # [num_normal, n_in]
        num_normal = normal_emb.size(0)
        
        # 调试信息：打印正常节点数量
        # print(f"[DEBUG] num_normal in batch: {num_normal}")
        
        # 如果正常节点数量少于2，无法计算InfoNCE损失
        if num_normal < 2:
            # print(f"[WARNING] num_normal={num_normal} < 2, returning 0.0 for uniformity_loss")
            return torch.tensor(0.0, device=emb.device)
        
        # L2 归一化，便于计算余弦相似度
        normal_emb_norm = F.normalize(normal_emb, p=2, dim=1)  # [num_normal, n_in]
        
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
