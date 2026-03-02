import torch
import math
import torch.nn as nn
from torch.nn import functional as F


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


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
        self.scale = att_size ** -0.5

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


class SGT(nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        hidden_dim,
        n_class,
        num_heads,
        ffn_dim,
        dropout_rate,
        attention_dropout_rate,
        args):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.node_emb = nn.Linear(self.input_dim,  self.hidden_dim)
        self.hop_emb = nn.Linear(self.input_dim,  self.hidden_dim)


        
        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))



        self.Linear_out = nn.Linear(int(self.hidden_dim / 2), 1)


        self.fusion_layer = nn.Linear(2 * self.hidden_dim, 1)


        self.args = args

        self.CEloss  = nn.CrossEntropyLoss()

        self.BCEloss  = nn.BCELoss()

        self.l2loss  = nn.MSELoss(reduction='mean')

        self.fc1 = nn.Linear(args.embedding_dim, int(args.embedding_dim / 2), bias=False)
        self.fc2 = nn.Linear(int(args.embedding_dim / 2), int(args.embedding_dim / 4), bias=False)
        self.fc3 = nn.Linear(int(args.embedding_dim / 4), 1, bias=False)
        self.fc4 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=False)
        self.act = nn.ReLU()


        self.cls_token_node = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.cls_token_hop = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.apply(lambda module: init_params(module, n_layers=n_layers))
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    def forward(self, batched_input_tokens, adj, normal_for_generation_idx, normal_for_train_idx, train_flag, args, sparse=False):
        
        # print(batched_data.shape)

        split_list = [self.args.sample_num_p + 1, self.args.sample_num_n + 1, self.args.sample_num_p + 1, self.args.sample_num_n + 1]
        # print(split_list)
        #正负样本的特征分离
        out = torch.split(batched_input_tokens, split_list, dim=1)


        node_p = out[0] # [b_size, (1+pk), d]
        node_n = out[1] # [b_size, (1+pk), d]

        hop_p = out[2] # [b_size, (1+pk), d]
        hop_n = out[3] # [b_size, (1+pk), d]

        node_p = self.node_emb(node_p)
        node_n = self.node_emb(node_n)

        hop_p = self.hop_emb(hop_p)
        hop_n = self.hop_emb(hop_n)

        
        if self.args.sample_num_n > 0:

            node_n = torch.split(node_n, [1, self.args.sample_num_n], dim=1)[1]
            node_n = torch.concat((self.cls_token_node.expand(batched_input_tokens.shape[0], -1, -1), node_n), dim=1)
            
            hop_n = torch.split(node_n, [1, self.args.sample_num_n], dim=1)[1]
            hop_n = torch.concat((self.cls_token_hop.expand(batched_input_tokens.shape[0], -1, -1), hop_n), dim=1)
            

        #交叉学习

        for i, l in enumerate(self.layers):

            node_n, node_n_attention_weights = self.layers[i](node_n)
            node_p, node_p_attention_weights = self.layers[i](node_p)

            hop_n, hop_n_attention_weights = self.layers[i](hop_n)
            hop_p, hop_p_attention_weights = self.layers[i](hop_p)
            if i == len(self.layers) - 1: # 拿到最后一层的注意力
                # node_n_attention_weights: [num_nodes, sample_num_n+1, sample_num_n+1]
                # node_p_attention_weights: [num_nodes, sample_num_p+1, sample_num_p+1]
                # hop_n_attention_weights: [num_nodes, sample_num_n+1, sample_num_n+1]
                # hop_p_attention_weights: [num_nodes, sample_num_p+1, sample_num_p+1]
                node_n_attention_weights = torch.mean(node_n_attention_weights, dim=1)
                node_p_attention_weights = torch.mean(node_p_attention_weights, dim=1)
                hop_n_attention_weights = torch.mean(hop_n_attention_weights, dim=1)
                hop_p_attention_weights = torch.mean(hop_p_attention_weights, dim=1)
                agg_attention_weights = [node_n_attention_weights, node_p_attention_weights, hop_n_attention_weights, hop_p_attention_weights]

        node_p = self.final_ln(node_p)
        node_n = self.final_ln(node_n)

        hop_p = self.final_ln(hop_p)
        hop_n = self.final_ln(hop_n)

        node_out_pre = node_p[:, 0, :] - node_n[:, 0, :]
        hop_out_pre = hop_p[:, 0, :] - hop_n[:, 0, :]


        emb = self.args.progregate_alpha*node_p[:, 0, :] + (1-self.args.progregate_alpha)*hop_p[:, 0, :]
        emb = emb - (self.args.progregate_alpha*node_n[:, 0, :] + (1-self.args.progregate_alpha)*hop_n[:, 0, :])
        emb = emb.unsqueeze(0)

        outlier_emb = None
        emb_combine = None
        normal_for_generation_emb = emb[:, normal_for_generation_idx, :]
        noise = torch.randn(normal_for_generation_emb.size(), device=self.device) * args.var + args.mean
        noised_normal_for_generation_emb = normal_for_generation_emb + noise
        
        # Add noise into the attribute of sampled abnormal nodes
        # degree = torch.sum(raw_adj[0, :, :], 0)[sample_abnormal_idx]
        # neigh_adj = raw_adj[0, sample_abnormal_idx, :] / torch.unsqueeze(degree, 1)

        neigh_adj = adj[0, normal_for_generation_idx, :]
        # emb[0, sample_abnormal_idx, :] =self.act(torch.mm(neigh_adj, emb[0, :, :]))
        # emb[0, sample_abnormal_idx, :] = self.fc4(emb[0, sample_abnormal_idx, :])

        outlier_emb = torch.mm(neigh_adj, emb[0, :, :])
        outlier_emb = self.act(self.fc4(outlier_emb))
        # emb_con = self.act(self.fc6(emb_con))

        emb_combine = torch.cat((emb[:, normal_for_train_idx, :], torch.unsqueeze(outlier_emb, 0)), 1)

        if train_flag:

            f_1 = self.fc1(emb_combine)
        else:
            f_1 = self.fc1(emb)
        f_1 = self.act(f_1)
        f_2 = self.fc2(f_1)
        f_2 = self.act(f_2)
        logits = self.fc3(f_2)
        emb[:, normal_for_generation_idx, :] = outlier_emb

        node_sum_p =  torch.sum(node_p[:, 1:-1, :], dim=1)/(node_p.shape[1]-1)

        hop_sum_p =  torch.sum(hop_p[:, 1:-1, :], dim=1)/(hop_p.shape[1]-1)  


        if self.args.sample_num_n > 0:

            con_loss = celoss(node_p[:, 0, :], node_sum_p, node_n[:, 1:-1, :], self.args.con_loss_temp, self.args.device) + celoss(hop_p[:, 0, :] , hop_sum_p, hop_n[:, 1:-1, :], self.args.con_loss_temp, self.args.device) 

        else:
            con_loss = 0
        
        gna_loss = torch.tensor(0.0, device=emb.device)
        reconstruction_loss = torch.tensor(0.0, device=emb.device)

        return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, agg_attention_weights, con_loss, gna_loss, reconstruction_loss



def celoss(target_re, pos_re, neg_re, temperature, device):
    l_pos = torch.einsum("nc,nc->n", [target_re, pos_re]).unsqueeze(-1)
    l_neg = torch.einsum("nc,nkc->nk", [target_re, neg_re.clone().detach()])


    logits = torch.cat([l_pos, l_neg], dim=1)

    logits /= temperature

    criterion = nn.CrossEntropyLoss().cuda(device)
    
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(device)

    loss = criterion(logits.cuda(device), labels)

    return loss
