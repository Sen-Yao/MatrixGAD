import torch
import torch.nn.functional as F

def check_token_collapse(tokens):
    # tokens: [N, K, D]
    # 计算每个节点内部，第 i 个 token 和 第 j 个 token 的相似度
    N, K, D = tokens.shape
    tokens_norm = F.normalize(tokens, p=2, dim=-1)
    sim_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2)) # [N, K, K]
    # 取上三角部分（排除对角线）的平均值
    mask = torch.triu(torch.ones(K, K), diagonal=1).to(tokens.device)
    mean_sim = (sim_matrix * mask).sum() / (N * (K * (K - 1) / 2))
    return mean_sim.item()