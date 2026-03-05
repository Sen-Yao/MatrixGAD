import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from utils import *


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


def calculate_cycle_density(adj):
    # 1. 转换格式并二值化
    if torch.is_tensor(adj):
        adj_np = adj.detach().cpu().numpy()
        if len(adj_np.shape) == 3: adj_np = adj_np[0]
        adj_sp = sp.csr_matrix(adj_np)
    else:
        adj_sp = adj.tocsr() # 确保是 CSR 格式

    # 【关键修改】强制二值化：所有非 0 权重归为 1
    adj_sp.data = np.ones_like(adj_sp.data)

    # 2. 移除自环
    adj_sp = adj_sp - sp.diags(adj_sp.diagonal())
    adj_sp.eliminate_zeros()

    # 【关键修改】确保对称性（处理无向图）
    # 如果 A[i,j] 有边但 A[j,i] 没存，这里补齐；如果都存了，依然是 1
    adj_sp = (adj_sp + adj_sp.T).astype(bool).astype(int)

    V = adj_sp.shape[0]
    # 无向图边数 = 非零元素 / 2
    E = adj_sp.nnz // 2 
    
    from scipy.sparse.csgraph import connected_components
    n_components, _ = connected_components(csgraph=adj_sp, directed=False)
    
    M = E - V + n_components
    
    # 修正密度定义：建议使用 M / (E + 1) 或 M / V
    # 如果要看结构复杂度，M / V 是可以的，但通常不会超过平均度数的一半
    cycle_density = M / V if V > 0 else 0
    print(f"Nodes: {V}, Edges: {E}, Avg Degree: {2*E/V:.2f}")
    print(f"Is symmetric: {(adj_sp != adj_sp.T).nnz == 0}")
    return {
        "edge_count": E,
        "node_count": V,
        "components": n_components,
        "cycle_density": cycle_density
    }


def multi_frequency_tokenization(features, adj, args):
    """
    创新点 1: 多频解耦 Tokenization
    将传统的 A^k X 替换为包含低通、高通和差分带通的频域序列
    """
    print("Multi-Frequency Tokenizing...")
    N, D = features.shape
    
    # 1. 第一个 Token 永远保持原始特征 (Identity)
    # 本质上包含了所有频率信息
    tokens_list = [features.unsqueeze(1)]
    
    # 缓存上一跳的特征，用于计算差分 (带通)
    prev_hop_feature = features
    
    for hop in range(1, args.pp_k + 1):
        # 基础的低通平滑特征 (A^k X)
        # 假设 node_neighborhood_feature 内部实现了 alpha 残差或纯 A^k X
        # 为了更纯粹的频域解耦，建议此处使用纯 A^k X (即 args.progregate_alpha=0)
        low_pass = node_neighborhood_feature(adj, features, hop, args.progregate_alpha)
        
        # --- 核心修改：频域分量提取 ---
        
        # A. 高通分量 (High-pass): X - A^k X 
        # 它直接度量了节点与其 k-hop 邻居的“离群程度”
        high_pass = features - low_pass
        
        # B. 局部差分分量 (Band-pass/Gradient): A^{k-1}X - A^k X
        # 它度量了随着搜索半径扩大，信息的“流失/演化速率”
        band_pass = prev_hop_feature - low_pass
        
        # 将不同频段的特征拼接或堆叠
        # 方法一：堆叠成序列 (推荐，让 Transformer 自己学习关注哪个频段)
        # 每个 hop 产生 2 个 token，或者根据需求进行组合
        tokens_list.append(high_pass.unsqueeze(1))
        tokens_list.append(band_pass.unsqueeze(1))
        
        # 更新缓存
        prev_hop_feature = low_pass

    # 最终形状为 (N, 1 + 2*pp_k, D)
    nodes_features = torch.cat(tokens_list, dim=1)
    
    # 打印前两个 token 的余弦相似度进行验证
    cos = torch.nn.CosineSimilarity(dim=1)
    sim = cos(nodes_features[:, 0, :], nodes_features[:, 1, :]).mean()
    print(f"Avg Cosine Similarity between Identity and High-pass Token: {sim:.4f}")
    diagnostic_tokens(nodes_features)
    return nodes_features


def hybrid_signal_residual_tokenization(features, adj, args):
    """
    创新点 1.5: 混合信号-残差序列 (Signal-Residual Hybrid)
    针对 Elliptic 的长链特征，保留低通重建基准，同时引入高通偏差。
    """
    print("Hybrid Frequency Tokenizing for Elliptic/Sparse Graph...")
    N, D = features.shape
    
    # 增加 pp_k。对于度数只有 1.5 的图，6跳太短了，建议尝试 12 或 16
    # 这里我们生成两组 Token：一组是 State (低通)，一组是 Deviation (高通)
    
    # 1. 初始化原始特征 (Identity)
    state_tokens = [features.unsqueeze(1)]
    deviation_tokens = []
    
    # 提前计算一个全局平均，用于构建“全局偏离”Token (可选)
    # global_mean = features.mean(dim=0, keepdim=True)
    
    for hop in range(1, args.pp_k + 1):
        # 计算低通平滑信号 (State)
        # 建议 alpha 保持在 0.1~0.2，确保 ID 信息不丢失
        low_pass = node_neighborhood_feature(adj, features, hop, args.progregate_alpha)
        state_tokens.append(low_pass.unsqueeze(1))
        
        # 计算高通偏差信号 (Deviation)
        # X - A^k X 反映了节点与该跳数邻居的差异
        high_pass = features - low_pass
        deviation_tokens.append(high_pass.unsqueeze(1))
    
    # 拼接：[Identity, State_1...k, Deviation_1...k]
    # 这样 AE 可以重建 State 部分，而 Discriminiator 可以利用 Deviation 寻找异常
    all_tokens = torch.cat(state_tokens + deviation_tokens, dim=1)
    
    # 监控：检查低通和高通的区分度

    return all_tokens

def longchainmulti_frequency_tokenization(features, adj, args):
    """
    长链感知型多频 Tokenization (针对 Elliptic 优化)
    1. 跳跃式采样 (Log-Hops): 解决度数低、邻居相似度高的问题。
    2. 增强型高通 (Amplified High-pass): 针对 alpha=0.1 的鲁棒版本。
    """
    print("Long-Chain Multi-Frequency Tokenizing...")
    N, D = features.shape
    
    # 保持 Identity 作为基准 (Token 0)
    tokens_list = [features.unsqueeze(1)]
    
    # 定义跳数序列。传统的 pp_k=6 只能看到 6 步。
    # 这里我们使用指数跳数，pp_k=6 时可以覆盖到 2^5=32 步以外的结构
    # 这样能让每个 Token 之间的差异增大，直接降低余弦相似度
    hop_steps = [2**i for i in range(args.pp_k)] # e.g., [1, 2, 4, 8, 16, 32]
    
    prev_low_pass = features
    
    for hop in hop_steps:
        # 获取低通信号
        # 注意：此处建议 alpha 保持 0.1，因为你需要 Identity 的锚点参与对比
        low_pass = node_neighborhood_feature(adj, features, hop, args.progregate_alpha)
        
        # 1. 改进版高通 (Global-Local Contrast): X - A^k X
        # 它代表了原始节点相对 k-step 邻居的“离群向量”
        high_pass = features - low_pass
        
        # 2. 改进版带通 (Propagation Velocity): A^{prev}X - A^{curr}X
        # 捕捉在更长尺度下结构特征的变化快慢
        band_pass = prev_low_pass - low_pass
        
        # 为了不让 Transformer 序列过长，我们可以只选最强的信号
        # 在 Elliptic 上，high_pass 包含关键的洗钱偏离信号
        tokens_list.append(high_pass.unsqueeze(1))
        tokens_list.append(band_pass.unsqueeze(1))
        
        # 更新
        prev_low_pass = low_pass
    # 结果包含 1 + 2*pp_k 个 Tokens
    nodes_features = torch.cat(tokens_list, dim=1)
    return nodes_features


def diagnostic_tokens(nodes_features):
    # 1. 计算 Token 间的交互矩阵 (Batch Size, K, K)
    # 观察不同位置的 Token 是否真的提供了新信息
    inner_sim = torch.matmul(nodes_features, nodes_features.transpose(-1, -2))
    # 归一化到相似度
    norm = torch.norm(nodes_features, dim=-1, keepdim=True)
    inner_sim = inner_sim / (torch.matmul(norm, norm.transpose(-1, -2)) + 1e-8)
    
    print(f"Token Avg Similarity Matrix (first 5 nodes):\n{inner_sim[:5].mean(0)}")
    
    # 2. 计算各维度方差 (判断是否过平滑)
    token_std = nodes_features.std(dim=0).mean() 
    print(f"Token Information Richness (Std): {token_std:.4f}")


def taylor_tokenization(features, adj, args):
    """
    创新方案：泰勒阶数 Tokenization
    不再堆叠不同的跳数，而是堆叠不同算子的处理结果：
    T0: Identity (X)
    T1: 一阶导 (Laplacian) -> L X = X - A^1 X
    T2: 二阶导 (Curvature) -> L^2 X = (X - A^1 X) - (A^1 X - A^2 X)
    T3: 节点中心性编码拼接到 X
    T4: 全局算子 (Chebyshev 简化版)
    """
    N, D = features.shape
    
    # 获取不同跳数的纯低通 (不带 alpha，纯结构)
    a1_x = node_neighborhood_feature(adj, features, 1, 0)
    a2_x = node_neighborhood_feature(adj, features, 2, 0)
    a4_x = node_neighborhood_feature(adj, features, 4, 0)

    # 计算不同性质的 Token
    t0 = features # Identity
    
    # T1: 局部偏离 (一阶拉普拉斯)
    t1 = t0 - a1_x 
    
    # T2: 结构曲率 (二阶拉普拉斯) 
    # 捕捉信号变化的“加速度”，这对发现洗钱中介节点（桥接点）极度有效
    t2 = t0 - 2 * a1_x + a2_x 
    
    # T3: 跨级扩散 (远距离偏差)
    t3 = t0 - a4_x
    
    # T4: 局部梯度
    t4 = a1_x - a2_x

    tokens = torch.stack([t0, t1, t2, t3, t4], dim=1) 
    # 此时只有 5 个 Token，但每个 Token 都有独特的物理含义
    diagnostic_tokens(tokens)
    return tokens

def multi_statistics_operator_tokenization(features, adj, args):
    """
    针对 Elliptic 的算子库 Tokenization
    目标：彻底打破 Token 间的高相关性，提供多维度的偏离视角。
    """
    N, D = features.shape
    
    # 辅助函数：获取不同类型的聚合
    def get_agg(type='mean'):
        # 你可以根据 adj 的类型（稀疏/稠密）实现 min/max
        # 这里的 mean 相当于 A^1 X
        return node_neighborhood_feature(adj, features, 1, 0)

    # T0: Identity (核心原特征)
    t0 = features
    
    # T1: Mean-Deviation (均值偏离/一阶导)
    # 反映节点与邻居的平均差异
    a_mean = get_agg('mean')
    t1 = t0 - a_mean
    
    # T2: 结构强度编码 (Degree Scaling)
    # 金融图中，度数本身就是极强的信号，直接乘进去增强异常点的模长
    deg = torch.sum(adj, dim=1, keepdim=True) if not torch.is_tensor(adj) else adj.sum(dim=-1).unsqueeze(-1)
    if len(deg.shape) == 3: deg = deg[0] # 处理 batch 维度
    t2 = torch.log(deg + 1).repeat(1, D)
    
    # T3: 二阶曲率 (Acceleration) 
    # (X - A1X) - (A1X - A2X) -> 捕捉结构突变
    a2_mean = node_neighborhood_feature(adj, features, 2, 0)
    t3 = (t0 - a_mean) - (a_mean - a2_mean)
    # T4: 局部对比增强 (Local Sharpness/Max-Contrast)
    # 在金融欺诈中，异常点往往在某些特定属性上远超邻居。
    # 这里我们模拟一个简单的“极值偏差”，或者使用 A^4 这种跨度较大的残差
    a4_mean = node_neighborhood_feature(adj, features, 4, 0)
    t4 = t0 - a4_mean
    # T5: 全局上下文锚点 (Global Anchor)
    # 捕捉节点相对于全图（正常群体）的绝对位置偏离
    global_center = features.mean(dim=0, keepdim=True)
    t5 = t0 - global_center
    # 最终组合：只保留这 6 个物理意义截然不同的 Token
    tokens = torch.stack([t0, t1, t2, t3, t4, t5], dim=1)
    
    # 这里的 tokens 形状为 (N, 6, D)
    diagnostic_tokens(tokens)
    return tokens


def get_attention_entropy(attn_weights):
    # attn_weights: (Heads, Seq_len, Seq_len) 针对单个节点
    # 我们关注节点自己的 token (t0) 对所有 token 的注意力分布
    # q_idx = 0 (t0), k_idx = 0...5
    dist = attn_weights[:, 0, :] # (Heads, 6)
    entropy = -torch.sum(dist * torch.log(dist + 1e-9), dim=-1) # (Heads,)
    return entropy.mean().item()