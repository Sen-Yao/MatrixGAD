import torch
import torch.nn.functional as F
import numpy as np

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

def print_token_cosine_similarity_matrix(tokens, num_nodes=3, num_tokens=None):
    """
    Compute and print cosine similarity matrices for the first few nodes and tokens
    
    Args:
        tokens: shape [N, K, D], N is number of nodes, K is number of tokens, D is feature dimension
        num_nodes: number of nodes to print, default is 3
        num_tokens: number of tokens to print, if None then print all tokens
    """
    N, K, D = tokens.shape
    
    if num_tokens is None:
        num_tokens = K
    
    # Ensure we don't exceed actual node and token counts
    num_nodes = min(num_nodes, N)
    num_tokens = min(num_tokens, K)
    
    # L2 normalize tokens
    tokens_norm = F.normalize(tokens, p=2, dim=-1)
    
    print(f"\n{'='*60}")
    print(f"Token Cosine Similarity Matrix (First {num_nodes} nodes, First {num_tokens} tokens)")
    print(f"Token shape: {tokens.shape}")
    print(f"{'='*60}")
    
    for node_idx in range(num_nodes):
        print(f"\nNode {node_idx} Token Cosine Similarity Matrix:")
        print("-" * 40)
        
        # Get current node's tokens and take first num_tokens
        node_tokens = tokens_norm[node_idx, :num_tokens, :]  # [num_tokens, D]
        
        # Compute cosine similarity matrix: [num_tokens, num_tokens]
        sim_matrix = torch.mm(node_tokens, node_tokens.transpose(0, 1))
        
        # Convert to numpy for better formatted printing
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        
        # Print header
        header = "       " + "  ".join([f"Tok{i:>2}" for i in range(num_tokens)])
        print(header)
        
        # Print similarity matrix
        for i in range(num_tokens):
            row_values = "  ".join([f"{sim_matrix_np[i, j]:>6.3f}" for j in range(num_tokens)])
            print(f"Tok{i:>2}  {row_values}")
    
    # Print statistics for all nodes
    print(f"\n{'='*60}")
    print("Token Similarity Statistics (All Nodes):")
    print("-" * 40)
    
    # Compute similarity matrices for all nodes
    all_sim_matrices = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # [N, K, K]
    
    # Compute statistics (excluding diagonal)
    mask = torch.triu(torch.ones(K, K), diagonal=1).to(tokens.device)
    
    # Compute mean similarity for each node
    mean_sims_per_node = (all_sim_matrices * mask).sum(dim=(1, 2)) / (K * (K - 1) / 2)
    
    print(f"Mean Token Similarity (across nodes): {mean_sims_per_node.mean().item():.4f}")
    print(f"Std Dev: {mean_sims_per_node.std().item():.4f}")
    print(f"Min: {mean_sims_per_node.min().item():.4f}")
    print(f"Max: {mean_sims_per_node.max().item():.4f}")
    print(f"{'='*60}\n")
    
    return all_sim_matrices
