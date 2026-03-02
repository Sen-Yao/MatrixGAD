from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import wandb
import numpy as np
import torch
import pandas as pd

def create_tsne_visualization(features, emb_last_epoch, labels, epoch, normal_for_train_idx, outlier_emb_last_epoch, args):
    """
    创建tsne可视化并保存到wandb
    
    Args:
        features: 原始特征 [num_nodes, feature_dim]
        embeddings: 模型生成的嵌入 [num_nodes, embedding_dim]
        labels: 真实标签 [num_nodes]
        node_types: 节点类型标签 [num_nodes]
        epoch: 当前epoch
        device: 设备
        normal_for_train_idx: 用于训练的正常节点索引
        normal_for_generation_idx: 用于生成异常节点的正常节点索引
        outlier_emb: 生成的异常节点嵌入 [num_generated_anomalies, embedding_dim]
    """
    # 准备tsne数据
    # 获取原始特征
    features = features.squeeze(0)
    
    # 获取嵌入（去掉batch维度）

    embeddings_last_epoch = emb_last_epoch.squeeze(0)  # [args.batch_size, embedding_dim]
    
    # 获取真实标签（去掉batch维度）
    labels = labels.squeeze(0)  # [args.batch_size]


    outlier_emb_len = len(outlier_emb_last_epoch)
    
    # 创建节点类型标签
    node_types = []
    print(f"nb_nodes: {args.batch_size}, outlier_emb_len: {outlier_emb_len}")
    for i in range(args.batch_size):
        if labels[i] == 1:
            # 真实异常点
            node_types.append("anomaly") 
        else:
            node_types.append("normal")
    
    # 添加生成的异常节点类型标签
    for i in range(outlier_emb_len):
        node_types.append("pesudo")
    # 将数据移到CPU并转换为numpy，使用detach()避免梯度问题
    print(f"\n\nStarting tsne visualization...")
    features = features.cpu().detach().numpy()
    embeddings_last_epoch = embeddings_last_epoch.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    # 创建用于可视化的嵌入和标签
    # 对于生成的离群点，我们需要将其添加到嵌入空间中，但不在特征空间中
    if outlier_emb_last_epoch is not None and len(outlier_emb_last_epoch) > 0:
        # 确保outlier_emb是numpy数组
        embeddings_last_epoch = np.concatenate([embeddings_last_epoch, outlier_emb_last_epoch.cpu().detach().numpy()], axis=0)

    

    # 创建tsne可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random', learning_rate=200.0)
    
    # 对过滤后的原始特征进行tsne
    features_2d = tsne.fit_transform(features)
    
    # 对过滤后的嵌入进行tsne
    embeddings_2d_last_epoch = tsne.fit_transform(embeddings_last_epoch)
    # 创建wandb表格
    # 原始特征空间的tsne
    feature_table_data = []
    for i in range(len(features_2d)):
        feature_table_data.append([
            float(features_2d[i, 0]),
            float(features_2d[i, 1]),
            node_types[i]
        ])
    
    feature_table = wandb.Table(
        columns=["TSNE_X", "TSNE_Y", "Node_Type"],
        data=feature_table_data
    )
    
    # 嵌入空间的tsne
    embedding_table_data_last_epoch = []
    for i in range(len(embeddings_2d_last_epoch)):
        # 对于生成的异常节点，使用默认标签和类型
        if i < len(node_types):
            node_type = node_types[i]
        else:
            node_type = "generated_anomaly"
        
        embedding_table_data_last_epoch.append([
            float(embeddings_2d_last_epoch[i, 0]),
            float(embeddings_2d_last_epoch[i, 1]),
            node_type
        ])
    
    embedding_table_last_epoch = wandb.Table(
        columns=["TSNE_X", "TSNE_Y", "Node_Type"],
        data=embedding_table_data_last_epoch
    )
    
    # 记录到wandb
    wandb.log({
        f"tsne_features": feature_table,
        f"tsne_embeddings_last_epoch": embedding_table_last_epoch,
    })
    print("tsne visualization done!\n")


def visualize_attention_weights(agg_attention_weights, labels, normal_for_train_idx, normal_for_generation_idx, 
                               outlier_emb, epoch, dataset_name, device, adj_matrix=None, args=None):
    """
    分析注意力权重，将原始注意力数据保存到wandb table中
    
    Args:
        agg_attention_weights: 注意力权重矩阵 [1, num_nodes, num_nodes]
        labels: 真实标签 [1, num_nodes]
        normal_for_train_idx: 用于训练的正常节点索引
        normal_for_generation_idx: 用于生成异常节点的正常节点索引
        outlier_emb: 生成的异常节点嵌入
        epoch: 当前epoch
        dataset_name: 数据集名称
        device: 设备
        adj_matrix: 邻接矩阵，用于判断节点间的邻居关系
    """
    print(f"\n\nStarting attention weights visualization...")
    if args.model_type != 'SGT':
        # 将注意力权重从GPU移到CPU并转换为numpy
        attention_weights = agg_attention_weights.squeeze(0).detach().cpu().numpy()  # [num_nodes, num_nodes]
        labels_np = labels.squeeze(0).detach().cpu().numpy()  # [num_nodes]
        
        # 获取节点数量
        num_nodes = attention_weights.shape[0]
        
        # 计算生成的异常节点数量
        if outlier_emb is None:
            outlier_emb_len = 0
        else:
            outlier_emb_len = len(outlier_emb)
        
        # 创建节点类型标签
        node_types = []
        for i in range(num_nodes):
            if i >= num_nodes - outlier_emb_len:
                node_types.append("generated_anomaly")
            elif labels_np[i] == 1:
                node_types.append("anomaly")
            else:
                node_types.append("normal")
        
        # 将节点类型转换为numpy数组
        node_types = np.array(node_types)
        
        # 分别获取正常节点和异常节点的索引
        normal_indices = np.where(node_types == "normal")[0]
        anomaly_indices = np.where(node_types == "anomaly")[0]
        generated_anomaly_indices = np.where(node_types == "generated_anomaly")[0]
        
        # print(f"节点统计: 正常={len(normal_indices)}, 异常={len(anomaly_indices)}, 生成异常={len(generated_anomaly_indices)}")
        
        # 1. 从正常点中选取前len(anomaly_indices)个，保证正常点和异常点数量相似
        if len(anomaly_indices) > 0 and len(normal_indices) > 0:
            max_sample_num = 50
            sampled_normal_count = min(len(anomaly_indices), len(normal_indices))
            sampled_normal_indices = normal_indices[:max_sample_num]
            sampled_anomaly_indices = anomaly_indices[:max_sample_num]
            
            # print(f"采样节点数量: {max_sample_num}")
            '''
            # 2. 记录每个采样出来的正常点关于其他全部正常点的注意力
            normal_to_normal_data = []
            for i, source_node in enumerate(sampled_normal_indices):
                for j, target_node in enumerate(sampled_normal_indices):
                    attention_value = attention_weights[source_node, target_node]
                    normal_to_normal_data.append([
                        "normal", "normal", source_node, target_node, attention_value
                    ])
            print("Finished saving normal to normal attention weights")
            # 3. 记录每个采样出来的正常点关于其他全部异常点的注意力
            normal_to_anomaly_data = []
            for i, source_node in enumerate(sampled_normal_indices):
                for j, target_node in enumerate(sampled_anomaly_indices):
                    attention_value = attention_weights[source_node, target_node]
                    normal_to_anomaly_data.append([
                        "normal", "abnormal", source_node, target_node, attention_value
                    ])
            print("Finished saving normal to anomaly attention weights")
            # 4. 记录每个异常点关于其他全部异常点的注意力
            anomaly_to_anomaly_data = []
            for i, source_node in enumerate(sampled_anomaly_indices):
                for j, target_node in enumerate(sampled_anomaly_indices):
                    attention_value = attention_weights[source_node, target_node]
                    anomaly_to_anomaly_data.append([
                        "abnormal", "abnormal", source_node, target_node, attention_value
                    ])
            '''
            
            # 记录前一百个正常点和异常点的注意力
            all_to_all_data = []
            # 拼接采样的正常点和异常点索引
            all_sampled_indices = np.concatenate([sampled_normal_indices, sampled_anomaly_indices])
            for i, source_node in enumerate(all_sampled_indices):
                for j, target_node in enumerate(all_sampled_indices):
                    attention_value = attention_weights[source_node, target_node]
                    all_to_all_data.append([
                        i,
                        j,
                        "normal" if source_node in sampled_normal_indices else "abnormal", 
                        "normal" if target_node in sampled_normal_indices else "abnormal", 
                        source_node, 
                        target_node, 
                        attention_value
                    ])
            df = pd.DataFrame(all_to_all_data, columns=["index_1", "index_2", "source_type", "target_type", "source_node", "target_node", "attention_weight"])

            # 按 source_type 分组并计算方差
            variance_by_type = df.groupby('source_type')['attention_weight'].var().to_dict()

            print(variance_by_type)
            print("Saving attention weights to wandb table...")
            # 保存到wandb table
            wandb.log({
                f"attention_tables/all_to_all_epoch_{epoch}": wandb.Table(
                    columns=["index_1", "index_2", "source_type", "target_type", "source_node", "target_node", "attention_weight"],
                    data=all_to_all_data
                )
            })
            
            # print(f"  正常->正常: {len(normal_to_normal_data)} 个数据点")
            # print(f"  正常->异常: {len(normal_to_anomaly_data)} 个数据点")
            # print(f"  异常->异常: {len(anomaly_to_anomaly_data)} 个数据点")
            
            # 添加画图功能：记录选定节点关于其他节点的注意力
            
            # 获取邻接矩阵信息（从原始数据中获取）
            # 这里需要从外部传入邻接矩阵，暂时使用一个简单的判断方法
            # 在实际使用时，您需要将邻接矩阵作为参数传入
            print(f"Analyzing attention weights from single node...")
            # 选择一个正常节点和一个异常节点进行分析

            if len(normal_indices) > 0 and len(anomaly_indices) > 0:
                selected_normal_node = normal_indices[0]  # 选择第一个正常节点
                selected_anomaly_node = anomaly_indices[0]  # 选择第一个异常节点
                
                # 分析正常节点的注意力分布
                normal_node_attention_data = []
                for target_node in range(num_nodes):
                    attention_value = attention_weights[selected_normal_node, target_node]
                    target_type = "normal" if target_node in normal_indices else ("anomaly" if target_node in anomaly_indices else "generated_anomaly")
                    
                    # 判断是否为邻居

                    is_neighbor = adj_matrix[selected_normal_node, target_node] > 0                
                    normal_node_attention_data.append([
                        target_node,  target_type, 
                        attention_value, is_neighbor
                    ])
                
                # 分析异常节点的注意力分布
                anomaly_node_attention_data = []
                for target_node in range(num_nodes):
                    attention_value = attention_weights[selected_anomaly_node, target_node]
                    target_type = "normal" if target_node in normal_indices else ("anomaly" if target_node in anomaly_indices else "generated_anomaly")
                    
                    # 判断是否为邻居
                    is_neighbor = adj_matrix[selected_anomaly_node, target_node] > 0
                    
                    anomaly_node_attention_data.append([
                        target_node,  target_type, 
                        attention_value, is_neighbor
                    ])
                
                # 保存到wandb table
                print(f"Saving single node attention weights to wandb table...")
                wandb.log({
                    f"attention_analysis/normal_node_attention_epoch_{epoch}": wandb.Table(
                        columns=["target_node", "target_type", "attention_weight", "is_neighbor"],
                        data=normal_node_attention_data
                    ),
                    f"attention_analysis/anomaly_node_attention_epoch_{epoch}": wandb.Table(
                        columns=["target_node", "target_type", "attention_weight", "is_neighbor"],
                        data=anomaly_node_attention_data
                    )
                })
            
        else:
            print("Warning: No enough normal nodes or anomaly nodes for analysis")
            wandb.log({
                f"attention_analysis/num_normal_nodes": len(normal_indices),
                f"attention_analysis/num_anomaly_nodes": len(anomaly_indices),
                f"attention_analysis/num_generated_anomaly_nodes": len(generated_anomaly_indices),
                f"attention_analysis/sampled_normal_count": 0,
                f"attention_analysis/normal_to_normal_pairs": 0,
                f"attention_analysis/normal_to_anomaly_pairs": 0,
                f"attention_analysis/anomaly_to_anomaly_pairs": 0,
            })
            print("Attention weights visualization done!")
            return {
                'normal_indices': normal_indices,
                'anomaly_indices': anomaly_indices,
                'generated_anomaly_indices': generated_anomaly_indices,
                'node_types': node_types
            }
    else:
        # 对于SGT模型，注意力权重比较复杂，暂时没有实现可视化
        return None