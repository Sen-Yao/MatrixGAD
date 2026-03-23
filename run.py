import torch.nn as nn

from MFMGAD import MFMGAD
from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import torch.utils.data as Data

import wandb
from visualization import create_tsne_visualization, visualize_attention_weights
from utils import send_notification

from playground import check_token_collapse, print_token_cosine_similarity_matrix
from diagnostics import compute_diagnostics, print_diagnostics
from diagnostics import compute_mfmgad_diagnostics, print_mfmgad_diagnostics, MFMGADDiagnosticCache


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument

def train(args):
    # Set random seed
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置设备
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available() and args.device >= 0:
        print(f'CUDA device name: {torch.cuda.get_device_name(args.device)}')
        print(f'CUDA device memory: {torch.cuda.get_device_properties(args.device).total_memory / 1024**3:.1f} GB')
    else:
        print('Using CPU for computation')

    # Load and preprocess data
    if args.dataset == 'dgraph':
        adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, _, _, normal_for_train_idx, normal_for_generation_idx = load_dgraph(train_rate=args.train_rate, val_rate=0.1, args=args)
        concated_input_features = nagphormer_tokenization(features, adj, args)
        
        # 分析多跳特征之间的相关性
        analyze_multihop_correlation(concated_input_features, ano_label, args.dataset)
        
        model = MFMGAD(features.shape[1], args.embedding_dim, 'prelu', args)
        features = features.to(device)
        adj = adj.to(device)
        labels = torch.tensor(labels).to(device)

        num_nodes = features.shape[0]
        ft_size = features.shape[1]
    else:
        adj, features, labels, all_idx, idx_train, idx_val, \
        idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, 0.1, args=args)

        if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
            features, _ = preprocess_features(features)
        else:
            features = features.todense()

    
        num_nodes = features.shape[0]
        ft_size = features.shape[1]

        adj = normalize_adj(adj)
        adj = (adj + sp.eye(adj.shape[0])).todense()
        features = torch.FloatTensor(features[np.newaxis])
        # adj = torch.FloatTensor(adj[np.newaxis])
        features = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj)
        # adj = adj.to_sparse_csr()
        adj = torch.FloatTensor(adj[np.newaxis])
        labels = torch.FloatTensor(labels[np.newaxis])

        # concated_input_features.shape: torch.Size([1, node_num, 2 * feature_dim])

        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)

        # Initialize model and optimiser
        concated_input_features = nagphormer_tokenization(features.squeeze(0), adj.squeeze(0), args)
        
        # 分析多跳特征之间的相关性
        analyze_multihop_correlation(concated_input_features, ano_label, args.dataset)
        
        model = MFMGAD(ft_size, args.embedding_dim, 'prelu', args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=int(0.1 * args.num_epoch) if args.warmup_updates == -1 else args.warmup_updates,
        tot_updates=args.num_epoch,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )

    # 损失函数设置
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
    xent = nn.CrossEntropyLoss()

    auc = 0
    ap = 0
    best_AUC = 0
    best_AP = 0
    best_model_state = None
    best_epoch = 0
    
    labels = labels.squeeze(0)

    all_node_indices = torch.arange(num_nodes)

    # 在半监督场景中，模型训练时允许访问全图的 feature 和被 normal_for_train_idx 允许的那些 label
    # 为了形式统一，这里将全图的 label 也提供给 Dataset，但是在实际训练中，只有 normal_for_train_idx 的那些 label 允许被使用！
    # 其中 all_node_indices 是用于计算 batch 内部的 normal_for_train_idx 的
    batch_data_train = Data.TensorDataset(concated_input_features, labels, all_node_indices)
    batch_data_val = Data.TensorDataset(concated_input_features[idx_val], labels[idx_val])
    batch_data_test = Data.TensorDataset(concated_input_features[idx_test], labels[idx_test])

    # 对于训练集需要分层采样

    all_indices = set(range(num_nodes))
    known_indices = set(normal_for_train_idx)
    unknown_indices = list(all_indices - known_indices)

    weights = torch.zeros(num_nodes)
    weights[normal_for_train_idx] = 1.0 / len(normal_for_train_idx)
    weights[unknown_indices] = 1.0 / len(unknown_indices)

    # 基于权重，实例化一个采样器
    # replacement=True 允许重复采样，这对于过采样少数类至关重要
    sampler = Data.WeightedRandomSampler(weights, num_samples=num_nodes, replacement=True)


    # 启用 pin_memory 和多 workers 加速数据加载
    num_workers = 4 if args.batch_size >= 1024 else 0
    train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, sampler=sampler, 
                                        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    normal_for_train_idx = torch.tensor(normal_for_train_idx, dtype=torch.long, device=device)
    normal_for_train_set = set(normal_for_train_idx.tolist())  # 预计算 set，避免重复创建

    # Create diagnostic cache to use the same batch of nodes every time
    mfmgad_diagnostic_cache = MFMGADDiagnosticCache()


    # Train model
    print(f"Start training! Total epochs: {args.num_epoch}")
    pbar = tqdm(total=args.num_epoch, desc='Training')
    total_time = 0
    for epoch in range(args.num_epoch + 1):
        dynamic_weights = get_dynamic_loss_weights(epoch, args)
        start_time = time.time()
        train_flag = True
        model.train()
        
        # 损失累加器
        batched_rec_loss = 0
        batched_ortho_loss = 0
        batched_consistency_loss = 0
        batched_contrast_loss = 0
        
        for batch_idx, item in enumerate(train_data_loader):
            concated_input_features = item[0].to(device)
            labels = item[1].to(device)
            batch_global_indices = item[2].to(device)

            optimizer.zero_grad()
            is_known_normal_mask = torch.isin(batch_global_indices, normal_for_train_idx)
            local_normal_for_train_idx = torch.nonzero(is_known_normal_mask, as_tuple=False).squeeze(-1)
            
            # MFMGAD 模型前向传播
            # 返回: (embeddings, logits, recon_loss, ortho_loss, consistency_loss, contrast_loss, suspicion_scores, attn_weights)
            emb, logits, loss_rec, ortho_loss, consistency_loss, contrast_loss, suspicion_scores, attn_weights = model(
                concated_input_features, None, None, local_normal_for_train_idx, train_flag, args
            )

            # 总损失：重构 + 正交 + 一致性 + 对比
            loss = (dynamic_weights['rec_loss_weight'] * loss_rec + 
                    dynamic_weights['ortho_loss_weight'] * ortho_loss +
                    dynamic_weights['consistency_weight'] * consistency_loss +
                    dynamic_weights['contrast_weight'] * contrast_loss)

            loss.backward()
            optimizer.step()
            
            batched_rec_loss += loss_rec
            batched_ortho_loss += ortho_loss
            batched_consistency_loss += consistency_loss
            batched_contrast_loss += contrast_loss

        batched_total_loss = batched_rec_loss + batched_ortho_loss + batched_consistency_loss + batched_contrast_loss
        end_time = time.time()
        total_time += end_time - start_time
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新进度条信息
        pbar.set_postfix({
            'Time': f'{total_time:.1f}s',
            'Epoch': f'{epoch+1}/{args.num_epoch}',
            'AUC': f'{auc:.4f}',
            'AP': f'{ap:.4f}'
        })
        pbar.update(1)
        
        if epoch % 2 == 0:
            wandb.log({
                "batched_total_loss": batched_total_loss.item(),
                "rec_loss": batched_rec_loss.item(),
                "ortho_loss": batched_ortho_loss.item(),
                "consistency_loss": batched_consistency_loss.item(),
                "contrast_loss": batched_contrast_loss.item(),
                "learning_rate": current_lr
            }, step=epoch)
        lr_scheduler.step()
        
        if epoch % 10 == 0:
            # ==========================================
            # 评估模式
            # ==========================================
            model.eval()
            train_flag = False

            all_batched_anomaly_scores = []
            with torch.no_grad():
                for _, item in enumerate(test_data_loader):
                    concated_input_features = item[0].to(device)
                    labels = item[1].to(device)
                    
                    # MFMGAD 推理模式
                    # 返回: (embeddings, logits, anomaly_scores, consistency_errors, suspicion_scores, attn_weights)
                    emb, logits, anomaly_scores, consistency_errors, suspicion_scores, attn_weights = model(
                        concated_input_features, None, None, normal_for_train_idx, train_flag, args
                    )
                    
                    if anomaly_scores is not None:
                        all_batched_anomaly_scores.append(anomaly_scores)
                    else:
                        # 如果 anomaly_scores 为空，使用 logits 作为备选
                        all_batched_anomaly_scores.append(logits.squeeze(0))
                
                # 拼接所有批次的异常得分
                concatenated_scores = torch.cat(all_batched_anomaly_scores, dim=0)
                scores = concatenated_scores.cpu().detach().numpy()
                
                auc = roc_auc_score(ano_label[idx_test], scores)
                ap = average_precision_score(ano_label[idx_test], scores, average='macro', pos_label=1, sample_weight=None)
            
            wandb.log({"AUC": auc, "AP": ap}, step=epoch)
            
            # ==========================================
            # MFMGAD 诊断信息
            # ==========================================
            if epoch % 20 == 0:
                # 计算诊断指标
                mfmgad_diagnostics = compute_mfmgad_diagnostics(
                    model, train_data_loader, ano_label, idx_test, device, args,
                    normal_for_train_idx=normal_for_train_idx, cache=mfmgad_diagnostic_cache
                )
                
                # 准备损失字典
                losses = {
                    'rec': batched_rec_loss.item() if hasattr(batched_rec_loss, 'item') else batched_rec_loss,
                    'ortho': batched_ortho_loss.item() if hasattr(batched_ortho_loss, 'item') else batched_ortho_loss,
                    'consistency': batched_consistency_loss.item() if hasattr(batched_consistency_loss, 'item') else batched_consistency_loss,
                    'contrast': batched_contrast_loss.item() if hasattr(batched_contrast_loss, 'item') else batched_contrast_loss
                }
                
                # 打印诊断信息
                print_mfmgad_diagnostics(
                    mfmgad_diagnostics, epoch, 
                    current_lr=current_lr, 
                    losses=losses, 
                    dynamic_weights=dynamic_weights
                )
            
            # 检查是否为最佳模型
            if auc > best_AUC and ap > best_AP:
                best_AUC = auc
                best_AP = ap
                best_model_state = model.state_dict().copy()
                best_epoch = epoch

    pbar.close()
    print(f"Training done! Total time: {total_time:.2f} seconds")
    print(f"Best AUC: {best_AUC:.4f}, Best AP: {best_AP:.4f}, Best Epoch: {best_epoch}")
        
        

if __name__ == "__main__":


    # 定义一个辅助函数，把各种字符串转成 Python 的 bool
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str,
                        default='reddit')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_split_seed', type=int, default=42)
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8192)

    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--var', type=float, default=0.0)
    parser.add_argument('--confidence_margin', type=float, default=2)
    parser.add_argument('--outlier_beta', type=float, default=0.3)
    parser.add_argument('--sample_rate', type=float, default=0.15)
    
    parser.add_argument('--model_type', type=str, default='PromptGAD')
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--progregate_alpha', type=float, default=0.2)
    parser.add_argument('--sample_num_p', type=int, default=7)
    parser.add_argument('--sample_num_n', type=int, default=7)
    parser.add_argument('--sample_size', type=int, default=10000)

    parser.add_argument('--GT_ffn_dim', type=int, default=256)
    parser.add_argument('--GT_dropout', type=float, default=0.4)
    parser.add_argument('--GT_attention_dropout', type=float, default=0.4)
    parser.add_argument('--GT_num_heads', type=int, default=2)
    parser.add_argument('--GT_num_layers', type=int, default=3)

    parser.add_argument('--proj_R_max', type=float, default=0.5)
    parser.add_argument('--proj_R_min', type=float, default=0.1)
    parser.add_argument('--ring_R_max', type=float, default=1)
    parser.add_argument('--ring_R_min', type=float, default=0.3)

    parser.add_argument('--rec_loss_weight', type=float, default=1)
    parser.add_argument('--bce_loss_weight', type=float, default=1.0)
    parser.add_argument('--margin_loss_weight', type=float, default=0)
    parser.add_argument('--con_loss_weight', type=float, default=0.1)
    parser.add_argument('--proj_loss_weight', type=float, default=0)
    parser.add_argument('--reconstruction_loss_weight', type=float, default=1.0)
    parser.add_argument('--ring_loss_weight', type=float, default=1.0)

    parser.add_argument('--lambda_rec_tok', type=float, default=1.0)
    parser.add_argument('--lambda_rec_emb', type=float, default=0.1)
    
    parser.add_argument('--con_loss_temp', type=float, default=10)
    parser.add_argument('--GNA_temp', type=float, default=1)

    # Prompt Token 相关参数
    parser.add_argument('--num_prompts', type=int, default=8, help='Number of learnable prompt tokens for frequency domain views')
    parser.add_argument('--ortho_loss_weight', type=float, default=0.1, help='Weight for orthogonal loss of prompt tokens')
    parser.add_argument('--uniformity_loss_weight', type=float, default=0.1, help='Weight for InfoNCE uniformity loss between normal nodes')
    parser.add_argument('--tokenizer_temp', type=float, default=0.1, help='Temperature parameter for tokenizer attention computation')
    parser.add_argument('--tokenizer_hallucination_ratio', type=float, default=2.0, help='Hallucination ratio multiplier for temperature during pseudo anomaly generation')
    parser.add_argument('--hallucination_prompt_ratio', type=float, default=0.2, help='Ratio of prompts to apply hallucination temperature during pseudo anomaly generation')
    parser.add_argument('--lambda_inter', type=float, default=0.1, help='Weight for inter-pattern dispersion loss in prompt-aware uniformity loss')
    parser.add_argument('--hallucination_temp_distance_scale', type=float, default=0.0, 
                        help='Scaling factor for distance-based temperature adjustment during pseudo-anomaly generation (0 = disable dynamic temperature)')

    parser.add_argument('--warmup_updates', type=int, default=50)
    parser.add_argument('--tot_updates', type=int, default=1000)
    parser.add_argument('--peak_lr', type=float, default=1e-4)    
    parser.add_argument('--end_lr', type=float, default=1e-4)

    parser.add_argument('--warmup_epoch', type=int, default=20)

    # ==================== MFMGAD 新增参数 ====================
    # Masked Frequency Prediction 相关
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='Ratio of frequency tokens to mask during training')
    parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for consistency loss in masked frequency prediction')
    
    # Mining 对比学习相关
    parser.add_argument('--mining_threshold_base', type=float, default=0.5, help='Base threshold for mining pseudo-anomalies')
    parser.add_argument('--mining_temperature', type=float, default=1.0, help='Temperature for soft label computation')
    parser.add_argument('--suspicion_lambda', type=float, default=0.5, help='Weight for consistency error in suspicion score')
    parser.add_argument('--suspicion_weight', type=float, default=1.0, help='Weight for suspicion in anomaly score')
    
    # 双分支损失权重
    parser.add_argument('--contrast_weight', type=float, default=1.0, help='Weight for contrastive loss in dual-branch learning')

    # Ablation Study
    parser.add_argument('--ablation_random_dir', type=str2bool, default=False, help='Ablation study: randomize perturbation direction')
    parser.add_argument('--margin_m', type=float, default=0.15, help='Margin parameter for BCE loss constraint (0.1~0.2)')



    args = parser.parse_args()

    if args.dataset in ['reddit', 'photo']:
        args.mean = 0.02
        args.var = 0.01
    else:
        args.mean = 0.0
        args.var = 0.0


    run = wandb.init(
        entity="HCCS",
        # Set the wandb project where this run will be logged.
        project="MatrixGAD",
        # Track hyperparameters and run metadata.
        config=args,
    )

    wandb.define_metric("AUC", summary="max")
    wandb.define_metric("AP", summary="max")
    wandb.define_metric("AUC", summary="last")
    wandb.define_metric("AP", summary="last")
    print('Dataset: ', args.dataset)
        
    try:
        train(args)
        start_time = time.time()
        wandb.finish()
        
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"显存不足!：{e}")
        send_notification(f"【VecFormer】出现显存不足!：{e}")
        wandb.log({"AUC.max": 0})
        wandb.log({"AP.max": 0})
        start_time = time.time()
        wandb.finish()
    
    except Exception as e:
        import traceback
        print(f"其他错误：{e}")
        traceback.print_exc()  # 打印详细的错误堆栈，包括出错的代码行
        wandb.log({"AUC.max": 0})
        start_time = time.time()
        wandb.finish()
    print(f"WandB finish took {time.time() - start_time:.2f} seconds")
