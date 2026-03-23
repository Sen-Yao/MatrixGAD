
# MFMGAD: Masked Frequency Modeling for Graph Anomaly Detection

## Overview

MFMGAD (Masked Frequency Modeling for Graph Anomaly Detection) is a novel graph anomaly detection framework that leverages masked frequency modeling and dual-branch learning to identify anomalous nodes in graphs. The core innovation lies in learning cross-frequency consistency patterns of normal nodes, which anomalies typically violate.

### Key Contributions

1. **Masked Frequency Prediction**: Randomly masks frequency-domain tokens and learns to predict them from visible tokens, establishing cross-frequency consistency as a normality indicator.

2. **Dual-branch Learning**: Combines consistency learning (for normal pattern modeling) with mining-based contrastive learning (for anomaly separation).

3. **Soft Label Mechanism**: Avoids false positives by using soft labels instead of hard pseudo-labels, respecting the heterogeneity of normal nodes.

4. **Double Verification**: Uses both consistency error and suspicion score to distinguish true anomalies from edge normal nodes.

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MFMGAD Framework                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入：节点 v 的多跳特征序列 [X⁰, X¹, X², ..., Xᵏ]                  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ Module 1: Prompt Tokenizer (频域视角提取)                     │  │
│   │                                                              │  │
│   │   输入序列 ─→ 8个可学习Prompt ─→ 频域token [O₁,...,O₈]        │  │
│   │                                                              │  │
│   │   每个Prompt = 一个频域滤波器                                 │  │
│   │   输出 = 节点在该频域视角下的表示                             │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ Module 2: Masked Frequency Prediction (一致性学习)            │  │
│   │                                                              │  │
│   │   随机 mask 若干 Prompt 输出                                 │  │
│   │   用剩余 Prompt 输出预测被 mask 的部分                        │  │
│   │                                                              │  │
│   │   ▸ 正常节点：预测误差小 (跨频一致)                           │  │
│   │   ▸ 异常节点：预测误差大 (跨频断裂)                           │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ Module 3: Transformer Encoder (表示学习)                     │  │
│   │                                                              │  │
│   │   将8个频域token编码为统一的节点嵌入                          │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ Module 4: Dual-branch Learning (双分支学习)                   │  │
│   │                                                              │  │
│   │   ┌──────────────────┐    ┌──────────────────────────┐       │  │
│   │   │ 分支A: 一致性学习 │    │ 分支B: Mining对比学习     │       │  │
│   │   │                  │    │                          │       │  │
│   │   │ 已标注正常节点    │    │ 已标注正常 ─→ 定义基线    │       │  │
│   │   │ 学习跨频预测     │    │ 未知节点 ─→ 计算可疑度    │       │  │
│   │   │                  │    │          ↓               │       │  │
│   │   │ 输出：一致性分数 │    │ 软标签对比学习            │       │  │
│   │   └──────────────────┘    └──────────────────────────┘       │  │
│   │                                                              │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ 异常得分融合                                                  │  │
│   │                                                              │  │
│   │   score(v) = α × 一致性误差(v) + β × 可疑度(v)               │  │
│   │                                                              │  │
│   │   ▸ 一致性误差高 + 可疑度高 ─→ 真异常                         │  │
│   │   ▸ 一致性误差低 + 可疑度高 ─→ 边缘正常（异质性）              │  │
│   │   ▸ 一致性误差低 + 可疑度低 ─→ 典型正常                       │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Graph Tokenization with Multi-hop Neighborhood Features

Given a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ with node feature matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, we first extract multi-hop neighborhood features for each node through a propagation process:

$$
\mathbf{X}^{(t)} = (1 - \alpha) \cdot \mathbf{A} \mathbf{X}^{(t-1)} + \alpha \cdot \mathbf{X}^{(0)}
$$

where:
- $\mathbf{X}^{(0)} = \mathbf{X}$ is the original node feature matrix
- $\mathbf{A}$ is the normalized adjacency matrix
- $\alpha \in [0, 1]$ is the propagation weight (default: 0.6)
- $t$ ranges from 1 to $k$ (default: $k=7$)

For each node $v_i$, we form a token sequence:
$$
\mathcal{T}_i = [\mathbf{X}_i^{(0)}, \mathbf{X}_i^{(1)}, \dots, \mathbf{X}_i^{(k)}] \in \mathbb{R}^{(k+1) \times D}
$$

## 3. Module 1: Prompt Tokenizer (Frequency-domain Perspective Extraction)

### 3.1 Learnable Frequency-domain Filters

We initialize $M$ learnable prompt tokens $\mathbf{P} \in \mathbb{R}^{M \times D}$ (default: $M=8$), where each prompt serves as a learnable frequency-domain filter that extracts different frequency characteristics from the token sequence.

### 3.2 Dynamic Filtering with Signed Attention

Similar to PromptGAD, we compute both magnitude and sign components:

**Magnitude (Importance):**
$$
\mathbf{S}_{\text{mag}} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{D}}
$$
$$
\text{magnitude} = \text{softmax}(\mathbf{S}_{\text{mag}} / \tau)
$$

**Sign (Direction):**
$$
\mathbf{S}_{\text{sign}} = \text{MLP}_q(\mathbf{Q}) \cdot \text{MLP}_k(\mathbf{K})^T
$$
$$
\text{sign} = \tanh(\mathbf{S}_{\text{sign}} / \tau)
$$

**Synthesized Filter Weights:**
$$
\text{attn_weights} = \text{magnitude} \odot \text{sign}
$$

### 3.3 Frequency-domain Token Extraction

$$
\mathbf{O} = \text{attn_weights} \cdot \mathcal{T}
$$

where $\mathbf{O} \in \mathbb{R}^{M \times D}$ is the frequency-domain token matrix, with each row $\mathbf{O}_i$ representing the node's representation in the $i$-th frequency perspective.

### 3.4 Orthogonality Constraint

To ensure diverse frequency perspectives, we impose an orthogonality constraint:

$$
\mathcal{L}_{\text{ortho}} = \frac{1}{M(M-1)} \sum_{i \neq j} |\cos(\mathbf{O}_i, \mathbf{O}_j)|
$$

## 4. Module 2: Masked Frequency Prediction

### 4.1 Random Frequency Masking

During training, we randomly mask a portion of frequency tokens (default: 25%):

$$
\text{num\_mask} = \lfloor M \times \text{mask\_ratio} \rfloor
$$

For each sample, we randomly select `num_mask` positions to mask, creating:
- **Visible tokens**: $\mathbf{O}_{\text{visible}} = \{\mathbf{O}_i : i \notin \text{masked\_indices}\}$
- **Masked tokens**: $\mathbf{O}_{\text{masked}} = \{\mathbf{O}_j : j \in \text{masked\_indices}\}$

### 4.2 Normal Baseline Computation

We maintain running averages of normal frequency tokens as baselines:

$$
\mathbf{B}_p = \text{mean}(\mathbf{O}_p^{(v)}) \quad \text{for } v \in \mathcal{N}_L
$$

where $\mathcal{N}_L$ is the set of labeled normal nodes, and $\mathbf{O}_p^{(v)}$ is the $p$-th frequency token of node $v$.

The baseline is updated using exponential moving average:
$$
\mathbf{B}_p^{(t)} = \beta \cdot \mathbf{B}_p^{(t-1)} + (1 - \beta) \cdot \mathbf{B}_p^{\text{new}}
$$

where $\beta = 0.9$ is the momentum.

### 4.3 Consistency Predictor

The predictor is a lightweight network (Cross-Attention + MLP) that predicts masked tokens from visible tokens:

$$
\hat{\mathbf{O}}_{\text{masked}} = \text{Predictor}(\mathbf{O}_{\text{visible}}, \mathbf{B})
$$

The predictor uses cross-attention where:
- **Query**: Normal baseline tokens $\mathbf{B}$
- **Key/Value**: Visible frequency tokens $\mathbf{O}_{\text{visible}}$

This design ensures predictions are biased toward "normal patterns".

### 4.4 Consistency Loss

We compute the prediction error only on masked positions:

$$
\mathcal{L}_{\text{consistency}} = \frac{1}{|\mathcal{M}| \cdot D} \sum_{i \in \mathcal{M}} \|\hat{\mathbf{O}}_i - \mathbf{O}_i\|^2
$$

where $\mathcal{M}$ is the set of masked indices.

**Key Insight**:
- **Normal nodes**: Small prediction error (cross-frequency consistency)
- **Anomalous nodes**: Large prediction error (cross-frequency inconsistency)

## 5. Module 3: Transformer Encoder

### 5.1 CLS Token Aggregation

We prepend a learnable CLS token to the frequency tokens:

$$
\mathbf{H}^{(0)} = [\text{CLS}, \mathbf{O}_1, \mathbf{O}_2, \dots, \mathbf{O}_M]
$$

### 5.2 Transformer Encoding

The sequence is processed through $L$ Transformer encoder layers (default: $L=3$):

$$
\mathbf{H}^{(l)} = \text{TransformerLayer}(\mathbf{H}^{(l-1)})
$$

Each layer contains:
- Multi-head self-attention (default: 2 heads)
- Feed-forward network
- Layer normalization
- Residual connections

### 5.3 Node Embedding

The CLS token's final representation serves as the node embedding:

$$
\mathbf{e}_v = \mathbf{H}^{(L)}[0] \in \mathbb{R}^D
$$

## 6. Module 4: Dual-branch Learning

### 6.1 Branch A: Consistency Learning

**Objective**: Learn what normal nodes should look like in terms of cross-frequency consistency.

$$
\mathcal{L}_{\text{consistency}} = \frac{1}{|\mathcal{N}_L|} \sum_{v \in \mathcal{N}_L} \|\hat{\mathbf{O}}_{\text{masked}}^{(v)} - \mathbf{O}_{\text{masked}}^{(v)}\|^2
$$

**Effect**:
- Normal nodes → Low prediction error → High consistency score
- Model learns "normal patterns should be predictable across frequencies"

### 6.2 Branch B: Mining-based Contrastive Learning

#### Step 1: Compute Dominant Prompt

For each node, find the most responsive prompt:

$$
\text{dominant}(v) = \arg\max_{p} \sum_{t=1}^{k} \text{attn\_weights}[v, p, t]
$$

#### Step 2: Compute Suspicion Score

For each unlabeled node $u$:

$$
\text{dist}(u) = \|\mathbf{e}_u - \mathbf{c}_{\text{dominant}(u)}\|_2
$$

where $\mathbf{c}_p$ is the center of normal nodes with dominant prompt $p$.

$$
\text{suspicion}(u) = \text{dist}(u) + \lambda \cdot \text{consistency\_error}(u)
$$

#### Step 3: Dynamic Threshold

The mining threshold evolves during training:

$$
\text{threshold}(\text{epoch}) = \text{base} + k \times (\text{epoch} - \text{warmup\_epochs})
$$

Early epochs: loose threshold (exploration)
Later epochs: strict threshold (exploitation)

#### Step 4: Soft Label Computation

Instead of hard pseudo-labels, we use soft labels:

$$
\text{soft\_label}(u) = \sigma\left(\frac{\text{suspicion}(u) - \text{threshold}}{\tau}\right)
$$

where $\sigma$ is the sigmoid function and $\tau$ is the temperature.

**Benefits of Soft Labels**:
- Respects uncertainty in pseudo-anomaly mining
- Gradual separation instead of hard decisions
- Reduces false positives from edge normal nodes

### 6.3 Contrastive Loss

$$
\mathcal{L}_{\text{contrast}} = \text{BCE}(\text{logits}, \text{soft\_labels})
$$

where:
- Logits are computed as similarity to normal center
- Soft labels weight the loss for each unlabeled node

## 7. The Double Verification Mechanism

### 7.1 Why Double Verification?

The key insight from RHO (Respect Heterogeneity of Normal nodes) is that normal nodes can be heterogeneous. Some normal nodes may appear "suspicious" in one metric but not another.

### 7.2 Four Scenarios

| Scenario | Suspicion | Consistency | Interpretation |
|----------|-----------|-------------|----------------|
| Typical Normal | Low | High | Normal center + predictable → Normal |
| Edge Normal | High | High | Far from center BUT predictable → Likely normal (heterogeneity) |
| Structural Anomaly | High | Low | Far from center + unpredictable → Anomaly |
| Community Anomaly | Very High | Low | Very far + unpredictable → Strong anomaly |

### 7.3 False Positive Protection

Edge normal nodes (the main source of false positives) are protected by:

1. **Consistency as Defense**: Even though they're far from the normal center (high suspicion), they maintain cross-frequency consistency (high consistency score).

2. **Soft Labels**: Instead of being forcefully pushed away as pseudo-anomalies, they receive moderate soft labels that preserve uncertainty.

3. **Combined Score**: The final anomaly score balances both metrics:
   
   $$
   \text{score}(v) = \alpha \cdot \text{consistency\_error}(v) + \beta \cdot \text{suspicion}(v)
   $$

## 8. Training Objective

### 8.1 Total Loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \alpha \cdot \mathcal{L}_{\text{consistency}} + \beta \cdot \mathcal{L}_{\text{contrast}} + \gamma \cdot \mathcal{L}_{\text{ortho}}
$$

### 8.2 Loss Components

1. **Reconstruction Loss**: Token reconstruction from embeddings
   $$
   \mathcal{L}_{\text{rec}} = \|\hat{\mathbf{O}} - \mathbf{O}\|^2
   $$

2. **Consistency Loss**: Cross-frequency prediction error (only on normal nodes)
   $$
   \mathcal{L}_{\text{consistency}} = \frac{1}{|\mathcal{N}_L|} \sum_{v \in \mathcal{N}_L} \|\hat{\mathbf{O}}_{\text{masked}}^{(v)} - \mathbf{O}_{\text{masked}}^{(v)}\|^2
   $$

3. **Contrastive Loss**: Soft-label contrastive learning
   $$
   \mathcal{L}_{\text{contrast}} = \text{BCE}(\text{logits}, \text{soft\_labels})
   $$

4. **Orthogonality Loss**: Diverse frequency perspectives
   $$
   \mathcal{L}_{\text{ortho}} = \frac{1}{M(M-1)} \sum_{i \neq j} |\cos(\mathbf{O}_i, \mathbf{O}_j)|
   $$

### 8.3 Self-Paced Training

**Phase 1: Warm-up (Epoch 1-30)**
- Only use labeled normal nodes
- Learn prompt filters (orthogonality constraint)
- Learn normal baselines
- Consistency predictor warm-up

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{ortho}} + \mathcal{L}_{\text{consistency}}
$$

**Phase 2: Mining + Contrast (Epoch 31-150)**
- Dynamic threshold: from loose to strict
- Mine pseudo-anomalies based on suspicion + consistency
- Soft-label contrastive learning

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{ortho}} + \mathcal{L}_{\text{consistency}} + \lambda(t) \cdot \mathcal{L}_{\text{contrast}}
$$

where $\lambda(t)$ gradually increases from 0.1 to 1.0.

**Phase 3: Fine-tuning (Epoch 151-200)**
- Freeze prompt filters
- Fine-tune anomaly scores
- Adjust normal baselines

## 9. Inference

### 9.1 Anomaly Score Computation

For a test node $v$:

$$
\text{score}(v) = \alpha \cdot \text{consistency\_norm}(v) + \beta \cdot \text{suspicion\_norm}(v)
$$

where:
- $\text{consistency\_norm}$: Normalized reconstruction error
- $\text{suspicion\_norm}$: Normalized distance to dominant prompt center

### 9.2 No Masking During Inference

During inference, we don't apply masking. The consistency error is approximated by the reconstruction error from the token decoder.

## 10. Hyperparameters

### 10.1 Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_prompts` | 8 | Number of learnable frequency-domain filters |
| `pp_k` | 7 | Number of propagation hops |
| `progregate_alpha` | 0.6 | Propagation retention weight |
| `GT_num_layers` | 3 | Transformer encoder layers |
| `GT_num_heads` | 2 | Attention heads |
| `GT_ffn_dim` | 256 | Feed-forward dimension |
| `GT_dropout` | 0.4 | Dropout rate |

### 10.2 Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4096 | Training batch size |
| `peak_lr` | 0.0003 | Peak learning rate |
| `end_lr` | 0.0001 | Final learning rate |
| `warmup_updates` | 30 | Warmup epochs |
| `num_epoch` | 200 | Total training epochs |

### 10.3 Loss Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rec_loss_weight` | 2.0 | Reconstruction loss weight |
| `consistency_weight` | 1.0 | Consistency loss weight |
| `contrast_weight` | 1.0 | Contrastive loss weight |
| `ortho_loss_weight` | 0.6 | Orthogonality loss weight |

### 10.4 Masked Frequency Prediction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask_ratio` | 0.25 | Ratio of tokens to mask |
| `tokenizer_temp` | 0.1 | Temperature for attention |

### 10.5 Mining

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mining_threshold_base` | 0.5 | Base threshold for pseudo-anomaly mining |
| `mining_temperature` | 1.0 | Temperature for soft labels |
| `suspicion_lambda` | 0.5 | Weight of consistency in suspicion score |

## 11. Comparison with PromptGAD

| Aspect | PromptGAD | MFMGAD |
|--------|-----------|--------|
| Pseudo-anomaly Generation | Temperature hallucination | No generation needed |
| Normal Pattern Learning | Uniformity loss | Masked prediction |
| Anomaly Signal | BCE with synthetic outliers | Consistency error + Suspicion |
| Heterogeneity Handling | Implicit | Explicit (soft labels) |
| False Positive Control | Via mining | Double verification |
| Training Complexity | Two-stage | Single-stage with warmup |

## 12. Implementation Details

### 12.1 Vectorized Operations

For GPU efficiency, the following operations are vectorized:

1. **Frequency Masking**: Uses `torch.topk` + `scatter_` instead of for-loops
2. **Suspicion Score**: Uses one-hot encoding + matrix multiplication for prompt center aggregation
3. **Batch Indexing**: Uses `torch.isin` instead of Python sets

### 12.2 Memory Optimization

1. **Classifier**: Only processes normal node embeddings during training
2. **Baseline Update**: Uses exponential moving average (no gradient storage)
3. **Soft Labels**: Computed on-the-fly without storing pseudo-labels

## 13. Usage

### Training Command

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --batch_size=4096 \
    --dataset=elliptic \
    --end_lr=0.0001 \
    --num_epoch=200 \
    --peak_lr=0.0003 \
    --pp_k=7 \
    --progregate_alpha=0.6 \
    --train_rate=0.15 \
    --warmup_updates=30 \
    --tokenizer_temp=0.1 \
    --ortho_loss_weight=0.6 \
    --rec_loss_weight=2 \
    --mask_ratio=0.25 \
    --consistency_weight=2.0 \
    --contrast_weight=0.5 \
    --mining_threshold_base=0.3 \
    --mining_temperature=0.5 \
    --suspicion_lambda=0.5
```

### Key Files

- `MFMGAD.py`: Model implementation
- `run.py`: Training script
- `utils.py`: Data loading and preprocessing

## 14. Theoretical Justification

### 14.1 Why Masked Frequency Prediction Works

The masked frequency prediction is grounded in the observation that normal nodes exhibit consistent patterns across different frequency domains (obtained from different propagation hops), while anomalous nodes show inconsistent patterns due to:

1. **Structural Anomalies**: Disconnected or loosely connected nodes have "jumpy" multi-hop features that violate smoothness assumptions.

2. **Attribute Anomalies**: Nodes with abnormal attributes propagate inconsistent signals that cannot be predicted from other frequency perspectives.

### 14.2 Connection to Frequency-domain Analysis

The multi-hop features $[\mathbf{X}^{(0)}, \mathbf{X}^{(1)}, \dots, \mathbf{X}^{(k)}]$ can be viewed as samples from different frequency bands:

- $\mathbf{X}^{(0)}$: Highest frequency (original signal)
- $\mathbf{X}^{(k)}$: Lower frequency (smoothed signal)

The learnable prompts act as adaptive filters that extract specific frequency components, and the consistency predictor learns the relationships between these components.

### 14.3 Why Double Verification Reduces False Positives

The double verification mechanism addresses the **edge normal node problem**:

- Edge normal nodes (e.g., bridge nodes, boundary nodes) are far from the normal center
- They would be flagged as anomalies by distance-based methods
- However, they maintain internal consistency (cross-frequency predictability)
- This consistency serves as a "defense" against false positives

By requiring both high suspicion AND low consistency, we filter out these edge cases while still catching true anomalies (which have both high suspicion AND low consistency).

## 15. Training Diagnostics

MFMGAD提供了丰富的诊断指标，用于监控训练过程中的模型健康状态和异常检测能力。

### 15.1 诊断指标及其物理意义

#### 15.1.1 嵌入空间分析

| 指标 | 名称 | 物理意义 |
|------|------|----------|
| `norm_avg_cos_sim` | 正常节点平均余弦相似度 | 衡量正常节点嵌入的聚集程度。值过高（>0.8）可能表示嵌入坍塌，所有正常节点被映射到相似的位置；值过低（<0.3）可能表示模型未能学习到正常节点的共同模式 |
| `norm_cos_sim_std` | 余弦相似度标准差 | 衡量正常节点嵌入的多样性。标准差应该适中，过小表示嵌入过于集中，过大表示嵌入过于分散 |
| `center_dist` | 正常/异常中心距离 | 正常节点中心与异常节点中心之间的欧氏距离。距离越大，表示模型能够更好地区分正常和异常节点 |
| `norm_intra_dist` | 正常类内距离 | 正常节点到正常中心的平均距离。反映正常节点的紧凑程度 |
| `abnorm_intra_dist` | 异常类内距离 | 异常节点到异常中心的平均距离。异常节点通常更加分散，因此这个值通常较大 |
| `separation_ratio` | 分离比 | `center_dist / norm_intra_dist`。衡量正常/异常分离程度相对于正常节点聚集程度的比值。值越大表示异常检测效果越好 |

#### 15.1.2 注意力权重分析

| 指标 | 名称 | 物理意义 |
|------|------|----------|
| `attn_mean` | 注意力均值 | 所有注意力权重的平均值。反映Prompt对输入token的平均关注程度 |
| `attn_std` | 注意力标准差 | 注意力权重的离散程度。较高的标准差表示Prompt能够区分不同位置的重要性 |
| `attn_orthogonality` | 注意力正交性 | 不同Prompt之间注意力向量的余弦相似度均值。值接近0表示不同Prompt学到了不同的特征模式（理想的正交性）；值较高表示Prompt之间存在冗余 |
| `attn_entropy_per_prompt` | Prompt选择熵 | 衡量每个节点对Prompt选择的多样性。高熵表示节点均匀地使用多个Prompt；低熵表示节点主要依赖少数几个Prompt |
| `norm_attn_mean` / `abnorm_attn_mean` | 正常/异常节点注意力均值 | 比较正常和异常节点的注意力模式差异。如果差异明显，说明模型能够通过注意力模式区分异常 |

#### 15.1.3 一致性误差分析

| 指标 | 名称 | 物理意义 |
|------|------|----------|
| `consistency_error_mean` | 一致性误差均值 | 重构误差的平均值。基于核心假设：正常节点的频域特征具有一致性，能够被准确重构 |
| `norm_consistency_mean` | 正常节点一致性误差 | 正常节点的平均重构误差。理想情况下应该较低，表示正常节点的频域模式一致 |
| `abnorm_consistency_mean` | 异常节点一致性误差 | 异常节点的平均重构误差。应该显著高于正常节点，表示异常节点的频域模式不一致 |
| `norm vs abnorm一致性差异` | 正常/异常一致性对比 | 核心诊断指标。如果 `abnorm_consistency_mean >> norm_consistency_mean`，说明模型成功学到了区分正常/异常的一致性模式 |

**关键洞察**：
- 正常节点：跨频域一致 → 重构误差小
- 异常节点：跨频域断裂 → 重构误差大

#### 15.1.4 可疑度分数分析

| 指标 | 名称 | 物理意义 |
|------|------|----------|
| `suspicion_mean` | 可疑度均值 | 所有节点的平均可疑程度。基于节点嵌入到正常中心的距离 |
| `suspicion_separation` | 可疑度分离度 | `abnorm_suspicion_mean - norm_suspicion_mean`。正分离度表示异常节点比正常节点更可疑，是期望的行为 |
| `norm_suspicion_mean` | 正常节点可疑度 | 正常节点到正常中心的平均距离。应该较低 |
| `abnorm_suspicion_mean` | 异常节点可疑度 | 异常节点到正常中心的平均距离。应该显著高于正常节点 |

**双重验证机制**：
```
异常得分 = α × 一致性误差 + β × 可疑度

真正异常：一致性误差高 + 可疑度高
边缘正常：一致性误差低 + 可疑度高（异质性）
典型正常：一致性误差低 + 可疑度低
```

#### 15.1.5 异常得分分析

| 指标 | 名称 | 物理意义 |
|------|------|----------|
| `anomaly_score_mean` | 异常得分均值 | 最终异常检测得分。结合了一致性误差和可疑度 |
| `anomaly_score_margin` | 异常得分边界 | `abnorm_anomaly_score_mean - norm_anomaly_score_mean`。边界越大，说明正常/异常分离越清晰，AUC通常越高 |

### 15.2 诊断输出示例

```
[MFMGAD-Diag@E20] lr=3.00e-04 | Loss(w): Rec=0.0234(1.0x), Ortho=0.0012(0.1x), Cons=0.0456(1.0x), Contrast=0.0234(1.0x)
  Embed: norm_cos_sim=0.452±0.123, center_dist=1.234, sep_ratio=2.345
  Attn: mean=0.0234, std=0.0123, ortho=0.0234, entropy=1.2345
        norm=0.0198±0.0089, abnorm=0.0312±0.0145
  Consistency: mean=0.1234±0.0234, range=[0.0012, 0.8765]
               norm=0.0876±0.0145, abnorm=0.2345±0.0456
  Suspicion: mean=0.3456±0.1234, sep=0.2345
              norm=0.2345±0.0987, abnorm=0.4690±0.1567
  AnomalyScore: mean=0.4567±0.2345, margin=0.3456
                 norm=0.2987±0.1234, abnorm=0.6443±0.1879
  Samples: normal=1234, abnormal=567
```

### 15.3 健康状态判断标准

| 指标 | 健康范围 | 异常信号 | 建议操作 |
|------|----------|----------|----------|
| `norm_avg_cos_sim` | 0.3-0.7 | >0.8 嵌入坍塌 | 增加正交损失权重或降低学习率 |
| `separation_ratio` | >1.5 | <1.0 分离不足 | 增加对比损失权重 |
| `attn_orthogonality` | <0.3 | >0.5 Prompt冗余 | 增加正交损失权重 |
| `consistency分离` | `abnorm > norm` | `norm >= abnorm` | 检查mask比例或增加一致性损失 |
| `suspicion_separation` | >0.1 | <0 或负值 | 检查正常中心更新逻辑 |
| `anomaly_score_margin` | >0.2 | <0.1 | 调整α/β权重或增加训练轮次 |

### 15.4 训练阶段诊断策略

**阶段1: Warm-up (Epoch 1-30)**
- 关注 `attn_orthogonality`：确保Prompt学习到多样化的频域视角
- 关注 `norm_avg_cos_sim`：避免嵌入坍塌

**阶段2: Mining + Contrast (Epoch 31-150)**
- 关注 `suspicion_separation`：验证伪异常挖掘的有效性
- 关注 `consistency分离`：验证双重验证机制的工作情况

**阶段3: Fine-tuning (Epoch 151-200)**
- 关注 `anomaly_score_margin`：最终检测性能指标
- 关注 `separation_ratio`：嵌入空间的最终质量
