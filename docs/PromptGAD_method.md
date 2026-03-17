# PromptGAD: Method

## Overview

PromptGAD is a graph anomaly detection framework that leverages prompt learning and frequency-domain perspective extraction to identify anomalous nodes in graphs. The model operates on the principle of learning discriminative frequency-domain representations from multi-hop neighborhood features, then using these representations to distinguish between normal and anomalous nodes through contrastive learning with synthetically generated outliers.

## 1 Graph Tokenization with Multi-hop Neighborhood Features

Given a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ with node feature matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ where $N$ is the number of nodes and $D$ is the feature dimension, we first extract multi-hop neighborhood features for each node.

For each node $v_i$, we compute its $k$-hop neighborhood features through a propagation process:

$$
\mathbf{X}^{(t)} = (1 - \alpha) \cdot \mathbf{A} \mathbf{X}^{(t-1)} + \alpha \cdot \mathbf{X}^{(0)}
$$

where:
- $\mathbf{X}^{(0)} = \mathbf{X}$ is the original node feature matrix
- $\mathbf{A}$ is the normalized adjacency matrix
- $\alpha \in [0, 1]$ is the propagation weight controlling the retention of original features
- $t$ ranges from 1 to $k$ (we use $k=7$ in practice)

For each node, we form a token sequence:
$$
\mathcal{T}_i = [\mathbf{X}_i^{(0)}, \mathbf{X}_i^{(1)}, \dots, \mathbf{X}_i^{(k)}]
$$

where $\mathbf{X}_i^{(t)} \in \mathbb{R}^D$ is the $t$-hop feature of node $v_i$.

## 2 Frequency-domain Perspective Prompt Tokenizer

To extract diverse frequency-domain perspectives from the token sequence, we introduce learnable prompt tokens and a dynamic filtering mechanism.

### 2.1 Learnable Prompt Tokens

We initialize $M$ learnable prompt tokens $\mathbf{P} \in \mathbb{R}^{M \times D}$, where $D$ is the original token dimension (same as input feature dimension) and $M=8$ in practice. These prompts serve as learnable queries to extract different frequency characteristics. Unlike the original implementation, we no longer project tokens to a separate embedding dimension. Instead, we work directly with the original token dimension $D$, eliminating the need for a separate embedding dimension.

### 2.2 Dynamic Filtering with Signed Attention

We compute both magnitude and sign components to form dynamic filters:

**Magnitude (Importance):**
$$
\mathbf{S}_{\text{mag}} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{D}}
$$
$$
\text{magnitude} = \text{softmax}(\mathbf{S}_{\text{mag}} / \tau)
$$

where $\mathbf{Q} = \mathbf{P}$ (expanded to batch size), $\mathbf{K} = \mathcal{T}_i$ (without projection), and $\tau$ is the temperature parameter.

**Sign (Direction/Mutation):**
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

where $\odot$ denotes element-wise multiplication.

### 2.3 Frequency-domain Token Extraction

Using the synthesized attention weights, we extract new frequency-domain tokens:
$$
\text{prompt_tokens} = \text{attn_weights} \cdot \mathbf{V}
$$

where $\mathbf{V} = \mathcal{T}_i$ (without projection), and we apply LayerNorm to the extracted tokens.

### 2.4 Orthogonality Loss

To encourage diverse frequency perspectives, we impose an orthogonality constraint on the filter weights:

$$
\text{ortho_loss} = \frac{1}{B(M^2 - M)} \sum_{b=1}^B \sum_{i \neq j} |\cos(\mathbf{w}_{b,i}, \mathbf{w}_{b,j})|
$$

where $\mathbf{w}_{b,i}$ is the normalized filter weight for the $i$-th prompt in batch $b$, and $\cos(\cdot, \cdot)$ is the cosine similarity.

## 3 Graph Transformer Encoding

The extracted frequency-domain tokens are processed through a Graph Transformer encoder. The encoder consists of $L$ layers (we use $L=3$), each containing:

- Multi-head self-attention with $H$ heads (we use $H=2$)
- Feed-forward network (FFN) with hidden dimension matching the input token dimension $D$
- Layer normalization and residual connections
- Dropout for regularization

For each layer:
$$
\mathbf{Z}^{(l)} = \text{LayerNorm}(\mathbf{Z}^{(l-1)} + \text{MultiHeadAttn}(\mathbf{Z}^{(l-1)}))
$$
$$
\mathbf{Z}^{(l+1)} = \text{LayerNorm}(\mathbf{Z}^{(l)} + \text{FFN}(\mathbf{Z}^{(l)}))
$$

After $L$ layers, we aggregate the final representations using attention pooling based on the last layer's attention weights, resulting in node embeddings $\mathbf{E} \in \mathbb{R}^{N \times D}$.

## 4 Reconstruction Learning and Synthetic Anomaly Generation

### 4.1 Token Reconstruction

We learn to reconstruct the frequency-domain tokens from the node embeddings:

$$
\hat{\text{tokens}} = \text{MLP}_{\text{dec}}(\mathbf{E})
$$

where $\hat{\text{tokens}} \in \mathbb{R}^{N \times (M \cdot D)}$. The reconstruction loss is:

$$
\text{loss}_{\text{rec}} = \text{MSE}(\hat{\text{tokens}}, \text{target_tokens})
$$

where $\text{target_tokens}$ are the flattened original frequency-domain tokens.

### 4.2 Pseudo-Anomaly Generation via Partial Temperature Hallucination

Instead of using reconstruction errors, we generate pseudo-anomalies using a partial temperature hallucination approach with **dominant-prompt selective temperature elevation** and **dynamic distance-based temperature scaling**.

#### 4.2.1 Dominant-Prompt Selective Temperature Elevation

For each normal node selected for pseudo-anomaly generation, we only apply an increased temperature parameter to its **dominant prompt** (the prompt with the largest attention weight) during the tokenizer process. **Importantly, the temperature elevation is only applied to the first `pp_k // 2` tokens (hops), while the later tokens use the normal temperature.** The process is as follows:

1. Select a subset of normal nodes for pseudo-anomaly generation based on the sample rate
2. For these selected nodes, compute the dominant prompt for each node:
   $$
   \text{dominant\_prompt}(i) = \arg\max_{p \in [1,M]} \left( \sum_{t=1}^k \text{attn\_weights}[i,p,t] \right)
   $$
3. Apply partial temperature elevation **only to the dominant prompt**:
   - For the dominant prompt of each node: use elevated temperature $\tau_{\text{hallucinated}} = \tau \times \text{hallucination\_ratio}$ only for the first `pp_k // 2` tokens, while using normal temperature $\tau$ for all other tokens
   - For all non-dominant prompts: use normal temperature $\tau$ for all tokens
4. Process the normal tokens through the tokenizer with mixed temperatures: normal tokens → Prompt extraction (with partial temperature elevation only for the dominant prompt) → Transformer encoding → CLS output, generating pseudo-anomaly samples
5. Compute the binary cross-entropy loss between normal node representations and the generated pseudo-anomaly representations

This approach creates meaningful yet artificial anomalies by selectively increasing the randomness and uncertainty in the token extraction process for each node's most important frequency-domain perspective (dominant prompt), particularly focusing on the earlier hops (first `pp_k // 2` tokens). By only elevating temperature for the dominant prompt, we ensure that the pseudo-anomaly generation targets the most representative frequency pattern for each node while maintaining stability in other perspectives.

#### 4.2.2 Dynamic Distance-Based Temperature Scaling

To further improve the quality of pseudo-anomalies, we introduce **dynamic temperature scaling based on the distance from each node to its dominant prompt center**. This allows nodes that are further from their cluster centers to have higher hallucination temperatures, potentially generating more diverse anomalies.

For each normal node $i$ selected for pseudo-anomaly generation:

1. Compute the L2-normalized embedding of the node: $\mathbf{e}_i = \text{normalize}(\text{emb}[i])$
2. Compute the embedding center $\mathbf{c}_p$ for the node's dominant prompt $p$ (based on training normal nodes, detached from gradient flow)
3. Compute the Euclidean distance from the node to its dominant prompt center:
   $$
   d_i = \|\mathbf{e}_i - \mathbf{c}_p\|_2
   $$
4. Normalize the distance using the maximum distance within the batch:
   $$
   \hat{d}_i = \frac{d_i}{\max_{j} d_j + \epsilon}
   $$
5. Compute the dynamic temperature ratio for each node:
   $$
   \text{dynamic\_ratio}_i = 1 + (\text{base\_hallucinated\_ratio} - 1) \times (1 + \alpha \times \hat{d}_i)
   $$
   where $\alpha$ is the `hallucination_temp_distance_scale` hyperparameter (default: 0.0, meaning disabled) and base_hallucinated_ratio is the `tokenizer_hallucination_ratio` (default: 2.0)
6. The final temperature for the dominant prompt of node $i$ becomes:
   $$
   \tau_{\text{hallucinated},i} = \tau \times \text{dynamic\_ratio}_i
   $$

When $\alpha = 0$, this reduces to the fixed temperature scaling approach. When $\alpha > 0$, nodes further from their cluster centers have higher hallucination temperatures, potentially generating more diverse pseudo-anomalies.

#### 4.2.3 Vectorized Batch Implementation

To ensure efficient computation even with dynamic temperature scaling, we implement a fully vectorized batch approach that requires only **2 tokenizer calls per batch**, instead of 2×N calls with a naive per-sample loop:

1. First tokenizer call: process all nodes with normal temperature (no elevation) to obtain `normal_prompt_tokens`
2. Second tokenizer call: process all nodes with `temperature_per_sample=dynamic_ratios` and `partial_temp=base_hallucinated_ratio × τ` to obtain `hallucinated_prompt_tokens`
3. Create a mask based on each node's dominant prompt
4. Mix the two tokenizer outputs using the mask:
   $$
   \text{mixed\_prompt\_tokens}[i,p] = \begin{cases}
   \text{hallucinated\_prompt\_tokens}[i,p] & \text{if } p = \text{dominant\_prompt}(i) \\
   \text{normal\_prompt\_tokens}[i,p] & \text{otherwise}
   \end{cases}
   $$

This vectorized implementation maintains high computational efficiency while supporting dynamic distance-based temperature scaling.

### 4.2.4 Gradient Flow Control with Detach()

To ensure proper gradient flow separation during pseudo-anomaly generation, we apply `.detach()` to the tokenizer outputs when generating pseudo-anomalies. This design decision has important implications for the training dynamics:

**Implementation Details:**
- When generating pseudo-anomalies, both `normal_prompt_tokens` and `hallucinated_prompt_tokens` are detached from the computation graph immediately after being produced by the tokenizer
- This detachment prevents gradients from the BCE loss from flowing back to the prompt tokens and tokenizer parameters
- The mixed `prompt_tokens` (combining normal and hallucinated tokens) are also in a detached state when passed to the Transformer encoder

**Gradient Flow Architecture:**
- **BCE Loss Gradients:** Only flow through the Transformer encoder and classification head (fc1, fc2, fc3 layers)
- **Prompt & Tokenizer Gradients:** Only updated through other loss components:
  - Reconstruction loss (`loss_rec`)
  - Orthogonality loss (`ortho_loss`)
  - Uniformity loss (`uniformity_loss`)

**Benefits of this Approach:**
1. **Clear Separation of Responsibilities:** The BCE loss focuses on teaching the encoder and classifier to distinguish patterns, while the prompt learning is guided by reconstruction and orthogonality objectives
2. **Stable Training:** Prevents potential conflicting gradients that could arise from having pseudo-anomaly generation influence the same prompt tokens used for normal pattern encoding
3. **Modular Optimization:** Allows each component to learn specialized representations without interference from the pseudo-anomaly generation process
4. **Consistent Prompt Behavior:** Ensures that prompt tokens maintain stable behavior for encoding normal patterns while the encoder learns to recognize deviations

This design ensures that the prompt tokens and tokenizer module learn robust frequency-domain representations through reconstruction and orthogonality constraints, while the Transformer encoder and classification head learn to distinguish between normal patterns and the artificially generated pseudo-anomalies.

## 5 Training Objective

The overall training objective combines multiple loss components:

$$
\mathcal{L} = w_{\text{bce}} \cdot \mathcal{L}_{\text{bce}} + w_{\text{rec}} \cdot \mathcal{L}_{\text{rec}} + w_{\text{ortho}} \cdot \mathcal{L}_{\text{ortho}} + w_{\text{uniform}} \cdot \mathcal{L}_{\text{uniform}}
$$

### 5.1 Binary Cross-Entropy Loss

For contrastive learning between normal nodes and synthetic outliers:

$$
\mathcal{L}_{\text{bce}} = \text{BCEWithLogits}(\text{logits}, \mathbf{y})
$$

where $\mathbf{y}$ contains 0 for normal nodes and 1 for synthetic outliers, and logits are predicted by a 3-layer MLP classifier.

### 5.2 Uniformity Loss

To encourage normal nodes to be well-separated in the embedding space, we use an InfoNCE-style uniformity loss:

$$
\mathcal{L}_{\text{uniform}} = \frac{1}{|\mathcal{N}|} \sum_{i \in \mathcal{N}} \log \sum_{j \in \mathcal{N}, j \neq i} \exp \left( \frac{\mathbf{e}_i^T \mathbf{e}_j}{\tau_{\text{GNA}}} \right)
$$

where $\mathcal{N}$ is the set of training normal nodes, $\mathbf{e}_i$ is the L2-normalized embedding of node $i$, and $\tau_{\text{GNA}}$ is the temperature parameter.

### 5.3 Prompt-aware Uniformity Loss (New)

To achieve implicit multi-normal-pattern modeling based on node dominant frequency-domain prompts, we introduce a prompt-aware uniformity loss that requires no extra parameters or clustering modules. This loss fully reuses existing attention weights and node embeddings, aligning with graph frequency-domain decomposition theory.

$$
\mathcal{L}_{\text{uniform}} = \mathcal{L}_{\text{intra-pattern}} + \lambda_{\text{inter}} \cdot \mathcal{L}_{\text{inter-pattern}}
$$

#### Intra-Pattern Aggregation Loss

Encourages nodes sharing the same dominant prompt (i.e., belonging to the same normal frequency pattern) to cluster tightly:

$$
\mathcal{L}_{\text{intra-pattern}} = \frac{1}{|\mathcal{N}|} \sum_{i \in \mathcal{N}} \log \left( \sum_{j \in \mathcal{P}(i), j \neq i} \exp\left( \frac{\mathbf{e}_i^T \mathbf{e}_j}{\tau_{\text{uniform}}} \right) \right)
$$

where $\mathcal{P}(i)$ is the set of other normal nodes sharing the same dominant prompt as node $i$. The dominant prompt for node $i$ is computed as:

$$
\text{dominant\_prompt}(i) = \arg\max_{p \in [1,M]} \left( \sum_{t=1}^k \text{attn\_weights}[i,p,t] \right)
$$

where $M$ is the total number of prompts (fixed to 8) and $\text{attn\_weights}[i,p,t]$ is the attention weight of prompt $p$ at hop $t$ for node $i$.

#### Inter-Pattern Dispersion Loss

Encourages different prompt-corresponding normal patterns to be separated, avoiding distribution overlap:

$$
\mathcal{L}_{\text{inter-pattern}} = \frac{1}{M(M-1)} \sum_{p=1}^M \sum_{q=p+1}^M \exp\left( \frac{\mathbf{c}_p^T \mathbf{c}_q}{\tau_{\text{uniform}}} \right)
$$

where $\mathbf{c}_p \in \mathbb{R}^D$ is the L2-normalized embedding center of all normal nodes corresponding to the $p$-th prompt, which is computed once per training step with `.detach()` to prevent gradient flow. The temperature parameter $\tau_{\text{uniform}}$ reuses the existing $\tau_{\text{GNA}}$ from the original uniformity loss.

The inter-pattern weight $\lambda_{\text{inter}}$ defaults to 0.1 and can be adjusted via the `--lambda_inter` command-line argument.

### 5.4 Dynamic Loss Weighting

We use polynomial decay learning rate scheduling with warmup. The learning rate starts at 0, linearly warms up to $\text{peak_lr}=5e-4$ over 50 epochs, then polynomially decays to $\text{end_lr}=3e-4$.

### 5.5 Inactive Components

Note that the ring loss component has been disabled ($w_{\text{ring}}=0$) and is not used in the current implementation. Additionally, the GCN and Discriminator modules defined in the codebase are not utilized in the actual forward pass.

## 6 Optimization

The model is optimized using AdamW with weight decay. Training is performed with a batch size of 32768 for 300 epochs. We use a weighted random sampler to balance the sampling of known normal nodes and unknown nodes during training.

For evaluation, we use AUC-ROC and Average Precision (AP) metrics on the test set.

## 7 诊断指标说明

在模型训练和评估过程中，我们通过以下诊断指标监控模型的训练状态和性能：

### 7.1 Logit 指标
- **norm_logits_mean**: 测试集中正常节点预测logits的平均值
- **abnorm_logits_mean**: 测试集中异常节点预测logits的平均值
- **logit_margin**: 异常节点logits平均值与正常节点logits平均值的差值，反映模型对正常/异常节点的区分能力
- **logit_std**: 所有测试节点logits的标准差，反映预测的离散程度

### 7.2 嵌入坍塌指标
- **avg_cos_sim**: 正常节点嵌入之间的平均余弦相似度。值越高表示嵌入越趋于相似（坍塌），理想值应适中
- **cos_sim_std**: 正常节点嵌入之间余弦相似度的标准差

### 7.3 欧氏距离分离指标
- **center_dist**: 正常节点嵌入中心与异常节点嵌入中心之间的欧氏距离
- **norm_intra_dist**: 正常节点到其嵌入中心的平均欧氏距离，反映正常节点簇的紧凑程度
- **abnorm_intra_dist**: 异常节点到其嵌入中心的平均欧氏距离，反映异常节点簇的紧凑程度
- **separation_ratio**: 中心距离与正常节点内距离的比值，值越大表示分离效果越好

### 7.4 三角几何指标
- **dist_norm_outlier**: 正常节点中心到伪异常中心的欧氏距离
- **dist_norm_abnorm**: 正常节点中心到真实异常中心的欧氏距离
- **dist_outlier_abnorm**: 伪异常中心到真实异常中心的欧氏距离
- **outlier_intra_dist**: 伪异常节点到其中心的平均欧氏距离
- **cos_sim_directions**: 正常→伪异常方向向量与正常→真实异常方向向量的余弦相似度
- **angle_degrees**: 正常→伪异常方向与正常→真实异常方向之间的夹角（度）
- **outlier_separation_ratio**: 正常→伪异常距离与正常节点内距离的比值
- **outlier_closer_to_abnorm**: 伪异常是否比正常节点更接近真实异常

### 7.5 伪异常质量指标
- **pseudo_anomaly_difficulty_coeff**: 伪异常难度系数，计算为正常节点与伪异常嵌入的平均余弦相似度。值越高表示伪异常越难区分（难度大）
- **pseudo_anomaly_authenticity_score**: 伪异常真实性分数，计算为伪异常与真实异常嵌入的平均余弦相似度。值越高表示伪异常与真实异常越相似（真实性高）
