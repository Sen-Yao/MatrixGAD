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
\text{new_tokens} = \text{attn_weights} \cdot \mathbf{V}
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

### 4.2 Reconstruction Error as Anomaly Direction

The reconstruction error is projected to the same token space:

$$
\mathbf{R} = \text{MLP}_{\text{proj}}(\hat{\text{tokens}} - \text{target_tokens})
$$

$$
\mathbf{R} = \text{normalize}(\mathbf{R})
$$

### 4.3 Synthetic Outlier Generation

For a subset of normal nodes (15% of training normal nodes), we generate synthetic outliers:

1. Add Gaussian noise to normal embeddings:
   $$
   \tilde{\mathbf{E}}_n = \mathbf{E}_n + \mathcal{N}(\mu, \sigma^2)
   $$
   where $\mu=0.02$, $\sigma=0.01$ for Reddit and Photo datasets, $\mu=0$, $\sigma=0$ otherwise.

2. Generate outliers using the projected reconstruction error:
   $$
   \mathbf{E}_{\text{out}} = \mathbf{E}_n + \beta \cdot \mathbf{R}
   $$
   where $\beta$ is the outlier magnitude coefficient (we use $\beta \in \{0.2, 0.3, 0.5\}$).

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

### 5.3 Dynamic Loss Weighting

We use polynomial decay learning rate scheduling with warmup. The learning rate starts at 0, linearly warms up to $\text{peak_lr}=5e-4$ over 50 epochs, then polynomially decays to $\text{end_lr}=3e-4$.

### 5.4 Inactive Components

Note that the ring loss component has been disabled ($w_{\text{ring}}=0$) and is not used in the current implementation. Additionally, the GCN and Discriminator modules defined in the codebase are not utilized in the actual forward pass.

## 6 Optimization

The model is optimized using AdamW with weight decay. Training is performed with a batch size of 32768 for 300 epochs. We use a weighted random sampler to balance the sampling of known normal nodes and unknown nodes during training.

For evaluation, we use AUC-ROC and Average Precision (AP) metrics on the test set.
