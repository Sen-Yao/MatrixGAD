# GGADFormer

## Requirements

- CUDA 11.8

To install requirements:

```bash
conda create -n GGADFormer python=3.8 -y
conda activate GGADFormer

# Install Pytorch (Depend on your CUDA version, see https://pytorch.org/get-started/previous-versions/)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Introduction

图异常检测（Graph Anomaly Detection）的核心任务是识别在图结构数据中行为模式或特征显著偏离大部分节点的实体。这些异常通常表现为结构性异常（如孤立点或意外的社群连接）、属性异常（节点的特征向量与众不同），亦或是两者的结合。

在众多现实场景中，标签稀缺是一个普遍挑战，由此催生了半监督图异常检测（Semi-supervised Graph Anomaly Detection）这一重要研究方向。该设定下，模型在训练时仅能利用极少数已知的“正常”节点作为先验知识，而绝大部分正常节点与所有的异常节点均是无标签的。

考虑到真实世界应用（如金融风控、网络安全入侵检测）中数据规模庞大、异常模式未知多变，使得全面标注的成本极其高昂。与之相对，获取少量正常样本的成本则低廉得多。因此，半监督图异常检测范式不仅具备高度的普适性，更是在资源受限的实际环境中解决问题的关键技术，具有重要的理论研究与实际应用价值。

在半监督图异常检测设定下，模型面临两大核心困境。其一，训练信号极度稀疏：模型仅能从有限的正常样本中归纳“正常”模式的分布，而对“异常”模式则完全未知，这极易导致模型对正常模式的认知产生偏差（bias），形成过于狭隘的决策边界，从而引发过拟合。其二，现有图神经网络（GNN）在处理大规模图时，普遍受限于过平滑与可扩展性（Scalability）的瓶颈。

为系统性地应对上述挑战，我们设计并提出了GGADFormer，一个基于 Transformer 架构的生成式图异常检测框架。GGADFormer 的先进性主要体現在其四大核心贡献：1) 一种解耦式的图结构感知输入编码策略，它将图拓扑信息的提取与节点表征学习分离，使得模型能够进行高效的小批量训练，天然地适用于大规模图；2) 图结构感知的输入编码策略，防止过度平滑；3) 一种新颖的生成式策略，在缺乏真实异常样本的情况下，**人工合成高质量的伪异常样本**；4) 多重学习机制，确保模型学习到鲁棒且具有区分性的节点表征。

GGADFormer 的创新点包括

- 首个基于 Transformer 的生成式半监督图异常检测模型
- 支持 mini-batch 训练，对大图有良好的可扩展性
- 相比传统的半监督异常检测方法采用的 15% 的划分比，GGADFormer 可以在 5% 的划分比的场景下达到非常优秀的性能


## Experiments

我们在 `Amazon`, `photo`, `reddit`, `ellptic`, `t_finance`, `tolokers`, `questions` 等多个不同的图异常检测常用的数据集上进行了试验。

数据集划分上，我们仅采用 5% 的节点作为训练集，模型仅能访问训练集中的正常节点的标签。

### 实验对比

我们使用了 5 个随机种子进行了实验，实验结果如下：

AUC:


|Dataset|Amazon|Reddit|photo|elliptic|t_finance|tolokers|questions
|-|-|-|-|-|-|-|-|
|GGAD|0.7514±0.0410|0.5274±0.0052|0.6114±0.0219|0.7006±0.0090|TBD|0.5382±0.0065|TBD
|GGADFormer|0.9324±0.0189|0.5629±0.0161|0.8268±0.0281|0.7221±0.0441|0.9077±0.0039|0.6534±0.0195|0.5568±0.0147

AP:

|Dataset|Amazon|Reddit|photo|elliptic|t_finance|tolokers|questions
|-|-|-|-|-|-|-|-|
|GGAD|0.3755±0.0749|0.0360±0.0003|0.1269±0.0091|0.2565±0.0200|TBD|0.2448±0.0039|TBD
|GGADFormer|0.8080±0.0088|0.0418±0.0042|0.5216±0.0419|0.2268±0.0755|0.6589±0.0323|0.3063±0.0138|0.0375±0.0020

## Run

```bash
python run.py --dataset=photo --num_epoch=200 --peak_lr=3e-4 --batch_size=128
```