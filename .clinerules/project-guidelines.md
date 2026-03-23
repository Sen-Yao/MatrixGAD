## Brief overview
MatrixGAD 项目规则 - 半监督图异常检测研究项目。目标是开发出一款超越现有 SOTA 模型（DOMINANT, AnomalyDAE, GGAD, RHO 等）的半监督图异常检测方法。

## 代码修改后行为
- 修改核心代码完成后，应尝试运行实验验证效果
- Conda 环境激活命令：`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate GGADFormer`
- 运行模型时，应考虑运行多个 seed（建议 5 个）来检验 seed 敏感性
- 实验结果应记录到 wandb

## 项目目标
- 超越现有模型在以下数据集上的 AUROC 和 AUPRC 指标
- 目标数据集：Amazon, Reddit, photo, elliptic, t_finance, tolokers, questions
- 目标 AUROC：Amazon > 0.94, Reddit > 0.58, photo > 0.90, elliptic > 0.77, t_finance > 0.90, tolokers > 0.67, questions > 0.61
- 目标 AUPRC：Amazon > 0.81, Reddit > 0.045, photo > 0.63, elliptic > 0.29, t_finance > 0.65, tolokers > 0.32, questions > 0.006

## 科研日记
- 日记目录：`docs/diary/`
- 用于记录科研路上的尝试与发现
- 文件命名格式：`YYYY-MM-DD-主题.md`
- 重要实验发现、失败尝试、关键洞察都应记录在此
- 示例：`2026-03-23-delta-vector-findings.md`

## 技术栈
- Python 3.8+
- PyTorch 2.0 + CUDA 11.8
- DGL (Deep Graph Library)
- wandb (实验跟踪)
- NumPy, SciPy, scikit-learn

## 代码风格
- 使用中文注释和文档字符串
- 函数应有清晰的文档说明参数和返回值
- 变量命名使用下划线风格 (snake_case)
- 类命名使用驼峰风格 (PascalCase)

## 项目结构
- 核心模型文件位于根目录：`PromptGAD.py`, `GGADFormer.py`, `model.py`
- 数据加载和工具函数：`utils.py`, `utils_tam.py`
- 训练入口：`run.py`
- 数据集目录：`dataset/`
- 文档目录：`docs/`
- 科研日记：`docs/diary/`
- 实验图表：`figs/`
- 基线模型：`GGAD/`

## 实验配置
- 默认训练集比例：5% (train_rate=0.05)
- 评估指标：AUROC, AUPRC (AP)
- 使用 wandb 进行实验追踪，项目名为 "MatrixGAD"，实体为 "HCCS"

## 数据集说明
- Amazon: 电商异常检测
- Reddit: 社交网络异常
- photo: 图片网络异常
- elliptic: 比特币交易异常
- t_finance: 金融交易异常
- tolokers: 众包平台异常
- questions: 问答平台异常