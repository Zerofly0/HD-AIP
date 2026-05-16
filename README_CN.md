# CT-AIP

本项目是用于抗炎肽（Anti-inflammatory Peptides, AIP）识别与预测的实验代码。代码包含两个预测分支：基于 CT-Net 的深度学习分支，以及融合蛋白语言模型表征和手工特征的机器学习分支。最终预测可通过两个分支的概率结果进行决策层融合。

English version: [README.md](README.md)

## 项目结构

```text
CT-AIP/
├── data/                         # 训练集和独立测试集
│   ├── AIP.fasta                  # 主训练数据
│   ├── BertAIP_test_dataset.fasta # 外部测试数据来源
│   └── ind_dataset.fasta          # 独立测试集
├── ct_src/                        # CT-Net 深度学习分支
│   ├── train.py                   # CT-Net 5 折训练
│   ├── predict_ct_ind.py          # CT-Net 独立测试集预测
│   ├── network.py                 # 网络结构
│   ├── data_utils.py              # FASTA 读取与数据集构建
│   ├── phy_utils.py               # 理化性质特征
│   ├── biovec.py                  # BioVec/k-mer 表征
│   ├── config.py                  # CT-Net 参数配置
│   └── models/                    # CT-Net 权重和训练资产
├── plm_src/                       # PLM + 手工特征机器学习分支
│   ├── extract_features.py        # ProtT5 + 理化特征 + CKSAAP 特征提取
│   ├── extract_esm2_3b.py         # ESM-2 3B 特征提取
│   ├── fusion_train.py            # LightGBM 训练与 OOF 预测
│   ├── evaluate_ind.py            # 机器学习分支独立测试集预测
│   ├── feature_utils.py           # 手工特征计算工具
│   └── config.py                  # PLM 分支参数配置
├── best_model/                    # 保存的最佳预测结果或模型输出
├── models/                        # 通用模型文件，如 biovec.model
├── output/                        # 特征缓存、LightGBM 模型和中间结果
├── extract_independent_dataset.py # 从外部数据整理独立测试集
├── fusion.py                      # 基于训练/交叉验证结果寻找融合策略
└── ind_eval.py                    # 独立测试集最终融合评估
```

## 环境依赖

建议使用 Python 3.9 或相近版本。运行蛋白语言模型特征提取时建议使用 GPU。

```bash
pip install numpy pandas scikit-learn torch transformers tqdm joblib lightgbm gensim
```

如果需要运行 ProtT5 或 ESM-2 3B 特征提取，请确保已安装与本机 CUDA 匹配的 PyTorch 版本。首次运行时，`transformers` 会自动下载模型权重。

## 数据格式

输入数据采用 FASTA 格式，标签写在标题行中：

```text
>seq_001|label=1
KLLKLLKKLLKLLK
>seq_002|label=0
GAGAGAGAGAGA
```

其中 `label=1` 表示抗炎肽，`label=0` 表示非抗炎肽。

## 使用已有模型进行独立测试

如果只是想在独立测试集上运行已有模型，不需要运行 `fusion.py`。`fusion.py` 主要用于重新训练后，基于训练集/交叉验证输出寻找融合权重和阈值。

独立测试流程如下：

### 1. 准备独立测试集

确认独立测试集位于：

```text
data/ind_dataset.fasta
```

如果需要从外部测试数据中整理独立测试集，可运行：

```bash
python extract_independent_dataset.py
```

### 2. 生成 CT-Net 分支预测

```bash
cd ct_src
python predict_ct_ind.py
```

该脚本会读取 `ct_src/models/` 中保存的 5 折模型权重，并生成：

```text
ct_src/ind_probs_ct.npy
```

### 3. 生成 PLM + 机器学习分支预测

```bash
cd ../plm_src
python extract_features.py
python extract_esm2_3b.py
python evaluate_ind.py
```

该流程会生成或读取独立测试集特征缓存，并输出：

```text
plm_src/ind_probs_ml.npy
```

注意：`plm_src/config.py` 中的 `FASTA_PATH` 和 `FEATURE_CACHE` 需要指向独立测试集，即 `data/ind_dataset.fasta` 及其对应缓存文件。

### 4. 进行独立测试集最终评估

```bash
cd ..
python ind_eval.py
```

`ind_eval.py` 会读取 `ct_src/ind_probs_ct.npy`、`plm_src/ind_probs_ml.npy` 和 `data/ind_dataset.fasta`，计算融合后的 ACC、AUC、MCC、Sn、Sp 和 Precision。

## 重新训练或使用自己的数据集

如果更换训练数据、重新训练模型，或希望在自己的数据集上重新确定融合权重和阈值，可以按以下流程运行。

### 1. 训练 CT-Net 分支

```bash
cd ct_src
python train.py
```

训练完成后会生成：

```text
ct_src/preds_ct_net_AIP.npz
ct_src/models/train_assets.joblib
ct_src/models/ct_net_best_model_fold_*.pth
```

### 2. 训练 PLM + 机器学习分支

```bash
cd ../plm_src
python extract_features.py
python extract_esm2_3b.py
python fusion_train.py
```

训练完成后会生成：

```text
plm_src/preds_plm_AIP.npz
output/models/lgbm_fold_*.pkl
output/models/top_300_indices.npy
output/cache/*.npz
```

### 3. 搜索融合策略

```bash
cd ..
python fusion.py
```

`fusion.py` 会读取 `ct_src/preds_ct_net_AIP.npz` 和 `plm_src/preds_plm_AIP.npz`，比较单模型结果，并搜索较优的融合权重和分类阈值。该步骤适用于重新训练或更换数据集后的模型选择，不是独立测试集预测的必需步骤。

## 主要方法

- **CT-Net 分支**：结合 BioVec/k-mer 表征、理化性质特征、CNN/ResNet 模块和 Transformer 模块，用于学习肽序列的局部与全局模式。
- **PLM 分支**：提取 ProtT5、ESM-2 3B、理化性质和 CKSAAP 特征，并使用 LightGBM 进行特征选择与分类。
- **决策层融合**：将 CT-Net 和 PLM 分支的预测概率进行加权融合，并使用设定阈值得到最终分类结果。

## 注意事项

- 直接进行独立测试时，请不要重新运行 `fusion.py`，除非你希望基于新训练结果重新确定融合策略。
- `plm_src/config.py` 中的路径配置会影响特征提取对象。训练集和独立测试集切换时，请确认配置文件指向正确 FASTA 和缓存文件。
- ESM-2 3B 模型较大，建议使用 GPU；如果显存不足，可以减小 `BATCH_SIZE` 或使用已缓存的 `.npz` 特征文件。
- 项目中已有的 `.npz`、`.npy`、`.pkl` 和 `.pth` 文件可用于复现实验结果或跳过部分耗时步骤。


