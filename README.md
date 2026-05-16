# CT-AIP

This repository contains the experimental code for anti-inflammatory peptide (AIP) identification in a research project. The project includes two prediction branches: a CT-Net deep learning branch and a machine learning branch that combines protein language model representations with handcrafted features. The final prediction can be obtained by decision-level fusion of the probabilities produced by the two branches.

Chinese version: [README_CN.md](README_CN.md)

## Project Structure

```text
CT-AIP/
├── data/                         # Training and independent test datasets
│   ├── AIP.fasta                  # Main training dataset
│   ├── BertAIP_test_dataset.fasta # External test data source
│   └── ind_dataset.fasta          # Independent test dataset
├── ct_src/                        # CT-Net deep learning branch
│   ├── train.py                   # 5-fold CT-Net training
│   ├── predict_ct_ind.py          # CT-Net prediction on the independent test set
│   ├── network.py                 # Network architecture
│   ├── data_utils.py              # FASTA reader and dataset builder
│   ├── phy_utils.py               # Physicochemical feature utilities
│   ├── biovec.py                  # BioVec/k-mer representation
│   ├── config.py                  # CT-Net configuration
│   └── models/                    # CT-Net weights and training assets
├── plm_src/                       # PLM + handcrafted feature branch
│   ├── extract_features.py        # ProtT5 + physicochemical + CKSAAP features
│   ├── extract_esm2_3b.py         # ESM-2 3B feature extraction
│   ├── fusion_train.py            # LightGBM training and OOF prediction
│   ├── evaluate_ind.py            # ML-branch prediction on the independent test set
│   ├── feature_utils.py           # Handcrafted feature utilities
│   └── config.py                  # PLM-branch configuration
├── best_model/                    # Saved best outputs or model results
├── models/                        # Shared model files, such as biovec.model
├── output/                        # Feature caches, LightGBM models, and intermediate outputs
├── extract_independent_dataset.py # Prepare the independent test set from external data
├── fusion.py                      # Search fusion strategy using training/CV outputs
└── ind_eval.py                    # Final independent test-set evaluation
```

## Requirements

Python 3.9 or a similar version is recommended. A GPU is recommended for protein language model feature extraction.

```bash
pip install numpy pandas scikit-learn torch transformers tqdm joblib lightgbm gensim
```

If ProtT5 or ESM-2 3B feature extraction is required, install a PyTorch version compatible with your CUDA environment. Model weights will be downloaded automatically by `transformers` during the first run.

## Data Format

Input data should be provided in FASTA format. Labels are stored in the header line:

```text
>seq_001|label=1
KLLKLLKKLLKLLK
>seq_002|label=0
GAGAGAGAGAGA
```

Here, `label=1` indicates an anti-inflammatory peptide, and `label=0` indicates a non-AIP sequence.

## Independent Testing with Existing Models

If you only want to run the existing models on an independent test set, you do not need to run `fusion.py`. The `fusion.py` script is mainly used after retraining to search fusion weights and classification thresholds from training or cross-validation outputs.

Use the following workflow for independent testing.

### 1. Prepare the Independent Test Set

Make sure the independent test set is available at:

```text
data/ind_dataset.fasta
```

If the independent test set needs to be prepared from external data, run:

```bash
python extract_independent_dataset.py
```

### 2. Generate CT-Net Predictions

```bash
cd ct_src
python predict_ct_ind.py
```

This script loads the 5-fold CT-Net weights from `ct_src/models/` and generates:

```text
ct_src/ind_probs_ct.npy
```

### 3. Generate PLM + Machine Learning Predictions

```bash
cd ../plm_src
python extract_features.py
python extract_esm2_3b.py
python evaluate_ind.py
```

This workflow generates or loads feature caches for the independent test set and produces:

```text
plm_src/ind_probs_ml.npy
```

Note: `FASTA_PATH` and `FEATURE_CACHE` in `plm_src/config.py` should point to the independent test set, namely `data/ind_dataset.fasta` and its corresponding feature cache.

### 4. Run Final Independent Test Evaluation

```bash
cd ..
python ind_eval.py
```

`ind_eval.py` reads `ct_src/ind_probs_ct.npy`, `plm_src/ind_probs_ml.npy`, and `data/ind_dataset.fasta`, then reports ACC, AUC, MCC, Sn, Sp, and Precision after probability-level fusion.

## Retraining or Using a Custom Dataset

If you replace the training data, retrain the models, or want to determine fusion weights and thresholds for your own dataset, use the workflow below.

### 1. Train the CT-Net Branch

```bash
cd ct_src
python train.py
```

The training script generates:

```text
ct_src/preds_ct_net_AIP.npz
ct_src/models/train_assets.joblib
ct_src/models/ct_net_best_model_fold_*.pth
```

### 2. Train the PLM + Machine Learning Branch

```bash
cd ../plm_src
python extract_features.py
python extract_esm2_3b.py
python fusion_train.py
```

The training workflow generates:

```text
plm_src/preds_plm_AIP.npz
output/models/lgbm_fold_*.pkl
output/models/top_300_indices.npy
output/cache/*.npz
```

### 3. Search the Fusion Strategy

```bash
cd ..
python fusion.py
```

`fusion.py` reads `ct_src/preds_ct_net_AIP.npz` and `plm_src/preds_plm_AIP.npz`, compares the two branches, and searches for suitable fusion weights and classification thresholds. This step is useful after retraining or dataset replacement, but it is not required for routine independent test-set prediction.

## Method Overview

- **CT-Net branch**: combines BioVec/k-mer representations, physicochemical features, CNN/ResNet modules, and Transformer modules to learn local and global peptide sequence patterns.
- **PLM branch**: extracts ProtT5, ESM-2 3B, physicochemical, and CKSAAP features, then applies LightGBM for feature selection and classification.
- **Decision-level fusion**: combines the prediction probabilities from the CT-Net and PLM branches with a fixed or searched fusion weight and threshold.

## Notes

- Do not run `fusion.py` for direct independent testing unless you intend to recompute the fusion strategy from new training outputs.
- The paths in `plm_src/config.py` determine which FASTA file is used for feature extraction. Check them carefully when switching between training and independent test data.
- ESM-2 3B is a large model. A GPU is recommended. If GPU memory is insufficient, reduce `BATCH_SIZE` or reuse cached `.npz` feature files.
- Existing `.npz`, `.npy`, `.pkl`, and `.pth` files can be used to reproduce results or skip time-consuming steps.


