import os
import torch

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

FASTA_PATH = os.path.join(DATA_DIR, "AIP.fasta")
FEATURE_CACHE = os.path.join(CACHE_DIR, "features_AIP.npz")

# 模型配置
PROTT5_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

# 训练配置
RANDOM_SEED = 2026
N_FOLDS = 5

# === Optuna 优化的最佳参数 ===
TOP_K_FEATURES = 210  # Optuna 建议保留 210 个特征

LGBM_PARAMS = {
    'n_estimators': 754,
    'learning_rate': 0.015841742665909857,
    'num_leaves': 93,
    'max_depth': 14,
    'min_child_samples': 55,
    'subsample': 0.7734469940873437,
    'colsample_bytree': 0.6613116556800112,
    'reg_alpha': 0.004917274892026019,
    'reg_lambda': 0.00010609021512078934,
    'class_weight': 'balanced',
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'random_state': RANDOM_SEED
}

# 创建目录
os.makedirs(CACHE_DIR, exist_ok=True)