import os
import torch

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "AIP.fasta")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Word2Vec
K_MER = 3           # 3个氨基酸作为一个词
VECTOR_SIZE = 128   # 词向量维度 (比之前的 100 稍大一点)
WINDOW_SIZE = 5     # 上下文窗口
W2V_EPOCHS = 30     # 词向量训练轮数

# deep-learning
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
NUM_EPOCHS = 150
HIDDEN_DIM = 128     # LSTM 隐藏层维度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

# --- 数据增强参数 ---
USE_AUGMENTATION = True   # 是否开启在线突变增强
AUGMENT_PROB = 0.5        # 每一条序列被执行增强的概率 (50% 的样本会被修改)
MUTATION_RATE = 0.05      # 序列中每个氨基酸被突变的概率 (例如 5% 的位点突变)