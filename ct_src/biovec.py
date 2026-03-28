from gensim.models import Word2Vec
import config
from data_utils import seq_to_kmers
import os

def train_biovec(seqs):
    """训练自定义的 Word2Vec 模型"""
    print(f">>> [BioVec] 开始训练 K-mer Embedding (K={config.K_MER})...")

    # 构建语料库
    corpus = [seq_to_kmers(s, k=config.K_MER) for s in seqs]

    # 初始化并训练
    model = Word2Vec(
        sentences=corpus,
        vector_size=config.VECTOR_SIZE,
        window=config.WINDOW_SIZE,
        min_count=1,
        workers=4,
        epochs=config.W2V_EPOCHS,
        sg=1  # 1=Skip-Gram (更适合小数据), 0=CBOW
    )

    # 保存模型
    save_path = os.path.join(config.MODEL_DIR, "biovec.model")
    model.save(save_path)
    print(f"✅ BioVec 模型已保存: {save_path}")
    print(f"   词表大小: {len(model.wv.key_to_index)}")

    return model