import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import config

# ============ 配置 ============
# ESM-2 3B 版本 (结构预测 SOTA)
MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
CACHE_FILE = os.path.join(config.CACHE_DIR, "ind_dataset_esm2_3b_features_AIP.npz")
BATCH_SIZE = 1
# ==============================

def main():
    if os.path.exists(CACHE_FILE):
        print(f"检测到已存在: {CACHE_FILE}，跳过提取。")
        return

    print(f"[Info] 加载 ESM-2 3B 模型: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    # 读取数据
    print(f"[Info] 读取数据: {config.FASTA_PATH}")
    ids, seqs = [], []
    with open(config.FASTA_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                ids.append(line[1:])
                seqs.append("")
            else:
                seqs[-1] += line

    # 提取特征
    print(">>> 开始提取 ESM-2 3B Embeddings (Mean + Max), 总样本数: {len(seqs)}<<<")
    emb_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), BATCH_SIZE)):
            batch_seqs = seqs[i: i + BATCH_SIZE]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=150)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            for j in range(len(batch_seqs)):
                mask = inputs['attention_mask'][j]
                seq_len = mask.sum().item() - 2  # ESM-2 的 0 号 token 是 <cls>，最后是 <eos>, 减去 cls 和 eos

                # [SeqLen, 2560]
                valid_tokens = outputs.last_hidden_state[j][1: 1 + seq_len]

                # Mean Pooling
                mean_emb = valid_tokens.mean(dim=0)
                # Max Pooling
                max_emb, _ = valid_tokens.max(dim=0)

                # Concat -> 5120 维
                combined = torch.cat([mean_emb, max_emb], dim=0)
                emb_list.append(combined.cpu().float().numpy())

    X_esm = np.array(emb_list)
    print(f"ESM-2 特征维度: {X_esm.shape}")

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    np.savez_compressed(CACHE_FILE, X=X_esm, ids=ids)
    print(f"√ ESM-2 特征已保存: {CACHE_FILE}")


if __name__ == "__main__":
    main()