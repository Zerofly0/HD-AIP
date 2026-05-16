import re
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.preprocessing import StandardScaler
import config
import feature_utils
import joblib

def parse_fasta(fp):
    ids, seqs, labels = [], [], []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                parts = line[1:].split("|")
                ids.append(parts[0])
                lbl = int(parts[1].split("=")[1]) if "label=" in parts[1] else -1
                labels.append(lbl)
                seqs.append("")
            else:
                seqs[-1] += line
        return ids, seqs, np.array(labels)


def main():
    print(f"[Step 1] 读取数据: {config.FASTA_PATH}")
    ids, seqs, labels = parse_fasta(config.FASTA_PATH)

    # --- 1. ProtT5 Embeddings ---
    print(f"[Step 2] 提取 ProtT5 Embeddings...")
    tokenizer = T5Tokenizer.from_pretrained(config.PROTT5_MODEL, do_lower_case=False, legacy=False)
    model = T5EncoderModel.from_pretrained(config.PROTT5_MODEL).to(config.DEVICE)
    if config.DEVICE == "cuda":
        model = model.half()
    model.eval()

    processed_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]
    emb_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(processed_seqs), config.BATCH_SIZE)):
            batch = processed_seqs[i: i + config.BATCH_SIZE]
            inputs = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest", return_tensors="pt")
            input_ids = inputs['input_ids'].to(config.DEVICE)
            att_mask = inputs['attention_mask'].to(config.DEVICE)
            out = model(input_ids=input_ids, attention_mask=att_mask)
            for j in range(len(batch)):
                seq_len = att_mask[j].sum().item()
                valid_tokens = out.last_hidden_state[j][:seq_len]

                # 1. Mean Pooling
                mean_emb = valid_tokens.mean(dim=0)

                # 2. Max Pooling
                max_emb, _ = valid_tokens.max(dim=0)

                # 3. Concat
                combined_emb = torch.cat([mean_emb, max_emb], dim=0)
                emb_list.append(combined_emb.cpu().float().numpy())

    X_emb = np.array(emb_list)

    # --- 2. 提取手工特征 (Phys + CKSAAP g=0,1,2,3,4) ---
    print("[Step 3] 计算 Phys + CKSAAP (Gaps 0-4)...")
    hand_list = []

    for seq in tqdm(seqs):
        feats = [
            feature_utils.get_physicochemical_features(seq),
            *[feature_utils.get_cksaap_features(seq, gap) for gap in range(5)]
        ]
        hand_list.append(np.hstack(feats))

    X_hand = np.array(hand_list)

    # 归一化
    # scaler = StandardScaler()
    # X_hand = scaler.fit_transform(X_hand)
    scaler = joblib.load(os.path.join(config.OUT_DIR, "models", "handcraft_scaler.pkl"))
    X_hand = scaler.transform(X_hand)
    os.makedirs(os.path.join(config.OUT_DIR, "models"), exist_ok=True)
    joblib.dump(scaler, os.path.join(config.OUT_DIR, "models", "handcraft_scaler.pkl"))

    # --- 3. 融合 ---
    X_final = np.hstack([X_emb, X_hand])
    print(f"[Step 4] 最终维度: {X_final.shape}")

    os.makedirs(os.path.dirname(config.FEATURE_CACHE), exist_ok=True)
    np.savez_compressed(
        config.FEATURE_CACHE,
        ids=ids, X=X_final, y=labels,
        feature_names=feature_utils.get_feature_names()
    )
    print("✅ 特征提取完成")


if __name__ == "__main__":
    main()