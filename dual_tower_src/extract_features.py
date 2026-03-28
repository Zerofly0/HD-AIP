import re
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.preprocessing import StandardScaler
import config
import feature_utils


def parse_fasta(fp):
    ids, seqs, labels = [], [], []
    with open(fp, "r", encoding="utf-8") as f:
        curr_id = None
        for line in f:
            if line.startswith(">"):
                parts = line[1:].split("|")
                curr_id = parts[0]
                lbl = int(parts[1].split("=")[1]) if "label=" in parts[1] else -1
                ids.append(curr_id)
                labels.append(lbl)
                seqs.append("")
            else:
                seqs[-1] += line.strip()
    return ids, seqs, np.array(labels)


def main():
    print(f"[Step 1] 读取数据: {config.FASTA_PATH}")
    ids, seqs, labels = parse_fasta(config.FASTA_PATH)

    # --- 1. ProtT5 Embeddings ---
    # 如果之前跑过 embedding 缓存，其实可以复用，不用重新加载模型
    # 这里为了代码完整性，写了加载逻辑。如果你想省时间，可以读取旧npz取前1024列
    print(f"[Step 2] 提取 ProtT5 Embeddings...")
    tokenizer = T5Tokenizer.from_pretrained(config.PROTT5_MODEL, do_lower_case=False, legacy=False)
    model = T5EncoderModel.from_pretrained(config.PROTT5_MODEL).to(config.DEVICE)
    if config.DEVICE == "cuda": model = model.half()
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
                seq_len = att_mask[j].sum()
                valid_tokens = out.last_hidden_state[j][:seq_len]

                # 1. Mean Pooling
                mean_emb = valid_tokens.mean(dim=0)

                # 2. Max Pooling (新增)
                max_emb, _ = valid_tokens.max(dim=0)

                # 3. Concat
                combined_emb = torch.cat([mean_emb, max_emb], dim=0)

                emb_list.append(combined_emb.cpu().float().numpy())

    X_emb = np.array(emb_list)

    # --- 2. 提取超级手工特征 (Phys + CKSAAP g=0,1,2,3,4) ---
    print("[Step 3] 计算 Phys + CKSAAP (Gaps 0-4)...")
    hand_list = []

    for seq in tqdm(seqs):
        feats = []
        # Phys
        feats.append(feature_utils.get_physicochemical_features(seq))
        # CKSAAP Gap 0 (DPC), 1, 2, 3, 4
        for gap in [0, 1, 2, 3, 4]:
            feats.append(feature_utils.get_cksaap_features(seq, gap))

        hand_list.append(np.hstack(feats))

    X_hand = np.array(hand_list)

    # 归一化
    scaler = StandardScaler()
    X_hand = scaler.fit_transform(X_hand)

    # --- 3. 融合 ---
    X_final = np.hstack([X_emb, X_hand])
    print(f"[Step 4] 最终维度: {X_final.shape}")

    np.savez_compressed(
        config.FEATURE_CACHE,
        ids=ids, X=X_final, y=labels,
        feature_names=feature_utils.get_feature_names()
    )
    print("✅ 特征提取完成")


if __name__ == "__main__":
    main()