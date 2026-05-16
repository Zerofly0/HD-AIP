import torch
import numpy as np
import os
import joblib
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix

import config
from data_utils import read_fasta, AIPDataset
from network import ParallelResNetTransformer


def main():
    print("========== CT-Net 独立测试集推断 ==========")

    # 1. 路径设置 (请确保 Ind_AIP.fasta 是你在前一步合并好的独立集)
    IND_FASTA_PATH = os.path.join(config.BASE_DIR, "data", "ind_dataset.fasta")
    ASSETS_PATH = "models/train_assets.joblib"

    if not os.path.exists(IND_FASTA_PATH):
        print(f"❌ 错误: 找不到独立测试集文件 {IND_FASTA_PATH}。请先执行合并脚本。")
        return
    if not os.path.exists(ASSETS_PATH):
        print(f"❌ 错误: 找不到训练资产 {ASSETS_PATH}。请确认 train.py 已成功跑完。")
        return

    # 2. 加载数据
    print(f">>> 1. 读取独立集数据: {IND_FASTA_PATH}")
    seqs, labels = read_fasta(IND_FASTA_PATH)
    labels = np.array(labels)

    # 3. 加载训练阶段固化的资产 (绝对不能在测试集上重新训练 BioVec!)
    print(">>> 2. 加载训练阶段锁定的 BioVec 词表与 Embedding...")
    assets = joblib.load(ASSETS_PATH)
    vocab = assets['vocab']
    padding_idx = assets['padding_idx']
    embedding_matrix = assets['embedding_matrix']

    vocab_size, embed_dim = embedding_matrix.shape
    phys_dim = 8
    global_max_len = max([len(s) - config.K_MER + 1 for s in seqs])

    # 4. 构建 DataLoader (注意: augment 必须为 False)
    test_ds = AIPDataset(seqs, labels, vocab, padding_idx, global_max_len, config.K_MER, augment=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # 5. 执行 5-Fold 模型集成推断
    print(">>> 3. 开始加载 5 折模型权重进行联合推断...")
    ind_probs = np.zeros(len(labels))

    model = ParallelResNetTransformer(vocab_size, embed_dim, phys_dim, config.HIDDEN_DIM, 1, embedding_matrix).to(
        config.DEVICE)

    for fold in range(5):
        weight_path = f"models/ct_net_best_model_fold_{fold + 1}.pth"
        if not os.path.exists(weight_path):
            print(f"❌ 错误: 找不到模型权重 {weight_path}")
            return

        # 加载单折模型权重
        model.load_state_dict(torch.load(weight_path, map_location=config.DEVICE))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for X_batch, X_phys, _ in test_loader:
                X_batch, X_phys = X_batch.to(config.DEVICE), X_phys.to(config.DEVICE)
                p = torch.sigmoid(model(X_batch, X_phys)).squeeze()
                if p.dim() == 0: p = p.unsqueeze(0)
                fold_probs.extend(p.cpu().numpy())

        ind_probs += np.array(fold_probs)
        print(f"   √ Fold {fold + 1} 推断完成")

    # 取 5 折模型的平均概率
    ind_probs = ind_probs / 5.0

    # 6. 保存概率供终极融合使用
    np.save("ind_probs_ct.npy", ind_probs)
    print(">>> ✅ CT-Net 独立集概率已保存至 ind_probs_ct.npy")

    # 7. (可选) 计算 CT-Net 单模型的独立集指标
    # 注意：这里我们简单使用 0.5 作为单模型的阈值进行初步查看，
    # 最终你的论文指标应该看融合后 (Fusion) 的表现。
    y_pred = (ind_probs > 0.5).astype(int)
    acc = accuracy_score(labels, y_pred)
    auc = roc_auc_score(labels, ind_probs)
    mcc = matthews_corrcoef(labels, y_pred)

    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n[CT-Net 单流独立集表现]")
    print(f"ACC: {acc:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f} | Sn: {sn:.4f} | Sp: {sp:.4f}")


if __name__ == "__main__":
    main()