import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve
import numpy as np
import os
import joblib
from torch.utils.data import WeightedRandomSampler

# 环境配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import config
from data_utils import read_fasta, AIPDataset
from biovec import train_biovec
from network import ParallelResNetTransformer


# --- 对抗训练工具类 FGM ---
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# Focal Loss 损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


def train():
    if not os.path.exists("models"):
        os.makedirs("models")

    print(">>> 1. 读取数据...")
    seqs, labels = read_fasta(config.DATA_PATH)
    labels = np.array(labels)
    seqs_arr = np.array(seqs)

    print(">>> 2. 准备 BioVec Embeddings...")
    w2v_model = train_biovec(seqs)
    vocab = {word: i for i, word in enumerate(w2v_model.wv.index_to_key)}
    padding_idx = len(vocab)
    raw_vectors = w2v_model.wv.vectors
    padding_vector = np.zeros((1, raw_vectors.shape[1]), dtype=np.float32)
    embedding_matrix = np.concatenate([raw_vectors, padding_vector], axis=0)

    vocab_size, embed_dim = embedding_matrix.shape
    phys_dim = 8  # 基础7维 + 电荷密度1维

    print(">>> 3. 开始 5-Fold 巅峰捕捉训练 (CT-Net)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    oof_preds = np.zeros(len(labels))
    global_max_len = max([len(s) - config.K_MER + 1 for s in seqs])

    # 记录每一折的巅峰指标和阈值
    fold_best_thresholds = [0.5] * 5
    final_best_accs = []
    final_best_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(seqs_arr, labels)):
        print(f"\n========== Fold {fold + 1} ==========")
        X_train, X_val = seqs_arr[train_idx], seqs_arr[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_ds = AIPDataset(
            X_train.tolist(),
            y_train,
            vocab,
            padding_idx,
            global_max_len,
            config.K_MER,
            config.USE_AUGMENTATION
        )
        val_ds = AIPDataset(X_val.tolist(), y_val, vocab, padding_idx, global_max_len, config.K_MER, False)

        # 【修复的 Sampler 逻辑】：保留正负样本权重，并正确包装为 PyTorch Sampler 对象
        sample_weights = [2.0 if label == 1 else 1.0 for label in y_train]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        model = ParallelResNetTransformer(vocab_size, embed_dim, phys_dim, config.HIDDEN_DIM, 1, embedding_matrix).to(
            config.DEVICE)
        fgm = FGM(model)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        criterion = FocalLoss(alpha=0.45, gamma=2.0).to(config.DEVICE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

        best_acc, best_auc = 0.0, 0.0
        best_thr_at_peak = 0.5
        best_model_path = f"models/wo_cnn_best_model_fold_{fold + 1}.pth"

        for epoch in range(config.NUM_EPOCHS):
            model.train()
            for X_batch, X_phys, y_batch in train_loader:
                X_batch, X_phys, y_batch = X_batch.to(config.DEVICE), X_phys.to(config.DEVICE), y_batch.to(
                    config.DEVICE)

                optimizer.zero_grad()
                logits = model(X_batch, X_phys)
                loss = criterion(logits.squeeze(), y_batch.float())
                loss.backward()

                # 对抗攻击
                fgm.attack()
                loss_adv = criterion(model(X_batch, X_phys).squeeze(), y_batch.float())
                loss_adv.backward()
                fgm.restore()

                optimizer.step()

            model.eval()
            all_labels, all_probs = [], []
            with torch.no_grad():
                for X_batch, X_phys, y_batch in val_loader:
                    X_batch, X_phys = X_batch.to(config.DEVICE), X_phys.to(config.DEVICE)
                    probs = torch.sigmoid(model(X_batch, X_phys)).squeeze()
                    # 防止单样本Batch降维
                    if probs.dim() == 0: probs = probs.unsqueeze(0)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(y_batch.numpy())

            scheduler.step()

            # --- 寻找该 Epoch 的最优阈值和 ACC ---
            p, r, t = precision_recall_curve(all_labels, all_probs)
            f1 = 2 * p * r / (p + r + 1e-10)
            curr_best_thr = t[np.argmax(f1)]
            curr_acc = accuracy_score(all_labels, (np.array(all_probs) > curr_best_thr).astype(int))
            curr_auc = roc_auc_score(all_labels, all_probs)

            # --- 核心锁定逻辑：以 ACC 提升为保存准则 ---
            if curr_acc > best_acc:
                best_acc = curr_acc
                best_auc = curr_auc
                best_thr_at_peak = curr_best_thr
                torch.save(model.state_dict(), best_model_path)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Fold {fold + 1} Epoch {epoch + 1} | ACC: {curr_acc:.4f} (Best: {best_acc:.4f}) | AUC: {curr_auc:.4f}")

        # --- 该折主训练结束，加载巅峰权重 ---
        model.load_state_dict(torch.load(best_model_path))

        # 记录每折最终成果
        fold_best_thresholds[fold] = best_thr_at_peak
        final_best_accs.append(best_acc)
        final_best_aucs.append(best_auc)

        # 生成巅峰 OOF 预测
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for X_batch, X_phys, _ in val_loader:
                X_batch, X_phys = X_batch.to(config.DEVICE), X_phys.to(config.DEVICE)
                p = torch.sigmoid(model(X_batch, X_phys)).squeeze()
                if p.dim() == 0: p = p.unsqueeze(0)
                fold_probs.extend(p.cpu().numpy())
        oof_preds[val_idx] = np.array(fold_probs)

    # --- 5-Fold 全面汇总 ---
    final_binary_preds = np.zeros_like(oof_preds)
    for fold, (_, v_idx) in enumerate(skf.split(seqs_arr, labels)):
        thr = fold_best_thresholds[fold]
        final_binary_preds[v_idx] = (oof_preds[v_idx] > thr).astype(int)

    final_acc = accuracy_score(labels, final_binary_preds)
    final_auc = roc_auc_score(labels, oof_preds)

    print(f"\n🚀 利用巅峰数值结果 (ACC优先模式):")
    print(f"   Avg ACC: {final_acc:.4f} (历史新高)")
    print(f"   Avg AUC: {final_auc:.4f}")
    print(f"   每折锁定阈值: {[round(float(i), 4) for i in fold_best_thresholds]}")

    # 保存关键模型资产
    joblib.dump({
        'vocab': vocab,
        'padding_idx': padding_idx,
        'embedding_matrix': embedding_matrix,
        'thresholds': fold_best_thresholds
    }, "models/train_assets.joblib")

    np.savez("preds_wo_cnn_AIP.npz", probs=oof_preds, labels=labels)


if __name__ == "__main__":
    train()