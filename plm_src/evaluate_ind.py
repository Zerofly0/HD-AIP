import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import config


def main():
    print(">>> 正在加载独立测试集特征...")
    # 加载独立集特征 (请确保你刚才生成了这两个文件)
    data_base = np.load(os.path.join(config.CACHE_DIR, "ind_dataset_features_AIP.npz"), allow_pickle=True)
    X_base = data_base["X"]
    y_true = data_base["y"]

    data_esm = np.load(os.path.join(config.CACHE_DIR, "ind_dataset_esm2_3b_features_AIP.npz"), allow_pickle=True)
    X_esm = data_esm["X"]

    X_final = np.hstack([X_base, X_esm])
    print(f"独立集总样本数: {len(y_true)}, 原始拼接维度: {X_final.shape[1]}")

    print(">>> 正在加载训练集规则 (Top-300 索引)...")
    top_indices = np.load(os.path.join(config.OUT_DIR, "models", "top_300_indices.npy"))
    X_selected = X_final[:, top_indices]

    print(">>> 正在加载 5-Fold 模型进行集成推断...")
    ind_probs = np.zeros(len(y_true))

    for fold in range(5):
        model_path = os.path.join(config.OUT_DIR, "models", f"lgbm_fold_{fold}.pkl")
        clf = joblib.load(model_path)
        # 累加 5 个模型的概率
        fold_probs = clf.predict_proba(X_selected)[:, 1]
        ind_probs += fold_probs

    # 取 5 折模型的平均概率
    ind_probs = ind_probs / 5.0

    np.save("ind_probs_ml.npy", ind_probs)

    # 评估指标
    y_pred = (ind_probs > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, ind_probs)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 40)
    print("🏆 独立测试集 (Independent Test Set) 最终评估结果")
    print("=" * 40)
    print(f"ACC (准确率):    {acc:.4f}")
    print(f"AUC (ROC曲线):   {auc:.4f}")
    print(f"MCC (马修斯系数): {mcc:.4f}")
    print(f"Sn  (敏感度):    {sn:.4f}")
    print(f"Sp  (特异度):    {sp:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()