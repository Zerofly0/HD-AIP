import os
import warnings
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import config
import joblib

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")


def main():
    print(">>> [1/3] 加载基础特征 (ProtT5 + Phys + CKSAAP)...")
    if not os.path.exists(config.FEATURE_CACHE):
        print(f"× 错误: 未找到基础特征文件 {config.FEATURE_CACHE}，请先运行 extract_features.py")
        return

    data_base = np.load(config.FEATURE_CACHE, allow_pickle=True)
    X_base = data_base["X"]
    y = data_base["y"]

    print(">>> [2/3] 加载大模型特征 (ESM-2 3B)...")
    esm_file = os.path.join(config.CACHE_DIR, "esm2_3b_features_AIP.npz")
    if not os.path.exists(esm_file):
        print(f"× 错误：未找到 ESM-2 特征文件 {esm_file}，请先运行 extract_esm2_3b.py")
        return

    data_esm = np.load(esm_file, allow_pickle=True)
    X_esm = data_esm["X"]

    # 简单校验样本数是否对齐，防止拼接错位
    if len(X_base) != len(X_esm):
        print("× 错误: 基础特征与 ESM-2 特征样本数量不一致，请检查！")
        return

    print(">>> [3/3] 执行特征级拼接 (Feature Concat)...")
    X_final = np.hstack([X_base, X_esm])
    print(f"   样本数: {len(y)}, 原始拼接总维度: {X_final.shape[1]}")

    # --- 特征筛选 ---
    print(">>> 正在基于 LightGBM 筛选 Top 300 核心特征...")

    selector = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1)
    selector.fit(X_final, y)

    top_indices = np.argsort(selector.feature_importances_)[::-1][:300]

    os.makedirs(os.path.join(config.OUT_DIR, "models"), exist_ok=True)
    np.save(os.path.join(config.OUT_DIR, "models", "top_300_indices.npy"), top_indices)

    X_selected = X_final[:, top_indices]
    print(f"   ✅ 已截取重要性排名前 300 的特征。")

    # --- 交叉验证与预测 ---
    oof_probs = np.zeros(len(y))

    params = config.LGBM_PARAMS.copy()
    params.update({
        'num_leaves': 128,
        'colsample_bytree': 0.5,
        'verbosity': -1,  # 屏蔽冗长的建树信息日志
        'n_jobs': -1,  # 开启多线程加速
        'random_state': 42  # 锁定随机种子，保证结果绝对可复现
    })

    print("\n>>> 开始生成 5-Fold OOF 预测结果 (用于最终 Fusion) <<<")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []

    for fold, (tr, val) in enumerate(skf.split(X_selected, y)):
        clf = LGBMClassifier(**params)
        clf.fit(X_selected[tr], y[tr])

        joblib.dump(clf, os.path.join(config.OUT_DIR, "models", f"lgbm_fold_{fold}.pkl"))

        # 记录验证集的概率
        fold_probs = clf.predict_proba(X_selected[val])[:, 1]
        oof_probs[val] = fold_probs

        # 实时评估
        fold_acc = accuracy_score(y[val], fold_probs > 0.5)
        fold_auc = roc_auc_score(y[val], fold_probs)
        accs.append(fold_acc)
        aucs.append(fold_auc)
        print(f"   Fold {fold + 1}: ACC = {fold_acc:.4f}, AUC = {fold_auc:.4f}")

    print("=" * 40)
    print(f"Dual-Tower 平均 ACC: {np.mean(accs):.4f}")
    print(f"Dual-Tower 平均 AUC: {np.mean(aucs):.4f}")
    print("=" * 40)

    save_path = "preds_plm_AIP.npz"
    np.savez(save_path, probs=oof_probs, labels=y)
    print(f"√ 集成数据已保存至: {save_path}")

if __name__ == "__main__":
    main()