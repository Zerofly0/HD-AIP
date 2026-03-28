import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
import config
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")


def main():
    # 1. 加载 ProtT5 + Phys + CKSAAP
    print("[1/3] 加载 ProtT5 + Phys + CKSAAP...")
    data_t5 = np.load(config.FEATURE_CACHE, allow_pickle=True)
    X_t5_phys = data_t5["X"]
    y = data_t5["y"]

    # 2. 加载 ESM-2 3B
    print("[2/3] 加载 ESM-2 3B...")
    esm_file = os.path.join(config.CACHE_DIR, "esm2_3b_features_AIP.npz")
    if not os.path.exists(esm_file):
        print("错误：请先运行 7_extract_esm2_3b.py")
        return
    data_esm = np.load(esm_file, allow_pickle=True)
    X_esm = data_esm["X"]

    # 3. 超级融合
    print(f"[3/3] 执行特征级 Fusion...")
    X_final = np.hstack([X_t5_phys, X_esm])
    print(f"原始特征维度: {X_final.shape[1]}")

    # 4. 特征筛选 (针对全集进行一次特征重要性评估)
    print(">>> 正在筛选 Top 300 核心特征...")
    selector = LGBMClassifier(n_estimators=100, random_state=42)
    selector.fit(X_final, y)
    top_indices = np.argsort(selector.feature_importances_)[::-1][:300]
    X_selected = X_final[:, top_indices]

    # 5. 准备 OOF (Out-of-Fold) 容器
    # 必须保存全量样本的预测概率，才能和另一个模型对齐
    oof_probs = np.zeros(len(y))

    # 设置模型参数
    params = config.LGBM_PARAMS.copy()
    params['num_leaves'] = 128
    params['colsample_bytree'] = 0.5

    # 6. 开始 5-Fold 验证并生成预测结果
    print(">>> 开始生成 OOF 预测结果 (用于后续 Fusion) <<<")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []

    for fold, (tr, val) in enumerate(skf.split(X_selected, y)):
        clf = LGBMClassifier(**params)
        clf.fit(X_selected[tr], y[tr])

        # 记录验证集的概率
        fold_probs = clf.predict_proba(X_selected[val])[:, 1]
        oof_probs[val] = fold_probs

        # 实时评估
        fold_acc = accuracy_score(y[val], fold_probs > 0.5)
        fold_auc = roc_auc_score(y[val], fold_probs)
        accs.append(fold_acc)
        aucs.append(fold_auc)
        print(f"Fold {fold + 1}: ACC={fold_acc:.4f}, AUC={fold_auc:.4f}")

    print("=" * 30)
    print(f"✅ 平均 ACC: {np.mean(accs):.4f}")
    print(f"✅ 平均 AUC: {np.mean(aucs):.4f}")
    print("=" * 30)

    # 7. 保存结果为 npz 文件，供集成脚本 main.py 调用
    save_path = "preds_old_model_AIP.npz"
    np.savez(save_path, probs=oof_probs, labels=y)
    print(f"🚀 集成数据已保存至: {save_path}")
    print("现在你可以运行 main.py 进行双流融合了！")


if __name__ == "__main__":
    main()