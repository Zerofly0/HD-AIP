import os
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # 新增：用于保持一致的特征筛选
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
import config

warnings.filterwarnings("ignore")


def evaluate_model(model_name, clf, X, y, skf):
    print(f"\n>>> 正在评估 {model_name} ...")
    accs, aucs, mccs = [], [], []

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        clf.fit(X[tr], y[tr])

        # 获取概率或决策边界用于计算 AUC
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X[val])[:, 1]
        else:
            probs = clf.decision_function(X[val])
            probs = (probs - probs.min()) / (probs.max() - probs.min())

        preds = clf.predict(X[val])

        fold_acc = accuracy_score(y[val], preds)
        fold_auc = roc_auc_score(y[val], probs)
        fold_mcc = matthews_corrcoef(y[val], preds)

        accs.append(fold_acc)
        aucs.append(fold_auc)
        mccs.append(fold_mcc)
        print(f"   Fold {fold + 1}: ACC={fold_acc:.4f}, AUC={fold_auc:.4f}, MCC={fold_mcc:.4f}")

    print(f"🏆 {model_name} 最终平均结果: ACC={np.mean(accs):.4f}, AUC={np.mean(aucs):.4f}, MCC={np.mean(mccs):.4f}")


def main():
    print(">>> [1/3] 加载基础特征 (ProtT5 + Phys + CKSAAP)...")
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

    # 简单校验样本数是否对齐
    if len(X_base) != len(X_esm):
        print("× 错误: 基础特征与 ESM-2 特征样本数量不一致，请检查！")
        return

    print(">>> [3/3] 执行特征级拼接 (Feature Concat) 并筛选 Top 300...")
    X_final = np.hstack([X_base, X_esm])
    print(f"   样本数: {len(y)}, 原始拼接总维度: {X_final.shape[1]}")

    # --- 统一特征筛选机制 ---
    selector = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1)
    selector.fit(X_final, y)

    top_indices = np.argsort(selector.feature_importances_)[::-1][:300]
    X_selected = X_final[:, top_indices]
    print(f"   ✅ 已截取重要性排名前 300 的统一特征，维度: {X_selected.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. SVM (线性核)
    svm_clf = SVC(kernel='linear', probability=True, random_state=42, max_iter=2000)
    # 2. Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # 3. XGBoost
    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')

    # 使用统一提取后的 X_selected 替换原本的 X_handcraft
    evaluate_model("SVM (Linear)", svm_clf, X_selected, y, skf)
    evaluate_model("Random Forest", rf_clf, X_selected, y, skf)
    evaluate_model("XGBoost", xgb_clf, X_selected, y, skf)


if __name__ == "__main__":
    main()