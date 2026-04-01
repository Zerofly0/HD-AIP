import os
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
    print(">>> 加载传统手工特征 (Phys + CKSAAP)...")
    data_base = np.load(config.FEATURE_CACHE, allow_pickle=True)
    X_base = data_base["X"]
    y = data_base["y"]

    # 剔除前 2048 维的 ProtT5 特征，仅保留传统手工特征
    X_handcraft = X_base[:, 2048:]
    print(f"   样本数: {len(y)}, 纯手工特征维度: {X_handcraft.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. SVM (线性核)
    svm_clf = SVC(kernel='linear', probability=True, random_state=42, max_iter=2000)
    # 2. Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # 3. XGBoost
    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')

    evaluate_model("SVM (Linear)", svm_clf, X_handcraft, y, skf)
    evaluate_model("Random Forest", rf_clf, X_handcraft, y, skf)
    evaluate_model("XGBoost", xgb_clf, X_handcraft, y, skf)


if __name__ == "__main__":
    main()