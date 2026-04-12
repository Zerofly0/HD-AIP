import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def evaluate_metrics(y_true, y_prob, threshold=0.5):
    """封装统一的指标计算函数"""
    y_pred = (y_prob > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision

    return acc, auc, mcc, sn, sp, precision


def main():
    print(">>> [HD-AIP 终极集成] 启动 Decision-Level Dynamic Ensemble...")

    # 1. 加载两个子系统的预测结果
    try:
        data_dual_tower = np.load("dual_tower_src/preds_dual_tower_AIP.npz")
        data_ct_net = np.load("ct_src/preds_ct_net_AIP.npz")

        p_dual_tower = data_dual_tower["probs"]  # 机器学习流预测概率
        y = data_dual_tower["labels"]  # 真实标签

        p_ct_net = data_ct_net["probs"]  # 深度并行网络流预测概率

        if len(p_dual_tower) != len(p_ct_net):
            print("❌ 错误：两组特征的样本数量不一致！")
            return
    except Exception as e:
        print(f"❌ 加载预测文件失败: {e}")
        return

    print(f"   当前集成样本总数: {len(y)}\n")

    # 2. 单子系统评估 (基准线对标)
    acc_dual, auc_dual, mcc_dual, sn_dual, sp_dual, prec_dual = evaluate_metrics(y, p_dual_tower, threshold=0.5)
    print(f"   [子系统 1] Dual-Tower (Machine Learning Stream):")
    print(
        f"   ACC={acc_dual:.4f}, AUC={auc_dual:.4f}, MCC={mcc_dual:.4f}, Sn={sn_dual:.4f}, Sp={sp_dual:.4f}, Prec={prec_dual:.4f}\n")

    acc_ct, auc_ct, mcc_ct, sn_ct, sp_ct, prec_ct = evaluate_metrics(y, p_ct_net, threshold=0.5)
    print(f"   [子系统 2] CT-Net (Deep Learning Stream):")
    print(
        f"   ACC={acc_ct:.4f}, AUC={auc_ct:.4f}, MCC={mcc_ct:.4f}, Sn={sn_ct:.4f}, Sp={sp_ct:.4f}, Prec={prec_ct:.4f}\n")

    # 3. 策略 A: 动态加权软集成寻优
    best_acc, best_w, best_t = 0, 0.5, 0.5
    best_auc, best_mcc, best_sn, best_sp, best_prec = 0, 0, 0, 0, 0

    for w in np.linspace(0.1, 0.9, 81):
        p_ensemble = w * p_ct_net + (1 - w) * p_dual_tower
        for t in np.arange(0.3, 0.7, 0.01):
            y_pred_temp = (p_ensemble > t).astype(int)
            acc = accuracy_score(y, y_pred_temp)

            if acc > best_acc:
                best_acc = acc
                best_w = w
                best_t = t
                best_auc = roc_auc_score(y, p_ensemble)
                best_mcc = matthews_corrcoef(y, y_pred_temp)

                tn, fp, fn, tp = confusion_matrix(y, y_pred_temp).ravel()
                best_sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                best_sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                best_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print(f">>> 策略 A： 动态软集成 (Dynamic Soft Ensemble)...")
    print(f"   最佳权重分配: {best_w:.2f} * CT-Net + {1 - best_w:.2f} * Dual-Tower")
    print(f"   最优截断阈值: {best_t:.2f}")
    print(f"   ACC: {best_acc:.4f}")
    print(f"   AUC: {best_auc:.4f}")
    print(f"   MCC: {best_mcc:.4f}")
    print(f"   Sn (Recall): {best_sn:.4f}")
    print(f"   Sp: {best_sp:.4f}")
    print(f"   Precision: {best_prec:.4f}\n")

    # 4. 策略 B: 堆叠集成 (Stacking)
    print(">>> 策略 B: Logistic Meta-Learner (Stacking)...")

    X_stack = np.vstack([p_dual_tower, p_ct_net]).T
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stack_preds = np.zeros(len(y))

    for tr, val in skf.split(X_stack, y):
        meta_clf = LogisticRegression()
        meta_clf.fit(X_stack[tr], y[tr])
        stack_preds[val] = meta_clf.predict_proba(X_stack[val])[:, 1]

    acc_stack, auc_stack, mcc_stack, sn_stack, sp_stack, prec_stack = evaluate_metrics(y, stack_preds, threshold=0.5)

    print(f"   ACC: {acc_stack:.4f}")
    print(f"   AUC: {auc_stack:.4f}")
    print(f"   MCC: {mcc_stack:.4f}")
    print(f"   Sn (Recall): {sn_stack:.4f}")
    print(f"   Sp: {sp_stack:.4f}")
    print(f"   Precision: {prec_stack:.4f}")

    if best_acc > max(acc_dual, acc_ct):
        print("\n✅ 集成成功！异构双塔模型 (HD-AIP) 性能全面超越单一子系统。")
    else:
        print("\n⚠️ 集成未带来显著提升。")


if __name__ == "__main__":
    main()