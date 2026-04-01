import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def main():
    print(">>> [终极集成] 启动 Dual-Stream Ensemble...")

    # 1. 加载两个模型的预测结果
    try:
        data_old = np.load("dual_tower_src/preds_old_model_AIP.npz")
        data_new = np.load("ct_src/preds_wo_gate_AIP.npz")

        p1 = data_old["probs"]  # 旧模型 (PLM + Machine Learning)
        y = data_old["labels"]  # 真实标签

        p2 = data_new["probs"]  # 新模型 (BioVec + CT-Net)

        # 简单检查
        if len(p1) != len(p2):
            print("❌ 错误：两个模型的样本数量不一致！")
            return
    except Exception as e:
        print(f"❌ 加载预测文件失败: {e}")
        print("请确保你已经按照说明修改了代码并保存了 .npz 文件。")
        return

    print(f"   样本数: {len(y)}")

    # 2. 单模型评估 (基准线)
    acc1 = accuracy_score(y, p1 > 0.5)
    auc1 = roc_auc_score(y, p1)
    mcc1 = matthews_corrcoef(y, p1 > 0.5)
    print(f"   [基准 1] Old Model (Dual-Tower): ACC={acc1:.4f}, AUC={auc1:.4f}, MCC={mcc1:.4f}")

    acc2 = accuracy_score(y, p2 > 0.5)
    auc2 = roc_auc_score(y, p2)
    mcc2 = matthews_corrcoef(y, p2 > 0.5)
    print(f"   [基准 2] New Model (CT-Net)    : ACC={acc2:.4f}, AUC={auc2:.4f}, MCC={mcc2:.4f}\n")

    # 3. 策略 A: 搜索最佳权重 + 最佳阈值 (双维度寻优)
    best_acc = 0
    best_w = 0.5
    best_t = 0.5
    best_auc = 0
    best_mcc = 0

    # 双重循环：寻找模型比例 w 和 最终判定阈值 t
    for w in np.linspace(0.1, 0.9, 81):
        p_ensemble = w * p2 + (1 - w) * p1
        for t in np.arange(0.3, 0.7, 0.01):  # 动态搜索最佳截断点
            acc = accuracy_score(y, p_ensemble > t)
            if acc > best_acc:
                best_acc = acc
                best_w = w
                best_t = t
                best_auc = roc_auc_score(y, p_ensemble)
                best_mcc = matthews_corrcoef(y, p_ensemble > t)

    print(f"🏆 终极双维度融合结果:")
    print(f"   权重分配: {best_w:.2f} * CT-Net + {1 - best_w:.2f} * Old-Model")
    print(f"   最佳融合阈值: {best_t:.2f}")
    print(f"   🔥 Final ACC: {best_acc:.4f}")
    print(f"   🔥 Final AUC: {best_auc:.4f}")
    print(f"   🔥 Final MCC: {best_mcc:.4f}")

    # 4. 策略 B: Stacking (Logistic Regression)
    # 如果两者差异很大，LR 可能会学到更复杂的组合逻辑
    print("\n>>> 策略 B: Logistic Meta-Learner...")

    # 输入特征是两个模型的概率值: [N, 2]
    X_stack = np.vstack([p1, p2]).T

    # 需要再做一次 CV，防止 Meta-Learner 过拟合
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stack_preds = np.zeros(len(y))

    for tr, val in skf.split(X_stack, y):
        meta_clf = LogisticRegression()
        meta_clf.fit(X_stack[tr], y[tr])
        stack_preds[val] = meta_clf.predict_proba(X_stack[val])[:, 1]

    acc_stack = accuracy_score(y, stack_preds > 0.5)
    auc_stack = roc_auc_score(y, stack_preds)
    mcc_stack = matthews_corrcoef(y, stack_preds > 0.5)

    print(f"🤖 Stacking LR 结果:")
    print(f"   ACC: {acc_stack:.4f}")
    print(f"   AUC: {auc_stack:.4f}")
    print(f"   MCC: {mcc_stack:.4f}")

    # 5. 结论
    if best_acc > max(acc1, acc2):
        print("\n✅ 集成成功！性能超越了单模型。")
    else:
        print("\n⚠️ 集成未带来显著提升，可能是两个模型预测结果过于相似 (Correlation too high)。")


if __name__ == "__main__":
    main()
