import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import os
def read_fasta(file_path):
    """
    读取特定格式的 FASTA 文件
    示例：
    >seq1|label=0
    DITVKNCVLKKSTNG

    """
    seqs, labels = [], []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                content = line[1:]
                parts = content.split('|')
                current_label = 0
                for part in parts:
                    if part.startswith("label="):
                        try:
                            current_label = int(part.split('=')[1])
                        except ValueError:
                            pass
                labels.append(current_label)
            else:
                seqs.append(line.upper())
    return seqs, np.array(labels)

def main():
    print("========== HD-AIP 独立测试集终极合成评估 ==========")

    # 1. 加载真实标签 (请确保路径指向你最终干净的测试集)
    TEST_FASTA = "data/ind_dataset.fasta"
    if not os.path.exists(TEST_FASTA):
        print(f"❌ 找不到测试集文件: {TEST_FASTA}")
        return

    _, labels = read_fasta(TEST_FASTA)
    y_true = np.array(labels)

    # 2. 加载两个子流的预测概率
    try:
        p_ct = np.load("ct_src/ind_probs_ct.npy")
        p_ml = np.load("plm_src/ind_probs_ml.npy")
    except FileNotFoundError:
        print("❌ 找不到概率文件！请先运行 predict_ct_ind.py 和 evaluate_ind.py。")
        return

    if len(y_true) != len(p_ct) or len(y_true) != len(p_ml):
        print("❌ 样本数量不匹配，请检查数据！")
        return

    # 3. 填入你在融合训练阶段 (fusion.py) 锁定的最优超参数！
    # ⚠️ 请将这里的数值替换为你当时 fusion.py 打印出来的"最佳权重分配"和"最优截断阈值"
    best_w = 0.52  # 假设这是 CT-Net 的最优权重
    best_t = 0.46  # 假设这是最优截断阈值

    w_ml = 1.0 - best_w

    print(f">>> 严格沿用训练集参数进行软集成: CT-Net({best_w}) + ML({w_ml:.2f}), 阈值={best_t}")

    # 4. 执行动态软集成 (Strategy A)
    p_final = (best_w * p_ct) + (w_ml * p_ml)
    y_pred = (p_final > best_t).astype(int)

    # 5. 计算各项严苛评估指标
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, p_final)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 6. 输出结果
    print("\n🏆 HD-AIP 独立集双流集成最终表现 🏆")
    print(f"   ACC (准确率)    : {acc:.4f}")
    print(f"   AUC (ROC曲线)   : {auc:.4f}")
    print(f"   MCC (马修斯系数): {mcc:.4f}")
    print(f"   Sn  (敏感度)    : {sn:.4f}")
    print(f"   Sp  (特异度)    : {sp:.4f}")
    print(f"   Precision       : {precision:.4f}")
    print("===================================================")


if __name__ == "__main__":
    main()