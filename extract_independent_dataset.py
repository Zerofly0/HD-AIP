import os


def get_seqs_from_fasta(fasta_file):
    """提取指定 FASTA 文件中的所有序列，存入集合作为黑名单"""
    seqs = set()
    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 忽略空行和表头
            if line and not line.startswith(">"):
                seqs.add(line.upper())
    return seqs


def extract_and_format_pure_test(train_fasta, test_fasta, pure_test_out):
    print(f">>> [1/2] 正在加载训练集 {train_fasta} 的序列作为黑名单...")
    train_seqs = get_seqs_from_fasta(train_fasta)
    print(f"    √ 训练集名单加载完毕，共 {len(train_seqs)} 条独立序列。")

    pure_count = 0
    overlap_count = 0

    print(f">>> [2/2] 正在筛选并格式化测试集 {test_fasta} ...")
    with open(test_fasta, 'r', encoding='utf-8') as fin, \
            open(pure_test_out, 'w', encoding='utf-8') as fout:

        lines = fin.read().splitlines()

        # 按 FASTA 格式每次读取两行 (表头和序列)
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break

            header = lines[i]
            seq = lines[i + 1].strip().upper()

            # 如果序列不在 AIP.fasta 中，则安全保留并进行格式转换
            if seq not in train_seqs:
                pure_count += 1

                # --- 核心格式转换逻辑 ---
                # 原始 header 示例: >test|972|pos 或 >test|1|neg
                if "pos" in header.lower():
                    label = 1
                elif "neg" in header.lower():
                    label = 0
                else:
                    label = 0  # Fallback 容错

                # 构造与 AIP.fasta 完全一致的 header 格式: >Ind_seqX|label=Y
                new_header = f">Ind_seq{pure_count}|label={label}"

                fout.write(f"{new_header}\n{seq}\n")
            else:
                overlap_count += 1

    print("\n========== 独立测试集提取与格式化报告 ==========")
    print(f"🚨 因数据泄露(重叠)而被剔除的测试序列: {overlap_count} 条")
    print(f"✅ 成功提取并格式化【纯净且未知的独立测试集】: {pure_count} 条")
    print(f"📁 新文件已保存至: {pure_test_out}")
    print("================================================")


if __name__ == "__main__":
    # 请确保这两个文件在当前目录下
    TRAIN_FASTA = "data/AIP.fasta"
    TEST_FASTA = "data/BertAIP_test_dataset.fasta"

    # 最终生成的完美测试集
    PURE_TEST_OUT = "data/ind_dataset.fasta"

    if os.path.exists(TRAIN_FASTA) and os.path.exists(TEST_FASTA):
        extract_and_format_pure_test(TRAIN_FASTA, TEST_FASTA, PURE_TEST_OUT)
    else:
        print("❌ 错误：找不到源文件，请确保 AIP.fasta 和 BertAIP_test_dataset.fasta 存在。")