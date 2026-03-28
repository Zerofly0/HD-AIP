import re
import numpy as np
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def clean_sequence(seq):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())


def get_physicochemical_features(seq):
    """宏观理化特征 (28维)"""
    clean_seq = clean_sequence(seq)
    if len(clean_seq) == 0: return np.zeros(28)
    analyser = ProteinAnalysis(clean_seq)
    feats = [
        len(clean_seq), analyser.molecular_weight(), analyser.gravy(),
        analyser.isoelectric_point(), analyser.aromaticity(), analyser.instability_index(),
        (seq.count('K') + seq.count('R')) / len(clean_seq),
        (seq.count('D') + seq.count('E')) / len(clean_seq)
    ]
    aa_counts = analyser.count_amino_acids()
    for aa in sorted(aa_counts.keys()):
        feats.append(aa_counts[aa] / len(clean_seq))
    return np.array(feats)


def get_cksaap_features(seq, gap):
    """
    计算 k-spaced 氨基酸对 (400维)
    gap=0 即为 DPC
    """
    clean_seq = clean_sequence(seq)
    length = len(clean_seq)
    if length < gap + 2: return np.zeros(400)

    aas = sorted("ACDEFGHIKLMNPQRSTVWY")
    pairs = ["".join(p) for p in product(aas, repeat=2)]
    pair_dict = {p: 0 for p in pairs}

    # 统计间隔为 gap 的对子
    for i in range(length - gap - 1):
        p = clean_seq[i] + clean_seq[i + gap + 1]
        if p in pair_dict:
            pair_dict[p] += 1

    total = length - gap - 1
    return np.array([pair_dict[p] / total for p in pairs])


def get_feature_names():
    """生成所有特征名字 (Emb_Mean/Max + Phys + CKSAAP + CTD)"""

    # 1. Embedding (2048维)
    names = [f"Emb_Mean_{i}" for i in range(1024)] + [f"Emb_Max_{i}" for i in range(1024)]

    # 2. Physicochemical (29维: 9个宏观性质 + 20个AAC)
    names += ["Length", "MolWeight", "Hydrophobicity", "pI", "Aromaticity", "Instability", "PosCharge", "NegCharge",
              "HydrophobicMoment"]
    names += [f"AAC_{aa}" for aa in sorted("ACDEFGHIKLMNPQRSTVWY")]

    # 3. CKSAAP (2000维: Gap 0, 1, 2, 3, 4)
    aas = sorted("ACDEFGHIKLMNPQRSTVWY")
    pairs = ["".join(p) for p in product(aas, repeat=2)]
    for g in [0, 1, 2, 3, 4]:
        names += [f"CKSAAP_g{g}_{p}" for p in pairs]

    # 4. CTD (57维: 3种性质 * 19个指标)
    # 对应 get_ctd_features 中的三个性质分组
    ctd_props = ["Hydrophobicity", "Charge", "Polarity"]

    for prop in ctd_props:
        # Composition (C): 3个 (Group 1, 2, 3)
        names += [f"CTD_{prop}_C1", f"CTD_{prop}_C2", f"CTD_{prop}_C3"]

        # Transition (T): 1个
        names += [f"CTD_{prop}_T"]

        # Distribution (D): 3个组 * 5个节点 (0, 25, 50, 75, 100%)
        for group in [1, 2, 3]:
            for p in [0, 25, 50, 75, 100]:
                names += [f"CTD_{prop}_D{group}_{p}"]

    return np.array(names)


import math


def get_hydrophobic_moment(seq, angle=100):
    """
    计算疏水力矩 (Hydrophobic Moment)
    angle=100 对应 alpha-helix
    """
    clean_seq = clean_sequence(seq)
    if len(clean_seq) == 0: return 0.0

    # Eisenberg 疏水性量表 (标准化)
    hydrophobicity_scale = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }

    h_vals = [hydrophobicity_scale.get(aa, 0) for aa in clean_seq]

    # 转换为弧度
    rad_angle = math.radians(angle)

    sin_sum = 0.0
    cos_sum = 0.0

    for i, h in enumerate(h_vals):
        sin_sum += h * math.sin(i * rad_angle)
        cos_sum += h * math.cos(i * rad_angle)

    moment = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(clean_seq)
    return moment


# 修改 get_physicochemical_features，加入 Moment
def get_physicochemical_features(seq):
    clean_seq = clean_sequence(seq)
    if len(clean_seq) == 0: return np.zeros(29)  # 注意维度变了 +1

    analyser = ProteinAnalysis(clean_seq)
    feats = [
        len(clean_seq),
        analyser.molecular_weight(),
        analyser.gravy(),
        analyser.isoelectric_point(),
        analyser.aromaticity(),
        analyser.instability_index(),
        (seq.count('K') + seq.count('R')) / len(clean_seq),
        (seq.count('D') + seq.count('E')) / len(clean_seq),
        get_hydrophobic_moment(seq)  # <--- 新增这一项
    ]

    aa_counts = analyser.count_amino_acids()
    for aa in sorted(aa_counts.keys()):
        feats.append(aa_counts[aa] / len(clean_seq))

    return np.array(feats)


def get_ctd_features(seq):
    """
    计算 CTD (Composition, Transition, Distribution) 特征
    基于 7 种理化性质:
    1. Hydrophobicity, 2. Normalized Van der Waals Volume, 3. Polarity,
    4. Polarizability, 5. Charge, 6. Secondary Structure, 7. Solvent Accessibility
    """
    clean_seq = clean_sequence(seq)
    N = len(clean_seq)
    if N == 0: return np.zeros(147)  # 7 props * (3 groups) * (1 C + 1 T + 5 D) = 147 dims

    # 定义性质分组 (Group 1, 2, 3)
    # 这里仅示例最关键的 'Hydrophobicity' (P1) 和 'Charge' (P2) 和 'Polarity' (P3)
    # 完整 CTD 有 7 类，为精简代码，我们实现这 3 类最核心的 (3 * 21 = 63 dims)

    # 1. Hydrophobicity (Polar, Neutral, Hydrophobic)
    # Group 1: R,K,E,D,Q,N (Polar)
    # Group 2: G,A,S,T,P,H,Y (Neutral)
    # Group 3: C,V,L,I,M,F,W (Hydrophobic)
    groups_hydro = {
        'R': 1, 'K': 1, 'E': 1, 'D': 1, 'Q': 1, 'N': 1,
        'G': 2, 'A': 2, 'S': 2, 'T': 2, 'P': 2, 'H': 2, 'Y': 2,
        'C': 3, 'V': 3, 'L': 3, 'I': 3, 'M': 3, 'F': 3, 'W': 3
    }

    # 2. Charge (Positive, Neutral, Negative)
    groups_charge = {
        'K': 1, 'R': 1,  # Positive
        'A': 2, 'N': 2, 'C': 2, 'Q': 2, 'G': 2, 'H': 2, 'I': 2, 'L': 2, 'M': 2, 'F': 2, 'P': 2, 'S': 2, 'T': 2, 'W': 2,
        'Y': 2, 'V': 2,  # Neutral
        'D': 3, 'E': 3  # Negative
    }

    # 3. Polarity (Polar, Non-polar) - 简化为2类或是按数值分
    # 这里用常见的 3 分类
    # G1 (4.9-6.2): L,I,F,W,C,M,V,Y
    # G2 (8.0-9.2): P,A,T,G,S
    # G3 (10.4-13.0): H,Q,R,K,N,E,D
    groups_polarity = {
        'L': 1, 'I': 1, 'F': 1, 'W': 1, 'C': 1, 'M': 1, 'V': 1, 'Y': 1,
        'P': 2, 'A': 2, 'T': 2, 'G': 2, 'S': 2,
        'H': 3, 'Q': 3, 'R': 3, 'K': 3, 'N': 3, 'E': 3, 'D': 3
    }

    properties = [groups_hydro, groups_charge, groups_polarity]
    ctd_vector = []

    for group_map in properties:
        # 映射序列为 1,2,3
        coded = [group_map.get(aa, 2) for aa in clean_seq]  # 默认归为中性

        # --- Composition (C) ---
        c = [coded.count(g) / N for g in [1, 2, 3]]
        ctd_vector.extend(c)

        # --- Transition (T) ---
        t = 0
        if N > 1:
            for i in range(N - 1):
                if coded[i] != coded[i + 1]:
                    t += 1
            ctd_vector.append(t / (N - 1))
        else:
            ctd_vector.append(0)

        # --- Distribution (D) ---
        # 记录 1,2,3 分别在 0%, 25%, 50%, 75%, 100% 出现的位置
        for g in [1, 2, 3]:
            indices = [i for i, x in enumerate(coded) if x == g]
            if not indices:
                ctd_vector.extend([0.0] * 5)
            else:
                # 归一化位置 (1-based / N)
                d = []
                for p in [0, 25, 50, 75, 100]:
                    # 找到第 p% 个出现的位置
                    k = int(len(indices) * p / 100)
                    if k >= len(indices): k = len(indices) - 1
                    d.append((indices[k] + 1) / N)
                ctd_vector.extend(d)

    return np.array(ctd_vector)