import re
import math
import numpy as np
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ================= 全局静态常量 (大幅提升循环提取速度) =================
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
# 预先生成 400 种氨基酸对 (用于 CKSAAP)
AA_PAIRS = ["".join(p) for p in product(AA_ALPHABET, repeat=2)]

# Eisenberg 疏水性量表
HYDROPHOBICITY_SCALE = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

# CTD 特征分组映射表
CTD_GROUPS_HYDRO = {
    'R': 1, 'K': 1, 'E': 1, 'D': 1, 'Q': 1, 'N': 1,
    'G': 2, 'A': 2, 'S': 2, 'T': 2, 'P': 2, 'H': 2, 'Y': 2,
    'C': 3, 'V': 3, 'L': 3, 'I': 3, 'M': 3, 'F': 3, 'W': 3
}

CTD_GROUPS_CHARGE = {
    'K': 1, 'R': 1,
    'A': 2, 'N': 2, 'C': 2, 'Q': 2, 'G': 2, 'H': 2, 'I': 2, 'L': 2, 'M': 2, 'F': 2, 'P': 2, 'S': 2, 'T': 2, 'W': 2, 'Y': 2, 'V': 2,
    'D': 3, 'E': 3
}

CTD_GROUPS_POLARITY = {
    'L': 1, 'I': 1, 'F': 1, 'W': 1, 'C': 1, 'M': 1, 'V': 1, 'Y': 1,
    'P': 2, 'A': 2, 'T': 2, 'G': 2, 'S': 2,
    'H': 3, 'Q': 3, 'R': 3, 'K': 3, 'N': 3, 'E': 3, 'D': 3
}

CTD_PROPERTIES = [CTD_GROUPS_HYDRO, CTD_GROUPS_CHARGE, CTD_GROUPS_POLARITY]
CTD_PROP_NAMES = ["Hydrophobicity", "Charge", "Polarity"]
# ======================================================================

def clean_sequence(seq):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())

def get_hydrophobic_moment(seq, angle=100):
    """计算疏水力矩 (Hydrophobic Moment)，默认 angle=100 对应 alpha-helix"""
    clean_seq = clean_sequence(seq)
    if not clean_seq: return 0.0

    h_vals = [HYDROPHOBICITY_SCALE.get(aa, 0) for aa in clean_seq]
    rad_angle = math.radians(angle)

    sin_sum = sum(h * math.sin(i * rad_angle) for i, h in enumerate(h_vals))
    cos_sum = sum(h * math.cos(i * rad_angle) for i, h in enumerate(h_vals))

    return math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(clean_seq)

def get_physicochemical_features(seq):
    """宏观理化特征 (29维，包含疏水力矩)"""
    clean_seq = clean_sequence(seq)
    length = len(clean_seq)
    if length == 0: return np.zeros(29)

    analyser = ProteinAnalysis(clean_seq)
    feats = [
        length,
        analyser.molecular_weight(),
        analyser.gravy(),
        analyser.isoelectric_point(),
        analyser.aromaticity(),
        analyser.instability_index(),
        (seq.count('K') + seq.count('R')) / length,
        (seq.count('D') + seq.count('E')) / length,
        get_hydrophobic_moment(seq)
    ]

    aa_counts = analyser.count_amino_acids()
    # 确保严格按照字母表顺序提取 AAC
    for aa in sorted(list(AA_ALPHABET)):
        feats.append(aa_counts.get(aa, 0) / length)

    return np.array(feats)

def get_cksaap_features(seq, gap):
    """计算 k-spaced 氨基酸对 (400维)"""
    clean_seq = clean_sequence(seq)
    length = len(clean_seq)
    if length < gap + 2: return np.zeros(400)

    # 字典初始化极其耗时，移到全局后使用 .copy() 或推导式可大幅加速
    pair_dict = {p: 0 for p in AA_PAIRS}

    for i in range(length - gap - 1):
        p = clean_seq[i] + clean_seq[i + gap + 1]
        if p in pair_dict:
            pair_dict[p] += 1

    total = length - gap - 1
    return np.array([pair_dict[p] / total for p in AA_PAIRS])

def get_feature_names():
    """生成所有特征名字"""
    # 1. Embedding (2048维)
    names = [f"Emb_Mean_{i}" for i in range(1024)] + [f"Emb_Max_{i}" for i in range(1024)]

    # 2. Physicochemical (29维)
    names += ["Length", "MolWeight", "Hydrophobicity", "pI", "Aromaticity", "Instability", "PosCharge", "NegCharge", "HydrophobicMoment"]
    names += [f"AAC_{aa}" for aa in sorted(list(AA_ALPHABET))]

    # 3. CKSAAP (2000维)
    for g in [0, 1, 2, 3, 4]:
        names += [f"CKSAAP_g{g}_{p}" for p in AA_PAIRS]

    return np.array(names)