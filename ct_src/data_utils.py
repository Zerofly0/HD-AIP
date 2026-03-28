import numpy as np
import torch
from torch.utils.data import Dataset
import random  # 引入 random
import config
from phy_utils import get_phys_features

# 标准 20 种氨基酸
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(file_path):
    """读取特定格式的 FASTA 文件"""
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

def seq_to_kmers(seq, k=3):
    if len(seq) < k: return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

class AIPDataset(Dataset):
    def __init__(self, sequences, labels, vocab, padding_idx, max_len, k=3, augment=False, hard_seqs=None,
                 error_df=None):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.k = k
        self.augment = augment
        self.hard_seqs = hard_seqs if hard_seqs is not None else []
        self.error_df = error_df
        self.weights = self._calculate_weights()

    def __len__(self):
        return len(self.sequences)

    def _calculate_weights(self):
        # 防御性判断：如果没有 error_df，则执行基础权重逻辑
        if self.error_df is None:
            return [2.0 if l == 1 else 1.0 for l in self.labels]

        prob_dict = dict(zip(self.error_df['sequence'], self.error_df['prob']))

        ws = []
        for seq, label in zip(self.sequences, self.labels):
            w = 1.0
            if label == 1:
                if seq in prob_dict:
                    # 如果是漏报(FN)，且概率在 0.3-0.5 之间，说明是“极难样本”
                    prob = prob_dict[seq]
                    if 0.3 <= prob <= 0.5:
                        w = 3.5  # 强力重采样
                    else:
                        w = 2.0
                else:
                    w = 1.2
            else:  # 负样本
                if seq in prob_dict:
                    prob = prob_dict[seq]
                    if 0.5 <= prob <= 0.7:  # 容易误报的负样本
                        w = 1.5
                w = 1.0
            ws.append(w)
        return ws

    def random_mutate(self, seq):
        """执行随机氨基酸点突变增强数据"""
        # 只有在 config 中定义的概率下才触发增强
        if random.random() > config.AUGMENT_PROB:
            return seq

        res = list(seq)
        for i in range(len(res)):
            # 每个点位有 MUTATION_RATE 的概率发生突变
            if random.random() < config.MUTATION_RATE:
                res[i] = random.choice("ACDEFGHIKLMNPQRSTVWY")
        return "".join(res)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.augment:
            seq = self.random_mutate(seq)

        # 1. BioVec 索引
        kmer_list = seq_to_kmers(seq, self.k)
        if not kmer_list: kmer_list = [seq] if seq else ["<UNK>"]
        indices = [self.vocab[kmer] if kmer in self.vocab else self.padding_idx for kmer in kmer_list]

        # 2. 理化特征增强
        phys_seq = seq[:len(indices)]
        phys_tensor = get_phys_features(phys_seq)  # [Seq, 7]
        if not isinstance(phys_tensor, torch.Tensor):
            phys_tensor = torch.tensor(phys_tensor, dtype=torch.float32)

        # 【关键改动】计算电荷密度 (假设第6列是侧链电荷)
        total_charge = torch.sum(phys_tensor[:, 6])
        charge_density = total_charge / len(seq)

        # 将电荷密度作为一个全局背景，扩充为 [Seq, 1] 维度的特征
        density_feat = torch.full((phys_tensor.size(0), 1), charge_density.item())
        # 拼接后特征变为 8 维
        phys_tensor = torch.cat([phys_tensor, density_feat], dim=1)

        # 3. Padding (注意 padding 维度改为 8)
        curr_len = len(indices)
        if curr_len > self.max_len:
            indices = indices[:self.max_len]
            phys_tensor = phys_tensor[:self.max_len, :]
            curr_len = self.max_len

        pad_len = self.max_len - curr_len
        if pad_len > 0:
            indices += [self.padding_idx] * pad_len
            # Padding 补零，维度为 [pad_len, 8]
            pad_phys = torch.zeros((pad_len, 8))
            phys_tensor = torch.cat([phys_tensor, pad_phys], dim=0)

        return torch.tensor(indices, dtype=torch.long), phys_tensor, torch.tensor(label, dtype=torch.float32)