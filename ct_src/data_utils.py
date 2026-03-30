import numpy as np
import torch
from torch.utils.data import Dataset
import random  # 引入 random
import config
from phy_utils import get_phys_features

# 标准 20 种氨基酸
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


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

def seq_to_kmers(seq, k=3):
    if len(seq) < k: return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

class AIPDataset(Dataset):
    def __init__(self, sequences, labels, vocab, padding_idx, max_len, k=3, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.k = k
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def random_mutate(self, seq):
        """执行随机氨基酸点突变增强数据"""
        if random.random() > config.AUGMENT_PROB:
            return seq

        res = list(seq)
        for i in range(len(res)):
            if random.random() < config.MUTATION_RATE:
                res[i] = random.choice(AMINO_ACIDS)
        return "".join(res)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.augment:
            seq = self.random_mutate(seq)

        # BioVec 索引
        kmer_list = seq_to_kmers(seq, self.k)
        if not kmer_list: kmer_list = [seq] if seq else ["<UNK>"]
        indices = [self.vocab[kmer] if kmer in self.vocab else self.padding_idx for kmer in kmer_list]

        # 理化特征增强[L,7]
        phys_seq = seq[:len(indices)]
        phys_tensor = get_phys_features(phys_seq)
        if not isinstance(phys_tensor, torch.Tensor):
            phys_tensor = torch.tensor(phys_tensor, dtype=torch.float32)

        # 计算电荷密度
        total_charge = torch.sum(phys_tensor[:, 6])
        charge_density = total_charge / len(seq)

        # 特征拼接，拼接后[L,8]
        density_feat = torch.full((phys_tensor.size(0), 1), charge_density.item())
        phys_tensor = torch.cat([phys_tensor, density_feat], dim=1)

        # Padding，长截短补
        curr_len = len(indices)
        if curr_len > self.max_len:
            indices = indices[:self.max_len]
            phys_tensor = phys_tensor[:self.max_len, :]
            curr_len = self.max_len

        pad_len = self.max_len - curr_len
        if pad_len > 0:
            indices += [self.padding_idx] * pad_len
            # Padding补零，维度为[pad_len,8]
            pad_phys = torch.zeros((pad_len, 8))
            phys_tensor = torch.cat([phys_tensor, pad_phys], dim=0)

        return torch.tensor(indices, dtype=torch.long), phys_tensor, torch.tensor(label, dtype=torch.float32)