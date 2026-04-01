import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

import config
from data_utils import read_fasta, AIPDataset
from biovec import train_biovec


# ================= 经典模型库 =================

class TextCNNBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, phys_dim, hidden_size, output_dim, embedding_matrix=None):
        super(TextCNNBaseline, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim + phys_dim, hidden_size, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 3, output_dim)

    def forward(self, x_idx, x_phys):
        emb = self.embedding(x_idx)
        x_input = torch.cat([emb, x_phys], dim=2).permute(0, 2, 1)
        conv_outs = [F.relu(conv(x_input)) for conv in self.convs]
        pooled_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]
        return self.fc(self.dropout(torch.cat(pooled_outs, dim=1)))


class BiLSTMBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, phys_dim, hidden_size, output_dim, embedding_matrix=None):
        super(BiLSTMBaseline, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.lstm = nn.LSTM(embed_dim + phys_dim, hidden_size, num_layers=2, bidirectional=True, batch_first=True,
                            dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, x_idx, x_phys):
        emb = self.embedding(x_idx)
        x_input = torch.cat([emb, x_phys], dim=2)
        lstm_out, _ = self.lstm(x_input)
        return self.fc(lstm_out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class StandardTransformerBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, phys_dim, hidden_size, output_dim, embedding_matrix=None):
        super(StandardTransformerBaseline, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.input_proj = nn.Linear(embed_dim + phys_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_size * 4,
                                                   dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x_idx, x_phys):
        emb = self.embedding(x_idx)
        x_input = self.input_proj(torch.cat([emb, x_phys], dim=2))
        t_out = self.transformer(self.pos_encoder(x_input)).mean(dim=1)
        return self.fc2(F.relu(self.fc1(t_out)))


# ================= 统一评估框架 =================

def evaluate_dl_model(ModelClass, model_name, seqs_arr, labels, vocab, padding_idx, global_max_len, vocab_size,
                      embed_dim, phys_dim, embedding_matrix):
    print(f"\n{'=' * 20} 正在训练与评估: {model_name} {'=' * 20}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    final_accs, final_aucs, final_mccs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(seqs_arr, labels)):
        X_train, X_val = seqs_arr[train_idx], seqs_arr[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_ds = AIPDataset(X_train.tolist(), y_train, vocab, padding_idx, global_max_len, config.K_MER,
                              augment=False)
        val_ds = AIPDataset(X_val.tolist(), y_val, vocab, padding_idx, global_max_len, config.K_MER, False)

        sample_weights = [2.0 if label == 1 else 1.0 for label in y_train]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        model = ModelClass(vocab_size, embed_dim, phys_dim, config.HIDDEN_DIM, 1, embedding_matrix).to(config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()  # 使用标准 BCE 加速训练

        best_acc, best_auc, best_mcc = 0.0, 0.0, 0.0

        for epoch in range(15):  # 基线对比无需太多 epoch
            model.train()
            for X_batch, X_phys, y_batch in train_loader:
                X_batch, X_phys, y_batch = X_batch.to(config.DEVICE), X_phys.to(config.DEVICE), y_batch.to(
                    config.DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X_batch, X_phys).squeeze(), y_batch.float())
                loss.backward()
                optimizer.step()

            model.eval()
            all_labels, all_probs = [], []
            with torch.no_grad():
                for X_batch, X_phys, y_batch in val_loader:
                    X_batch, X_phys = X_batch.to(config.DEVICE), X_phys.to(config.DEVICE)
                    probs = torch.sigmoid(model(X_batch, X_phys)).squeeze()
                    if probs.dim() == 0: probs = probs.unsqueeze(0)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(y_batch.numpy())

            preds = (np.array(all_probs) > 0.5).astype(int)
            curr_acc = accuracy_score(all_labels, preds)
            curr_auc = roc_auc_score(all_labels, all_probs)
            curr_mcc = matthews_corrcoef(all_labels, preds)

            if curr_acc > best_acc:
                best_acc, best_auc, best_mcc = curr_acc, curr_auc, curr_mcc

        print(f"   Fold {fold + 1}: ACC={best_acc:.4f}, AUC={best_auc:.4f}, MCC={best_mcc:.4f}")
        final_accs.append(best_acc)
        final_aucs.append(best_auc)
        final_mccs.append(best_mcc)

    print(
        f"🏆 {model_name} 最终平均结果: ACC={np.mean(final_accs):.4f}, AUC={np.mean(final_aucs):.4f}, MCC={np.mean(final_mccs):.4f}")


def main():
    print(">>> 准备 BioVec 数据与特征环境...")
    seqs, labels = read_fasta(config.DATA_PATH)
    labels = np.array(labels)
    seqs_arr = np.array(seqs)

    w2v_model = train_biovec(seqs)
    vocab = {word: i for i, word in enumerate(w2v_model.wv.index_to_key)}
    padding_idx = len(vocab)
    raw_vectors = w2v_model.wv.vectors
    padding_vector = np.zeros((1, raw_vectors.shape[1]), dtype=np.float32)
    embedding_matrix = np.concatenate([raw_vectors, padding_vector], axis=0)

    vocab_size, embed_dim = embedding_matrix.shape
    phys_dim = 8
    global_max_len = max([len(s) - config.K_MER + 1 for s in seqs])

    # 依次执行三种经典深度学习模型的评估
    evaluate_dl_model(TextCNNBaseline, "TextCNN", seqs_arr, labels, vocab, padding_idx, global_max_len, vocab_size,
                      embed_dim, phys_dim, embedding_matrix)
    evaluate_dl_model(BiLSTMBaseline, "BiLSTM", seqs_arr, labels, vocab, padding_idx, global_max_len, vocab_size,
                      embed_dim, phys_dim, embedding_matrix)
    evaluate_dl_model(StandardTransformerBaseline, "Standard Transformer", seqs_arr, labels, vocab, padding_idx,
                      global_max_len, vocab_size, embed_dim, phys_dim, embedding_matrix)


if __name__ == "__main__":
    main()