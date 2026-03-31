import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)# 不需要反向传播

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ParallelResNetTransformer(nn.Module):
    """
    CT-Net: A Dual-Branch Network with Physicochemical Gating
    """
    def __init__(self, vocab_size, embed_dim, phys_dim, hidden_size, output_dim, embedding_matrix=None):
        super(ParallelResNetTransformer, self).__init__()

        # 1. Semantic Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = True # 可微调

        self.embed_dropout = nn.Dropout(0.2)

        # 2. Physicochemical Feature Extraction (MLP)
        self.phys_ext = nn.Sequential(
            nn.Linear(phys_dim, phys_dim * 2),
            nn.ReLU(),
            nn.Linear(phys_dim * 2, phys_dim)
        )

        # === Branch 1: Residual CNN (Local Motifs) ===
        self.input_dim = embed_dim + phys_dim
        self.res_cnn = nn.Sequential(
            ResidualBlock(self.input_dim, hidden_size),
            ResidualBlock(hidden_size, hidden_size)
        )
        self.cnn_dropout = nn.Dropout(0.3)

        # === Branch 2: Transformer (Global Dependencies) ===
        self.trans_dim = embed_dim  # 消除硬编码 128
        self.pos_encoder = PositionalEncoding(self.trans_dim, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.trans_dim,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # === Fusion & Classification ===
        self.fusion_dim = hidden_size + self.trans_dim
        self.fc1 = nn.Linear(self.fusion_dim, 64)
        self.fc_dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x_idx, x_phys):
        # 1. 特征增强 (Feature Enhancement)
        x_phys_enhanced = self.phys_ext(x_phys)

        emb = self.embedding(x_idx)
        emb = self.embed_dropout(emb)
        x_interact = emb

        # 2. Branch 1: CNN (提取局部特征)
        x_cnn_input = torch.cat([x_interact, x_phys_enhanced], dim=2)
        x_cnn = x_cnn_input.permute(0, 2, 1)
        c = self.res_cnn(x_cnn)
        c = self.cnn_dropout(c)
        c_out = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)

        # 3. Branch 2: Transformer (提取全局依赖)
        t = self.pos_encoder(x_interact)
        t_out = self.transformer(t)
        t_out = t_out.mean(dim=1)

        # 4. 特征级拼接与分类 (Feature Fusion)
        fusion = torch.cat([c_out, t_out], dim=1)
        out = F.relu(self.fc1(fusion))
        out = self.fc_dropout(out)
        logits = self.fc2(out)

        return logits