import torch
import torch.nn as nn


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return self.pe[:, :x.size(1), :]


class MultiScalePositionalEmbedding(nn.Module):
    """
    A simple multi-scale positional embedding by summing embeddings at multiple scales.
    """

    def __init__(self, d_model, scales=[1, 2, 4], max_len=5000):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Parameter(torch.randn(1, max_len, d_model)) for _ in scales
        ])

    def forward(self, x):
        seq_len = x.size(1)
        pe_sum = sum(pe[:, :seq_len, :] for pe in self.embeddings)
        return pe_sum
