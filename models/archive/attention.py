import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbSparseAttention(nn.Module):
    """
    A simplified version of ProbSparse attention.
    This implementation computes the full attention scores but then masks out
    entries below a threshold (based on mean and std) to simulate sparsity.
    """

    def __init__(self, dropout=0.1, sparsity_factor=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.sparsity_factor = sparsity_factor

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch, heads, seq_len, d_k]
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # Define threshold: mean plus some multiple of std.
        threshold = scores.mean(
            dim=-1, keepdim=True
        ) + self.sparsity_factor * scores.std(dim=-1, keepdim=True)
        sparse_mask = scores >= threshold
        sparse_scores = scores.masked_fill(~sparse_mask, -1e9)
        attn = F.softmax(sparse_scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output


class LinearAttention(nn.Module):
    """
    A simple linearized attention mechanism using a kernel feature map (ELU+1).
    This replaces the standard quadratic attention with a mechanism having linear complexity.
    """

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Assuming Q, K, V: [batch, heads, seq_len, d_k]
        Q_prime = F.elu(Q) + 1  # shape: same as Q
        K_prime = F.elu(K) + 1
        # Aggregate K_prime and V by contracting over the sequence.
        KV = torch.einsum("bhlk,bhlv->bhlv", K_prime, V)
        # Compute a normalization factor.
        Z = 1 / (
            torch.einsum("bhlk,bhl->bhl", Q_prime, K_prime.sum(dim=-2)) + 1e-6
        )
        output = torch.einsum("bhlk,bhlv->bhlv", Q_prime, KV) * Z.unsqueeze(-1)
        return output
