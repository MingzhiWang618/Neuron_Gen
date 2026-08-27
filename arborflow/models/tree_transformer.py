"""Full-attention Transformer with rooted-tree relative biases."""

from __future__ import annotations

import math

import torch
from torch import nn


class TreeSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, *, attention_dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, d_model = values.shape
        qkv = self.qkv(values).reshape(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attention_bias
        scores = scores.masked_fill(
            ~padding_mask[:, None, None, :], -torch.finfo(scores.dtype).max
        )
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, d_model)
        return self.output(attended) * padding_mask.unsqueeze(-1)


class TreeTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = TreeSelfAttention(
            d_model, num_heads, attention_dropout=dropout
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        values = values + self.attention(
            self.attention_norm(values), padding_mask, attention_bias
        )
        values = values + self.feedforward(self.feedforward_norm(values))
        return values * padding_mask.unsqueeze(-1)


class TreeTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TreeTransformerLayer(
                d_model,
                num_heads,
                feedforward_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values, padding_mask, attention_bias)
        return self.output_norm(values) * padding_mask.unsqueeze(-1)
