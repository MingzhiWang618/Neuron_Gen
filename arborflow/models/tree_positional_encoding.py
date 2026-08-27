"""Learned full-attention biases derived solely from rooted-tree relations."""

from __future__ import annotations

import torch
from torch import nn


class TreeAttentionBias(nn.Module):
    """Per-head shortest-path and ancestor/descendant/sibling attention bias."""

    def __init__(self, num_heads: int, *, max_tree_distance: int = 16) -> None:
        super().__init__()
        self.max_tree_distance = max_tree_distance
        self.distance_embedding = nn.Embedding(max_tree_distance + 2, num_heads)
        self.relation_embedding = nn.Embedding(5, num_heads)

    def forward(
        self, shortest_path_distance: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        distance = shortest_path_distance.clamp(0, self.max_tree_distance + 1)
        bias = self.distance_embedding(distance) + self.relation_embedding(relation)
        return bias.permute(0, 3, 1, 2)
