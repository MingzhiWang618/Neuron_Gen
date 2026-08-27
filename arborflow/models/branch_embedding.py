"""Embedding of geometry, state, type, depth, and rooted path features."""

from __future__ import annotations

import torch
from torch import nn

from arborflow.data.dynamic_batch import CONTINUOUS_FEATURE_DIM, GeometryBatch


class BranchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        max_swc_type: int = 31,
        max_depth: int = 64,
        path_buckets: int = 257,
    ) -> None:
        super().__init__()
        self.max_swc_type = max_swc_type
        self.max_depth = max_depth
        self.path_buckets = path_buckets
        self.continuous = nn.Sequential(
            nn.Linear(CONTINUOUS_FEATURE_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.type_embedding = nn.Embedding(max_swc_type + 1, d_model)
        self.depth_embedding = nn.Embedding(max_depth + 1, d_model)
        self.child_embedding = nn.Embedding(3, d_model)
        self.path_embedding = nn.Embedding(path_buckets, d_model)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, batch: GeometryBatch) -> torch.Tensor:
        result = self.continuous(batch.continuous_features)
        result = result + self.type_embedding(batch.swc_type.clamp(0, self.max_swc_type))
        result = result + self.depth_embedding(batch.depth.clamp(0, self.max_depth))
        result = result + self.child_embedding(batch.child_position.clamp(0, 2))
        result = result + self.path_embedding(batch.path_code.remainder(self.path_buckets))
        result = self.normalization(result)
        return result * batch.padding_mask.unsqueeze(-1)
