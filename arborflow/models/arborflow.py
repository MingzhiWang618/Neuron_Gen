"""Milestone 4 geometry-only ArborFlow model with oracle topology."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from arborflow.data.dynamic_batch import GeometryBatch
from arborflow.models.branch_embedding import BranchEmbedding
from arborflow.models.tree_positional_encoding import TreeAttentionBias
from arborflow.models.tree_transformer import TreeTransformer
from arborflow.models.velocity_head import VelocityHead


@dataclass(frozen=True)
class GeometryModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.0
    max_swc_type: int = 31
    max_depth: int = 64
    path_buckets: int = 257
    max_tree_distance: int = 16

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("model dimensions and layer count must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.feedforward_dim < self.d_model:
            raise ValueError("feedforward_dim must be at least d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_depth < 1 or self.path_buckets < 2 or self.max_tree_distance < 1:
            raise ValueError("tree positional limits are invalid")


class GeometryFlowModel(nn.Module):
    """Predict geometry velocity for every currently born branch token."""

    def __init__(self, config: GeometryModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeometryModelConfig()
        self.embedding = BranchEmbedding(
            self.config.d_model,
            max_swc_type=self.config.max_swc_type,
            max_depth=self.config.max_depth,
            path_buckets=self.config.path_buckets,
        )
        self.tree_bias = TreeAttentionBias(
            self.config.num_heads,
            max_tree_distance=self.config.max_tree_distance,
        )
        self.backbone = TreeTransformer(
            self.config.d_model,
            self.config.num_heads,
            self.config.num_layers,
            self.config.feedforward_dim,
            dropout=self.config.dropout,
        )
        self.velocity_head = VelocityHead(self.config.d_model)

    def encode(self, batch: GeometryBatch) -> torch.Tensor:
        values = self.embedding(batch)
        bias = self.tree_bias(batch.shortest_path_distance, batch.relation)
        return self.backbone(values, batch.padding_mask, bias)

    def forward(self, batch: GeometryBatch) -> torch.Tensor:
        return self.velocity_head(self.encode(batch), batch.padding_mask)
