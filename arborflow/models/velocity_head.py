"""Per-branch geometry velocity prediction."""

from __future__ import annotations

import torch
from torch import nn

from arborflow.data.dynamic_batch import GEOMETRY_DIM


class VelocityHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, GEOMETRY_DIM),
        )

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.network(values) * padding_mask.unsqueeze(-1)
