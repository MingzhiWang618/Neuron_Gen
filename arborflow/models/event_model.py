"""Milestone 5 event model with the same dynamic Tree Transformer backbone."""

from __future__ import annotations

import torch

from arborflow.data.event_dataset import EventBatch
from arborflow.models.arborflow import GeometryFlowModel, GeometryModelConfig
from arborflow.models.event_head import EventHead


class EventFlowModel(GeometryFlowModel):
    """Predict WAIT/EXTEND/SPLIT/STOP without a geometry velocity head."""

    def __init__(self, config: GeometryModelConfig | None = None) -> None:
        super().__init__(config)
        del self.velocity_head
        self.event_head = EventHead(self.config.d_model)

    def forward(self, batch: EventBatch) -> torch.Tensor:
        values = self.encode(batch)
        return self.event_head(values, batch.padding_mask)
