"""Training loops for staged ArborFlow objectives."""

from arborflow.training.event_trainer import (
    EventTrainingConfig,
    EventTrainingResult,
    train_event_model,
)
from arborflow.training.geometry_trainer import (
    GeometryTrainingConfig,
    GeometryTrainingResult,
    train_geometry_flow,
)

__all__ = [
    "EventTrainingConfig",
    "EventTrainingResult",
    "GeometryTrainingConfig",
    "GeometryTrainingResult",
    "train_event_model",
    "train_geometry_flow",
]
