"""Discrete frontier-event vocabulary, loss, and classification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch.nn import functional as F

from arborflow.structures.tree_events import EventType


class EventClass(IntEnum):
    WAIT = 0
    EXTEND = 1
    SPLIT = 2
    STOP = 3


NUM_EVENT_CLASSES = len(EventClass)
IGNORE_EVENT_INDEX = -100


def event_class(event_type: EventType) -> EventClass:
    return EventClass[event_type.value]


def masked_event_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    frontier_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy over current frontier leaves only."""

    selected = frontier_mask & (labels != IGNORE_EVENT_INDEX)
    if not bool(selected.any()):
        raise ValueError("event loss requires at least one frontier target")
    return F.cross_entropy(logits[selected], labels[selected], weight=class_weights)


@dataclass(frozen=True)
class EventMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    confusion: tuple[tuple[int, ...], ...]
    class_support: tuple[int, ...]
    finite: bool


def confusion_matrix(
    predicted: torch.Tensor,
    labels: torch.Tensor,
    frontier_mask: torch.Tensor,
) -> torch.Tensor:
    selected = frontier_mask & (labels != IGNORE_EVENT_INDEX)
    target = labels[selected].to(torch.long)
    guess = predicted[selected].to(torch.long)
    indices = target * NUM_EVENT_CLASSES + guess
    return torch.bincount(
        indices, minlength=NUM_EVENT_CLASSES * NUM_EVENT_CLASSES
    ).reshape(NUM_EVENT_CLASSES, NUM_EVENT_CLASSES)


def macro_f1_from_confusion(matrix: torch.Tensor) -> float:
    matrix = matrix.to(torch.float64)
    true_positive = matrix.diag()
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    f1 = torch.where(denominator > 0.0, 2.0 * true_positive / denominator, 0.0)
    return float(f1.mean())


def event_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    frontier_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> EventMetrics:
    loss = masked_event_cross_entropy(
        logits, labels, frontier_mask, class_weights=class_weights
    )
    matrix = confusion_matrix(logits.argmax(dim=-1), labels, frontier_mask)
    support = matrix.sum(dim=1)
    total = int(matrix.sum())
    accuracy = float(matrix.diag().sum()) / max(total, 1)
    return EventMetrics(
        loss=float(loss.detach()),
        accuracy=accuracy,
        macro_f1=macro_f1_from_confusion(matrix),
        confusion=tuple(tuple(int(value) for value in row) for row in matrix.tolist()),
        class_support=tuple(int(value) for value in support.tolist()),
        finite=bool(torch.isfinite(loss) & torch.isfinite(logits).all()),
    )
