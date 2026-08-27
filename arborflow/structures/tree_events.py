"""Discrete topology events and compact trajectory records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from arborflow.structures.branch import BezierPrimitive


class EventType(str, Enum):
    EXTEND = "EXTEND"
    SPLIT = "SPLIT"
    STOP = "STOP"


class PruneActionKind(str, Enum):
    TERMINAL_BRANCH = "TERMINAL_BRANCH"
    TERMINAL_SIBLING_PAIR = "TERMINAL_SIBLING_PAIR"


class PruningStrategy(str, Enum):
    UNIFORM = "uniform"
    DEEP = "deep"
    SHORT = "short"
    LONG = "long"


@dataclass(frozen=True)
class TreeEvent:
    event_type: EventType
    parent_branch_id: int | None
    new_branches: tuple[BezierPrimitive, ...]
    event_time: float

    @property
    def new_branch_ids(self) -> tuple[int, ...]:
        return tuple(branch.primitive_id for branch in self.new_branches)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type.value,
            "parent_branch_id": self.parent_branch_id,
            "new_branch_ids": list(self.new_branch_ids),
            "event_time": self.event_time,
        }


@dataclass(frozen=True)
class PruningStep:
    step_index: int
    action_kind: PruneActionKind
    strategy: PruningStrategy
    parent_branch_id: int | None
    removed_branch_ids: tuple[int, ...]
    remaining_branch_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "action_kind": self.action_kind.value,
            "strategy": self.strategy.value,
            "parent_branch_id": self.parent_branch_id,
            "removed_branch_ids": list(self.removed_branch_ids),
            "remaining_branch_count": self.remaining_branch_count,
        }


@dataclass(frozen=True)
class PruningTrajectory:
    seed: int
    initial_branch_ids: tuple[int, ...]
    steps: tuple[PruningStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "initial_branch_ids": list(self.initial_branch_ids),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class GrowthTrajectory:
    source_pruning_seed: int
    events: tuple[TreeEvent, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_pruning_seed": self.source_pruning_seed,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True)
class GrowthReplayState:
    present_branch_ids: tuple[int, ...]
    active_leaf_ids: tuple[int, ...]
    stopped_branch_ids: tuple[int, ...]

