"""Variable-size tree states used by oracle replay and later model stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from arborflow.structures.branch import immutable_float_array, immutable_int_array


@dataclass(frozen=True, eq=False)
class BranchState:
    """One currently present branch token in parent-relative coordinates."""

    primitive_id: int
    source_branch_id: int
    parent_index: int
    start: NDArray[np.float64]
    control_offsets: NDArray[np.float64]
    radius_start: float
    radius_end: float
    swc_type: int
    depth: int
    birth_time: float
    age: float
    virtual: bool = False
    continuation: bool = False

    def __post_init__(self) -> None:
        start = immutable_float_array(self.start)
        offsets = immutable_float_array(self.control_offsets)
        if start.shape != (3,):
            raise ValueError("branch-state start must have shape [3]")
        if offsets.shape != (3, 3):
            raise ValueError("branch-state control_offsets must have shape [3, 3]")
        if np.any(~np.isfinite(start)) or np.any(~np.isfinite(offsets)):
            raise ValueError("branch-state geometry must be finite")
        if self.parent_index < -1:
            raise ValueError("parent_index must be -1 or a valid state index")
        if not 0.0 <= self.birth_time < 1.0:
            raise ValueError("birth_time must be in [0, 1)")
        if not 0.0 <= self.age <= 1.0:
            raise ValueError("branch age must be in [0, 1]")
        if (
            not np.isfinite(self.radius_start)
            or not np.isfinite(self.radius_end)
            or self.radius_start <= 0.0
            or self.radius_end <= 0.0
        ):
            raise ValueError("branch-state radii must be positive")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "control_offsets", offsets)

    @property
    def control_points(self) -> NDArray[np.float64]:
        points = np.concatenate(
            (self.start[None, :], self.start + self.control_offsets), axis=0
        )
        points.setflags(write=False)
        return points

    @property
    def end(self) -> NDArray[np.float64]:
        value = self.start + self.control_offsets[2]
        value.setflags(write=False)
        return value


@dataclass(frozen=True, eq=False)
class EmbeddedTreeState:
    """A snapshot whose token count grows dynamically as events are replayed.

    NumPy arrays are used in the data/oracle layer. Milestone 4 can convert them to
    padded PyTorch tensors without changing the variable-size state contract.
    """

    branches: tuple[BranchState, ...]
    parent_index: NDArray[np.int64]
    active_leaf_mask: NDArray[np.bool_]
    stopped_mask: NDArray[np.bool_]
    global_time: float

    def __post_init__(self) -> None:
        parent_index = immutable_int_array(self.parent_index)
        active = np.asarray(self.active_leaf_mask, dtype=np.bool_).copy()
        stopped = np.asarray(self.stopped_mask, dtype=np.bool_).copy()
        size = len(self.branches)
        if parent_index.shape != (size,):
            raise ValueError("parent_index must have shape [num_branches]")
        if active.shape != (size,) or stopped.shape != (size,):
            raise ValueError("state masks must have shape [num_branches]")
        if np.any(active & stopped):
            raise ValueError("a branch cannot be active and stopped simultaneously")
        for index, branch in enumerate(self.branches):
            if branch.parent_index != int(parent_index[index]):
                raise ValueError("branch and state parent indices disagree")
            if branch.parent_index >= index:
                raise ValueError("parents must precede children in insertion order")
        if not 0.0 <= self.global_time <= 1.0:
            raise ValueError("global_time must be in [0, 1]")
        active.setflags(write=False)
        stopped.setflags(write=False)
        object.__setattr__(self, "branches", tuple(self.branches))
        object.__setattr__(self, "parent_index", parent_index)
        object.__setattr__(self, "active_leaf_mask", active)
        object.__setattr__(self, "stopped_mask", stopped)

    @property
    def primitive_ids(self) -> tuple[int, ...]:
        return tuple(branch.primitive_id for branch in self.branches)

    def to_dict(self, *, include_geometry: bool = False) -> dict[str, object]:
        records: list[dict[str, object]] = []
        for index, branch in enumerate(self.branches):
            record: dict[str, object] = {
                "primitive_id": branch.primitive_id,
                "parent_index": branch.parent_index,
                "birth_time": branch.birth_time,
                "age": branch.age,
                "active_leaf": bool(self.active_leaf_mask[index]),
                "stopped": bool(self.stopped_mask[index]),
            }
            if include_geometry:
                record.update(
                    {
                        "start": branch.start.tolist(),
                        "control_offsets": branch.control_offsets.tolist(),
                        "radius_start": branch.radius_start,
                        "radius_end": branch.radius_end,
                    }
                )
            records.append(record)
        return {"global_time": self.global_time, "branches": records}
