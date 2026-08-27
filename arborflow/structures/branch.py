"""Geometry-bearing branch records used by the data and model layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def immutable_float_array(value: object, *, shape_tail: tuple[int, ...] = ()) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if shape_tail and (
        array.ndim < len(shape_tail) or array.shape[-len(shape_tail) :] != shape_tail
    ):
        raise ValueError(f"expected array ending in shape {shape_tail}, got {array.shape}")
    array.setflags(write=False)
    return array


def immutable_int_array(value: object) -> IntArray:
    array = np.asarray(value, dtype=np.int64).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Branch:
    """A maximal SWC path between two topological key nodes.

    ``source_node_ids`` and ``point_swc_types`` are provenance needed for an exact
    decomposition round-trip. Virtual branches intentionally have no source IDs.
    """

    branch_id: int
    parent_id: int | None
    children_ids: tuple[int, ...]
    points: FloatArray
    radii: FloatArray
    swc_type: int
    depth: int
    virtual: bool = False
    continuation: bool = False
    source_node_ids: tuple[int, ...] = ()
    point_swc_types: IntArray | None = None

    def __post_init__(self) -> None:
        points = immutable_float_array(self.points, shape_tail=(3,))
        radii = immutable_float_array(self.radii)
        if points.ndim != 2 or len(points) < 2:
            raise ValueError("branch points must have shape [N, 3] with N >= 2")
        if radii.shape != (len(points),):
            raise ValueError("branch radii must have shape [N]")
        if np.any(~np.isfinite(points)) or np.any(~np.isfinite(radii)):
            raise ValueError("branch geometry must be finite")
        if np.any(radii <= 0):
            raise ValueError("branch radii must be positive")
        if self.depth < 0:
            raise ValueError("branch depth cannot be negative")
        if self.branch_id in self.children_ids:
            raise ValueError("a branch cannot be its own child")
        if self.point_swc_types is None:
            point_types = np.full(len(points), self.swc_type, dtype=np.int64)
            point_types.setflags(write=False)
        else:
            point_types = immutable_int_array(self.point_swc_types)
            if point_types.shape != (len(points),):
                raise ValueError("point_swc_types must have shape [N]")
        if self.source_node_ids and len(self.source_node_ids) != len(points):
            raise ValueError("source_node_ids must be empty or have one ID per point")
        if not self.virtual and not self.source_node_ids:
            raise ValueError("real branches require source_node_ids")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "point_swc_types", point_types)
        object.__setattr__(self, "children_ids", tuple(self.children_ids))
        object.__setattr__(self, "source_node_ids", tuple(self.source_node_ids))

    @property
    def start(self) -> FloatArray:
        return self.points[0]

    @property
    def end(self) -> FloatArray:
        return self.points[-1]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def direction(self) -> FloatArray:
        delta = self.end - self.start
        norm = float(np.linalg.norm(delta))
        if norm == 0:
            result = np.zeros(3, dtype=np.float64)
        else:
            result = delta / norm
        result.setflags(write=False)
        return result


@dataclass(frozen=True, eq=False)
class BezierPrimitive:
    """A cubic branch primitive in parent-relative coordinates.

    ``control_offsets`` stores ``(P1-P0, P2-P0, P3-P0)``. The absolute start is
    retained because Milestone 1 has no dynamic state from which to recover it yet.
    """

    primitive_id: int
    source_branch_id: int
    parent_id: int | None
    children_ids: tuple[int, ...]
    start: FloatArray
    control_offsets: FloatArray
    radius_start: float
    radius_end: float
    swc_type: int
    depth: int
    virtual: bool = False
    continuation: bool = False

    def __post_init__(self) -> None:
        start = immutable_float_array(self.start)
        offsets = immutable_float_array(self.control_offsets)
        if start.shape != (3,):
            raise ValueError("primitive start must have shape [3]")
        if offsets.shape != (3, 3):
            raise ValueError("control_offsets must have shape [3, 3]")
        if not np.isfinite(self.radius_start) or not np.isfinite(self.radius_end):
            raise ValueError("primitive radii must be finite")
        if self.radius_start <= 0 or self.radius_end <= 0:
            raise ValueError("primitive radii must be positive")
        if self.depth < 0:
            raise ValueError("primitive depth cannot be negative")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "control_offsets", offsets)
        object.__setattr__(self, "children_ids", tuple(self.children_ids))

    @property
    def control_points(self) -> FloatArray:
        points = np.concatenate((self.start[None, :], self.start + self.control_offsets), axis=0)
        points.setflags(write=False)
        return points

    @property
    def end(self) -> FloatArray:
        result = self.start + self.control_offsets[2]
        result.setflags(write=False)
        return result
