"""Adaptive cubic Bézier fitting for branch polylines."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from arborflow.structures.branch import BezierPrimitive, Branch
from arborflow.structures.embedded_tree import BezierTree, EmbeddedTree
from arborflow.structures.tree_invariants import (
    assert_valid_embedded_tree,
    validate_bezier_tree,
)


@dataclass(frozen=True)
class BezierFitConfig:
    max_rmse_um: float = 1.5
    max_error_um: float = 3.0
    max_primitive_length_um: float = 80.0

    def __post_init__(self) -> None:
        if self.max_rmse_um <= 0 or self.max_error_um <= 0:
            raise ValueError("Bézier error thresholds must be positive")
        if self.max_primitive_length_um <= 0:
            raise ValueError("max_primitive_length_um must be positive")


@dataclass(frozen=True)
class PrimitiveFitStats:
    primitive_id: int
    source_branch_id: int
    polyline_length_um: float
    rmse_um: float
    max_error_um: float


@dataclass(frozen=True)
class BezierFitReport:
    stats: tuple[PrimitiveFitStats, ...]

    @property
    def max_rmse_um(self) -> float:
        return max((item.rmse_um for item in self.stats), default=0.0)

    @property
    def max_error_um(self) -> float:
        return max((item.max_error_um for item in self.stats), default=0.0)


@dataclass(frozen=True)
class _FittedSegment:
    control_points: NDArray[np.float64]
    radius_start: float
    radius_end: float
    length: float
    rmse: float
    max_error: float
    swc_type: int


def evaluate_cubic(
    control_points: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate one cubic Bézier curve at parameters in ``[0, 1]``."""

    controls = np.asarray(control_points, dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64).reshape(-1, 1)
    if controls.shape != (4, 3):
        raise ValueError("control_points must have shape [4, 3]")
    one_minus = 1.0 - values
    return (
        one_minus**3 * controls[0]
        + 3.0 * one_minus**2 * values * controls[1]
        + 3.0 * one_minus * values**2 * controls[2]
        + values**3 * controls[3]
    )


def _chord_parameters(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    edge_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1), np.cumsum(edge_lengths)))
    total = float(cumulative[-1])
    if total == 0:
        return np.linspace(0.0, 1.0, len(points)), 0.0
    return cumulative / total, total


def _fit_cubic(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    parameters, _ = _chord_parameters(points)
    start = points[0]
    end = points[-1]
    if len(points) == 2:
        controls = np.stack(
            (start, start + (end - start) / 3.0, start + 2.0 * (end - start) / 3.0, end)
        )
    else:
        t = parameters
        one_minus = 1.0 - t
        coefficients = np.stack((3.0 * one_minus**2 * t, 3.0 * one_minus * t**2), axis=1)
        target = points - one_minus[:, None] ** 3 * start - t[:, None] ** 3 * end
        interior, _, _, _ = np.linalg.lstsq(coefficients, target, rcond=None)
        controls = np.vstack((start, interior, end))
    residuals = np.linalg.norm(evaluate_cubic(controls, parameters) - points, axis=1)
    return controls, residuals


def _interpolate_at_distance(
    points: NDArray[np.float64],
    radii: NDArray[np.float64],
    cumulative: NDArray[np.float64],
    distance: float,
) -> tuple[NDArray[np.float64], float]:
    if distance <= 0:
        return points[0].copy(), float(radii[0])
    if distance >= cumulative[-1]:
        return points[-1].copy(), float(radii[-1])
    right = int(np.searchsorted(cumulative, distance, side="right"))
    left = right - 1
    span = float(cumulative[right] - cumulative[left])
    fraction = 0.0 if span == 0 else float((distance - cumulative[left]) / span)
    point = (1.0 - fraction) * points[left] + fraction * points[right]
    radius = (1.0 - fraction) * radii[left] + fraction * radii[right]
    return point, float(radius)


def _length_limited_chunks(
    points: NDArray[np.float64],
    radii: NDArray[np.float64],
    max_length: float,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    edge_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1), np.cumsum(edge_lengths)))
    total = float(cumulative[-1])
    if total <= max_length:
        return [(points, radii)]
    boundaries = np.linspace(0.0, total, math.ceil(total / max_length) + 1)
    chunks: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for start_distance, end_distance in zip(boundaries[:-1], boundaries[1:]):
        chunk_points: list[NDArray[np.float64]] = []
        chunk_radii: list[float] = []
        start_point, start_radius = _interpolate_at_distance(
            points, radii, cumulative, float(start_distance)
        )
        chunk_points.append(start_point)
        chunk_radii.append(start_radius)
        internal = np.flatnonzero(
            (cumulative > start_distance + 1e-12)
            & (cumulative < end_distance - 1e-12)
        )
        for index in internal:
            chunk_points.append(points[index].copy())
            chunk_radii.append(float(radii[index]))
        end_point, end_radius = _interpolate_at_distance(
            points, radii, cumulative, float(end_distance)
        )
        chunk_points.append(end_point)
        chunk_radii.append(end_radius)
        chunks.append(
            (
                np.asarray(chunk_points, dtype=np.float64),
                np.asarray(chunk_radii, dtype=np.float64),
            )
        )
    return chunks


def _fit_adaptive(
    points: NDArray[np.float64],
    radii: NDArray[np.float64],
    config: BezierFitConfig,
    swc_type: int,
) -> list[_FittedSegment]:
    pending: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = [(points, radii)]
    accepted: list[_FittedSegment] = []
    while pending:
        segment_points, segment_radii = pending.pop()
        controls, residuals = _fit_cubic(segment_points)
        _, length = _chord_parameters(segment_points)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        max_error = float(residuals.max(initial=0.0))
        fits = rmse <= config.max_rmse_um and max_error <= config.max_error_um
        if fits or len(segment_points) <= 2:
            accepted.append(
                _FittedSegment(
                    controls,
                    float(segment_radii[0]),
                    float(segment_radii[-1]),
                    length,
                    rmse,
                    max_error,
                    swc_type,
                )
            )
            continue
        split_index = int(np.argmax(residuals))
        split_index = min(max(split_index, 1), len(segment_points) - 2)
        left = (segment_points[: split_index + 1], segment_radii[: split_index + 1])
        right = (segment_points[split_index:], segment_radii[split_index:])
        pending.append(right)
        pending.append(left)
    return accepted


def _fit_branch(branch: Branch, config: BezierFitConfig) -> list[_FittedSegment]:
    if branch.virtual:
        controls = np.repeat(branch.start[None, :], 4, axis=0)
        return [
            _FittedSegment(
                controls,
                float(branch.radii[0]),
                float(branch.radii[-1]),
                0.0,
                0.0,
                0.0,
                branch.swc_type,
            )
        ]
    fitted: list[_FittedSegment] = []
    point_types = branch.point_swc_types
    assert point_types is not None
    edge_types = point_types[1:]
    group_start = 0
    for edge_index in range(1, len(edge_types) + 1):
        if edge_index < len(edge_types) and edge_types[edge_index] == edge_types[group_start]:
            continue
        typed_points = branch.points[group_start : edge_index + 1]
        typed_radii = branch.radii[group_start : edge_index + 1]
        swc_type = int(edge_types[group_start])
        for points, radii in _length_limited_chunks(
            typed_points, typed_radii, config.max_primitive_length_um
        ):
            fitted.extend(_fit_adaptive(points, radii, config, swc_type))
        group_start = edge_index
    return fitted


def fit_bezier_tree(
    tree: EmbeddedTree,
    config: BezierFitConfig | None = None,
) -> tuple[BezierTree, BezierFitReport]:
    """Fit every branch and connect continuation primitives into one dynamic tree."""

    config = config or BezierFitConfig()
    assert_valid_embedded_tree(tree)
    branch_segments = {branch.branch_id: _fit_branch(branch, config) for branch in tree.branches}
    branch_to_ids: dict[int, tuple[int, ...]] = {}
    next_id = 0
    for branch in tree.branches:
        count = len(branch_segments[branch.branch_id])
        branch_to_ids[branch.branch_id] = tuple(range(next_id, next_id + count))
        next_id += count

    branches = tree.by_id()
    primitives: dict[int, BezierPrimitive] = {}
    stats: list[PrimitiveFitStats] = []
    for branch in tree.branches:
        primitive_ids = branch_to_ids[branch.branch_id]
        segments = branch_segments[branch.branch_id]
        for index, (primitive_id, segment) in enumerate(zip(primitive_ids, segments)):
            if index > 0:
                parent_id = primitive_ids[index - 1]
            elif branch.parent_id is None:
                parent_id = None
            else:
                parent_id = branch_to_ids[branch.parent_id][-1]
            if index + 1 < len(primitive_ids):
                children_ids = (primitive_ids[index + 1],)
            else:
                children_ids = tuple(
                    branch_to_ids[child_id][0] for child_id in branch.children_ids
                )
            controls = segment.control_points
            primitives[primitive_id] = BezierPrimitive(
                primitive_id=primitive_id,
                source_branch_id=branch.branch_id,
                parent_id=parent_id,
                children_ids=children_ids,
                start=controls[0],
                control_offsets=controls[1:] - controls[0],
                radius_start=segment.radius_start,
                radius_end=segment.radius_end,
                swc_type=segment.swc_type,
                depth=0,
                virtual=branch.virtual,
                continuation=branch.continuation or index > 0,
            )
            stats.append(
                PrimitiveFitStats(
                    primitive_id,
                    branch.branch_id,
                    segment.length,
                    segment.rmse,
                    segment.max_error,
                )
            )

    root_primitive_ids = tuple(branch_to_ids[branch_id][0] for branch_id in tree.root_branch_ids)
    depths: dict[int, int] = {}
    queue = deque((primitive_id, 0) for primitive_id in root_primitive_ids)
    while queue:
        primitive_id, depth = queue.popleft()
        if primitive_id in depths:
            raise ValueError(f"primitive {primitive_id} has repeated ownership")
        depths[primitive_id] = depth
        queue.extend((child_id, depth + 1) for child_id in primitives[primitive_id].children_ids)
    bezier_tree = BezierTree(
        root=tree.root,
        primitives=tuple(
            replace(primitives[index], depth=depths[index]) for index in sorted(primitives)
        ),
        root_primitive_ids=root_primitive_ids,
        branch_to_primitive_ids=tuple(
            (branch_id, branch_to_ids[branch_id]) for branch_id in sorted(branch_to_ids)
        ),
        source=tree.source,
        comments=tree.comments,
    )
    primitive_report = validate_bezier_tree(bezier_tree)
    if not primitive_report.valid:
        raise ValueError("invalid fitted Bézier tree: " + "; ".join(primitive_report.errors))
    return bezier_tree, BezierFitReport(tuple(stats))
