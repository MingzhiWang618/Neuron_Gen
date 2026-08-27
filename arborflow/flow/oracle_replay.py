"""Event-driven oracle replay with dynamic branch insertion and exact targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arborflow.data.trajectory_builder import replay_growth_trajectory
from arborflow.flow.geometry_path import (
    BranchGeometryPath,
    OracleGeometryConfig,
    build_geometry_paths,
)
from arborflow.structures.branch import BezierPrimitive
from arborflow.structures.dynamic_state import BranchState, EmbeddedTreeState
from arborflow.structures.embedded_tree import BezierTree
from arborflow.structures.tree_events import EventType, GrowthTrajectory
from arborflow.structures.tree_invariants import validate_bezier_tree


@dataclass(frozen=True)
class OracleReplayReport:
    valid: bool
    errors: tuple[str, ...]
    topology_exact: bool
    dynamic_indices_stable: bool
    snapshot_count: int
    final_primitive_count: int
    max_final_control_error_um: float
    max_final_radius_error_um: float
    max_continuity_error_um: float
    final_state: EmbeddedTreeState

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "topology_exact": self.topology_exact,
            "dynamic_indices_stable": self.dynamic_indices_stable,
            "snapshot_count": self.snapshot_count,
            "final_primitive_count": self.final_primitive_count,
            "max_final_control_error_um": self.max_final_control_error_um,
            "max_final_radius_error_um": self.max_final_radius_error_um,
            "max_continuity_error_um": self.max_continuity_error_um,
            "final_state": self.final_state.to_dict(include_geometry=False),
        }


class OracleReplay:
    """Replay ground-truth events while geometry follows its analytic oracle path."""

    def __init__(
        self,
        tree: BezierTree,
        trajectory: GrowthTrajectory,
        *,
        geometry_seed: int,
        geometry_config: OracleGeometryConfig | None = None,
    ) -> None:
        tree_report = validate_bezier_tree(tree, require_binary=True)
        if not tree_report.valid:
            raise ValueError("oracle replay requires a valid binary tree: " + "; ".join(
                tree_report.errors
            ))
        growth_report = replay_growth_trajectory(tree, trajectory)
        if not growth_report.valid:
            raise ValueError("oracle replay requires a valid growth trajectory: " + "; ".join(
                growth_report.errors
            ))
        self.tree = tree
        self.trajectory = trajectory
        self.paths: dict[int, BranchGeometryPath] = build_geometry_paths(
            tree,
            trajectory,
            seed=geometry_seed,
            config=geometry_config,
        )

    def _discrete_state(
        self, global_time: float
    ) -> tuple[list[int], list[int], set[int], set[int]]:
        insertion_ids: list[int] = []
        parent_indices: list[int] = []
        index_by_id: dict[int, int] = {}
        active: set[int] = set()
        stopped: set[int] = set()
        for event in self.trajectory.events:
            if event.event_time > global_time:
                break
            if event.event_type is EventType.STOP:
                branch_id = event.parent_branch_id
                if branch_id is None or branch_id not in active:
                    raise RuntimeError("validated STOP event lost its active branch")
                active.remove(branch_id)
                stopped.add(branch_id)
                continue
            parent_index = (
                -1
                if event.parent_branch_id is None
                else index_by_id[event.parent_branch_id]
            )
            if event.parent_branch_id is not None:
                active.remove(event.parent_branch_id)
            for branch in event.new_branches:
                branch_id = branch.primitive_id
                index_by_id[branch_id] = len(insertion_ids)
                insertion_ids.append(branch_id)
                parent_indices.append(parent_index)
                active.add(branch_id)
        return insertion_ids, parent_indices, active, stopped

    def state_at(self, global_time: float) -> EmbeddedTreeState:
        """Return the exact variable-size state at any normalized global time."""

        if not 0.0 <= global_time <= 1.0:
            raise ValueError("global_time must be in [0, 1]")
        primitive_ids, parent_indices, active, stopped = self._discrete_state(global_time)
        target = self.tree.by_id()
        branches: list[BranchState] = []
        for index, primitive_id in enumerate(primitive_ids):
            primitive = target[primitive_id]
            parent_index = parent_indices[index]
            start = (
                np.asarray(self.tree.root.position, dtype=np.float64)
                if parent_index == -1
                else branches[parent_index].end
            )
            offsets, radii, age = self.paths[primitive_id].interpolate(global_time)
            branches.append(
                BranchState(
                    primitive_id=primitive_id,
                    source_branch_id=primitive.source_branch_id,
                    parent_index=parent_index,
                    start=start,
                    control_offsets=offsets,
                    radius_start=float(radii[0]),
                    radius_end=float(radii[1]),
                    swc_type=primitive.swc_type,
                    depth=primitive.depth,
                    birth_time=self.paths[primitive_id].birth_time,
                    age=age,
                    virtual=primitive.virtual,
                    continuation=primitive.continuation,
                )
            )
        return EmbeddedTreeState(
            branches=tuple(branches),
            parent_index=np.asarray(parent_indices, dtype=np.int64),
            active_leaf_mask=np.asarray(
                [primitive_id in active for primitive_id in primitive_ids], dtype=np.bool_
            ),
            stopped_mask=np.asarray(
                [primitive_id in stopped for primitive_id in primitive_ids], dtype=np.bool_
            ),
            global_time=global_time,
        )

    def materialize_tree(self, state: EmbeddedTreeState) -> BezierTree:
        """Convert a replay snapshot back to a topology-bearing Bézier tree."""

        target = self.tree.by_id()
        present = set(state.primitive_ids)
        primitives: list[BezierPrimitive] = []
        for branch in state.branches:
            original = target[branch.primitive_id]
            primitives.append(
                BezierPrimitive(
                    primitive_id=branch.primitive_id,
                    source_branch_id=branch.source_branch_id,
                    parent_id=original.parent_id,
                    children_ids=tuple(
                        child_id for child_id in original.children_ids if child_id in present
                    ),
                    start=branch.start,
                    control_offsets=branch.control_offsets,
                    radius_start=branch.radius_start,
                    radius_end=branch.radius_end,
                    swc_type=branch.swc_type,
                    depth=branch.depth,
                    virtual=branch.virtual,
                    continuation=branch.continuation,
                )
            )
        mapping = tuple(
            (branch_id, tuple(item for item in primitive_ids if item in present))
            for branch_id, primitive_ids in self.tree.branch_to_primitive_ids
            if any(item in present for item in primitive_ids)
        )
        return BezierTree(
            root=self.tree.root,
            primitives=tuple(primitives),
            root_primitive_ids=tuple(
                primitive_id
                for primitive_id in self.tree.root_primitive_ids
                if primitive_id in present
            ),
            branch_to_primitive_ids=mapping,
            source=self.tree.source,
            comments=self.tree.comments,
        )

    def replay(self) -> OracleReplayReport:
        """Check every event boundary and the final analytic reconstruction."""

        times = (0.0,) + tuple(event.event_time for event in self.trajectory.events) + (1.0,)
        errors: list[str] = []
        previous_ids: tuple[int, ...] = ()
        dynamic_indices_stable = True
        max_continuity_error = 0.0
        target = self.tree.by_id()
        for snapshot_index, global_time in enumerate(times):
            state = self.state_at(global_time)
            present_ids = set(state.primitive_ids)
            if state.primitive_ids[: len(previous_ids)] != previous_ids:
                dynamic_indices_stable = False
                errors.append(f"snapshot {snapshot_index} reordered existing branch indices")
            previous_ids = state.primitive_ids
            for index, branch in enumerate(state.branches):
                expected_start = (
                    np.asarray(self.tree.root.position, dtype=np.float64)
                    if branch.parent_index == -1
                    else state.branches[branch.parent_index].end
                )
                continuity_error = float(np.linalg.norm(branch.start - expected_start))
                max_continuity_error = max(max_continuity_error, continuity_error)
                if continuity_error > 1e-8:
                    errors.append(
                        f"snapshot {snapshot_index} branch {branch.primitive_id} is discontinuous"
                    )
                present_children = tuple(
                    child_id
                    for child_id in target[branch.primitive_id].children_ids
                    if child_id in present_ids
                )
                expected_active = not present_children and not state.stopped_mask[index]
                if bool(state.active_leaf_mask[index]) != expected_active:
                    errors.append(
                        f"snapshot {snapshot_index} branch {branch.primitive_id} has bad leaf state"
                    )

        final_state = self.state_at(1.0)
        final_tree = self.materialize_tree(final_state)
        final_by_id = final_tree.by_id()
        topology_exact = (
            final_tree.root_primitive_ids == self.tree.root_primitive_ids
            and set(final_by_id) == set(target)
            and all(
                final_by_id[item].parent_id == primitive.parent_id
                and final_by_id[item].children_ids == primitive.children_ids
                for item, primitive in target.items()
            )
        )
        if not topology_exact:
            errors.append("final topology differs from the oracle target")
        control_errors = [
            float(
                np.max(
                    np.linalg.norm(
                        final_by_id[item].control_points - primitive.control_points,
                        axis=1,
                    )
                )
            )
            for item, primitive in target.items()
        ]
        radius_errors = [
            max(
                abs(final_by_id[item].radius_start - primitive.radius_start),
                abs(final_by_id[item].radius_end - primitive.radius_end),
            )
            for item, primitive in target.items()
        ]
        max_control_error = max(control_errors, default=0.0)
        max_radius_error = max(radius_errors, default=0.0)
        if max_control_error > 1e-8:
            errors.append("final control points do not recover the oracle geometry")
        if max_radius_error > 1e-10:
            errors.append("final radii do not recover the oracle geometry")
        if tuple(final_state.primitive_ids) != previous_ids:
            errors.append("final replay state disagrees with the last event snapshot")
        if np.any(final_state.active_leaf_mask):
            errors.append("final replay still has active leaves")
        invariant_report = validate_bezier_tree(final_tree, require_binary=True)
        errors.extend(f"final tree: {message}" for message in invariant_report.errors)
        return OracleReplayReport(
            valid=not errors,
            errors=tuple(errors),
            topology_exact=topology_exact,
            dynamic_indices_stable=dynamic_indices_stable,
            snapshot_count=len(times),
            final_primitive_count=len(final_state.branches),
            max_final_control_error_um=max_control_error,
            max_final_radius_error_um=max_radius_error,
            max_continuity_error_um=max_continuity_error,
            final_state=final_state,
        )
