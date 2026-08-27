"""Oracle probability paths from near-zero birth seeds to data branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from arborflow.data.trajectory_builder import branch_birth_times
from arborflow.structures.branch import BezierPrimitive, immutable_float_array
from arborflow.structures.embedded_tree import BezierTree
from arborflow.structures.tree_events import GrowthTrajectory


@dataclass(frozen=True)
class OracleGeometryConfig:
    """Configuration for reproducible isotropic branch birth seeds."""

    birth_noise_sigma_um: float = 0.01

    def __post_init__(self) -> None:
        if not np.isfinite(self.birth_noise_sigma_um) or self.birth_noise_sigma_um < 0.0:
            raise ValueError("birth_noise_sigma_um cannot be negative")


def branch_age(global_time: float, birth_time: float) -> float:
    """Return clipped normalized branch age from the task-book definition."""

    if not 0.0 <= global_time <= 1.0:
        raise ValueError("global_time must be in [0, 1]")
    if not 0.0 <= birth_time < 1.0:
        raise ValueError("birth_time must be in [0, 1)")
    return float(np.clip((global_time - birth_time) / (1.0 - birth_time), 0.0, 1.0))


@dataclass(frozen=True, eq=False)
class BranchGeometryPath:
    """A constant-velocity path in parent-relative branch parameter space."""

    primitive_id: int
    birth_time: float
    seed_control_offsets: NDArray[np.float64]
    target_control_offsets: NDArray[np.float64]
    seed_radii: NDArray[np.float64]
    target_radii: NDArray[np.float64]

    def __post_init__(self) -> None:
        seed_offsets = immutable_float_array(self.seed_control_offsets)
        target_offsets = immutable_float_array(self.target_control_offsets)
        seed_radii = immutable_float_array(self.seed_radii)
        target_radii = immutable_float_array(self.target_radii)
        if seed_offsets.shape != (3, 3) or target_offsets.shape != (3, 3):
            raise ValueError("geometry-path offsets must have shape [3, 3]")
        if seed_radii.shape != (2,) or target_radii.shape != (2,):
            raise ValueError("geometry-path radii must have shape [2]")
        arrays = (seed_offsets, target_offsets, seed_radii, target_radii)
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("geometry-path parameters must be finite")
        if not 0.0 <= self.birth_time < 1.0:
            raise ValueError("birth_time must be in [0, 1)")
        if np.any(seed_radii <= 0.0) or np.any(target_radii <= 0.0):
            raise ValueError("geometry-path radii must be positive")
        object.__setattr__(self, "seed_control_offsets", seed_offsets)
        object.__setattr__(self, "target_control_offsets", target_offsets)
        object.__setattr__(self, "seed_radii", seed_radii)
        object.__setattr__(self, "target_radii", target_radii)

    @property
    def control_velocity(self) -> NDArray[np.float64]:
        value = (self.target_control_offsets - self.seed_control_offsets) / (
            1.0 - self.birth_time
        )
        value.setflags(write=False)
        return value

    @property
    def radius_velocity(self) -> NDArray[np.float64]:
        value = (self.target_radii - self.seed_radii) / (1.0 - self.birth_time)
        value.setflags(write=False)
        return value

    def interpolate(
        self, global_time: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        age = branch_age(global_time, self.birth_time)
        offsets = (1.0 - age) * self.seed_control_offsets + age * self.target_control_offsets
        radii = (1.0 - age) * self.seed_radii + age * self.target_radii
        offsets.setflags(write=False)
        radii.setflags(write=False)
        return offsets, radii, age


def _birth_seed(
    primitive: BezierPrimitive,
    generator: np.random.Generator,
    config: OracleGeometryConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    offsets = generator.normal(
        loc=0.0,
        scale=config.birth_noise_sigma_um,
        size=(3, 3),
    )
    radii = np.asarray(
        (primitive.radius_start, primitive.radius_start), dtype=np.float64
    )
    return offsets, radii


def build_geometry_paths(
    tree: BezierTree,
    trajectory: GrowthTrajectory,
    *,
    seed: int,
    config: OracleGeometryConfig | None = None,
) -> dict[int, BranchGeometryPath]:
    """Create exactly one deterministic oracle path for every target primitive."""

    config = config or OracleGeometryConfig()
    births = branch_birth_times(trajectory)
    primitives = tree.by_id()
    if set(births) != set(primitives):
        missing = sorted(set(primitives) - set(births))
        extra = sorted(set(births) - set(primitives))
        raise ValueError(f"trajectory births do not match tree; missing={missing}, extra={extra}")
    event_branches = {
        branch.primitive_id: branch
        for event in trajectory.events
        for branch in event.new_branches
    }
    for primitive_id, target in primitives.items():
        event_branch = event_branches[primitive_id]
        scalar_metadata_equal = (
            event_branch.source_branch_id == target.source_branch_id
            and event_branch.parent_id == target.parent_id
            and event_branch.children_ids == target.children_ids
            and event_branch.radius_start == target.radius_start
            and event_branch.radius_end == target.radius_end
            and event_branch.swc_type == target.swc_type
            and event_branch.depth == target.depth
            and event_branch.virtual == target.virtual
            and event_branch.continuation == target.continuation
        )
        geometry_equal = np.array_equal(event_branch.start, target.start) and np.array_equal(
            event_branch.control_offsets, target.control_offsets
        )
        if not scalar_metadata_equal or not geometry_equal:
            raise ValueError(
                f"trajectory branch {primitive_id} does not match its geometry target"
            )
    generator = np.random.default_rng(seed)
    result: dict[int, BranchGeometryPath] = {}
    for primitive_id in sorted(primitives):
        primitive = primitives[primitive_id]
        seed_offsets, seed_radii = _birth_seed(primitive, generator, config)
        target_radii = np.asarray(
            (primitive.radius_start, primitive.radius_end), dtype=np.float64
        )
        result[primitive_id] = BranchGeometryPath(
            primitive_id=primitive_id,
            birth_time=births[primitive_id],
            seed_control_offsets=seed_offsets,
            target_control_offsets=primitive.control_offsets,
            seed_radii=seed_radii,
            target_radii=target_radii,
        )
    return result
