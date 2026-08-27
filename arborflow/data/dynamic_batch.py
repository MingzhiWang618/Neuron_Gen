"""Variable-size geometry samples, tree relations, padding, and SO(3) augmentation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import NDArray

from arborflow.data.normalization import GeometryNormalizer
from arborflow.flow.oracle_replay import OracleReplay


GEOMETRY_DIM = 11
CONTINUOUS_FEATURE_DIM = 20


@dataclass(frozen=True)
class GeometryStateSample:
    continuous_features: NDArray[np.float32]
    current_geometry: NDArray[np.float32]
    target_geometry: NDArray[np.float32]
    target_velocity: NDArray[np.float32]
    swc_type: NDArray[np.int64]
    depth: NDArray[np.int64]
    child_position: NDArray[np.int64]
    path_code: NDArray[np.int64]
    parent_index: NDArray[np.int64]
    global_time: float

    @property
    def branch_count(self) -> int:
        return len(self.parent_index)


@dataclass(frozen=True)
class GeometryBatch:
    continuous_features: torch.Tensor
    current_geometry: torch.Tensor
    target_geometry: torch.Tensor
    target_velocity: torch.Tensor
    swc_type: torch.Tensor
    depth: torch.Tensor
    child_position: torch.Tensor
    path_code: torch.Tensor
    parent_index: torch.Tensor
    shortest_path_distance: torch.Tensor
    relation: torch.Tensor
    padding_mask: torch.Tensor
    global_time: torch.Tensor

    def to(self, device: torch.device | str) -> GeometryBatch:
        return replace(
            self,
            **{
                field: getattr(self, field).to(device)
                for field in self.__dataclass_fields__
            },
        )


def _primitive_position(replay: OracleReplay, primitive_id: int) -> int:
    primitive = replay.tree.by_id()[primitive_id]
    siblings = (
        replay.tree.root_primitive_ids
        if primitive.parent_id is None
        else replay.tree.by_id()[primitive.parent_id].children_ids
    )
    return siblings.index(primitive_id) + 1


def geometry_state_sample(
    replay: OracleReplay,
    global_time: float,
    normalizer: GeometryNormalizer,
    *,
    path_buckets: int = 257,
) -> GeometryStateSample:
    """Convert one oracle snapshot and its analytic targets to model arrays."""

    if path_buckets < 2:
        raise ValueError("path_buckets must be at least two")
    state = replay.state_at(global_time)
    if not state.branches:
        raise ValueError("geometry training requires at least one born branch")
    count = len(state.branches)
    current = np.empty((count, GEOMETRY_DIM), dtype=np.float64)
    target = np.empty_like(current)
    velocity = np.empty_like(current)
    direction = np.empty((count, 3), dtype=np.float64)
    path_length = np.empty((count, 1), dtype=np.float64)
    swc_type = np.empty(count, dtype=np.int64)
    depth = np.empty(count, dtype=np.int64)
    child_position = np.empty(count, dtype=np.int64)
    path_code = np.empty(count, dtype=np.int64)
    target_primitives = replay.tree.by_id()
    for index, branch in enumerate(state.branches):
        primitive_id = branch.primitive_id
        primitive = target_primitives[primitive_id]
        path = replay.paths[primitive_id]
        current[index] = np.concatenate(
            (branch.control_offsets.reshape(-1), (branch.radius_start, branch.radius_end))
        )
        target[index] = np.concatenate(
            (
                primitive.control_offsets.reshape(-1),
                (primitive.radius_start, primitive.radius_end),
            )
        )
        velocity[index] = np.concatenate(
            (path.control_velocity.reshape(-1), path.radius_velocity)
        )
        endpoint = branch.control_offsets[2]
        endpoint_norm = float(np.linalg.norm(endpoint))
        direction[index] = endpoint / endpoint_norm if endpoint_norm > 1e-12 else 0.0
        polygon = np.vstack((np.zeros((1, 3)), branch.control_offsets))
        path_length[index, 0] = float(
            np.linalg.norm(np.diff(polygon, axis=0), axis=1).sum()
        ) / normalizer.coordinate_scale_um
        swc_type[index] = primitive.swc_type
        depth[index] = primitive.depth
        child_position[index] = _primitive_position(replay, primitive_id)
        parent_index = int(state.parent_index[index])
        parent_code = 0 if parent_index == -1 else int(path_code[parent_index])
        path_code[index] = (
            parent_code * 3 + int(child_position[index])
        ) % path_buckets
    normalized_current = normalizer.normalize_geometry(current)
    normalized_target = normalizer.normalize_geometry(target)
    normalized_velocity = normalizer.normalize_geometry(velocity)
    continuous = np.concatenate(
        (
            normalized_current,
            direction,
            path_length,
            np.asarray([[branch.age] for branch in state.branches]),
            np.asarray([[branch.birth_time] for branch in state.branches]),
            np.full((count, 1), global_time),
            state.active_leaf_mask[:, None],
            state.stopped_mask[:, None],
        ),
        axis=1,
        dtype=np.float64,
    )
    if continuous.shape != (count, CONTINUOUS_FEATURE_DIM):
        raise RuntimeError("continuous feature contract changed unexpectedly")
    return GeometryStateSample(
        continuous.astype(np.float32),
        normalized_current.astype(np.float32),
        normalized_target.astype(np.float32),
        normalized_velocity.astype(np.float32),
        swc_type,
        depth,
        child_position,
        path_code,
        state.parent_index.astype(np.int64, copy=True),
        global_time,
    )


def _tree_relations(
    parent_index: NDArray[np.int64], max_distance: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return clipped shortest-path distances and directed relation classes."""

    count = len(parent_index)
    ancestor_paths: list[list[int]] = []
    for index in range(count):
        path = [index]
        parent = int(parent_index[index])
        while parent != -1:
            path.append(parent)
            parent = int(parent_index[parent])
        path.append(-1)
        ancestor_paths.append(path)
    distances = np.empty((count, count), dtype=np.int64)
    relations = np.zeros((count, count), dtype=np.int64)
    for left in range(count):
        left_positions = {item: position for position, item in enumerate(ancestor_paths[left])}
        for right in range(count):
            right_path = ancestor_paths[right]
            common = next(item for item in right_path if item in left_positions)
            distance = left_positions[common] + right_path.index(common)
            distances[left, right] = min(distance, max_distance + 1)
            if left == right:
                relations[left, right] = 1
            elif left in right_path[1:]:
                relations[left, right] = 2
            elif right in ancestor_paths[left][1:]:
                relations[left, right] = 3
            elif parent_index[left] == parent_index[right]:
                relations[left, right] = 4
    return distances, relations


def collate_geometry_samples(
    samples: list[GeometryStateSample], *, max_tree_distance: int = 16
) -> GeometryBatch:
    if not samples:
        raise ValueError("cannot collate an empty geometry batch")
    batch_size = len(samples)
    max_branches = max(sample.branch_count for sample in samples)

    def zeros(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    continuous = zeros((batch_size, max_branches, CONTINUOUS_FEATURE_DIM), torch.float32)
    current = zeros((batch_size, max_branches, GEOMETRY_DIM), torch.float32)
    target = zeros((batch_size, max_branches, GEOMETRY_DIM), torch.float32)
    velocity = zeros((batch_size, max_branches, GEOMETRY_DIM), torch.float32)
    swc_type = zeros((batch_size, max_branches), torch.long)
    depth = zeros((batch_size, max_branches), torch.long)
    child_position = zeros((batch_size, max_branches), torch.long)
    path_code = zeros((batch_size, max_branches), torch.long)
    parent_index = torch.full((batch_size, max_branches), -1, dtype=torch.long)
    distances = zeros((batch_size, max_branches, max_branches), torch.long)
    relations = zeros((batch_size, max_branches, max_branches), torch.long)
    mask = zeros((batch_size, max_branches), torch.bool)
    times = zeros((batch_size,), torch.float32)
    for batch_index, sample in enumerate(samples):
        count = sample.branch_count
        continuous[batch_index, :count] = torch.from_numpy(sample.continuous_features)
        current[batch_index, :count] = torch.from_numpy(sample.current_geometry)
        target[batch_index, :count] = torch.from_numpy(sample.target_geometry)
        velocity[batch_index, :count] = torch.from_numpy(sample.target_velocity)
        swc_type[batch_index, :count] = torch.from_numpy(sample.swc_type)
        depth[batch_index, :count] = torch.from_numpy(sample.depth)
        child_position[batch_index, :count] = torch.from_numpy(sample.child_position)
        path_code[batch_index, :count] = torch.from_numpy(sample.path_code)
        parent_index[batch_index, :count] = torch.from_numpy(sample.parent_index)
        sample_distances, sample_relations = _tree_relations(
            sample.parent_index, max_tree_distance
        )
        distances[batch_index, :count, :count] = torch.from_numpy(sample_distances)
        relations[batch_index, :count, :count] = torch.from_numpy(sample_relations)
        mask[batch_index, :count] = True
        times[batch_index] = sample.global_time
    return GeometryBatch(
        continuous,
        current,
        target,
        velocity,
        swc_type,
        depth,
        child_position,
        path_code,
        parent_index,
        distances,
        relations,
        mask,
        times,
    )


def random_so3(
    batch_size: int, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample proper 3D rotations from normalized random quaternions."""

    quaternion = torch.randn((batch_size, 4), generator=generator)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(batch_size, 3, 3)


def _rotate_vectors(values: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    original_shape = values.shape
    vectors = values.reshape(values.shape[0], values.shape[1], -1, 3)
    rotated = torch.einsum("bnvc,bdc->bnvd", vectors, rotations)
    return rotated.reshape(original_shape)


def rotate_geometry_batch(
    batch: GeometryBatch, *, generator: torch.Generator | None = None
) -> GeometryBatch:
    """Apply one independent SO(3) rotation per tree to all vector quantities."""

    rotations = random_so3(len(batch.global_time), generator=generator).to(
        batch.continuous_features.device
    )
    continuous = batch.continuous_features.clone()
    continuous[..., :9] = _rotate_vectors(continuous[..., :9], rotations)
    continuous[..., 11:14] = _rotate_vectors(continuous[..., 11:14], rotations)
    current = batch.current_geometry.clone()
    target = batch.target_geometry.clone()
    velocity = batch.target_velocity.clone()
    current[..., :9] = _rotate_vectors(current[..., :9], rotations)
    target[..., :9] = _rotate_vectors(target[..., :9], rotations)
    velocity[..., :9] = _rotate_vectors(velocity[..., :9], rotations)
    return replace(
        batch,
        continuous_features=continuous,
        current_geometry=current,
        target_geometry=target,
        target_velocity=velocity,
    )
