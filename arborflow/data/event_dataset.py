"""Oracle-geometry frontier snapshots for Milestone 5 event supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from arborflow.data.dynamic_batch import (
    GeometryBatch,
    GeometryStateSample,
    collate_geometry_samples,
    geometry_state_sample,
)
from arborflow.data.geometry_dataset import GeometryTreeRecord
from arborflow.data.normalization import GeometryNormalizer
from arborflow.data.trajectory_builder import (
    PruningConfig,
    build_pruning_trajectories,
    resample_event_times,
    reverse_to_growth,
)
from arborflow.flow.event_process import IGNORE_EVENT_INDEX, EventClass, event_class
from arborflow.flow.geometry_path import OracleGeometryConfig
from arborflow.flow.oracle_replay import OracleReplay


@dataclass(frozen=True)
class EventDatasetConfig:
    trajectories_per_neuron: int = 8
    samples_per_neuron: int = 32
    base_seed: int = 0
    event_time_jitter: float = 0.25
    path_buckets: int = 257
    resample_each_epoch: bool = True

    def __post_init__(self) -> None:
        if self.trajectories_per_neuron < 1 or self.samples_per_neuron < 1:
            raise ValueError("trajectory and sample counts must be positive")


@dataclass(frozen=True)
class EventStateSample(GeometryStateSample):
    event_labels: np.ndarray
    frontier_mask: np.ndarray


@dataclass(frozen=True)
class EventBatch(GeometryBatch):
    event_labels: torch.Tensor
    frontier_mask: torch.Tensor


def _fixed_oracle_geometry(sample: GeometryStateSample) -> GeometryStateSample:
    """Replace interpolated geometry features with complete ground-truth geometry."""

    continuous = sample.continuous_features.copy()
    continuous[:, :11] = sample.target_geometry
    target_offsets = sample.target_geometry[:, :9].reshape(-1, 3, 3)
    endpoint = target_offsets[:, 2]
    endpoint_norm = np.linalg.norm(endpoint, axis=1, keepdims=True)
    continuous[:, 11:14] = np.divide(
        endpoint,
        endpoint_norm,
        out=np.zeros_like(endpoint),
        where=endpoint_norm > 1e-12,
    )
    polygon = np.concatenate(
        (np.zeros((len(target_offsets), 1, 3)), target_offsets), axis=1
    )
    continuous[:, 14] = np.linalg.norm(
        np.diff(polygon, axis=1), axis=2
    ).sum(axis=1)
    zeros = np.zeros_like(sample.target_velocity)
    return GeometryStateSample(
        continuous_features=continuous,
        current_geometry=sample.target_geometry.copy(),
        target_geometry=sample.target_geometry.copy(),
        target_velocity=zeros,
        swc_type=sample.swc_type.copy(),
        depth=sample.depth.copy(),
        child_position=sample.child_position.copy(),
        path_code=sample.path_code.copy(),
        parent_index=sample.parent_index.copy(),
        global_time=sample.global_time,
    )


def event_state_sample(
    replay: OracleReplay,
    event_index: int,
    normalizer: GeometryNormalizer,
    *,
    path_buckets: int = 257,
) -> EventStateSample:
    """Label the target leaf and WAIT alternatives immediately before one event."""

    event = replay.trajectory.events[event_index]
    if event.parent_branch_id is None:
        raise ValueError("the deterministic root creation event is not predicted")
    global_time = float(np.nextafter(event.event_time, 0.0))
    state = replay.state_at(global_time)
    base = _fixed_oracle_geometry(
        geometry_state_sample(
            replay,
            global_time,
            normalizer,
            path_buckets=path_buckets,
        )
    )
    frontier = state.active_leaf_mask.astype(np.bool_, copy=True)
    labels = np.full(len(state.branches), IGNORE_EVENT_INDEX, dtype=np.int64)
    labels[frontier] = int(EventClass.WAIT)
    try:
        target_index = state.primitive_ids.index(event.parent_branch_id)
    except ValueError as error:
        raise RuntimeError("next event target is absent from its pre-event state") from error
    if not frontier[target_index]:
        raise RuntimeError("next event target is not an active frontier leaf")
    labels[target_index] = int(event_class(event.event_type))
    return EventStateSample(
        **base.__dict__,
        event_labels=labels,
        frontier_mask=frontier,
    )


def collate_event_samples(
    samples: list[EventStateSample], *, max_tree_distance: int = 16
) -> EventBatch:
    if not samples:
        raise ValueError("cannot collate an empty event batch")
    geometry = collate_geometry_samples(samples, max_tree_distance=max_tree_distance)
    batch_size = len(samples)
    max_branches = geometry.padding_mask.shape[1]
    labels = torch.full(
        (batch_size, max_branches), IGNORE_EVENT_INDEX, dtype=torch.long
    )
    frontier = torch.zeros((batch_size, max_branches), dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = sample.branch_count
        labels[index, :count] = torch.from_numpy(sample.event_labels)
        frontier[index, :count] = torch.from_numpy(sample.frontier_mask)
    values = [
        getattr(geometry, field_name)
        for field_name in GeometryBatch.__dataclass_fields__
    ]
    return EventBatch(*values, labels, frontier)


class EventFlowDataset(Dataset[EventStateSample]):
    """Sample next-event decisions while all present geometry stays oracle-fixed."""

    def __init__(
        self,
        records: list[GeometryTreeRecord],
        *,
        normalizer: GeometryNormalizer,
        config: EventDatasetConfig | None = None,
    ) -> None:
        if not records:
            raise ValueError("event dataset requires at least one tree")
        self.records = tuple(records)
        self.normalizer = normalizer
        self.config = config or EventDatasetConfig()
        self.epoch = 0
        self.replays: list[tuple[OracleReplay, ...]] = []
        self.event_indices: list[tuple[tuple[int, ...], ...]] = []
        for tree_index, record in enumerate(self.records):
            trajectory_seed = self.config.base_seed + tree_index * 100_003
            pruning = build_pruning_trajectories(
                record.tree,
                base_seed=trajectory_seed,
                config=PruningConfig(
                    trajectories_per_neuron=self.config.trajectories_per_neuron
                ),
            )
            tree_replays: list[OracleReplay] = []
            tree_indices: list[tuple[int, ...]] = []
            for trajectory_index, destruction in enumerate(pruning):
                replay_seed = trajectory_seed + trajectory_index + 1
                growth = resample_event_times(
                    reverse_to_growth(record.tree, destruction),
                    seed=replay_seed,
                    maximum_jitter=self.config.event_time_jitter,
                )
                replay = OracleReplay(
                    record.tree,
                    growth,
                    geometry_seed=replay_seed + 1_000_003,
                    geometry_config=OracleGeometryConfig(0.0),
                )
                eligible = tuple(
                    index
                    for index, event in enumerate(growth.events)
                    if event.parent_branch_id is not None
                )
                if eligible:
                    tree_replays.append(replay)
                    tree_indices.append(eligible)
            if not tree_replays:
                raise ValueError(f"tree {record.path} has no predictable frontier events")
            self.replays.append(tuple(tree_replays))
            self.event_indices.append(tuple(tree_indices))

    def __len__(self) -> int:
        return len(self.records) * self.config.samples_per_neuron

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch) if self.config.resample_each_epoch else 0

    def __getitem__(self, index: int) -> EventStateSample:
        tree_index = index % len(self.records)
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [self.config.base_seed, self.epoch, index, tree_index, 5]
            )
        )
        replay_index = int(generator.integers(0, len(self.replays[tree_index])))
        candidates = self.event_indices[tree_index][replay_index]
        event_index = candidates[int(generator.integers(0, len(candidates)))]
        return event_state_sample(
            self.replays[tree_index][replay_index],
            event_index,
            self.normalizer,
            path_buckets=self.config.path_buckets,
        )

    def class_counts(self) -> tuple[int, ...]:
        counts = np.zeros(len(EventClass), dtype=np.int64)
        saved_epoch = self.epoch
        self.set_epoch(0)
        try:
            for index in range(len(self)):
                sample = self[index]
                selected = sample.event_labels[sample.frontier_mask]
                counts += np.bincount(selected, minlength=len(EventClass))
        finally:
            self.epoch = saved_epoch
        return tuple(int(value) for value in counts)
