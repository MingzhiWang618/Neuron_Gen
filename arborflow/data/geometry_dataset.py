"""Oracle-topology dataset for geometry flow matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from arborflow.data.bezier_fitting import BezierFitConfig, BezierFitReport, fit_bezier_tree
from arborflow.data.binary_normalization import normalize_binary
from arborflow.data.branch_decomposition import decompose_swc
from arborflow.data.dynamic_batch import GeometryStateSample, geometry_state_sample
from arborflow.data.normalization import GeometryNormalizer
from arborflow.data.swc_io import read_swc
from arborflow.data.swc_validation import SwcValidationConfig, clean_swc
from arborflow.data.trajectory_builder import (
    PruningConfig,
    build_pruning_trajectories,
    resample_event_times,
    reverse_to_growth,
)
from arborflow.flow.geometry_path import OracleGeometryConfig
from arborflow.flow.oracle_replay import OracleReplay
from arborflow.structures.embedded_tree import BezierTree


@dataclass(frozen=True)
class GeometryTreeRecord:
    path: Path
    tree: BezierTree
    fit_report: BezierFitReport


@dataclass(frozen=True)
class GeometryDatasetConfig:
    trajectories_per_neuron: int = 8
    samples_per_neuron: int = 4
    base_seed: int = 0
    birth_noise_sigma_um: float = 0.01
    event_time_jitter: float = 0.25
    path_buckets: int = 257
    resample_each_epoch: bool = True

    def __post_init__(self) -> None:
        if self.trajectories_per_neuron < 1 or self.samples_per_neuron < 1:
            raise ValueError("trajectory and sample counts must be positive")


def prepare_geometry_tree(
    path: str | Path,
    *,
    validation_config: SwcValidationConfig | None = None,
    fit_config: BezierFitConfig | None = None,
) -> GeometryTreeRecord:
    source_path = Path(path)
    cleaning = clean_swc(read_swc(source_path), validation_config or SwcValidationConfig())
    if not cleaning.success or cleaning.morphology is None:
        report = cleaning.after or cleaning.before
        messages = [issue.message for issue in report.errors]
        raise ValueError(f"{source_path}: SWC cleaning failed: {'; '.join(messages)}")
    normalized, _ = normalize_binary(decompose_swc(cleaning.morphology))
    tree, fit_report = fit_bezier_tree(normalized, fit_config)
    return GeometryTreeRecord(source_path, tree, fit_report)


class GeometryFlowDataset(Dataset[GeometryStateSample]):
    """Sample random continuous times while retaining ground-truth topology events."""

    def __init__(
        self,
        records: list[GeometryTreeRecord],
        *,
        normalizer: GeometryNormalizer,
        config: GeometryDatasetConfig | None = None,
    ) -> None:
        if not records:
            raise ValueError("geometry dataset requires at least one tree")
        self.records = tuple(records)
        self.normalizer = normalizer
        self.config = config or GeometryDatasetConfig()
        self.epoch = 0
        self.replays: list[tuple[OracleReplay, ...]] = []
        for tree_index, record in enumerate(self.records):
            trajectory_seed = self.config.base_seed + tree_index * 100_003
            pruning_trajectories = build_pruning_trajectories(
                record.tree,
                base_seed=trajectory_seed,
                config=PruningConfig(
                    trajectories_per_neuron=self.config.trajectories_per_neuron
                ),
            )
            tree_replays: list[OracleReplay] = []
            for trajectory_index, pruning in enumerate(pruning_trajectories):
                replay_seed = trajectory_seed + trajectory_index + 1
                growth = resample_event_times(
                    reverse_to_growth(record.tree, pruning),
                    seed=replay_seed,
                    maximum_jitter=self.config.event_time_jitter,
                )
                tree_replays.append(
                    OracleReplay(
                        record.tree,
                        growth,
                        geometry_seed=replay_seed + 1_000_003,
                        geometry_config=OracleGeometryConfig(
                            self.config.birth_noise_sigma_um
                        ),
                    )
                )
            self.replays.append(tuple(tree_replays))

    def __len__(self) -> int:
        return len(self.records) * self.config.samples_per_neuron

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch) if self.config.resample_each_epoch else 0

    def __getitem__(self, index: int) -> GeometryStateSample:
        tree_index = index % len(self.records)
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [self.config.base_seed, self.epoch, index, tree_index]
            )
        )
        replays = self.replays[tree_index]
        replay = replays[int(generator.integers(0, len(replays)))]
        first_birth = min(path.birth_time for path in replay.paths.values())
        lower = min(first_birth + 1e-5, 1.0 - 2e-5)
        global_time = float(generator.uniform(lower, 1.0 - 1e-5))
        return geometry_state_sample(
            replay,
            global_time,
            self.normalizer,
            path_buckets=self.config.path_buckets,
        )
