"""Finite dynamic topology sampling with an empirical oracle-geometry provider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from arborflow.data.dynamic_batch import CONTINUOUS_FEATURE_DIM, GEOMETRY_DIM
from arborflow.data.event_dataset import (
    EventBatch,
    EventStateSample,
    collate_event_samples,
)
from arborflow.data.geometry_dataset import GeometryTreeRecord
from arborflow.data.normalization import GeometryNormalizer
from arborflow.flow.event_process import IGNORE_EVENT_INDEX, EventClass
from arborflow.models.event_model import EventFlowModel


@dataclass(frozen=True, eq=False)
class BranchGeometryTemplate:
    geometry: NDArray[np.float32]
    swc_type: int

    def __post_init__(self) -> None:
        geometry = np.asarray(self.geometry, dtype=np.float32).copy()
        if geometry.shape != (GEOMETRY_DIM,) or np.any(~np.isfinite(geometry)):
            raise ValueError("branch template geometry must be finite with shape [11]")
        geometry.setflags(write=False)
        object.__setattr__(self, "geometry", geometry)


@dataclass(frozen=True)
class OracleGeometryBank:
    root_templates: tuple[BranchGeometryTemplate, ...]
    child_templates: tuple[BranchGeometryTemplate, ...]

    @classmethod
    def from_records(
        cls,
        records: list[GeometryTreeRecord],
        normalizer: GeometryNormalizer,
    ) -> OracleGeometryBank:
        roots: list[BranchGeometryTemplate] = []
        children: list[BranchGeometryTemplate] = []
        for record in records:
            for primitive in record.tree.primitives:
                raw = np.concatenate(
                    (
                        primitive.control_offsets.reshape(-1),
                        (primitive.radius_start, primitive.radius_end),
                    )
                )
                template = BranchGeometryTemplate(
                    normalizer.normalize_geometry(raw).astype(np.float32),
                    primitive.swc_type,
                )
                (roots if primitive.parent_id is None else children).append(template)
        if not roots or not children:
            raise ValueError("oracle geometry bank requires root and non-root primitives")
        return cls(tuple(roots), tuple(children))


@dataclass(frozen=True)
class EventSamplerConfig:
    max_branches: int = 1000
    max_depth: int = 64
    max_steps: int = 200
    temperature: float = 1.0
    prior_correction_strength: float = 1.0

    def __post_init__(self) -> None:
        if self.max_branches < 1 or self.max_depth < 1 or self.max_steps < 1:
            raise ValueError("event sampler limits must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.prior_correction_strength <= 1.0:
            raise ValueError("prior_correction_strength must be in [0, 1]")


@dataclass(frozen=True)
class SampledEvent:
    step: int
    branch_index: int | None
    event_class: EventClass
    global_time: float


@dataclass(frozen=True)
class EventSamplingResult:
    branch_count: int
    leaf_count: int
    maximum_depth: int
    event_sequence: tuple[SampledEvent, ...]
    wait_steps: int
    forced_stop_count: int
    termination_reason: str
    seed: int

    @property
    def forced_termination(self) -> bool:
        return self.forced_stop_count > 0


@dataclass
class _GeneratedBranch:
    template: BranchGeometryTemplate
    parent_index: int
    depth: int
    child_position: int
    path_code: int
    birth_time: float
    stopped: bool = False


def _state_sample(
    branches: list[_GeneratedBranch],
    active: set[int],
    global_time: float,
) -> EventStateSample:
    count = len(branches)
    geometry = np.stack([branch.template.geometry for branch in branches])
    continuous = np.zeros((count, CONTINUOUS_FEATURE_DIM), dtype=np.float32)
    continuous[:, :11] = geometry
    offsets = geometry[:, :9].reshape(count, 3, 3)
    endpoint = offsets[:, 2]
    endpoint_norm = np.linalg.norm(endpoint, axis=1, keepdims=True)
    continuous[:, 11:14] = np.divide(
        endpoint,
        endpoint_norm,
        out=np.zeros_like(endpoint),
        where=endpoint_norm > 1e-12,
    )
    polygon = np.concatenate((np.zeros((count, 1, 3)), offsets), axis=1)
    continuous[:, 14] = np.linalg.norm(np.diff(polygon, axis=1), axis=2).sum(axis=1)
    birth_times = np.asarray([branch.birth_time for branch in branches], dtype=np.float32)
    continuous[:, 15] = np.clip(
        (global_time - birth_times) / np.maximum(1.0 - birth_times, 1e-6), 0.0, 1.0
    )
    continuous[:, 16] = birth_times
    continuous[:, 17] = global_time
    frontier = np.asarray([index in active for index in range(count)], dtype=np.bool_)
    stopped = np.asarray([branch.stopped for branch in branches], dtype=np.bool_)
    continuous[:, 18] = frontier
    continuous[:, 19] = stopped
    labels = np.full(count, IGNORE_EVENT_INDEX, dtype=np.int64)
    return EventStateSample(
        continuous_features=continuous,
        current_geometry=geometry.copy(),
        target_geometry=geometry.copy(),
        target_velocity=np.zeros_like(geometry),
        swc_type=np.asarray([branch.template.swc_type for branch in branches]),
        depth=np.asarray([branch.depth for branch in branches]),
        child_position=np.asarray([branch.child_position for branch in branches]),
        path_code=np.asarray([branch.path_code for branch in branches]),
        parent_index=np.asarray([branch.parent_index for branch in branches]),
        global_time=global_time,
        event_labels=labels,
        frontier_mask=frontier,
    )


def _draw_index(probabilities: NDArray[np.float64], generator: np.random.Generator) -> int:
    probabilities = probabilities / probabilities.sum()
    return int(generator.choice(len(probabilities), p=probabilities))


def sample_event_tree(
    model: EventFlowModel,
    geometry_bank: OracleGeometryBank,
    *,
    config: EventSamplerConfig | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    class_priors: tuple[float, ...] | None = None,
) -> EventSamplingResult:
    """Sample topology while branch geometry is supplied by a fixed empirical bank."""

    config = config or EventSamplerConfig()
    generator = np.random.default_rng(seed)
    root = geometry_bank.root_templates[
        int(generator.integers(0, len(geometry_bank.root_templates)))
    ]
    branches = [_GeneratedBranch(root, -1, 0, 1, 1, 0.0)]
    active: set[int] = {0}
    events: list[SampledEvent] = []
    wait_steps = 0
    forced_stop_count = 0
    termination_reason = "max_steps"
    target_device = torch.device(device)
    model.eval()
    log_prior: torch.Tensor | None = None
    if class_priors is not None:
        if len(class_priors) != len(EventClass) or any(value <= 0.0 for value in class_priors):
            raise ValueError("class_priors must contain four positive probabilities")
        prior = torch.tensor(class_priors, dtype=torch.float32, device=target_device)
        prior = prior / prior.sum()
        log_prior = config.prior_correction_strength * prior.log()

    for step in range(config.max_steps):
        if not active:
            if forced_stop_count == 0:
                termination_reason = "all_stopped"
            break
        global_time = min((step + 1) / (config.max_steps + 1), 1.0 - 1e-5)
        sample = _state_sample(branches, active, global_time)
        batch: EventBatch = collate_event_samples(
            [sample], max_tree_distance=model.config.max_tree_distance
        ).to(target_device)
        with torch.no_grad():
            logits = model(batch)[0].float()
            if log_prior is not None:
                # Inverse-frequency CE learns approximately uniform-prior posteriors.
                # Restore the empirical frontier-event prior before free sampling.
                logits = logits + log_prior
            probabilities = torch.softmax(
                logits / config.temperature, dim=-1
            ).cpu().numpy()
        active_indices = np.asarray(sorted(active), dtype=np.int64)
        frontier_probabilities = probabilities[active_indices]
        all_wait_probability = float(
            np.prod(frontier_probabilities[:, int(EventClass.WAIT)])
        )
        if float(generator.random()) < all_wait_probability:
            wait_steps += 1
            events.append(SampledEvent(step, None, EventClass.WAIT, global_time))
            continue
        hazards = 1.0 - frontier_probabilities[:, int(EventClass.WAIT)]
        selected_position = _draw_index(hazards.astype(np.float64), generator)
        branch_index = int(active_indices[selected_position])
        non_wait = frontier_probabilities[selected_position, 1:].astype(np.float64)
        predicted = EventClass(1 + _draw_index(non_wait, generator))
        events.append(SampledEvent(step, branch_index, predicted, global_time))

        if predicted is EventClass.STOP:
            active.remove(branch_index)
            branches[branch_index].stopped = True
            continue
        child_count = 1 if predicted is EventClass.EXTEND else 2
        child_depth = branches[branch_index].depth + 1
        if child_depth > config.max_depth:
            active.remove(branch_index)
            branches[branch_index].stopped = True
            forced_stop_count += 1
            termination_reason = "max_depth"
            continue
        if len(branches) + child_count > config.max_branches:
            forced_stop_count += len(active)
            for active_index in active:
                branches[active_index].stopped = True
            active.clear()
            termination_reason = "max_branches"
            break
        active.remove(branch_index)
        parent_path = branches[branch_index].path_code
        for child_offset in range(child_count):
            template = geometry_bank.child_templates[
                int(generator.integers(0, len(geometry_bank.child_templates)))
            ]
            child_position = child_offset + 1
            path_code = (parent_path * 3 + child_position) % model.config.path_buckets
            branches.append(
                _GeneratedBranch(
                    template,
                    branch_index,
                    child_depth,
                    child_position,
                    path_code,
                    global_time,
                )
            )
            active.add(len(branches) - 1)
    else:
        forced_stop_count += len(active)
        for active_index in active:
            branches[active_index].stopped = True
        active.clear()

    leaf_count = sum(
        not any(branch.parent_index == index for branch in branches)
        for index in range(len(branches))
    )
    return EventSamplingResult(
        branch_count=len(branches),
        leaf_count=leaf_count,
        maximum_depth=max(branch.depth for branch in branches),
        event_sequence=tuple(events),
        wait_steps=wait_steps,
        forced_stop_count=forced_stop_count,
        termination_reason=termination_reason,
        seed=seed,
    )
