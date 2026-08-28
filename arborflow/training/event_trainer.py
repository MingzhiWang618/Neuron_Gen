"""Stage-B trainer: oracle geometry with frontier event classification only."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable

import torch
from torch.utils.data import DataLoader

from arborflow.data.dynamic_batch import rotate_geometry_batch
from arborflow.data.event_dataset import (
    EventBatch,
    EventFlowDataset,
    collate_event_samples,
)
from arborflow.flow.event_process import (
    NUM_EVENT_CLASSES,
    EventClass,
    EventMetrics,
    event_metrics,
    macro_f1_from_confusion,
    masked_event_cross_entropy,
)
from arborflow.models.arborflow import GeometryModelConfig
from arborflow.models.event_model import EventFlowModel
from arborflow.training.geometry_trainer import _amp_dtype


@dataclass(frozen=True)
class EventTrainingConfig:
    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    random_rotation: bool = True
    mixed_precision: bool = True
    seed: int = 0
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning rate and gradient clip norm must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass(frozen=True)
class EventTrainingResult:
    model: EventFlowModel
    history: tuple[dict[str, object], ...]
    initial_train: EventMetrics
    initial_validation: EventMetrics
    best_validation: EventMetrics
    best_train_macro_f1: float
    best_validation_macro_f1: float
    majority_class: int
    majority_baseline_macro_f1: float
    class_counts: tuple[int, ...]
    class_weights: tuple[float, ...]
    max_unclipped_gradient_norm: float
    all_finite: bool
    device: str
    amp_enabled: bool
    amp_dtype: str


def _loader(
    dataset: EventFlowDataset,
    config: EventTrainingConfig,
    *,
    shuffle: bool,
    max_tree_distance: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        collate_fn=partial(
            collate_event_samples, max_tree_distance=max_tree_distance
        ),
    )


def _balanced_class_weights(counts: tuple[int, ...]) -> torch.Tensor:
    values = torch.tensor(counts, dtype=torch.float64)
    if bool((values <= 0).any()):
        missing = [index for index, count in enumerate(counts) if count <= 0]
        raise ValueError(f"event training data is missing classes {missing}")
    weights = values.sum() / values
    return (weights / weights.mean()).to(torch.float32)


def evaluate_event_model(
    model: EventFlowModel,
    loader: DataLoader,
    dataset: EventFlowDataset,
    device: torch.device,
) -> EventMetrics:
    model.eval()
    dataset.set_epoch(0)
    loss_sum = 0.0
    event_count = 0
    matrix = torch.zeros((NUM_EVENT_CLASSES, NUM_EVENT_CLASSES), dtype=torch.long)
    finite = True
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            metrics = event_metrics(logits, batch.event_labels, batch.frontier_mask)
            count = sum(metrics.class_support)
            loss_sum += metrics.loss * count
            event_count += count
            matrix += torch.tensor(metrics.confusion, dtype=torch.long)
            finite &= metrics.finite
    support = matrix.sum(dim=1)
    total = int(matrix.sum())
    return EventMetrics(
        loss=loss_sum / max(event_count, 1),
        accuracy=float(matrix.diag().sum()) / max(total, 1),
        macro_f1=macro_f1_from_confusion(matrix),
        confusion=tuple(tuple(int(value) for value in row) for row in matrix.tolist()),
        class_support=tuple(int(value) for value in support.tolist()),
        finite=finite,
    )


def train_event_model(
    train_dataset: EventFlowDataset,
    validation_dataset: EventFlowDataset,
    *,
    model_config: GeometryModelConfig | None = None,
    training_config: EventTrainingConfig | None = None,
    device: str | torch.device = "cpu",
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> EventTrainingResult:
    """Train only the event head/backbone; geometry is fixed to oracle targets."""

    model_config = model_config or GeometryModelConfig()
    training_config = training_config or EventTrainingConfig()
    torch.manual_seed(training_config.seed)
    target_device = torch.device(device)
    model = EventFlowModel(model_config).to(target_device)
    train_loader = _loader(
        train_dataset,
        training_config,
        shuffle=True,
        max_tree_distance=model_config.max_tree_distance,
    )
    validation_loader = _loader(
        validation_dataset,
        training_config,
        shuffle=False,
        max_tree_distance=model_config.max_tree_distance,
    )
    counts = train_dataset.class_counts()
    class_weights = _balanced_class_weights(counts).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    use_amp = training_config.mixed_precision and target_device.type == "cuda"
    amp_dtype = _amp_dtype(target_device, use_amp)
    try:
        scaler = torch.amp.GradScaler(
            target_device.type, enabled=use_amp, init_scale=256.0
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=256.0)

    initial_train = evaluate_event_model(model, train_loader, train_dataset, target_device)
    initial_validation = evaluate_event_model(
        model, validation_loader, validation_dataset, target_device
    )
    validation_support = torch.tensor(initial_validation.class_support, dtype=torch.long)
    majority_class = int(validation_support.argmax())
    majority_matrix = torch.zeros(
        (NUM_EVENT_CLASSES, NUM_EVENT_CLASSES), dtype=torch.long
    )
    majority_matrix[:, majority_class] = validation_support
    majority_f1 = macro_f1_from_confusion(majority_matrix)

    best_state = copy.deepcopy(model.state_dict())
    best_train_f1 = initial_train.macro_f1
    best_validation_f1 = initial_validation.macro_f1
    max_gradient_norm = 0.0
    all_finite = initial_train.finite and initial_validation.finite
    history: list[dict[str, object]] = []
    rotation_generator = torch.Generator().manual_seed(training_config.seed + 12_337)

    for epoch in range(training_config.epochs):
        model.train()
        train_dataset.set_epoch(epoch + 1)
        optimizer.zero_grad(set_to_none=True)
        optimization_losses: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            batch = batch.to(target_device)
            if training_config.random_rotation:
                batch = rotate_geometry_batch(batch, generator=rotation_generator)
            with torch.autocast(
                device_type=target_device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(batch)
                loss = masked_event_cross_entropy(
                    logits,
                    batch.event_labels,
                    batch.frontier_mask,
                    class_weights=class_weights,
                )
                scaled_loss = loss / training_config.gradient_accumulation_steps
            finite_step = bool(torch.isfinite(loss) and torch.isfinite(logits).all())
            all_finite &= finite_step
            if not finite_step:
                raise FloatingPointError(f"non-finite event step at epoch {epoch}")
            scaler.scale(scaled_loss).backward()
            update = (
                (batch_index + 1) % training_config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if update:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.gradient_clip_norm
                )
                if not math.isfinite(float(gradient_norm)):
                    raise FloatingPointError(f"non-finite event gradient at epoch {epoch}")
                max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            optimization_losses.append(float(loss.detach()))

        train_metrics = evaluate_event_model(
            model, train_loader, train_dataset, target_device
        )
        validation_metrics = evaluate_event_model(
            model, validation_loader, validation_dataset, target_device
        )
        best_train_f1 = max(best_train_f1, train_metrics.macro_f1)
        if validation_metrics.macro_f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.macro_f1
            best_state = copy.deepcopy(model.state_dict())
        all_finite &= train_metrics.finite and validation_metrics.finite
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "optimization_loss": sum(optimization_losses)
            / max(len(optimization_losses), 1),
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "train_macro_f1": train_metrics.macro_f1,
            "validation_loss": validation_metrics.loss,
            "validation_accuracy": validation_metrics.accuracy,
            "validation_macro_f1": validation_metrics.macro_f1,
            "validation_confusion": validation_metrics.confusion,
            "finite": train_metrics.finite and validation_metrics.finite,
        }
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

    model.load_state_dict(best_state)
    best_validation = evaluate_event_model(
        model, validation_loader, validation_dataset, target_device
    )
    return EventTrainingResult(
        model=model,
        history=tuple(history),
        initial_train=initial_train,
        initial_validation=initial_validation,
        best_validation=best_validation,
        best_train_macro_f1=best_train_f1,
        best_validation_macro_f1=best_validation_f1,
        majority_class=majority_class,
        majority_baseline_macro_f1=majority_f1,
        class_counts=counts,
        class_weights=tuple(float(value) for value in class_weights.cpu()),
        max_unclipped_gradient_norm=max_gradient_norm,
        all_finite=all_finite,
        device=str(target_device),
        amp_enabled=use_amp,
        amp_dtype=str(amp_dtype).removeprefix("torch."),
    )


def checkpoint_payload(
    result: EventTrainingResult,
    model_config: GeometryModelConfig,
    training_config: EventTrainingConfig,
    train_dataset: EventFlowDataset,
) -> dict[str, object]:
    return {
        "model_state_dict": result.model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "normalizer": train_dataset.normalizer.to_dict(),
        "event_classes": [item.name for item in EventClass],
        "event_class_counts": result.class_counts,
        "event_class_weights": result.class_weights,
        "milestone": 5,
        "oracle_geometry": True,
    }
