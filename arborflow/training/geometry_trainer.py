"""Numerically guarded Stage-A trainer for oracle-event geometry flow."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from functools import partial

import torch
from torch.utils.data import DataLoader

from arborflow.data.dynamic_batch import (
    GeometryBatch,
    collate_geometry_samples,
    rotate_geometry_batch,
)
from arborflow.data.geometry_dataset import GeometryFlowDataset
from arborflow.flow.hybrid_objective import geometry_metrics, masked_velocity_mse
from arborflow.models.arborflow import GeometryFlowModel, GeometryModelConfig


@dataclass(frozen=True)
class GeometryTrainingConfig:
    epochs: int = 80
    batch_size: int = 2
    learning_rate: float = 3e-4
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
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass(frozen=True)
class GeometryTrainingResult:
    model: GeometryFlowModel
    history: tuple[dict[str, float | bool | int], ...]
    initial_train: dict[str, float | bool]
    initial_validation: dict[str, float | bool]
    best_train_loss: float
    best_validation_loss: float
    best_validation_control_rmse_um: float
    max_unclipped_gradient_norm: float
    max_abs_prediction: float
    max_abs_target_velocity: float
    all_finite: bool
    device: str
    amp_enabled: bool
    amp_dtype: str


def _amp_dtype(device: torch.device, enabled: bool) -> torch.dtype:
    """Prefer BF16 on supported CUDA devices for its FP32-like dynamic range."""

    if not enabled:
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _loader(
    dataset: GeometryFlowDataset,
    config: GeometryTrainingConfig,
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
            collate_geometry_samples, max_tree_distance=max_tree_distance
        ),
    )


def evaluate_geometry_flow(
    model: GeometryFlowModel,
    loader: DataLoader,
    dataset: GeometryFlowDataset,
    device: torch.device,
) -> dict[str, float | bool]:
    model.eval()
    losses: list[float] = []
    control_errors: list[float] = []
    radius_errors: list[float] = []
    finite = True
    dataset.set_epoch(0)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            metrics = geometry_metrics(
                prediction,
                batch,
                coordinate_scale_um=dataset.normalizer.coordinate_scale_um,
                radius_scale_um=dataset.normalizer.radius_scale_um,
            )
            losses.append(metrics.loss)
            control_errors.append(metrics.control_rmse_um)
            radius_errors.append(metrics.radius_rmse_um)
            finite &= metrics.finite
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "control_rmse_um": float(
            sum(control_errors) / max(len(control_errors), 1)
        ),
        "radius_rmse_um": float(sum(radius_errors) / max(len(radius_errors), 1)),
        "finite": finite,
    }


def train_geometry_flow(
    train_dataset: GeometryFlowDataset,
    validation_dataset: GeometryFlowDataset,
    *,
    model_config: GeometryModelConfig | None = None,
    training_config: GeometryTrainingConfig | None = None,
    device: str | torch.device = "cpu",
) -> GeometryTrainingResult:
    """Train only velocity while topology and birth times remain oracle-provided."""

    model_config = model_config or GeometryModelConfig()
    training_config = training_config or GeometryTrainingConfig()
    torch.manual_seed(training_config.seed)
    target_device = torch.device(device)
    model = GeometryFlowModel(model_config).to(target_device)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    use_amp = training_config.mixed_precision and target_device.type == "cuda"
    amp_dtype = _amp_dtype(target_device, use_amp)
    try:
        scaler = torch.amp.GradScaler(
            target_device.type,
            enabled=use_amp,
            init_scale=256.0,
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=256.0)
    initial_train = evaluate_geometry_flow(
        model, train_loader, train_dataset, target_device
    )
    initial_validation = evaluate_geometry_flow(
        model, validation_loader, validation_dataset, target_device
    )
    history: list[dict[str, float | bool | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float(initial_validation["loss"])
    best_validation_control = float(initial_validation["control_rmse_um"])
    best_train_loss = float(initial_train["loss"])
    max_gradient_norm = 0.0
    max_abs_prediction = 0.0
    max_abs_target_velocity = 0.0
    all_finite = bool(initial_train["finite"] and initial_validation["finite"])
    rotation_generator = torch.Generator().manual_seed(training_config.seed + 7_919)

    for epoch in range(training_config.epochs):
        model.train()
        train_dataset.set_epoch(epoch + 1)
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            batch = batch.to(target_device)
            if training_config.random_rotation:
                batch = rotate_geometry_batch(batch, generator=rotation_generator)
            with torch.autocast(
                device_type=target_device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                prediction = model(batch)
                loss = masked_velocity_mse(
                    prediction, batch.target_velocity, batch.padding_mask
                )
                scaled_loss = loss / training_config.gradient_accumulation_steps
            finite_step = bool(
                torch.isfinite(loss) and torch.isfinite(prediction).all()
            )
            all_finite &= finite_step
            if not finite_step:
                raise FloatingPointError(f"non-finite geometry step at epoch {epoch}")
            max_abs_prediction = max(
                max_abs_prediction, float(prediction.detach().abs().max())
            )
            max_abs_target_velocity = max(
                max_abs_target_velocity,
                float(batch.target_velocity.detach().abs().max()),
            )
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
                finite_gradient = math.isfinite(float(gradient_norm))
                all_finite &= finite_gradient
                if not finite_gradient:
                    raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
                max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.detach()))

        train_metrics = evaluate_geometry_flow(
            model, train_loader, train_dataset, target_device
        )
        validation_metrics = evaluate_geometry_flow(
            model, validation_loader, validation_dataset, target_device
        )
        best_train_loss = min(best_train_loss, float(train_metrics["loss"]))
        if float(validation_metrics["loss"]) < best_validation_loss:
            best_validation_loss = float(validation_metrics["loss"])
            best_state = copy.deepcopy(model.state_dict())
        best_validation_control = min(
            best_validation_control,
            float(validation_metrics["control_rmse_um"]),
        )
        all_finite &= bool(train_metrics["finite"] and validation_metrics["finite"])
        history.append(
            {
                "epoch": epoch + 1,
                "optimization_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                "train_loss": float(train_metrics["loss"]),
                "train_control_rmse_um": float(train_metrics["control_rmse_um"]),
                "train_radius_rmse_um": float(train_metrics["radius_rmse_um"]),
                "validation_loss": float(validation_metrics["loss"]),
                "validation_control_rmse_um": float(
                    validation_metrics["control_rmse_um"]
                ),
                "validation_radius_rmse_um": float(
                    validation_metrics["radius_rmse_um"]
                ),
                "finite": bool(train_metrics["finite"] and validation_metrics["finite"]),
            }
        )
    model.load_state_dict(best_state)
    return GeometryTrainingResult(
        model=model,
        history=tuple(history),
        initial_train=initial_train,
        initial_validation=initial_validation,
        best_train_loss=best_train_loss,
        best_validation_loss=best_validation_loss,
        best_validation_control_rmse_um=best_validation_control,
        max_unclipped_gradient_norm=max_gradient_norm,
        max_abs_prediction=max_abs_prediction,
        max_abs_target_velocity=max_abs_target_velocity,
        all_finite=all_finite,
        device=str(target_device),
        amp_enabled=use_amp,
        amp_dtype=str(amp_dtype).removeprefix("torch."),
    )


def checkpoint_payload(
    result: GeometryTrainingResult,
    model_config: GeometryModelConfig,
    training_config: GeometryTrainingConfig,
    train_dataset: GeometryFlowDataset,
) -> dict[str, object]:
    return {
        "model_state_dict": result.model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "normalizer": train_dataset.normalizer.to_dict(),
        "milestone": 4,
        "oracle_events": True,
    }
