"""Command-line Stage-A pipeline for oracle-event geometry flow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from arborflow.data.bezier_fitting import BezierFitConfig
from arborflow.data.geometry_dataset import (
    GeometryDatasetConfig,
    GeometryFlowDataset,
    GeometryTreeRecord,
    prepare_geometry_tree,
)
from arborflow.data.normalization import GeometryNormalizer
from arborflow.data.swc_validation import SwcValidationConfig
from arborflow.models.arborflow import GeometryModelConfig
from arborflow.training.geometry_trainer import (
    GeometryTrainingConfig,
    checkpoint_payload,
    train_geometry_flow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train geometry velocity with oracle topology events"
    )
    parser.add_argument("input", type=Path, help="SWC file or directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--no-random-rotation", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--trajectories-per-neuron", type=int, default=8)
    parser.add_argument("--samples-per-neuron", type=int, default=4)
    parser.add_argument("--birth-noise-sigma-um", type=float, default=0.01)
    parser.add_argument("--event-time-jitter", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-tree-distance", type=int, default=16)
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--min-real-branches", type=int, default=5)
    parser.add_argument("--max-real-branches", type=int, default=1000)
    parser.add_argument("--allow-planar", action="store_true")
    parser.add_argument("--max-bezier-rmse-um", type=float, default=1.5)
    parser.add_argument("--max-bezier-error-um", type=float, default=3.0)
    parser.add_argument("--max-primitive-length-um", type=float, default=80.0)
    parser.add_argument("--overfit-loss-ratio", type=float, default=0.25)
    parser.add_argument("--validation-error-ratio", type=float, default=0.95)
    return parser


def _source_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"input does not exist: {path}")
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() == ".swc"
    )


def _prepare_records(args: argparse.Namespace) -> tuple[list[GeometryTreeRecord], list[str]]:
    validation_config = SwcValidationConfig(
        require_3d=not args.allow_planar,
        min_nodes=args.min_nodes,
        min_real_branches=args.min_real_branches,
        max_real_branches=args.max_real_branches,
    )
    fit_config = BezierFitConfig(
        max_rmse_um=args.max_bezier_rmse_um,
        max_error_um=args.max_bezier_error_um,
        max_primitive_length_um=args.max_primitive_length_um,
    )
    records: list[GeometryTreeRecord] = []
    failures: list[str] = []
    for source_path in _source_paths(args.input):
        if len(records) >= args.max_samples:
            break
        try:
            records.append(
                prepare_geometry_tree(
                    source_path,
                    validation_config=validation_config,
                    fit_config=fit_config,
                )
            )
        except (ValueError, OSError) as error:
            failures.append(str(error))
    if not records:
        raise ValueError("no valid SWC samples were prepared")
    return records, failures


def _split_records(
    records: list[GeometryTreeRecord], args: argparse.Namespace
) -> tuple[list[GeometryTreeRecord], list[GeometryTreeRecord]]:
    if args.overfit:
        return records, records
    if len(records) < 2:
        raise ValueError("a validation split requires at least two valid SWCs")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(records), generator=generator).tolist()
    validation_count = max(1, round(len(records) * args.validation_fraction))
    validation_indices = set(order[:validation_count])
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [
        record for index, record in enumerate(records) if index in validation_indices
    ]
    return train, validation


def _resolved_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def run(args: argparse.Namespace) -> int:
    if args.max_samples < 1:
        raise ValueError("max_samples must be positive")
    if args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    records, failures = _prepare_records(args)
    train_records, validation_records = _split_records(records, args)
    normalizer = GeometryNormalizer.from_trees(
        [record.tree for record in train_records]
    )
    dataset_config = GeometryDatasetConfig(
        trajectories_per_neuron=args.trajectories_per_neuron,
        samples_per_neuron=args.samples_per_neuron,
        base_seed=args.seed,
        birth_noise_sigma_um=args.birth_noise_sigma_um,
        event_time_jitter=args.event_time_jitter,
        resample_each_epoch=not args.overfit,
    )
    train_dataset = GeometryFlowDataset(
        train_records, normalizer=normalizer, config=dataset_config
    )
    validation_dataset = GeometryFlowDataset(
        validation_records, normalizer=normalizer, config=dataset_config
    )
    model_config = GeometryModelConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_tree_distance=args.max_tree_distance,
    )
    training_config = GeometryTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        random_rotation=not args.no_random_rotation,
        mixed_precision=not args.no_mixed_precision,
        seed=args.seed,
    )
    device = _resolved_device(args.device)
    result = train_geometry_flow(
        train_dataset,
        validation_dataset,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )
    initial_train_loss = float(result.initial_train["loss"])
    train_loss_ratio = result.best_train_loss / max(initial_train_loss, 1e-12)
    initial_validation_control = float(
        result.initial_validation["control_rmse_um"]
    )
    validation_control_ratio = result.best_validation_control_rmse_um / max(
        initial_validation_control, 1e-12
    )
    validation_decreased = validation_control_ratio < 1.0
    validation_pass = validation_control_ratio <= args.validation_error_ratio
    overfit_pass = (
        not args.overfit
        or (
            len(train_records) >= args.max_samples
            and train_loss_ratio <= args.overfit_loss_ratio
        )
    )
    rotation_pass = not training_config.random_rotation or result.all_finite
    success = (
        result.all_finite
        and validation_pass
        and overfit_pass
        and rotation_pass
    )
    history = list(result.history)
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        checkpoint_payload(result, model_config, training_config, train_dataset),
        args.output / "geometry_flow.pt",
    )
    summary = {
        "success": success,
        "mode": "overfit" if args.overfit else "train_validation_split",
        "prepared_sample_count": len(records),
        "train_sample_count": len(train_records),
        "validation_sample_count": len(validation_records),
        "preparation_failure_count": len(failures),
        "preparation_failures": failures,
        "train_sources": [str(record.path) for record in train_records],
        "validation_sources": [str(record.path) for record in validation_records],
        "train_primitive_count": sum(
            len(record.tree.primitives) for record in train_records
        ),
        "normalizer": normalizer.to_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "device": result.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "amp_enabled": result.amp_enabled,
        "amp_dtype": result.amp_dtype,
        "model_parameter_count": sum(
            parameter.numel() for parameter in result.model.parameters()
        ),
        "initial_train": result.initial_train,
        "initial_validation": result.initial_validation,
        "best_train_loss": result.best_train_loss,
        "best_validation_loss": result.best_validation_loss,
        "best_validation_control_rmse_um": result.best_validation_control_rmse_um,
        "train_loss_ratio": train_loss_ratio,
        "validation_control_ratio": validation_control_ratio,
        "validation_geometry_error_decreased": validation_decreased,
        "validation_acceptance_pass": validation_pass,
        "overfit_pass": overfit_pass if args.overfit else None,
        "rotation_augmentation_pass": (
            result.all_finite if training_config.random_rotation else None
        ),
        "rotation_augmentation_exercised": training_config.random_rotation,
        "all_finite": result.all_finite,
        "max_unclipped_gradient_norm": result.max_unclipped_gradient_norm,
        "max_abs_prediction": result.max_abs_prediction,
        "max_abs_target_velocity": result.max_abs_target_velocity,
        "prediction_to_target_scale_ratio": result.max_abs_prediction
        / max(result.max_abs_target_velocity, 1e-12),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if success else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
