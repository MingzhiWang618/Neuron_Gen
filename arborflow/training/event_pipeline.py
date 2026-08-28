"""Command-line Stage-B pipeline for event learning with oracle geometry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from arborflow.data.event_dataset import EventDatasetConfig, EventFlowDataset
from arborflow.data.geometry_dataset import GeometryTreeRecord
from arborflow.data.normalization import GeometryNormalizer
from arborflow.flow.event_process import EventClass
from arborflow.flow.event_sampler import (
    EventSamplerConfig,
    OracleGeometryBank,
    sample_event_tree,
)
from arborflow.models.arborflow import GeometryModelConfig
from arborflow.training.event_trainer import (
    EventTrainingConfig,
    checkpoint_payload,
    train_event_model,
)
from arborflow.training.geometry_pipeline import (
    _prepare_records,
    _resolved_device,
    _split_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train WAIT/EXTEND/SPLIT/STOP events with oracle geometry"
    )
    parser.add_argument("input", type=Path, help="SWC file or directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--no-random-rotation", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--trajectories-per-neuron", type=int, default=4)
    parser.add_argument("--samples-per-neuron", type=int, default=32)
    parser.add_argument("--event-time-jitter", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-tree-distance", type=int, default=16)
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--min-real-branches", type=int, default=5)
    parser.add_argument("--max-real-branches", type=int, default=1000)
    parser.add_argument("--allow-planar", action="store_true")
    parser.add_argument("--max-bezier-rmse-um", type=float, default=1.5)
    parser.add_argument("--max-bezier-error-um", type=float, default=3.0)
    parser.add_argument("--max-primitive-length-um", type=float, default=80.0)
    parser.add_argument("--minimum-macro-f1-margin", type=float, default=0.02)
    parser.add_argument("--num-generated-samples", type=int, default=64)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--prior-correction-strength", type=float, default=1.0)
    parser.add_argument("--max-branches", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--maximum-forced-termination-rate", type=float, default=1.0)
    parser.add_argument("--minimum-size-ratio", type=float, default=0.5)
    parser.add_argument("--maximum-size-ratio", type=float, default=2.0)
    return parser


def _target_distribution(records: list[GeometryTreeRecord]) -> dict[str, float]:
    counts = [len(record.tree.primitives) for record in records]
    depths = [max(primitive.depth for primitive in record.tree.primitives) for record in records]
    return {
        "sample_count": len(records),
        "branch_count_mean": float(np.mean(counts)),
        "branch_count_std": float(np.std(counts)),
        "maximum_depth_mean": float(np.mean(depths)),
        "maximum_depth_std": float(np.std(depths)),
    }


def _sampling_report(
    model,
    train_records: list[GeometryTreeRecord],
    validation_records: list[GeometryTreeRecord],
    normalizer: GeometryNormalizer,
    args: argparse.Namespace,
    device: torch.device,
    class_counts: tuple[int, ...],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bank = OracleGeometryBank.from_records(train_records, normalizer)
    config = EventSamplerConfig(
        max_branches=args.max_branches,
        max_depth=args.max_depth,
        max_steps=args.max_steps,
        temperature=args.sampling_temperature,
        prior_correction_strength=args.prior_correction_strength,
    )
    class_priors = tuple(
        count / sum(class_counts) for count in class_counts
    )
    results = [
        sample_event_tree(
            model,
            bank,
            config=config,
            device=device,
            seed=args.seed + 1_000_003 + index,
            class_priors=class_priors,
        )
        for index in range(args.num_generated_samples)
    ]
    generated_counts = np.asarray([item.branch_count for item in results], dtype=float)
    generated_depths = np.asarray([item.maximum_depth for item in results], dtype=float)
    target = _target_distribution(validation_records)
    branch_ratio = float(generated_counts.mean()) / max(
        float(target["branch_count_mean"]), 1e-12
    )
    depth_ratio = float(generated_depths.mean()) / max(
        float(target["maximum_depth_mean"]), 1e-12
    )
    forced_rate = float(np.mean([item.forced_termination for item in results]))
    reasons = Counter(item.termination_reason for item in results)
    report: dict[str, object] = {
        "oracle_geometry_provider": True,
        "class_priors": class_priors,
        "prior_correction_strength": args.prior_correction_strength,
        "generated_sample_count": len(results),
        "generated_branch_count_mean": float(generated_counts.mean()),
        "generated_branch_count_std": float(generated_counts.std()),
        "generated_maximum_depth_mean": float(generated_depths.mean()),
        "generated_maximum_depth_std": float(generated_depths.std()),
        "target": target,
        "branch_count_mean_ratio": branch_ratio,
        "maximum_depth_mean_ratio": depth_ratio,
        "forced_termination_rate": forced_rate,
        "termination_reasons": dict(sorted(reasons.items())),
        "finite_termination_pass": all(
            len(item.event_sequence) <= args.max_steps for item in results
        ),
    }
    records: list[dict[str, object]] = []
    for item in results:
        records.append(
            {
                "seed": item.seed,
                "branch_count": item.branch_count,
                "leaf_count": item.leaf_count,
                "maximum_depth": item.maximum_depth,
                "wait_steps": item.wait_steps,
                "forced_stop_count": item.forced_stop_count,
                "termination_reason": item.termination_reason,
                "events": [
                    {
                        "step": event.step,
                        "branch_index": event.branch_index,
                        "event_class": event.event_class.name,
                        "global_time": event.global_time,
                    }
                    for event in item.event_sequence
                ],
            }
        )
    return report, records


def run(args: argparse.Namespace) -> int:
    if args.max_samples < 2 and not args.overfit:
        raise ValueError("event validation requires at least two requested samples")
    if args.num_generated_samples < 1 or args.num_threads < 1:
        raise ValueError("sample and thread counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    records, failures = _prepare_records(args)
    train_records, validation_records = _split_records(records, args)
    normalizer = GeometryNormalizer.from_trees(
        [record.tree for record in train_records]
    )
    dataset_config = EventDatasetConfig(
        trajectories_per_neuron=args.trajectories_per_neuron,
        samples_per_neuron=args.samples_per_neuron,
        base_seed=args.seed,
        event_time_jitter=args.event_time_jitter,
        resample_each_epoch=not args.overfit,
    )
    train_dataset = EventFlowDataset(
        train_records, normalizer=normalizer, config=dataset_config
    )
    validation_dataset = EventFlowDataset(
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
    training_config = EventTrainingConfig(
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
    result = train_event_model(
        train_dataset,
        validation_dataset,
        model_config=model_config,
        training_config=training_config,
        device=device,
        progress_callback=lambda record: print(
            "epoch={epoch} train_macro_f1={train_macro_f1:.4f} "
            "validation_macro_f1={validation_macro_f1:.4f}".format(**record),
            flush=True,
        ),
    )
    sampling, sample_records = _sampling_report(
        result.model,
        train_records,
        validation_records,
        normalizer,
        args,
        device,
        result.class_counts,
    )
    macro_margin = (
        result.best_validation_macro_f1 - result.majority_baseline_macro_f1
    )
    macro_pass = macro_margin >= args.minimum_macro_f1_margin
    size_pass = (
        args.minimum_size_ratio
        <= float(sampling["branch_count_mean_ratio"])
        <= args.maximum_size_ratio
        and args.minimum_size_ratio
        <= float(sampling["maximum_depth_mean_ratio"])
        <= args.maximum_size_ratio
    )
    termination_pass = (
        bool(sampling["finite_termination_pass"])
        and float(sampling["forced_termination_rate"])
        <= args.maximum_forced_termination_rate
    )
    success = result.all_finite and macro_pass and size_pass and termination_pass
    torch.save(
        checkpoint_payload(result, model_config, training_config, train_dataset),
        args.output / "event_model.pt",
    )
    (args.output / "history.json").write_text(
        json.dumps(list(result.history), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "samples.json").write_text(
        json.dumps(sample_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "success": success,
        "prepared_sample_count": len(records),
        "train_sample_count": len(train_records),
        "validation_sample_count": len(validation_records),
        "preparation_failure_count": len(failures),
        "preparation_failures": failures,
        "train_sources": [str(record.path) for record in train_records],
        "validation_sources": [str(record.path) for record in validation_records],
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
        "event_classes": [item.name for item in EventClass],
        "class_counts": result.class_counts,
        "class_weights": result.class_weights,
        "initial_train": asdict(result.initial_train),
        "initial_validation": asdict(result.initial_validation),
        "best_validation": asdict(result.best_validation),
        "best_train_macro_f1": result.best_train_macro_f1,
        "best_validation_macro_f1": result.best_validation_macro_f1,
        "majority_class": EventClass(result.majority_class).name,
        "majority_baseline_macro_f1": result.majority_baseline_macro_f1,
        "macro_f1_margin": macro_margin,
        "macro_f1_acceptance_pass": macro_pass,
        "sampling": sampling,
        "sampling_distribution_pass": size_pass,
        "sampling_termination_pass": termination_pass,
        "all_finite": result.all_finite,
        "max_unclipped_gradient_norm": result.max_unclipped_gradient_norm,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if success else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
