"""Command-line pipeline for Milestone 2 trajectory construction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from arborflow.data.bezier_fitting import BezierFitConfig, fit_bezier_tree
from arborflow.data.binary_normalization import normalize_binary
from arborflow.data.branch_decomposition import decompose_swc
from arborflow.data.swc_io import read_swc
from arborflow.data.swc_validation import SwcValidationConfig, clean_swc
from arborflow.data.trajectory_builder import (
    PruningConfig,
    build_pruning_trajectories,
    reverse_to_growth,
    validate_pruning_trajectory,
)
from arborflow.visualization.render_trajectory import render_pruning_trajectory_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build legal birth-death tree trajectories")
    parser.add_argument("input", type=Path, help="clean or raw SWC input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--min-real-branches", type=int, default=5)
    parser.add_argument("--max-real-branches", type=int, default=1000)
    parser.add_argument("--allow-planar", action="store_true")
    parser.add_argument("--max-bezier-rmse-um", type=float, default=1.5)
    parser.add_argument("--max-bezier-error-um", type=float, default=3.0)
    parser.add_argument("--max-primitive-length-um", type=float, default=80.0)
    parser.add_argument("--max-visualization-frames", type=int, default=12)
    return parser


def run(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    source = read_swc(args.input)
    cleaning = clean_swc(
        source,
        SwcValidationConfig(
            require_3d=not args.allow_planar,
            min_nodes=args.min_nodes,
            min_real_branches=args.min_real_branches,
            max_real_branches=args.max_real_branches,
        ),
    )
    if not cleaning.success or cleaning.morphology is None:
        (args.output / "summary.json").write_text(
            json.dumps({"success": False, "cleaning": cleaning.to_dict()}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 1
    normalized, normalization_map = normalize_binary(decompose_swc(cleaning.morphology))
    tree, fit_report = fit_bezier_tree(
        normalized,
        BezierFitConfig(
            max_rmse_um=args.max_bezier_rmse_um,
            max_error_um=args.max_bezier_error_um,
            max_primitive_length_um=args.max_primitive_length_um,
        ),
    )
    pruning_config = PruningConfig(trajectories_per_neuron=args.num_trajectories)
    trajectories = build_pruning_trajectories(
        tree, base_seed=args.seed, config=pruning_config
    )
    all_valid = True
    valid_count = 0
    strategy_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    orders: set[tuple[tuple[int, ...], ...]] = set()
    step_counts: list[int] = []
    growth_event_counts: list[int] = []
    for index, pruning in enumerate(trajectories):
        growth = reverse_to_growth(tree, pruning)
        validation = validate_pruning_trajectory(tree, pruning)
        all_valid &= validation.valid
        valid_count += int(validation.valid)
        strategy_counts.update(step.strategy.value for step in pruning.steps)
        event_counts.update(event.event_type.value for event in growth.events)
        step_counts.append(len(pruning.steps))
        growth_event_counts.append(len(growth.events))
        orders.add(tuple(step.removed_branch_ids for step in pruning.steps))
        record = {
            "valid": validation.valid,
            "errors": list(validation.errors),
            "pruning": pruning.to_dict(),
            "growth": growth.to_dict(),
        }
        (args.output / f"trajectory_{index:03d}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        render_pruning_trajectory_svg(
            tree,
            pruning,
            args.output / f"trajectory_{index:03d}.svg",
            max_frames=args.max_visualization_frames,
        )
    summary = {
        "success": all_valid,
        "source": str(args.input),
        "base_seed": args.seed,
        "trajectory_count": len(trajectories),
        "valid_trajectory_count": valid_count,
        "unique_pruning_orders": len(orders),
        "pruning_steps_min": min(step_counts, default=0),
        "pruning_steps_max": max(step_counts, default=0),
        "growth_events_min": min(growth_event_counts, default=0),
        "growth_events_max": max(growth_event_counts, default=0),
        "primitive_count": len(tree.primitives),
        "virtual_branch_count": len(normalization_map.virtual_branch_ids),
        "max_bezier_rmse_um": fit_report.max_rmse_um,
        "max_bezier_error_um": fit_report.max_error_um,
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if all_valid else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
