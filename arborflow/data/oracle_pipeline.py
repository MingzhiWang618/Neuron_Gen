"""Command-line pipeline for Milestone 3 analytic oracle replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arborflow.data.bezier_fitting import BezierFitConfig, fit_bezier_tree
from arborflow.data.binary_normalization import normalize_binary
from arborflow.data.branch_decomposition import decompose_swc
from arborflow.data.swc_io import read_swc, write_swc
from arborflow.data.swc_validation import SwcValidationConfig, clean_swc
from arborflow.data.trajectory_builder import (
    PruningConfig,
    build_pruning_trajectories,
    resample_event_times,
    reverse_to_growth,
)
from arborflow.flow.geometry_path import OracleGeometryConfig
from arborflow.flow.oracle_replay import OracleReplay
from arborflow.reconstruction.tree_to_swc import bezier_tree_to_swc
from arborflow.structures.tree_invariants import critical_topology_signature
from arborflow.visualization.render_oracle import render_oracle_replay_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay complete birth-death growth with oracle events and geometry"
    )
    parser.add_argument("input", type=Path, help="clean or raw SWC input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--birth-noise-sigma-um", type=float, default=0.01)
    parser.add_argument("--event-time-jitter", type=float, default=0.25)
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--min-real-branches", type=int, default=5)
    parser.add_argument("--max-real-branches", type=int, default=1000)
    parser.add_argument("--allow-planar", action="store_true")
    parser.add_argument("--max-bezier-rmse-um", type=float, default=1.5)
    parser.add_argument("--max-bezier-error-um", type=float, default=3.0)
    parser.add_argument("--max-primitive-length-um", type=float, default=80.0)
    parser.add_argument("--sample-spacing-um", type=float, default=2.0)
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
    trajectories = build_pruning_trajectories(
        tree,
        base_seed=args.seed,
        config=PruningConfig(trajectories_per_neuron=args.num_trajectories),
    )
    source_signature = critical_topology_signature(cleaning.morphology)
    records: list[dict[str, object]] = []
    all_valid = True
    for index, pruning in enumerate(trajectories):
        growth = reverse_to_growth(tree, pruning)
        growth = resample_event_times(
            growth,
            seed=args.seed + index,
            maximum_jitter=args.event_time_jitter,
        )
        geometry_seed = args.seed + 1_000_003 * (index + 1)
        replay = OracleReplay(
            tree,
            growth,
            geometry_seed=geometry_seed,
            geometry_config=OracleGeometryConfig(args.birth_noise_sigma_um),
        )
        report = replay.replay()
        final_tree = replay.materialize_tree(report.final_state)
        reconstructed = bezier_tree_to_swc(
            final_tree, sample_spacing_um=args.sample_spacing_um
        )
        swc_topology_exact = critical_topology_signature(reconstructed) == source_signature
        valid = report.valid and swc_topology_exact
        all_valid &= valid
        swc_path = args.output / f"oracle_{index:03d}.swc"
        write_swc(reconstructed, swc_path)
        render_oracle_replay_svg(
            replay,
            args.output / f"oracle_{index:03d}.svg",
            max_frames=args.max_visualization_frames,
        )
        record = {
            "valid": valid,
            "geometry_seed": geometry_seed,
            "swc_topology_exact": swc_topology_exact,
            "growth": growth.to_dict(),
            "replay": report.to_dict(),
        }
        records.append(record)
        (args.output / f"oracle_{index:03d}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary = {
        "success": all_valid,
        "source": str(args.input),
        "base_seed": args.seed,
        "trajectory_count": len(trajectories),
        "valid_replay_count": sum(bool(record["valid"]) for record in records),
        "primitive_count": len(tree.primitives),
        "virtual_branch_count": len(normalization_map.virtual_branch_ids),
        "max_bezier_rmse_um": fit_report.max_rmse_um,
        "max_bezier_error_um": fit_report.max_error_um,
        "max_oracle_control_error_um": max(
            float(record["replay"]["max_final_control_error_um"])
            for record in records
        ),
        "max_oracle_radius_error_um": max(
            float(record["replay"]["max_final_radius_error_um"])
            for record in records
        ),
        "max_continuity_error_um": max(
            float(record["replay"]["max_continuity_error_um"])
            for record in records
        ),
        "topology_exact_count": sum(
            bool(record["replay"]["topology_exact"]) for record in records
        ),
        "swc_topology_exact_count": sum(
            bool(record["swc_topology_exact"]) for record in records
        ),
        "dynamic_indices_stable_count": sum(
            bool(record["replay"]["dynamic_indices_stable"]) for record in records
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if all_valid else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
