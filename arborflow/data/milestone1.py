"""End-to-end Milestone 1 inspection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arborflow.data.bezier_fitting import BezierFitConfig, fit_bezier_tree
from arborflow.data.binary_normalization import denormalize_binary, normalize_binary
from arborflow.data.branch_decomposition import branch_tree_to_swc_exact, decompose_swc
from arborflow.data.swc_io import SwcMorphology, read_swc, write_swc
from arborflow.data.swc_validation import SwcValidationConfig, clean_swc, validate_swc
from arborflow.reconstruction.tree_to_swc import bezier_tree_to_swc
from arborflow.structures.tree_invariants import (
    critical_topology_signature,
    validate_bezier_tree,
    validate_embedded_tree,
)
from arborflow.visualization.render_tree import render_swc_comparison_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ArborFlow Milestone 1 round-trip")
    parser.add_argument("input", type=Path, help="input SWC file")
    parser.add_argument("--output", type=Path, required=True, help="artifact directory")
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--min-real-branches", type=int, default=5)
    parser.add_argument("--max-real-branches", type=int, default=1000)
    parser.add_argument("--allow-planar", action="store_true")
    parser.add_argument("--max-bezier-rmse-um", type=float, default=1.5)
    parser.add_argument("--max-bezier-error-um", type=float, default=3.0)
    parser.add_argument("--max-primitive-length-um", type=float, default=80.0)
    parser.add_argument("--export-sample-spacing-um", type=float, default=2.0)
    return parser


def _morphio_readable(path: Path) -> bool | None:
    try:
        import morphio  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        morphio.Morphology(str(path))
    except Exception:  # MorphIO exposes several parser-specific exception types.
        return False
    return True


def _bifurcation_count(morphology: SwcMorphology) -> int:
    child_counts: dict[int, int] = {}
    root_ids = {node.node_id for node in morphology if node.parent_id == -1}
    for node in morphology:
        if node.parent_id != -1:
            child_counts[node.parent_id] = child_counts.get(node.parent_id, 0) + 1
    return sum(
        child_count >= 2 and node_id not in root_ids
        for node_id, child_count in child_counts.items()
    )


def run(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    source = read_swc(args.input)
    validation_config = SwcValidationConfig(
        require_3d=not args.allow_planar,
        min_nodes=args.min_nodes,
        min_real_branches=args.min_real_branches,
        max_real_branches=args.max_real_branches,
    )
    cleaning = clean_swc(source, validation_config)
    if not cleaning.success or cleaning.morphology is None:
        (args.output / "milestone1.json").write_text(
            json.dumps({"success": False, "cleaning": cleaning.to_dict()}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 1
    cleaned = cleaning.morphology
    write_swc(cleaned, args.output / "cleaned.swc")

    tree = decompose_swc(cleaned)
    exact = branch_tree_to_swc_exact(tree)
    exact_roundtrip = exact.nodes == cleaned.nodes
    write_swc(exact, args.output / "exact_roundtrip.swc")

    normalized, normalization_map = normalize_binary(tree)
    restored = denormalize_binary(normalized, normalization_map)
    binary_roundtrip = branch_tree_to_swc_exact(restored).nodes == cleaned.nodes
    binary_report = validate_embedded_tree(normalized, require_binary=True)

    fit_config = BezierFitConfig(
        max_rmse_um=args.max_bezier_rmse_um,
        max_error_um=args.max_bezier_error_um,
        max_primitive_length_um=args.max_primitive_length_um,
    )
    bezier_tree, fit_report = fit_bezier_tree(normalized, fit_config)
    bezier_invariants = validate_bezier_tree(bezier_tree, require_binary=True)
    reconstructed = bezier_tree_to_swc(
        bezier_tree, sample_spacing_um=args.export_sample_spacing_um
    )
    reconstructed_path = args.output / "bezier_reconstruction.swc"
    write_swc(reconstructed, reconstructed_path)
    render_swc_comparison_svg(
        cleaned, reconstructed, args.output / "original_vs_reconstruction.svg"
    )

    relaxed = SwcValidationConfig(
        require_3d=False,
        min_nodes=1,
        min_real_branches=0,
        max_real_branches=max(args.max_real_branches, 2**31 - 1),
    )
    reconstruction_validation = validate_swc(reconstructed, relaxed)
    source_bifurcations = _bifurcation_count(cleaned)
    reconstruction_bifurcations = _bifurcation_count(reconstructed)
    bifurcations_preserved = source_bifurcations == reconstruction_bifurcations
    topology_preserved = (
        critical_topology_signature(cleaned)
        == critical_topology_signature(reconstructed)
    )
    fit_within_thresholds = (
        fit_report.max_rmse_um <= fit_config.max_rmse_um
        and fit_report.max_error_um <= fit_config.max_error_um
        and all(
            item.polyline_length_um <= fit_config.max_primitive_length_um + 1e-8
            for item in fit_report.stats
        )
    )
    success = all(
        (
            exact_roundtrip,
            binary_roundtrip,
            binary_report.valid,
            bezier_invariants.valid,
            reconstruction_validation.valid,
            topology_preserved,
            bifurcations_preserved,
            fit_within_thresholds,
        )
    )
    morphio_readable = _morphio_readable(reconstructed_path)
    metrics = {
        "success": success,
        "milestone1_acceptance_complete": success and morphio_readable is True,
        "cleaning": cleaning.to_dict(),
        "exact_roundtrip": exact_roundtrip,
        "binary_roundtrip": binary_roundtrip,
        "binary_tree_valid": binary_report.valid,
        "binary_tree_errors": binary_report.errors,
        "bezier_tree_valid": bezier_invariants.valid,
        "bezier_tree_errors": bezier_invariants.errors,
        "parent_child_continuity_error_um": bezier_invariants.max_continuity_error,
        "topology_preserved": topology_preserved,
        "bifurcations_preserved": bifurcations_preserved,
        "source_bifurcations": source_bifurcations,
        "reconstruction_bifurcations": reconstruction_bifurcations,
        "source_nodes": len(cleaned),
        "source_real_branches": cleaning.after.real_branch_count if cleaning.after else None,
        "decomposed_branches": len(tree.branches),
        "normalized_branches": len(normalized.branches),
        "virtual_branches": len(normalization_map.virtual_branch_ids),
        "bezier_primitives": len(bezier_tree.primitives),
        "max_bezier_rmse_um": fit_report.max_rmse_um,
        "max_bezier_error_um": fit_report.max_error_um,
        "fit_within_thresholds": fit_within_thresholds,
        "reconstruction_validation": reconstruction_validation.to_dict(),
        "morphio_readable": morphio_readable,
    }
    (args.output / "milestone1.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if success else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
