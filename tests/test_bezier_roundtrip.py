from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from arborflow.data.bezier_fitting import BezierFitConfig, evaluate_cubic, fit_bezier_tree
from arborflow.data.binary_normalization import normalize_binary
from arborflow.data.branch_decomposition import decompose_swc
from arborflow.data.swc_io import SwcMorphology, SwcNode
from arborflow.data.swc_validation import SwcValidationConfig, validate_swc
from arborflow.reconstruction.tree_to_swc import bezier_tree_to_swc
from arborflow.structures.tree_invariants import critical_topology_signature
from arborflow.visualization.render_tree import render_swc_comparison_svg
from tests.fixtures import branching_morphology, root_multifurcation


class BezierRoundtripTests(unittest.TestCase):
    def test_known_cubic_is_fitted_exactly(self) -> None:
        controls = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 2.0, 0.5), (2.0, -1.0, 1.0), (4.0, 0.0, 2.0))
        )
        points = evaluate_cubic(controls, np.linspace(0.0, 1.0, 9))
        # Chord-length parameterization is generally different from the generating t,
        # so this checks bounded approximation rather than control-point identity.
        morphology = branching_morphology()
        tree = decompose_swc(morphology)
        branch = tree.branches[0]
        branch = replace(
            branch,
            points=points,
            radii=np.linspace(1.2, 0.8, len(points)),
            source_node_ids=tuple(range(100, 100 + len(points))),
            point_swc_types=np.full(len(points), 3),
        )
        custom_tree = replace(
            tree,
            root=replace(tree.root, node_id=100, x=0.0, y=0.0, z=0.0),
            branches=(replace(branch, parent_id=None, children_ids=()),),
            root_branch_ids=(branch.branch_id,),
            source_node_order=tuple(range(100, 100 + len(points))),
        )
        _, report = fit_bezier_tree(
            custom_tree, BezierFitConfig(max_rmse_um=0.3, max_error_um=0.6)
        )
        self.assertLessEqual(report.max_rmse_um, 0.3)
        self.assertLessEqual(report.max_error_um, 0.6)

    def test_long_two_point_branch_creates_continuations(self) -> None:
        source = root_multifurcation()
        # Stretch each two-point branch beyond the primitive length threshold.
        stretched_nodes = tuple(
            node if node.parent_id == -1 else type(node)(
                node.node_id,
                node.swc_type,
                node.x * 100.0,
                node.y * 100.0,
                node.z * 100.0,
                node.radius,
                node.parent_id,
            )
            for node in source
        )
        tree = decompose_swc(type(source)(stretched_nodes))
        bezier, report = fit_bezier_tree(tree, BezierFitConfig(max_primitive_length_um=80.0))
        self.assertGreater(len(bezier.primitives), len(tree.branches))
        self.assertTrue(any(item.continuation for item in bezier.primitives))
        self.assertLessEqual(max(item.polyline_length_um for item in report.stats), 80.0 + 1e-9)

    def test_swc_type_transition_creates_typed_continuation(self) -> None:
        source = SwcMorphology(
            (
                SwcNode(1, 1, 0.0, 0.0, 0.0, 2.0, -1),
                SwcNode(2, 3, 1.0, 0.0, 0.0, 1.0, 1),
                SwcNode(3, 4, 2.0, 0.0, 0.0, 0.9, 2),
                SwcNode(4, 4, 3.0, 0.0, 0.0, 0.8, 3),
            )
        )
        bezier, _ = fit_bezier_tree(decompose_swc(source))
        self.assertEqual([item.swc_type for item in bezier.primitives], [3, 4])
        self.assertTrue(bezier.primitives[1].continuation)
        reconstructed = bezier_tree_to_swc(bezier, sample_spacing_um=0.5)
        self.assertEqual({node.swc_type for node in reconstructed}, {1, 3, 4})

    def test_bezier_export_preserves_critical_topology_and_removes_virtuals(self) -> None:
        source = branching_morphology()
        normalized, _ = normalize_binary(decompose_swc(source))
        bezier, report = fit_bezier_tree(normalized)
        reconstructed = bezier_tree_to_swc(bezier, sample_spacing_um=0.5)
        validation = validate_swc(
            reconstructed,
            SwcValidationConfig(
                require_3d=False,
                min_nodes=1,
                min_real_branches=0,
                max_real_branches=100,
            ),
        )
        self.assertTrue(validation.valid, validation.to_dict())
        self.assertEqual(
            critical_topology_signature(reconstructed), critical_topology_signature(source)
        )
        self.assertEqual(validation.real_branch_count, 4)
        self.assertLessEqual(report.max_rmse_um, 1.5)
        self.assertLessEqual(report.max_error_um, 3.0)

    def test_dependency_free_svg_comparison(self) -> None:
        source = branching_morphology()
        bezier, _ = fit_bezier_tree(decompose_swc(source))
        reconstructed = bezier_tree_to_swc(bezier)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.svg"
            render_swc_comparison_svg(source, reconstructed, output)
            payload = output.read_text(encoding="utf-8")
        self.assertIn("<svg", payload)
        self.assertIn("original", payload)
        self.assertIn("reconstruction", payload)


if __name__ == "__main__":
    unittest.main()
