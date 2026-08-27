from __future__ import annotations

import unittest

from arborflow.data.swc_io import SwcMorphology, SwcNode, parse_swc_lines
from arborflow.data.swc_validation import SwcValidationConfig, clean_swc, validate_swc


def node(node_id: int, parent_id: int, xyz: tuple[float, float, float]) -> SwcNode:
    return SwcNode(node_id, 3 if parent_id != -1 else 1, *xyz, 1.0, parent_id)


TEST_CONFIG = SwcValidationConfig(
    require_3d=False,
    min_nodes=1,
    min_real_branches=0,
    max_real_branches=100,
)


class SwcValidationTests(unittest.TestCase):
    def test_valid_tree(self) -> None:
        morphology = SwcMorphology(
            (
                node(1, -1, (0, 0, 0)),
                node(2, 1, (1, 0, 0)),
                node(3, 2, (2, 1, 0)),
                node(4, 2, (2, -1, 0)),
            )
        )
        report = validate_swc(morphology, TEST_CONFIG)
        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(report.real_branch_count, 3)

    def test_duplicate_id_is_explicit_error(self) -> None:
        morphology = SwcMorphology(
            (node(1, -1, (0, 0, 0)), node(1, -1, (1, 0, 0)))
        )
        report = validate_swc(morphology, TEST_CONFIG)
        self.assertIn("duplicate_node_id", {issue.code for issue in report.errors})

    def test_invalid_parent_is_not_repaired(self) -> None:
        morphology = SwcMorphology((node(1, -1, (0, 0, 0)), node(2, 99, (1, 0, 0))))
        result = clean_swc(morphology, TEST_CONFIG)
        self.assertFalse(result.success)
        self.assertIsNone(result.morphology)
        self.assertIn("invalid_parent_id", {issue.code for issue in result.before.errors})

    def test_cycle_is_detected(self) -> None:
        morphology = SwcMorphology((node(1, 2, (0, 0, 0)), node(2, 1, (1, 0, 0))))
        config = SwcValidationConfig(
            require_3d=False,
            require_single_root=False,
            min_nodes=1,
            min_real_branches=0,
            max_real_branches=100,
        )
        report = validate_swc(morphology, config)
        self.assertIn("cycle", {issue.code for issue in report.errors})

    def test_zero_length_node_is_collapsed_and_children_reparented(self) -> None:
        morphology = SwcMorphology(
            (
                node(1, -1, (0, 0, 0)),
                node(2, 1, (0, 0, 0)),
                node(3, 2, (1, 0, 0)),
            )
        )
        result = clean_swc(morphology, TEST_CONFIG)
        self.assertTrue(result.success, result.to_dict())
        assert result.morphology is not None
        self.assertEqual([sample.node_id for sample in result.morphology], [1, 3])
        self.assertEqual(result.morphology.nodes[1].parent_id, 1)
        self.assertEqual(result.actions[0].node_id, 2)

    def test_nonlocal_duplicate_coordinate_is_not_silently_merged(self) -> None:
        morphology = SwcMorphology(
            (
                node(1, -1, (0, 0, 0)),
                node(2, 1, (1, 1, 0)),
                node(3, 1, (1, -1, 0)),
                node(4, 2, (2, 0, 0)),
                node(5, 3, (2, 0, 0)),
            )
        )
        result = clean_swc(morphology, TEST_CONFIG)
        self.assertFalse(result.success)
        self.assertIn(
            "ambiguous_duplicate_coordinate", {issue.code for issue in result.before.errors}
        )

    def test_duplicate_tolerance_uses_euclidean_distance_across_hash_buckets(self) -> None:
        config = SwcValidationConfig(
            require_3d=False,
            min_nodes=1,
            min_real_branches=0,
            max_real_branches=100,
            coordinate_tolerance_um=1.0,
        )
        near_across_boundary = SwcMorphology(
            (
                node(1, -1, (0, 0, 0)),
                node(2, 1, (0.9, 0, 0)),
                node(3, 1, (1.1, 0, 0)),
            )
        )
        report = validate_swc(near_across_boundary, config)
        self.assertIn("ambiguous_duplicate_coordinate", {issue.code for issue in report.errors})

        diagonal_beyond_radius = SwcMorphology(
            (
                node(1, -1, (0, 0, 0)),
                node(2, 1, (1.1, 0, 0)),
                node(3, 1, (1.9, 0.9, 0.9)),
            )
        )
        report = validate_swc(diagonal_beyond_radius, config)
        self.assertNotIn(
            "ambiguous_duplicate_coordinate", {issue.code for issue in report.errors}
        )

    def test_admission_filters_run_after_cleaning(self) -> None:
        morphology = parse_swc_lines(
            ["1 1 0 0 0 1 -1", "2 3 1 0 0 1 1", "3 3 2 0 0 1 2"]
        )
        strict = SwcValidationConfig(
            require_3d=False,
            min_nodes=4,
            min_real_branches=0,
            max_real_branches=100,
        )
        result = clean_swc(morphology, strict)
        self.assertFalse(result.success)
        assert result.after is not None
        self.assertIn("too_few_nodes", {issue.code for issue in result.after.errors})


if __name__ == "__main__":
    unittest.main()
