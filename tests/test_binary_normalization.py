from __future__ import annotations

import unittest

from arborflow.data.binary_normalization import denormalize_binary, normalize_binary
from arborflow.data.branch_decomposition import branch_tree_to_swc_exact, decompose_swc
from arborflow.structures.tree_invariants import validate_embedded_tree
from tests.fixtures import branching_morphology, root_multifurcation


class BinaryNormalizationTests(unittest.TestCase):
    def _check_roundtrip(self, morphology, expected_virtual: int) -> None:
        original = decompose_swc(morphology)
        normalized, mapping = normalize_binary(original)
        report = validate_embedded_tree(normalized, require_binary=True)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(sum(branch.virtual for branch in normalized.branches), expected_virtual)
        restored = denormalize_binary(normalized, mapping)
        recovered = branch_tree_to_swc_exact(restored)
        self.assertEqual(recovered.nodes, morphology.nodes)

    def test_internal_trifurcation_is_reversible(self) -> None:
        self._check_roundtrip(branching_morphology(), expected_virtual=1)

    def test_root_four_way_fanout_is_reversible(self) -> None:
        self._check_roundtrip(root_multifurcation(), expected_virtual=2)

    def test_normalization_is_deterministic(self) -> None:
        tree = decompose_swc(root_multifurcation())
        first, first_map = normalize_binary(tree)
        second, second_map = normalize_binary(tree)
        self.assertEqual(first_map, second_map)
        self.assertEqual(first.root_branch_ids, second.root_branch_ids)
        self.assertEqual(
            [(item.branch_id, item.parent_id, item.children_ids) for item in first.branches],
            [(item.branch_id, item.parent_id, item.children_ids) for item in second.branches],
        )


if __name__ == "__main__":
    unittest.main()

