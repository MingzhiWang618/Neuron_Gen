from __future__ import annotations

import unittest

from arborflow.data.branch_decomposition import branch_tree_to_swc_exact, decompose_swc
from arborflow.data.swc_io import SwcMorphology, SwcNode
from arborflow.structures.tree_invariants import validate_embedded_tree
from tests.fixtures import branching_morphology


class BranchDecompositionTests(unittest.TestCase):
    def test_key_node_paths_and_parentage(self) -> None:
        tree = decompose_swc(branching_morphology())
        self.assertEqual(len(tree.branches), 4)
        trunk = tree.branches[0]
        self.assertEqual(trunk.source_node_ids, (1, 2, 3))
        self.assertIsNone(trunk.parent_id)
        self.assertEqual(len(trunk.children_ids), 3)
        self.assertEqual({branch.depth for branch in tree.branches[1:]}, {1})
        report = validate_embedded_tree(tree)
        self.assertTrue(report.valid, report.errors)
        self.assertAlmostEqual(report.max_continuity_error, 0.0)

    def test_exact_roundtrip_preserves_every_swc_field_and_order(self) -> None:
        source = branching_morphology()
        recovered = branch_tree_to_swc_exact(decompose_swc(source))
        self.assertEqual(recovered.nodes, source.nodes)
        self.assertEqual(recovered.comments, source.comments)

    def test_deep_tree_has_no_python_recursion_limit(self) -> None:
        nodes = [SwcNode(1, 1, 0.0, 0.0, 0.0, 2.0, -1)]
        current = 1
        next_id = 2
        for depth in range(1050):
            trunk_id = next_id
            leaf_id = next_id + 1
            next_id += 2
            nodes.append(SwcNode(trunk_id, 3, depth + 1.0, 0.0, 0.1, 1.0, current))
            nodes.append(SwcNode(leaf_id, 3, depth + 0.5, 1.0, 0.2, 0.8, current))
            current = trunk_id
        morphology = SwcMorphology(tuple(nodes))
        tree = decompose_swc(morphology)
        self.assertGreater(max(branch.depth for branch in tree.branches), 1000)
        self.assertEqual(branch_tree_to_swc_exact(tree).nodes, morphology.nodes)


if __name__ == "__main__":
    unittest.main()
