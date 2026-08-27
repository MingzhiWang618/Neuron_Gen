from __future__ import annotations

import unittest

import numpy as np

from arborflow.data.bezier_fitting import fit_bezier_tree
from arborflow.data.binary_normalization import denormalize_binary, normalize_binary
from arborflow.data.branch_decomposition import branch_tree_to_swc_exact, decompose_swc
from arborflow.data.swc_io import SwcMorphology, SwcNode
from arborflow.reconstruction.tree_to_swc import bezier_tree_to_swc
from arborflow.structures.tree_invariants import (
    critical_topology_signature,
    validate_bezier_tree,
    validate_embedded_tree,
)


class Milestone1PropertyTests(unittest.TestCase):
    def test_random_rooted_trees_preserve_topology(self) -> None:
        for seed in range(20):
            rng = np.random.default_rng(seed)
            nodes = [SwcNode(1, 1, 0.0, 0.0, 0.0, 3.0, -1)]
            positions = {1: np.zeros(3)}
            for node_id in range(2, 62):
                parent_id = int(rng.integers(1, node_id))
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                position = positions[parent_id] + direction * float(rng.uniform(0.5, 3.0))
                positions[node_id] = position
                nodes.append(
                    SwcNode(
                        node_id,
                        int(rng.choice((2, 3, 4))),
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        float(rng.uniform(0.2, 1.2)),
                        parent_id,
                    )
                )
            morphology = SwcMorphology(tuple(nodes))
            decomposed = decompose_swc(morphology)
            self.assertEqual(branch_tree_to_swc_exact(decomposed).nodes, morphology.nodes)
            normalized, mapping = normalize_binary(decomposed)
            self.assertTrue(validate_embedded_tree(normalized, require_binary=True).valid)
            restored = denormalize_binary(normalized, mapping)
            self.assertEqual(branch_tree_to_swc_exact(restored).nodes, morphology.nodes)
            bezier, _ = fit_bezier_tree(normalized)
            self.assertTrue(validate_bezier_tree(bezier, require_binary=True).valid)
            reconstructed = bezier_tree_to_swc(bezier)
            self.assertEqual(
                critical_topology_signature(reconstructed),
                critical_topology_signature(morphology),
            )


if __name__ == "__main__":
    unittest.main()
