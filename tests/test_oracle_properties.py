from __future__ import annotations

import unittest

import numpy as np

from arborflow.data.bezier_fitting import fit_bezier_tree
from arborflow.data.binary_normalization import normalize_binary
from arborflow.data.branch_decomposition import decompose_swc
from arborflow.data.swc_io import SwcMorphology, SwcNode
from arborflow.data.trajectory_builder import (
    PruningConfig,
    build_pruning_trajectories,
    reverse_to_growth,
)
from arborflow.flow.oracle_replay import OracleReplay


class OraclePropertyTests(unittest.TestCase):
    def test_random_trees_replay_without_index_or_geometry_errors(self) -> None:
        for tree_seed in range(8):
            generator = np.random.default_rng(tree_seed)
            nodes = [SwcNode(1, 1, 0.0, 0.0, 0.0, 3.0, -1)]
            positions = {1: np.zeros(3)}
            for node_id in range(2, 42):
                parent_id = int(generator.integers(1, node_id))
                direction = generator.normal(size=3)
                direction /= np.linalg.norm(direction)
                position = positions[parent_id] + direction * float(
                    generator.uniform(0.5, 3.0)
                )
                positions[node_id] = position
                nodes.append(
                    SwcNode(
                        node_id,
                        int(generator.choice((2, 3, 4))),
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        float(generator.uniform(0.2, 1.2)),
                        parent_id,
                    )
                )
            normalized, _ = normalize_binary(
                decompose_swc(SwcMorphology(tuple(nodes)))
            )
            tree, _ = fit_bezier_tree(normalized)
            trajectories = build_pruning_trajectories(
                tree,
                base_seed=tree_seed,
                config=PruningConfig(trajectories_per_neuron=2),
            )
            for trajectory_index, pruning in enumerate(trajectories):
                replay = OracleReplay(
                    tree,
                    reverse_to_growth(tree, pruning),
                    geometry_seed=10_000 * tree_seed + trajectory_index,
                )
                report = replay.replay()
                self.assertTrue(report.valid, report.errors)
                self.assertTrue(report.topology_exact)
                self.assertTrue(report.dynamic_indices_stable)
                self.assertLessEqual(report.max_final_control_error_um, 1e-12)


if __name__ == "__main__":
    unittest.main()
