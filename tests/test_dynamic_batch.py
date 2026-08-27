from __future__ import annotations

import importlib.util
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from arborflow.data.dynamic_batch import (
        collate_geometry_samples,
        geometry_state_sample,
        random_so3,
        rotate_geometry_batch,
    )
    from arborflow.data.normalization import GeometryNormalizer
    from arborflow.data.trajectory_builder import (
        build_pruning_trajectory,
        reverse_to_growth,
    )
    from arborflow.flow.oracle_replay import OracleReplay
    from tests.test_pruning import primitive_tree


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch model extra is not installed")
class DynamicBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        tree = primitive_tree(max_length=1.0)
        growth = reverse_to_growth(tree, build_pruning_trajectory(tree, seed=5))
        self.replay = OracleReplay(tree, growth, geometry_seed=12)
        self.normalizer = GeometryNormalizer.from_trees([tree])

    def test_padding_and_tree_relations_follow_current_dynamic_size(self) -> None:
        first_time = self.replay.trajectory.events[0].event_time
        early = geometry_state_sample(
            self.replay, first_time, self.normalizer
        )
        late = geometry_state_sample(self.replay, 0.999, self.normalizer)
        batch = collate_geometry_samples([early, late], max_tree_distance=8)
        self.assertEqual(tuple(batch.continuous_features.shape[:2]), (2, late.branch_count))
        self.assertEqual(int(batch.padding_mask[0].sum()), early.branch_count)
        self.assertEqual(int(batch.padding_mask[1].sum()), late.branch_count)
        self.assertTrue(torch.all(batch.relation[1].diagonal() == 1))
        for index, parent in enumerate(late.parent_index):
            if parent >= 0:
                self.assertEqual(int(batch.shortest_path_distance[1, index, parent]), 1)
                self.assertEqual(int(batch.relation[1, parent, index]), 2)
                self.assertEqual(int(batch.relation[1, index, parent]), 3)

    def test_so3_rotation_preserves_vector_norms_and_has_positive_determinant(self) -> None:
        sample = geometry_state_sample(self.replay, 0.8, self.normalizer)
        batch = collate_geometry_samples([sample])
        rotations = random_so3(16, generator=torch.Generator().manual_seed(3))
        determinants = torch.linalg.det(rotations)
        torch.testing.assert_close(determinants, torch.ones_like(determinants), atol=1e-5, rtol=0)
        rotated = rotate_geometry_batch(
            batch, generator=torch.Generator().manual_seed(4)
        )
        original_norm = batch.current_geometry[..., :9].reshape(1, -1, 3, 3).norm(dim=-1)
        rotated_norm = rotated.current_geometry[..., :9].reshape(1, -1, 3, 3).norm(dim=-1)
        torch.testing.assert_close(original_norm, rotated_norm, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(
            batch.padding_mask.numpy(), rotated.padding_mask.numpy()
        )


if __name__ == "__main__":
    unittest.main()
