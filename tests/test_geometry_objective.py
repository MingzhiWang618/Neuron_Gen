from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from arborflow.data.dynamic_batch import (
        collate_geometry_samples,
        geometry_state_sample,
    )
    from arborflow.data.normalization import GeometryNormalizer
    from arborflow.data.trajectory_builder import (
        build_pruning_trajectory,
        reverse_to_growth,
    )
    from arborflow.flow.hybrid_objective import geometry_metrics
    from arborflow.flow.oracle_replay import OracleReplay
    from tests.test_pruning import primitive_tree


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch model extra is not installed")
class GeometryObjectiveTests(unittest.TestCase):
    def test_oracle_velocity_reconstructs_target_at_machine_precision(self) -> None:
        tree = primitive_tree(max_length=1.0)
        growth = reverse_to_growth(tree, build_pruning_trajectory(tree, seed=7))
        replay = OracleReplay(tree, growth, geometry_seed=14)
        normalizer = GeometryNormalizer.from_trees([tree])
        sample = geometry_state_sample(replay, 0.63, normalizer)
        batch = collate_geometry_samples([sample])
        metrics = geometry_metrics(
            batch.target_velocity,
            batch,
            coordinate_scale_um=normalizer.coordinate_scale_um,
            radius_scale_um=normalizer.radius_scale_um,
        )
        self.assertLessEqual(metrics.loss, 1e-12)
        self.assertLessEqual(metrics.control_rmse_um, 1e-6)
        self.assertLessEqual(metrics.radius_rmse_um, 1e-6)
        self.assertTrue(metrics.finite)
        self.assertTrue(torch.isfinite(batch.target_velocity).all())


if __name__ == "__main__":
    unittest.main()
