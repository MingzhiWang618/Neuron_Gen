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
    from arborflow.flow.hybrid_objective import masked_velocity_mse
    from arborflow.flow.oracle_replay import OracleReplay
    from arborflow.models.arborflow import GeometryFlowModel, GeometryModelConfig
    from tests.test_pruning import primitive_tree


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch model extra is not installed")
class ModelForwardTests(unittest.TestCase):
    def test_variable_size_forward_backward_is_masked_and_finite(self) -> None:
        tree = primitive_tree(max_length=1.0)
        growth = reverse_to_growth(tree, build_pruning_trajectory(tree, seed=6))
        replay = OracleReplay(tree, growth, geometry_seed=13)
        normalizer = GeometryNormalizer.from_trees([tree])
        samples = [
            geometry_state_sample(replay, global_time, normalizer)
            for global_time in (0.35, 0.85)
        ]
        batch = collate_geometry_samples(samples, max_tree_distance=8)
        model = GeometryFlowModel(
            GeometryModelConfig(
                d_model=32,
                num_heads=4,
                num_layers=2,
                feedforward_dim=64,
                max_tree_distance=8,
            )
        )
        prediction = model(batch)
        self.assertEqual(prediction.shape, batch.target_velocity.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.all(prediction[~batch.padding_mask] == 0))
        loss = masked_velocity_mse(
            prediction, batch.target_velocity, batch.padding_mask
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
