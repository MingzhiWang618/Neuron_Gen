from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from arborflow.data.bezier_fitting import BezierFitReport
    from arborflow.data.geometry_dataset import (
        GeometryDatasetConfig,
        GeometryFlowDataset,
        GeometryTreeRecord,
    )
    from arborflow.data.normalization import GeometryNormalizer
    from arborflow.models.arborflow import GeometryModelConfig
    from arborflow.training.geometry_trainer import (
        GeometryTrainingConfig,
        train_geometry_flow,
    )
    from tests.test_pruning import primitive_tree


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch model extra is not installed")
class GeometryTrainingTests(unittest.TestCase):
    def _dataset(self, *, samples: int = 2) -> GeometryFlowDataset:
        tree = primitive_tree(max_length=1.0)
        record = GeometryTreeRecord(Path("fixture.swc"), tree, BezierFitReport(()))
        return GeometryFlowDataset(
            [record],
            normalizer=GeometryNormalizer.from_trees([tree]),
            config=GeometryDatasetConfig(
                trajectories_per_neuron=1,
                samples_per_neuron=samples,
                base_seed=31,
                resample_each_epoch=False,
            ),
        )

    def test_tiny_fixed_dataset_overfits_without_nonfinite_values(self) -> None:
        dataset = self._dataset(samples=2)
        result = train_geometry_flow(
            dataset,
            dataset,
            model_config=GeometryModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                feedforward_dim=32,
                max_tree_distance=8,
            ),
            training_config=GeometryTrainingConfig(
                epochs=30,
                batch_size=2,
                learning_rate=2e-3,
                random_rotation=False,
                mixed_precision=False,
                seed=32,
            ),
        )
        self.assertTrue(result.all_finite)
        self.assertFalse(result.amp_enabled)
        self.assertEqual(result.amp_dtype, "float32")
        self.assertLess(
            result.best_train_loss,
            float(result.initial_train["loss"]),
        )
        self.assertLess(
            result.best_validation_control_rmse_um,
            float(result.initial_validation["control_rmse_um"]),
        )

    def test_rotation_augmented_training_remains_finite(self) -> None:
        dataset = self._dataset(samples=1)
        result = train_geometry_flow(
            dataset,
            dataset,
            model_config=GeometryModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                feedforward_dim=32,
                max_tree_distance=8,
            ),
            training_config=GeometryTrainingConfig(
                epochs=3,
                batch_size=1,
                learning_rate=1e-3,
                random_rotation=True,
                mixed_precision=False,
                seed=33,
            ),
        )
        self.assertTrue(result.all_finite)
        self.assertGreater(result.max_abs_target_velocity, 0.0)


if __name__ == "__main__":
    unittest.main()
