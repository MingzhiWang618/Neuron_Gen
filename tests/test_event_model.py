from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import numpy as np
    import torch

    from arborflow.data.bezier_fitting import BezierFitReport
    from arborflow.data.event_dataset import (
        EventDatasetConfig,
        EventFlowDataset,
        collate_event_samples,
    )
    from arborflow.data.geometry_dataset import GeometryTreeRecord
    from arborflow.data.normalization import GeometryNormalizer
    from arborflow.flow.event_process import (
        EventClass,
        event_metrics,
        masked_event_cross_entropy,
    )
    from arborflow.models.arborflow import GeometryModelConfig
    from arborflow.models.event_model import EventFlowModel
    from arborflow.training.event_trainer import (
        EventTrainingConfig,
        train_event_model,
    )
    from tests.test_pruning import primitive_tree


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch model extra is not installed")
class EventModelTests(unittest.TestCase):
    def _dataset(self, *, samples: int = 32) -> EventFlowDataset:
        tree = primitive_tree(max_length=1.0)
        record = GeometryTreeRecord(Path("fixture.swc"), tree, BezierFitReport(()))
        return EventFlowDataset(
            [record],
            normalizer=GeometryNormalizer.from_trees([tree]),
            config=EventDatasetConfig(
                trajectories_per_neuron=4,
                samples_per_neuron=samples,
                base_seed=71,
                resample_each_epoch=False,
            ),
        )

    def test_oracle_geometry_labels_only_the_current_frontier(self) -> None:
        dataset = self._dataset()
        labels_seen: set[int] = set()
        for index in range(len(dataset)):
            sample = dataset[index]
            np.testing.assert_array_equal(
                sample.current_geometry, sample.target_geometry
            )
            self.assertTrue(np.all(sample.event_labels[~sample.frontier_mask] == -100))
            frontier_labels = sample.event_labels[sample.frontier_mask]
            self.assertEqual(
                int(np.count_nonzero(frontier_labels != int(EventClass.WAIT))), 1
            )
            labels_seen.update(int(value) for value in frontier_labels)
        self.assertEqual(labels_seen, {int(item) for item in EventClass})

    def test_variable_frontier_forward_and_loss_are_masked_and_finite(self) -> None:
        dataset = self._dataset()
        samples = [dataset[index] for index in range(8)]
        batch = collate_event_samples(samples, max_tree_distance=8)
        model = EventFlowModel(
            GeometryModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                feedforward_dim=32,
                max_tree_distance=8,
            )
        )
        logits = model(batch)
        self.assertEqual(logits.shape, (*batch.padding_mask.shape, len(EventClass)))
        loss = masked_event_cross_entropy(
            logits, batch.event_labels, batch.frontier_mask
        )
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(
            all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )

    def test_macro_f1_counts_all_four_classes(self) -> None:
        labels = torch.tensor([[0, 1, 2, 3]])
        frontier = torch.ones_like(labels, dtype=torch.bool)
        logits = torch.full((1, 4, 4), -4.0)
        logits[0, torch.arange(4), labels[0]] = 4.0
        metrics = event_metrics(logits, labels, frontier)
        self.assertAlmostEqual(metrics.macro_f1, 1.0)
        self.assertAlmostEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.class_support, (1, 1, 1, 1))

    def test_tiny_event_training_is_finite(self) -> None:
        dataset = self._dataset(samples=32)
        result = train_event_model(
            dataset,
            dataset,
            model_config=GeometryModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                feedforward_dim=32,
                max_tree_distance=8,
            ),
            training_config=EventTrainingConfig(
                epochs=2,
                batch_size=8,
                learning_rate=1e-3,
                random_rotation=False,
                mixed_precision=False,
                seed=72,
            ),
        )
        self.assertTrue(result.all_finite)
        self.assertEqual(len(result.class_counts), len(EventClass))
        self.assertEqual(result.amp_dtype, "float32")
        self.assertGreaterEqual(result.best_validation_macro_f1, 0.0)


if __name__ == "__main__":
    unittest.main()
