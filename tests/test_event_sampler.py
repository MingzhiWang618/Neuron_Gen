from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from arborflow.data.bezier_fitting import BezierFitReport
    from arborflow.data.geometry_dataset import GeometryTreeRecord
    from arborflow.data.normalization import GeometryNormalizer
    from arborflow.flow.event_process import EventClass
    from arborflow.flow.event_sampler import (
        EventSamplerConfig,
        OracleGeometryBank,
        sample_event_tree,
    )
    from arborflow.models.arborflow import GeometryModelConfig
    from tests.test_pruning import primitive_tree


class _SplitThenStop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = GeometryModelConfig(
            d_model=16, num_heads=4, num_layers=1, feedforward_dim=32
        )

    def forward(self, batch):
        batch_size, token_count = batch.padding_mask.shape
        logits = torch.full((batch_size, token_count, len(EventClass)), -20.0)
        event = EventClass.SPLIT if token_count == 1 else EventClass.STOP
        logits[..., int(event)] = 20.0
        return logits.to(batch.padding_mask.device)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch model extra is not installed")
class EventSamplerTests(unittest.TestCase):
    def test_dynamic_split_inserts_children_then_stops_finitely(self) -> None:
        tree = primitive_tree(max_length=1.0)
        records = [GeometryTreeRecord(Path("fixture.swc"), tree, BezierFitReport(()))]
        normalizer = GeometryNormalizer.from_trees([tree])
        bank = OracleGeometryBank.from_records(records, normalizer)
        result = sample_event_tree(
            _SplitThenStop(),
            bank,
            config=EventSamplerConfig(max_branches=16, max_depth=8, max_steps=8),
            seed=91,
            class_priors=(0.25, 0.25, 0.25, 0.25),
        )
        self.assertEqual(result.branch_count, 3)
        self.assertEqual(result.leaf_count, 2)
        self.assertEqual(result.maximum_depth, 1)
        self.assertEqual(result.termination_reason, "all_stopped")
        self.assertFalse(result.forced_termination)
        self.assertEqual(
            [event.event_class for event in result.event_sequence],
            [EventClass.SPLIT, EventClass.STOP, EventClass.STOP],
        )

    def test_wait_policy_hits_max_steps_without_infinite_loop(self) -> None:
        model = _SplitThenStop()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        tree = primitive_tree()
        records = [GeometryTreeRecord(Path("fixture.swc"), tree, BezierFitReport(()))]
        normalizer = GeometryNormalizer.from_trees([tree])
        bank = OracleGeometryBank.from_records(records, normalizer)

        class WaitModel(_SplitThenStop):
            def forward(self, batch):
                logits = super().forward(batch)
                logits.fill_(-20.0)
                logits[..., int(EventClass.WAIT)] = 20.0
                return logits

        result = sample_event_tree(
            WaitModel(), bank, config=EventSamplerConfig(max_steps=5), seed=7
        )
        self.assertEqual(len(result.event_sequence), 5)
        self.assertEqual(result.wait_steps, 5)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertTrue(result.forced_termination)


if __name__ == "__main__":
    unittest.main()
