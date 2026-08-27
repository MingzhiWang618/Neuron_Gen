from __future__ import annotations

import unittest
from dataclasses import replace

from arborflow.data.bezier_fitting import BezierFitConfig, fit_bezier_tree
from arborflow.data.binary_normalization import normalize_binary
from arborflow.data.branch_decomposition import decompose_swc
from arborflow.data.trajectory_builder import (
    PruningConfig,
    build_pruning_trajectories,
    build_pruning_trajectory,
    validate_pruning_trajectory,
)
from arborflow.structures.tree_events import PruneActionKind
from tests.fixtures import balanced_morphology, branching_morphology


def primitive_tree(*, max_length: float = 80.0):
    normalized, _ = normalize_binary(decompose_swc(branching_morphology()))
    tree, _ = fit_bezier_tree(
        normalized, BezierFitConfig(max_primitive_length_um=max_length)
    )
    return tree


class PruningTests(unittest.TestCase):
    def test_every_step_is_legal_and_reversal_recovers_target(self) -> None:
        tree = primitive_tree(max_length=1.0)
        trajectory = build_pruning_trajectory(tree, seed=17)
        report = validate_pruning_trajectory(tree, trajectory)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(trajectory.steps[-1].remaining_branch_count, 0)
        self.assertEqual(
            {step.action_kind for step in trajectory.steps},
            {
                PruneActionKind.TERMINAL_BRANCH,
                PruneActionKind.TERMINAL_SIBLING_PAIR,
            },
        )
        removed = [
            branch_id for step in trajectory.steps for branch_id in step.removed_branch_ids
        ]
        self.assertEqual(sorted(removed), sorted(tree.by_id()))
        self.assertEqual(len(removed), len(set(removed)))

    def test_same_seed_is_bitwise_reproducible(self) -> None:
        tree = primitive_tree()
        first = build_pruning_trajectory(tree, seed=123)
        second = build_pruning_trajectory(tree, seed=123)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_tampered_internal_deletion_is_rejected(self) -> None:
        tree = primitive_tree()
        trajectory = build_pruning_trajectory(tree, seed=123)
        first = trajectory.steps[0]
        internal_id = next(
            primitive.primitive_id for primitive in tree.primitives if primitive.children_ids
        )
        tampered_step = replace(
            first,
            removed_branch_ids=(internal_id,),
            action_kind=PruneActionKind.TERMINAL_BRANCH,
        )
        tampered = replace(trajectory, steps=(tampered_step, *trajectory.steps[1:]))
        report = validate_pruning_trajectory(tree, tampered)
        self.assertFalse(report.valid)
        self.assertIn("not a legal leaf-pruning action", report.errors[0])

    def test_eight_seeded_trajectories_include_multiple_legal_orders(self) -> None:
        normalized, _ = normalize_binary(decompose_swc(balanced_morphology()))
        tree, _ = fit_bezier_tree(normalized)
        trajectories = build_pruning_trajectories(
            tree,
            base_seed=9,
            config=PruningConfig(trajectories_per_neuron=8),
        )
        self.assertEqual(len(trajectories), 8)
        orders = {
            tuple(step.removed_branch_ids for step in trajectory.steps)
            for trajectory in trajectories
        }
        self.assertGreater(len(orders), 1)
        for trajectory in trajectories:
            report = validate_pruning_trajectory(tree, trajectory)
            self.assertTrue(report.valid, report.errors)


if __name__ == "__main__":
    unittest.main()
