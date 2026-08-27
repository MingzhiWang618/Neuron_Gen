from __future__ import annotations

import unittest

from arborflow.data.trajectory_builder import (
    branch_birth_times,
    build_pruning_trajectory,
    replay_growth_trajectory,
    resample_event_times,
    reverse_to_growth,
)
from arborflow.structures.tree_events import EventType
from tests.test_pruning import primitive_tree


class GrowthReversalTests(unittest.TestCase):
    def test_event_sequence_has_extend_split_and_stop(self) -> None:
        tree = primitive_tree(max_length=1.0)
        pruning = build_pruning_trajectory(tree, seed=21)
        growth = reverse_to_growth(tree, pruning)
        event_types = {event.event_type for event in growth.events}
        self.assertEqual(event_types, {EventType.EXTEND, EventType.SPLIT, EventType.STOP})
        self.assertTrue(
            all(
                left.event_time < right.event_time
                for left, right in zip(growth.events, growth.events[1:])
            )
        )

    def test_structural_replay_recovers_all_primitives_and_stops_all_tips(self) -> None:
        tree = primitive_tree(max_length=1.0)
        pruning = build_pruning_trajectory(tree, seed=8)
        growth = reverse_to_growth(tree, pruning)
        report = replay_growth_trajectory(tree, growth)
        self.assertTrue(report.valid, report.errors)
        assert report.final_state is not None
        self.assertEqual(set(report.final_state.present_branch_ids), set(tree.by_id()))
        self.assertFalse(report.final_state.active_leaf_ids)
        expected_tips = {
            item.primitive_id for item in tree.primitives if not item.children_ids
        }
        self.assertEqual(set(report.final_state.stopped_branch_ids), expected_tips)

    def test_every_child_birth_occurs_after_its_parent(self) -> None:
        tree = primitive_tree(max_length=1.0)
        growth = reverse_to_growth(tree, build_pruning_trajectory(tree, seed=42))
        birth_time = {
            branch.primitive_id: event.event_time
            for event in growth.events
            for branch in event.new_branches
        }
        for primitive in tree.primitives:
            if primitive.parent_id is not None:
                self.assertLess(birth_time[primitive.parent_id], birth_time[primitive.primitive_id])

    def test_continuous_event_times_can_be_resampled_without_reordering(self) -> None:
        tree = primitive_tree(max_length=1.0)
        growth = reverse_to_growth(tree, build_pruning_trajectory(tree, seed=42))
        first = resample_event_times(growth, seed=1)
        second = resample_event_times(growth, seed=2)
        self.assertEqual(
            [event.event_type for event in first.events],
            [event.event_type for event in second.events],
        )
        self.assertNotEqual(
            [event.event_time for event in first.events],
            [event.event_time for event in second.events],
        )
        self.assertTrue(
            all(
                left.event_time < right.event_time
                for left, right in zip(first.events, first.events[1:])
            )
        )
        births = branch_birth_times(first)
        for primitive in tree.primitives:
            if primitive.parent_id is not None:
                self.assertLess(births[primitive.parent_id], births[primitive.primitive_id])


if __name__ == "__main__":
    unittest.main()
