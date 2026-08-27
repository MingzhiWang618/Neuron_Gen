from __future__ import annotations

import unittest

import numpy as np

from arborflow.data.trajectory_builder import build_pruning_trajectory, reverse_to_growth
from arborflow.flow.geometry_path import OracleGeometryConfig
from arborflow.flow.oracle_replay import OracleReplay
from tests.test_pruning import primitive_tree


class OracleReplayTests(unittest.TestCase):
    def _replay(self, *, noise: float = 0.01) -> OracleReplay:
        tree = primitive_tree(max_length=1.0)
        growth = reverse_to_growth(tree, build_pruning_trajectory(tree, seed=17))
        return OracleReplay(
            tree,
            growth,
            geometry_seed=91,
            geometry_config=OracleGeometryConfig(noise),
        )

    def test_replay_starts_at_root_anchor_and_inserts_at_event_boundary(self) -> None:
        replay = self._replay(noise=0.0)
        initial = replay.state_at(0.0)
        self.assertFalse(initial.branches)
        first_event = replay.trajectory.events[0]
        before = replay.state_at(first_event.event_time - 1e-12)
        after = replay.state_at(first_event.event_time)
        self.assertFalse(before.branches)
        self.assertEqual(after.primitive_ids, first_event.new_branch_ids)
        self.assertTrue(all(branch.age == 0.0 for branch in after.branches))
        self.assertTrue(np.all(after.parent_index == -1))

    def test_all_intermediate_branches_are_continuous_and_indices_are_stable(self) -> None:
        replay = self._replay()
        previous_ids: tuple[int, ...] = ()
        for global_time in np.linspace(0.0, 1.0, 41):
            state = replay.state_at(float(global_time))
            self.assertEqual(state.primitive_ids[: len(previous_ids)], previous_ids)
            previous_ids = state.primitive_ids
            for branch in state.branches:
                expected = (
                    np.asarray(replay.tree.root.position)
                    if branch.parent_index == -1
                    else state.branches[branch.parent_index].end
                )
                np.testing.assert_allclose(branch.start, expected, atol=0.0)

    def test_nonzero_seed_noise_still_recovers_exact_target(self) -> None:
        replay = self._replay(noise=0.05)
        report = replay.replay()
        self.assertTrue(report.valid, report.errors)
        self.assertTrue(report.topology_exact)
        self.assertTrue(report.dynamic_indices_stable)
        self.assertEqual(report.final_primitive_count, len(replay.tree.primitives))
        self.assertLessEqual(report.max_final_control_error_um, 1e-12)
        self.assertLessEqual(report.max_final_radius_error_um, 1e-12)
        self.assertEqual(report.max_continuity_error_um, 0.0)
        self.assertFalse(np.any(report.final_state.active_leaf_mask))

    def test_materialized_final_tree_matches_all_target_parameters(self) -> None:
        replay = self._replay()
        final = replay.materialize_tree(replay.state_at(1.0))
        target = replay.tree.by_id()
        for primitive in final.primitives:
            expected = target[primitive.primitive_id]
            self.assertEqual(primitive.parent_id, expected.parent_id)
            self.assertEqual(primitive.children_ids, expected.children_ids)
            np.testing.assert_allclose(
                primitive.control_points, expected.control_points, atol=1e-12
            )


if __name__ == "__main__":
    unittest.main()
