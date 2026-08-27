from __future__ import annotations

import unittest

import numpy as np

from arborflow.data.trajectory_builder import build_pruning_trajectory, reverse_to_growth
from arborflow.flow.geometry_path import (
    OracleGeometryConfig,
    branch_age,
    build_geometry_paths,
)
from tests.test_pruning import primitive_tree


class GeometryPathTests(unittest.TestCase):
    def test_branch_age_is_clipped_and_reaches_one(self) -> None:
        self.assertEqual(branch_age(0.1, 0.2), 0.0)
        self.assertEqual(branch_age(0.2, 0.2), 0.0)
        self.assertAlmostEqual(branch_age(0.6, 0.2), 0.5)
        self.assertEqual(branch_age(1.0, 0.2), 1.0)

    def test_zero_noise_birth_is_zero_length_and_final_geometry_is_exact(self) -> None:
        tree = primitive_tree(max_length=1.0)
        growth = reverse_to_growth(tree, build_pruning_trajectory(tree, seed=3))
        paths = build_geometry_paths(
            tree,
            growth,
            seed=11,
            config=OracleGeometryConfig(birth_noise_sigma_um=0.0),
        )
        for primitive_id, path in paths.items():
            birth_offsets, _, birth_age_value = path.interpolate(path.birth_time)
            final_offsets, final_radii, final_age_value = path.interpolate(1.0)
            self.assertTrue(np.array_equal(birth_offsets, np.zeros((3, 3))))
            self.assertEqual(birth_age_value, 0.0)
            self.assertEqual(final_age_value, 1.0)
            np.testing.assert_allclose(
                final_offsets, tree.by_id()[primitive_id].control_offsets, atol=0.0
            )
            np.testing.assert_allclose(
                final_radii,
                (
                    tree.by_id()[primitive_id].radius_start,
                    tree.by_id()[primitive_id].radius_end,
                ),
                atol=0.0,
            )

    def test_analytic_velocity_integrates_to_interpolation(self) -> None:
        tree = primitive_tree()
        growth = reverse_to_growth(tree, build_pruning_trajectory(tree, seed=4))
        path = next(iter(build_geometry_paths(tree, growth, seed=22).values()))
        global_time = path.birth_time + 0.37 * (1.0 - path.birth_time)
        offsets, radii, _ = path.interpolate(global_time)
        elapsed = global_time - path.birth_time
        np.testing.assert_allclose(
            offsets,
            path.seed_control_offsets + elapsed * path.control_velocity,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            radii,
            path.seed_radii + elapsed * path.radius_velocity,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
