from __future__ import annotations

import unittest

import numpy as np

from arborflow.data.normalization import GeometryNormalizer
from tests.test_pruning import primitive_tree


class GeometryNormalizationTests(unittest.TestCase):
    def test_geometry_roundtrip_uses_scalar_rotation_safe_scales(self) -> None:
        tree = primitive_tree(max_length=1.0)
        normalizer = GeometryNormalizer.from_trees([tree])
        primitive = tree.primitives[0]
        geometry = np.concatenate(
            (
                primitive.control_offsets.reshape(-1),
                (primitive.radius_start, primitive.radius_end),
            )
        )[None, :]
        normalized = normalizer.normalize_geometry(geometry)
        recovered = normalizer.denormalize_geometry(normalized)
        np.testing.assert_allclose(recovered, geometry, atol=1e-12)
        self.assertGreater(normalizer.coordinate_scale_um, 0.0)
        self.assertGreater(normalizer.radius_scale_um, 0.0)


if __name__ == "__main__":
    unittest.main()
