"""Rotation-safe scale normalization for parent-relative branch geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from arborflow.structures.embedded_tree import BezierTree


@dataclass(frozen=True)
class GeometryNormalizer:
    """Scalar coordinate/radius scales that preserve SO(3) structure."""

    coordinate_scale_um: float
    radius_scale_um: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.coordinate_scale_um) or self.coordinate_scale_um <= 0.0:
            raise ValueError("coordinate_scale_um must be finite and positive")
        if not np.isfinite(self.radius_scale_um) or self.radius_scale_um <= 0.0:
            raise ValueError("radius_scale_um must be finite and positive")

    @classmethod
    def from_trees(cls, trees: list[BezierTree] | tuple[BezierTree, ...]) -> GeometryNormalizer:
        primitives = [primitive for tree in trees for primitive in tree.primitives]
        if not primitives:
            raise ValueError("normalization requires at least one primitive")
        coordinate_values = np.concatenate(
            [primitive.control_offsets.reshape(-1) for primitive in primitives]
        )
        radius_values = np.asarray(
            [
                radius
                for primitive in primitives
                for radius in (primitive.radius_start, primitive.radius_end)
            ],
            dtype=np.float64,
        )
        coordinate_scale = float(np.sqrt(np.mean(coordinate_values**2)))
        radius_scale = float(np.sqrt(np.mean(radius_values**2)))
        return cls(max(coordinate_scale, 1e-6), max(radius_scale, 1e-6))

    def normalize_geometry(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64).copy()
        if array.shape[-1] != 11:
            raise ValueError("geometry vectors must end in 11 values")
        array[..., :9] /= self.coordinate_scale_um
        array[..., 9:] /= self.radius_scale_um
        return array

    def denormalize_geometry(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64).copy()
        if array.shape[-1] != 11:
            raise ValueError("geometry vectors must end in 11 values")
        array[..., :9] *= self.coordinate_scale_um
        array[..., 9:] *= self.radius_scale_um
        return array

    def to_dict(self) -> dict[str, float]:
        return {
            "coordinate_scale_um": self.coordinate_scale_um,
            "radius_scale_um": self.radius_scale_um,
        }
