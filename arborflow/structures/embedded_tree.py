"""Rooted embedded trees at polyline and Bézier resolutions."""

from __future__ import annotations

from dataclasses import dataclass

from arborflow.data.swc_io import SwcNode
from arborflow.structures.branch import BezierPrimitive, Branch


@dataclass(frozen=True)
class EmbeddedTree:
    """A single SWC root anchor plus a variable number of branch tokens."""

    root: SwcNode
    branches: tuple[Branch, ...]
    root_branch_ids: tuple[int, ...]
    source_node_order: tuple[int, ...]
    source: str | None = None
    comments: tuple[str, ...] = ()

    def by_id(self) -> dict[int, Branch]:
        result = {branch.branch_id: branch for branch in self.branches}
        if len(result) != len(self.branches):
            raise ValueError("duplicate branch IDs")
        return result


@dataclass(frozen=True)
class BezierTree:
    """An embedded tree represented by relative cubic Bézier primitives."""

    root: SwcNode
    primitives: tuple[BezierPrimitive, ...]
    root_primitive_ids: tuple[int, ...]
    branch_to_primitive_ids: tuple[tuple[int, tuple[int, ...]], ...]
    source: str | None = None
    comments: tuple[str, ...] = ()

    def by_id(self) -> dict[int, BezierPrimitive]:
        result = {primitive.primitive_id: primitive for primitive in self.primitives}
        if len(result) != len(self.primitives):
            raise ValueError("duplicate primitive IDs")
        return result

    def primitives_for_branch(self, branch_id: int) -> tuple[int, ...]:
        return dict(self.branch_to_primitive_ids)[branch_id]

