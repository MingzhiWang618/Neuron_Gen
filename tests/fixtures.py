from __future__ import annotations

from arborflow.data.swc_io import SwcMorphology, SwcNode


def branching_morphology() -> SwcMorphology:
    """One internal trifurcation plus curved, typed sample paths."""

    rows = (
        # id, type, x, y, z, radius, parent
        (1, 1, 0.0, 0.0, 0.0, 3.0, -1),
        (2, 3, 1.0, 0.0, 0.1, 1.2, 1),
        (3, 3, 2.0, 0.0, 0.2, 1.1, 2),
        (4, 3, 3.0, 1.0, 0.5, 0.9, 3),
        (5, 3, 4.0, 2.0, 1.0, 0.7, 4),
        (6, 4, 3.0, -1.0, 0.4, 0.9, 3),
        (7, 4, 4.0, -2.0, 0.8, 0.7, 6),
        (8, 2, 3.0, 0.0, -1.0, 0.8, 3),
        (9, 2, 4.0, 0.0, -2.0, 0.6, 8),
    )
    return SwcMorphology(
        tuple(SwcNode(*row) for row in rows),
        source="branching.swc",
        comments=("test fixture",),
    )


def root_multifurcation() -> SwcMorphology:
    rows = [(1, 1, 0.0, 0.0, 0.0, 3.0, -1)]
    for index, point in enumerate(
        ((2.0, 0.0, 0.0), (0.0, 2.0, 0.2), (-2.0, 0.0, 0.4), (0.0, -2.0, 0.6)),
        start=2,
    ):
        rows.append((index, 3, *point, 1.0, 1))
    return SwcMorphology(tuple(SwcNode(*row) for row in rows), source="root_multi.swc")


def balanced_morphology() -> SwcMorphology:
    rows = (
        (1, 1, 0.0, 0.0, 0.0, 3.0, -1),
        (2, 3, 1.0, 0.0, 0.1, 1.2, 1),
        (3, 3, 2.0, 1.0, 0.2, 1.0, 2),
        (4, 3, 2.0, -1.0, 0.3, 1.0, 2),
        (5, 3, 3.0, 1.5, 0.5, 0.7, 3),
        (6, 3, 3.0, 0.5, 0.6, 0.7, 3),
        (7, 3, 3.0, -0.5, 0.7, 0.7, 4),
        (8, 3, 3.0, -1.5, 0.8, 0.7, 4),
    )
    return SwcMorphology(
        tuple(SwcNode(*row) for row in rows), source="balanced.swc"
    )
