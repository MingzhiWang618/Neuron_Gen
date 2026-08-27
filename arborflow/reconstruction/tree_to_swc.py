"""Export a variable-size Bézier tree to a standard SWC morphology."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from arborflow.data.bezier_fitting import evaluate_cubic
from arborflow.data.swc_io import SwcMorphology, SwcNode
from arborflow.structures.branch import BezierPrimitive
from arborflow.structures.embedded_tree import BezierTree


def _approximate_length(primitive: BezierPrimitive, samples: int = 32) -> float:
    points = evaluate_cubic(
        primitive.control_points, np.linspace(0.0, 1.0, samples + 1)
    )
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def bezier_tree_to_swc(
    tree: BezierTree,
    *,
    sample_spacing_um: float = 2.0,
    minimum_edges_per_primitive: int = 2,
) -> SwcMorphology:
    """Sample real primitives and remove virtual branches during SWC export."""

    if sample_spacing_um <= 0:
        raise ValueError("sample_spacing_um must be positive")
    if minimum_edges_per_primitive < 1:
        raise ValueError("minimum_edges_per_primitive must be at least one")
    primitives = tree.by_id()
    primitive_ids = set(primitives)
    roots = set(tree.root_primitive_ids)
    if roots != {item.primitive_id for item in tree.primitives if item.parent_id is None}:
        raise ValueError("root primitive IDs do not match parent pointers")
    for primitive in tree.primitives:
        if primitive.parent_id is not None and primitive.parent_id not in primitive_ids:
            raise ValueError(f"primitive {primitive.primitive_id} has a missing parent")
        if not set(primitive.children_ids) <= primitive_ids:
            raise ValueError(f"primitive {primitive.primitive_id} has a missing child")
        for child_id in primitive.children_ids:
            if primitives[child_id].parent_id != primitive.primitive_id:
                raise ValueError("primitive parent/child pointers disagree")

    root = SwcNode(
        node_id=1,
        swc_type=tree.root.swc_type,
        x=tree.root.x,
        y=tree.root.y,
        z=tree.root.z,
        radius=tree.root.radius,
        parent_id=-1,
    )
    output = [root]
    endpoint_node_ids: dict[int, int] = {}
    visited: set[int] = set()
    queue = deque(tree.root_primitive_ids)
    while queue:
        primitive_id = queue.popleft()
        if primitive_id in visited:
            raise ValueError(f"cycle or repeated primitive ownership at {primitive_id}")
        visited.add(primitive_id)
        primitive = primitives[primitive_id]
        parent_swc_id = (
            root.node_id
            if primitive.parent_id is None
            else endpoint_node_ids[primitive.parent_id]
        )
        if primitive.virtual:
            endpoint_node_ids[primitive_id] = parent_swc_id
        else:
            length = _approximate_length(primitive)
            edge_count = max(
                minimum_edges_per_primitive,
                int(math.ceil(length / sample_spacing_um)),
            )
            parameters = np.linspace(0.0, 1.0, edge_count + 1)[1:]
            points = evaluate_cubic(primitive.control_points, parameters)
            radii = (
                (1.0 - parameters) * primitive.radius_start
                + parameters * primitive.radius_end
            )
            for point, radius in zip(points, radii):
                node_id = len(output) + 1
                output.append(
                    SwcNode(
                        node_id=node_id,
                        swc_type=primitive.swc_type,
                        x=float(point[0]),
                        y=float(point[1]),
                        z=float(point[2]),
                        radius=float(radius),
                        parent_id=parent_swc_id,
                    )
                )
                parent_swc_id = node_id
            endpoint_node_ids[primitive_id] = parent_swc_id
        queue.extend(primitive.children_ids)
    missing = primitive_ids - visited
    if missing:
        raise ValueError(f"unreachable primitives: {sorted(missing)}")
    comments = tuple(tree.comments) + (
        "Reconstructed by Birth-Death ArborFlow from cubic Bezier primitives",
    )
    return SwcMorphology(tuple(output), source=tree.source, comments=comments)

