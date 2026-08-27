"""Reversible decomposition of a clean SWC tree into maximal branches."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from arborflow.data.swc_io import SwcMorphology, SwcNode
from arborflow.data.swc_validation import SwcValidationConfig, validate_swc
from arborflow.structures.branch import Branch
from arborflow.structures.embedded_tree import EmbeddedTree
from arborflow.structures.tree_invariants import assert_valid_embedded_tree


class BranchDecompositionError(ValueError):
    """Raised when an SWC cannot be safely represented as branches."""


_DECOMPOSITION_CONFIG = SwcValidationConfig(
    require_3d=False,
    require_single_root=True,
    require_connected=True,
    min_nodes=1,
    min_real_branches=0,
    max_real_branches=2**31 - 1,
)


def decompose_swc(morphology: SwcMorphology) -> EmbeddedTree:
    """Convert a validated SWC into paths between topological key nodes.

    Key nodes are the root, terminations, and any sample with out-degree other than
    one. Branch IDs are assigned deterministically by a pre-order traversal whose
    siblings are ordered by their source node IDs.
    """

    validation = validate_swc(morphology, _DECOMPOSITION_CONFIG)
    errors = list(validation.errors)
    zero_length = [
        issue for issue in validation.issues if issue.code == "zero_length_edge"
    ]
    if errors or zero_length:
        messages = [issue.message for issue in (*errors, *zero_length)]
        raise BranchDecompositionError(
            "SWC must be cleaned before branch decomposition: " + "; ".join(messages)
        )

    by_id = morphology.by_id()
    root_id = validation.root_ids[0]
    children: dict[int, list[int]] = {node_id: [] for node_id in by_id}
    for node in morphology:
        if node.parent_id != -1:
            children[node.parent_id].append(node.node_id)
    for node_children in children.values():
        node_children.sort()

    branches: dict[int, Branch] = {}
    branch_children: dict[int, list[int]] = defaultdict(list)
    root_branch_ids: list[int] = []
    next_branch_id = 0
    pending = [
        (root_id, child_id, None, 0) for child_id in reversed(children[root_id])
    ]
    while pending:
        start_node_id, first_child_id, parent_id, depth = pending.pop()
        branch_id = next_branch_id
        next_branch_id += 1
        path_ids = [start_node_id, first_child_id]
        current = first_child_id
        while len(children[current]) == 1:
            current = children[current][0]
            path_ids.append(current)
        path_nodes = [by_id[node_id] for node_id in path_ids]
        branches[branch_id] = Branch(
            branch_id=branch_id,
            parent_id=parent_id,
            children_ids=(),
            points=np.asarray([node.position for node in path_nodes], dtype=np.float64),
            radii=np.asarray([node.radius for node in path_nodes], dtype=np.float64),
            swc_type=path_nodes[1].swc_type,
            depth=depth,
            virtual=False,
            continuation=False,
            source_node_ids=tuple(path_ids),
            point_swc_types=np.asarray(
                [node.swc_type for node in path_nodes], dtype=np.int64
            ),
        )
        if parent_id is None:
            root_branch_ids.append(branch_id)
        else:
            branch_children[parent_id].append(branch_id)
        pending.extend(
            (current, child_id, branch_id, depth + 1)
            for child_id in reversed(children[current])
        )
    branches = {
        branch_id: replace(branch, children_ids=tuple(branch_children[branch_id]))
        for branch_id, branch in branches.items()
    }
    tree = EmbeddedTree(
        root=by_id[root_id],
        branches=tuple(branches[index] for index in sorted(branches)),
        root_branch_ids=tuple(root_branch_ids),
        source_node_order=tuple(node.node_id for node in morphology),
        source=morphology.source,
        comments=morphology.comments,
    )
    assert_valid_embedded_tree(tree)
    return tree


def branch_tree_to_swc_exact(tree: EmbeddedTree) -> SwcMorphology:
    """Invert branch decomposition exactly using retained SWC provenance.

    Virtual normalization branches are skipped. This function is for lossless data
    round-trips; generated Bézier trees use ``bezier_tree_to_swc`` instead.
    """

    assert_valid_embedded_tree(tree)
    recovered: dict[int, SwcNode] = {tree.root.node_id: tree.root}
    for branch in tree.branches:
        if branch.virtual:
            continue
        if len(branch.source_node_ids) != len(branch.points):
            raise BranchDecompositionError(
                f"branch {branch.branch_id} lacks exact SWC provenance"
            )
        point_types = branch.point_swc_types
        assert point_types is not None
        for index, source_id in enumerate(branch.source_node_ids):
            if index == 0:
                if source_id not in recovered:
                    raise BranchDecompositionError(
                        f"branch {branch.branch_id} starts at unknown SWC node {source_id}"
                    )
                existing = recovered[source_id]
                if not np.allclose(existing.position, branch.points[index], atol=1e-12):
                    raise BranchDecompositionError(
                        f"conflicting geometry for shared SWC node {source_id}"
                    )
                continue
            candidate = SwcNode(
                node_id=source_id,
                swc_type=int(point_types[index]),
                x=float(branch.points[index, 0]),
                y=float(branch.points[index, 1]),
                z=float(branch.points[index, 2]),
                radius=float(branch.radii[index]),
                parent_id=branch.source_node_ids[index - 1],
            )
            existing = recovered.get(source_id)
            if existing is not None and existing != candidate:
                raise BranchDecompositionError(
                    f"conflicting provenance for shared SWC node {source_id}"
                )
            recovered[source_id] = candidate

    missing = set(tree.source_node_order) - set(recovered)
    extra = set(recovered) - set(tree.source_node_order)
    if missing or extra:
        raise BranchDecompositionError(
            f"SWC provenance mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return SwcMorphology(
        tuple(recovered[node_id] for node_id in tree.source_node_order),
        source=tree.source,
        comments=tree.comments,
    )
