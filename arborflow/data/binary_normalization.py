"""Deterministic and reversible conversion of multifurcations to binary trees."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from arborflow.structures.branch import Branch
from arborflow.structures.embedded_tree import EmbeddedTree
from arborflow.structures.tree_invariants import assert_valid_embedded_tree


@dataclass(frozen=True)
class BinaryNormalizationMap:
    """Information needed to remove virtual branches without ambiguity."""

    virtual_branch_ids: tuple[int, ...]
    original_root_branch_ids: tuple[int, ...]
    original_parent_ids: tuple[tuple[int, int | None], ...]
    original_children_ids: tuple[tuple[int, tuple[int, ...]], ...]
    original_depths: tuple[tuple[int, int], ...]


def _stable_child_key(branch: Branch) -> tuple[float, ...]:
    """Stable direction/length ordering independent of SWC row order."""

    direction = branch.direction
    return (
        -round(float(direction[0]), 12),
        -round(float(direction[1]), 12),
        -round(float(direction[2]), 12),
        -round(branch.length, 12),
        float(branch.source_node_ids[-1]),
        float(branch.branch_id),
    )


def _virtual_branch(
    branch_id: int,
    parent_id: int | None,
    junction: np.ndarray,
    radius: float,
    swc_type: int,
) -> Branch:
    return Branch(
        branch_id=branch_id,
        parent_id=parent_id,
        children_ids=(),
        points=np.stack((junction, junction), axis=0),
        radii=np.asarray((radius, radius), dtype=np.float64),
        swc_type=swc_type,
        depth=0,
        virtual=True,
        continuation=True,
        source_node_ids=(),
        point_swc_types=np.asarray((swc_type, swc_type), dtype=np.int64),
    )


def normalize_binary(tree: EmbeddedTree) -> tuple[EmbeddedTree, BinaryNormalizationMap]:
    """Right-factor every fan-out into a deterministic binary continuation chain."""

    assert_valid_embedded_tree(tree)
    original = tree.by_id()
    mutable = dict(original)
    next_id = max(mutable, default=-1) + 1
    virtual_ids: list[int] = []

    def normalize_children(
        owner_id: int | None,
        child_ids: tuple[int, ...],
        junction: np.ndarray,
        radius: float,
        swc_type: int,
    ) -> tuple[int, ...]:
        nonlocal next_id
        ordered = tuple(
            sorted(child_ids, key=lambda child_id: _stable_child_key(mutable[child_id]))
        )
        if len(ordered) <= 2:
            for child_id in ordered:
                mutable[child_id] = replace(mutable[child_id], parent_id=owner_id)
            return ordered

        current_owner = owner_id
        top_children: tuple[int, ...] | None = None
        for index in range(len(ordered) - 2):
            virtual_id = next_id
            next_id += 1
            virtual_ids.append(virtual_id)
            virtual = _virtual_branch(
                virtual_id, current_owner, junction, radius, swc_type
            )
            left_child = ordered[index]
            mutable[left_child] = replace(mutable[left_child], parent_id=current_owner)
            mutable[virtual_id] = virtual
            pair = (left_child, virtual_id)
            if current_owner is None:
                top_children = pair
            else:
                mutable[current_owner] = replace(
                    mutable[current_owner], children_ids=pair
                )
            current_owner = virtual_id
        final_pair = ordered[-2:]
        for child_id in final_pair:
            mutable[child_id] = replace(mutable[child_id], parent_id=current_owner)
        mutable[current_owner] = replace(mutable[current_owner], children_ids=final_pair)
        if owner_id is not None:
            top_children = mutable[owner_id].children_ids
        assert top_children is not None
        return top_children

    root_branch_ids = normalize_children(
        None,
        tree.root_branch_ids,
        np.asarray(tree.root.position, dtype=np.float64),
        tree.root.radius,
        tree.root.swc_type,
    )
    for branch_id in sorted(original):
        branch = mutable[branch_id]
        normalized_children = normalize_children(
            branch_id,
            tuple(original[branch_id].children_ids),
            branch.end,
            float(branch.radii[-1]),
            branch.swc_type,
        )
        mutable[branch_id] = replace(branch, children_ids=normalized_children)

    depths: dict[int, int] = {}
    queue = deque((branch_id, 0) for branch_id in root_branch_ids)
    while queue:
        branch_id, depth = queue.popleft()
        if branch_id in depths:
            raise ValueError(f"normalization created repeated branch {branch_id}")
        depths[branch_id] = depth
        queue.extend((child_id, depth + 1) for child_id in mutable[branch_id].children_ids)
    normalized = EmbeddedTree(
        root=tree.root,
        branches=tuple(
            replace(mutable[branch_id], depth=depths[branch_id])
            for branch_id in sorted(mutable)
        ),
        root_branch_ids=root_branch_ids,
        source_node_order=tree.source_node_order,
        source=tree.source,
        comments=tree.comments,
    )
    assert_valid_embedded_tree(normalized, require_binary=True)
    mapping = BinaryNormalizationMap(
        virtual_branch_ids=tuple(virtual_ids),
        original_root_branch_ids=tree.root_branch_ids,
        original_parent_ids=tuple(
            (branch.branch_id, branch.parent_id) for branch in tree.branches
        ),
        original_children_ids=tuple(
            (branch.branch_id, branch.children_ids) for branch in tree.branches
        ),
        original_depths=tuple((branch.branch_id, branch.depth) for branch in tree.branches),
    )
    return normalized, mapping


def denormalize_binary(
    tree: EmbeddedTree, mapping: BinaryNormalizationMap
) -> EmbeddedTree:
    """Remove only the virtual branches introduced by ``normalize_binary``."""

    assert_valid_embedded_tree(tree, require_binary=True)
    virtual_ids = set(mapping.virtual_branch_ids)
    branches = tree.by_id()
    if not virtual_ids <= set(branches):
        raise ValueError("normalization map references missing virtual branches")
    if any(not branches[branch_id].virtual for branch_id in virtual_ids):
        raise ValueError("normalization map would remove a real branch")
    parent_ids = dict(mapping.original_parent_ids)
    children_ids = dict(mapping.original_children_ids)
    depths = dict(mapping.original_depths)
    real_ids = set(branches) - virtual_ids
    if real_ids != set(parent_ids) or real_ids != set(children_ids):
        raise ValueError("normalization map does not match the real branch set")
    restored = EmbeddedTree(
        root=tree.root,
        branches=tuple(
            replace(
                branches[branch_id],
                parent_id=parent_ids[branch_id],
                children_ids=children_ids[branch_id],
                depth=depths[branch_id],
            )
            for branch_id in sorted(real_ids)
        ),
        root_branch_ids=mapping.original_root_branch_ids,
        source_node_order=tree.source_node_order,
        source=tree.source,
        comments=tree.comments,
    )
    assert_valid_embedded_tree(restored)
    return restored
