"""Structural and geometric invariants for embedded branch trees."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from arborflow.data.swc_io import SwcMorphology
from arborflow.structures.embedded_tree import BezierTree, EmbeddedTree


@dataclass
class TreeInvariantReport:
    errors: list[str] = field(default_factory=list)
    max_continuity_error: float = 0.0

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_embedded_tree(
    tree: EmbeddedTree,
    *,
    require_binary: bool = False,
    continuity_tolerance: float = 1e-8,
) -> TreeInvariantReport:
    report = TreeInvariantReport()
    branches = tree.by_id()
    branch_ids = set(branches)
    roots = set(tree.root_branch_ids)
    if len(roots) != len(tree.root_branch_ids):
        report.errors.append("root_branch_ids contains duplicates")
    if not roots <= branch_ids:
        report.errors.append("root_branch_ids references missing branches")
    declared_roots = {branch.branch_id for branch in tree.branches if branch.parent_id is None}
    if roots != declared_roots:
        report.errors.append("root_branch_ids does not match branches with parent_id=None")
    if require_binary and len(roots) > 2:
        report.errors.append("binary tree has more than two root branches")

    for branch in tree.branches:
        if branch.parent_id is not None and branch.parent_id not in branch_ids:
            report.errors.append(f"branch {branch.branch_id} has missing parent {branch.parent_id}")
        if len(set(branch.children_ids)) != len(branch.children_ids):
            report.errors.append(f"branch {branch.branch_id} has duplicate children")
        if not set(branch.children_ids) <= branch_ids:
            report.errors.append(f"branch {branch.branch_id} references a missing child")
        if require_binary and len(branch.children_ids) > 2:
            report.errors.append(f"branch {branch.branch_id} has more than two children")
        if not branch.virtual and branch.length <= continuity_tolerance:
            report.errors.append(f"real branch {branch.branch_id} has zero length")
        if branch.parent_id is None or branch.parent_id in branch_ids:
            expected_start = (
                np.asarray(tree.root.position, dtype=np.float64)
                if branch.parent_id is None
                else branches[branch.parent_id].end
            )
            continuity_error = float(np.linalg.norm(branch.start - expected_start))
            report.max_continuity_error = max(
                report.max_continuity_error, continuity_error
            )
            if continuity_error > continuity_tolerance:
                report.errors.append(
                    f"branch {branch.branch_id} is discontinuous from its parent by "
                    f"{continuity_error:.6g}"
                )
        for child_id in branch.children_ids:
            if child_id in branches and branches[child_id].parent_id != branch.branch_id:
                report.errors.append(
                    f"branch {branch.branch_id} and child {child_id} disagree on parentage"
                )

    visited: set[int] = set()
    queue = deque(tree.root_branch_ids)
    while queue:
        branch_id = queue.popleft()
        if branch_id in visited:
            report.errors.append(f"cycle or repeated ownership at branch {branch_id}")
            continue
        visited.add(branch_id)
        if branch_id in branches:
            queue.extend(branches[branch_id].children_ids)
    missing = branch_ids - visited
    if missing:
        report.errors.append(f"unreachable branches: {sorted(missing)}")
    return report


def assert_valid_embedded_tree(
    tree: EmbeddedTree,
    *,
    require_binary: bool = False,
    continuity_tolerance: float = 1e-8,
) -> None:
    report = validate_embedded_tree(
        tree,
        require_binary=require_binary,
        continuity_tolerance=continuity_tolerance,
    )
    if not report.valid:
        raise ValueError("invalid embedded tree: " + "; ".join(report.errors))


def validate_bezier_tree(
    tree: BezierTree,
    *,
    require_binary: bool = False,
    continuity_tolerance: float = 1e-8,
) -> TreeInvariantReport:
    report = TreeInvariantReport()
    primitives = tree.by_id()
    primitive_ids = set(primitives)
    roots = set(tree.root_primitive_ids)
    declared_roots = {
        primitive.primitive_id for primitive in tree.primitives if primitive.parent_id is None
    }
    if roots != declared_roots:
        report.errors.append("root_primitive_ids does not match parent pointers")
    if require_binary and len(roots) > 2:
        report.errors.append("binary tree has more than two root primitives")
    for primitive in tree.primitives:
        if primitive.parent_id is not None and primitive.parent_id not in primitive_ids:
            report.errors.append(
                f"primitive {primitive.primitive_id} has missing parent {primitive.parent_id}"
            )
        if not set(primitive.children_ids) <= primitive_ids:
            report.errors.append(f"primitive {primitive.primitive_id} has a missing child")
        if require_binary and len(primitive.children_ids) > 2:
            report.errors.append(
                f"primitive {primitive.primitive_id} has more than two children"
            )
        if primitive.parent_id is None or primitive.parent_id in primitive_ids:
            expected_start = (
                np.asarray(tree.root.position, dtype=np.float64)
                if primitive.parent_id is None
                else primitives[primitive.parent_id].end
            )
            error = float(np.linalg.norm(primitive.start - expected_start))
            report.max_continuity_error = max(report.max_continuity_error, error)
            if error > continuity_tolerance:
                report.errors.append(
                    f"primitive {primitive.primitive_id} is discontinuous by {error:.6g}"
                )
        for child_id in primitive.children_ids:
            if (
                child_id in primitives
                and primitives[child_id].parent_id != primitive.primitive_id
            ):
                report.errors.append(
                    f"primitive {primitive.primitive_id} and child {child_id} disagree"
                )
    visited: set[int] = set()
    queue = deque(tree.root_primitive_ids)
    while queue:
        primitive_id = queue.popleft()
        if primitive_id in visited:
            report.errors.append(f"cycle or repeated ownership at primitive {primitive_id}")
            continue
        visited.add(primitive_id)
        if primitive_id in primitives:
            queue.extend(primitives[primitive_id].children_ids)
    missing = primitive_ids - visited
    if missing:
        report.errors.append(f"unreachable primitives: {sorted(missing)}")
    return report


def critical_topology_signature(morphology: SwcMorphology) -> str:
    """Canonical topology digest after suppressing degree-one sample nodes."""

    by_id = morphology.by_id()
    roots = [node.node_id for node in morphology if node.parent_id == -1]
    if len(roots) != 1:
        raise ValueError("topology signature requires exactly one root")
    children: dict[int, list[int]] = {node_id: [] for node_id in by_id}
    for node in morphology:
        if node.parent_id != -1:
            if node.parent_id not in by_id:
                raise ValueError(f"missing parent {node.parent_id}")
            children[node.parent_id].append(node.node_id)

    contracted: dict[int, list[int]] = {}
    critical_nodes = {roots[0]} | {
        node_id for node_id, node_children in children.items() if len(node_children) != 1
    }
    for node_id in critical_nodes:
        endpoints: list[int] = []
        for child_id in children[node_id]:
            current = child_id
            chain_seen = {node_id}
            while current not in critical_nodes:
                if current in chain_seen:
                    raise ValueError("cycle in morphology")
                chain_seen.add(current)
                current = children[current][0]
            endpoints.append(current)
        contracted[node_id] = endpoints

    root_id = roots[0]
    state: dict[int, int] = {}
    order: list[int] = []
    stack = [(root_id, False)]
    while stack:
        node_id, expanded = stack.pop()
        if expanded:
            state[node_id] = 2
            order.append(node_id)
            continue
        if state.get(node_id) == 1:
            raise ValueError("cycle in morphology")
        if state.get(node_id) == 2:
            continue
        state[node_id] = 1
        stack.append((node_id, True))
        stack.extend((child_id, False) for child_id in contracted[node_id])
    if len(state) != len(critical_nodes):
        raise ValueError("morphology is disconnected")
    signatures: dict[int, str] = {}
    for node_id in order:
        children_payload = ",".join(
            sorted(signatures[child_id] for child_id in contracted[node_id])
        )
        signatures[node_id] = sha256(f"({children_payload})".encode()).hexdigest()
    return signatures[root_id]
