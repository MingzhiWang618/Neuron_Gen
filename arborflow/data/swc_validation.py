"""Validation and conservative, auditable cleaning for rooted SWC trees."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from itertools import product
from typing import Iterable

from arborflow.data.swc_io import SwcMorphology, SwcNode


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    node_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["severity"] = self.severity.value
        result["node_ids"] = list(self.node_ids)
        return result


@dataclass(frozen=True)
class SwcValidationConfig:
    require_3d: bool = True
    require_single_root: bool = True
    require_connected: bool = True
    min_nodes: int = 30
    min_real_branches: int = 5
    max_real_branches: int = 1000
    coordinate_tolerance_um: float = 1e-6

    def __post_init__(self) -> None:
        if self.min_nodes < 1:
            raise ValueError("min_nodes must be positive")
        if self.min_real_branches < 0:
            raise ValueError("min_real_branches cannot be negative")
        if self.max_real_branches < self.min_real_branches:
            raise ValueError("max_real_branches must be >= min_real_branches")
        if self.coordinate_tolerance_um < 0:
            raise ValueError("coordinate_tolerance_um cannot be negative")


@dataclass
class ValidationReport:
    source: str | None
    node_count: int
    root_ids: list[int] = field(default_factory=list)
    real_branch_count: int | None = None
    coordinate_rank: int | None = None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        node_ids: Iterable[int] = (),
    ) -> None:
        self.issues.append(ValidationIssue(code, severity, message, tuple(node_ids)))

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "valid": self.valid,
            "node_count": self.node_count,
            "root_ids": self.root_ids,
            "real_branch_count": self.real_branch_count,
            "coordinate_rank": self.coordinate_rank,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class CleaningAction:
    action: str
    node_id: int
    details: str


@dataclass
class CleaningResult:
    morphology: SwcMorphology | None
    before: ValidationReport
    after: ValidationReport | None
    actions: list[CleaningAction]

    @property
    def success(self) -> bool:
        return self.morphology is not None and self.after is not None and self.after.valid

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "before": self.before.to_dict(),
            "after": None if self.after is None else self.after.to_dict(),
            "actions": [asdict(action) for action in self.actions],
        }


def _distance_squared(left: SwcNode, right: SwcNode) -> float:
    return sum((a - b) ** 2 for a, b in zip(left.position, right.position))


def _duplicate_coordinate_components(
    nodes: tuple[SwcNode, ...], tolerance: float
) -> list[list[int]]:
    """Find coordinate clusters using an exact tolerance check and spatial hashing."""

    finite_nodes = [
        node for node in nodes if all(math.isfinite(value) for value in node.position)
    ]
    if tolerance == 0:
        exact: dict[tuple[float, float, float], list[int]] = defaultdict(list)
        for node in finite_nodes:
            exact[node.position].append(node.node_id)
        return [group for group in exact.values() if len(group) > 1]

    parents = list(range(len(finite_nodes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    threshold_sq = tolerance * tolerance
    neighbor_offsets = tuple(product((-1, 0, 1), repeat=3))
    for index, node in enumerate(finite_nodes):
        bucket = tuple(math.floor(value / tolerance) for value in node.position)
        for offset in neighbor_offsets:
            neighbor = tuple(value + delta for value, delta in zip(bucket, offset))
            for other_index in buckets.get(neighbor, ()):
                if _distance_squared(node, finite_nodes[other_index]) <= threshold_sq:
                    union(index, other_index)
        buckets[bucket].append(index)

    components: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(finite_nodes):
        components[find(index)].append(node.node_id)
    return [group for group in components.values() if len(group) > 1]


def _coordinate_rank(nodes: tuple[SwcNode, ...], tolerance: float) -> int:
    """Compute affine rank without requiring NumPy in the validation core."""

    if len(nodes) < 2:
        return 0
    origin = nodes[0].position
    vectors = [
        [node.position[axis] - origin[axis] for axis in range(3)] for node in nodes[1:]
    ]
    basis: list[list[float]] = []
    threshold_sq = tolerance * tolerance
    for vector in vectors:
        residual = list(vector)
        for unit in basis:
            projection = sum(a * b for a, b in zip(residual, unit))
            residual = [a - projection * b for a, b in zip(residual, unit)]
        norm_sq = sum(value * value for value in residual)
        if norm_sq > threshold_sq:
            norm = math.sqrt(norm_sq)
            basis.append([value / norm for value in residual])
        if len(basis) == 3:
            break
    return len(basis)


def _find_cycle(node_ids: set[int], parent_by_id: dict[int, int]) -> list[int] | None:
    done: set[int] = set()
    for start in sorted(node_ids):
        if start in done:
            continue
        path: list[int] = []
        positions: dict[int, int] = {}
        current = start
        while current in node_ids and current not in done:
            if current in positions:
                return path[positions[current] :] + [current]
            positions[current] = len(path)
            path.append(current)
            parent = parent_by_id[current]
            if parent == -1:
                break
            current = parent
        done.update(path)
    return None


def _real_branch_count(root_id: int, children: dict[int, list[int]]) -> int:
    """Count maximal paths between root/bifurcation/termination key nodes."""

    count = 0
    stack = [root_id]
    seen: set[int] = set()
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node_children = children.get(node_id, [])
        if node_id == root_id or len(node_children) != 1:
            count += len(node_children)
        stack.extend(node_children)
    return count


def validate_swc(
    morphology: SwcMorphology,
    config: SwcValidationConfig | None = None,
) -> ValidationReport:
    """Validate topology, geometry, and dataset admission constraints."""

    config = config or SwcValidationConfig()
    report = ValidationReport(morphology.source, len(morphology))
    nodes = morphology.nodes
    id_counts = Counter(node.node_id for node in nodes)
    duplicate_ids = sorted(node_id for node_id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        report.add(
            "duplicate_node_id",
            Severity.ERROR,
            f"node IDs must be unique; duplicates: {duplicate_ids}",
            duplicate_ids,
        )
    nonpositive_ids = sorted(node.node_id for node in nodes if node.node_id <= 0)
    if nonpositive_ids:
        report.add(
            "nonpositive_node_id",
            Severity.ERROR,
            "node IDs must be positive integers",
            nonpositive_ids,
        )
    report.root_ids = [node.node_id for node in nodes if node.parent_id == -1]
    if config.require_single_root and len(report.root_ids) != 1:
        report.add(
            "root_count",
            Severity.ERROR,
            f"expected exactly one root (parent -1), found {len(report.root_ids)}",
            report.root_ids,
        )
    finite_failures = [
        node.node_id
        for node in nodes
        if not all(math.isfinite(value) for value in (*node.position, node.radius))
    ]
    if finite_failures:
        report.add(
            "nonfinite_geometry",
            Severity.ERROR,
            "coordinates and radii must be finite",
            finite_failures,
        )
    bad_radii = [
        node.node_id
        for node in nodes
        if node.radius <= 0 or not math.isfinite(node.radius)
    ]
    if bad_radii:
        report.add(
            "nonpositive_radius",
            Severity.ERROR,
            "radii must be finite and strictly positive",
            bad_radii,
        )
    node_ids = set(id_counts)
    invalid_parents = sorted(
        node.node_id
        for node in nodes
        if node.parent_id != -1 and node.parent_id not in node_ids
    )
    if invalid_parents:
        report.add(
            "invalid_parent_id",
            Severity.ERROR,
            "one or more parent IDs do not reference an existing node",
            invalid_parents,
        )
    self_parents = sorted(node.node_id for node in nodes if node.parent_id == node.node_id)
    if self_parents:
        report.add(
            "self_parent",
            Severity.ERROR,
            "a node cannot be its own parent",
            self_parents,
        )
    topology_indexable = not duplicate_ids and not invalid_parents and not self_parents
    by_id: dict[int, SwcNode] = {}
    children: dict[int, list[int]] = defaultdict(list)
    if topology_indexable:
        by_id = {node.node_id: node for node in nodes}
        for node in nodes:
            if node.parent_id != -1:
                children[node.parent_id].append(node.node_id)
        parent_by_id = {node.node_id: node.parent_id for node in nodes}
        cycle = _find_cycle(node_ids, parent_by_id)
        if cycle:
            report.add("cycle", Severity.ERROR, "parent pointers contain a cycle", cycle)
        if config.require_connected and len(report.root_ids) == 1 and not cycle:
            visited: set[int] = set()
            queue = deque(report.root_ids)
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                queue.extend(children.get(current, ()))
            disconnected = sorted(node_ids - visited)
            if disconnected:
                report.add(
                    "disconnected",
                    Severity.ERROR,
                    "all nodes must be reachable from the unique root",
                    disconnected,
                )
        tolerance_sq = config.coordinate_tolerance_um**2
        zero_edges = sorted(
            node.node_id
            for node in nodes
            if node.parent_id != -1
            and _distance_squared(node, by_id[node.parent_id]) <= tolerance_sq
        )
        if zero_edges:
            report.add(
                "zero_length_edge",
                Severity.WARNING,
                "parent-child samples share a coordinate and can be conservatively collapsed",
                zero_edges,
            )
        ambiguous_groups: list[list[int]] = []
        zero_edge_set = set(zero_edges)
        for group in _duplicate_coordinate_components(
            nodes, config.coordinate_tolerance_um
        ):
            group_set = set(group)
            removable = {
                node_id
                for node_id in group
                if node_id in zero_edge_set and by_id[node_id].parent_id in group_set
            }
            if len(group_set - removable) > 1:
                ambiguous_groups.append(sorted(group))
        if ambiguous_groups:
            involved = sorted({item for group in ambiguous_groups for item in group})
            report.add(
                "ambiguous_duplicate_coordinate",
                Severity.ERROR,
                "non-parent duplicate coordinates cannot be merged without changing topology",
                involved,
            )
        if len(report.root_ids) == 1 and not cycle:
            report.real_branch_count = _real_branch_count(report.root_ids[0], children)
    report.coordinate_rank = _coordinate_rank(nodes, config.coordinate_tolerance_um)
    if config.require_3d and report.coordinate_rank < 3:
        report.add(
            "not_3d",
            Severity.ERROR,
            f"coordinate affine rank is {report.coordinate_rank}, expected 3",
        )
    if len(nodes) < config.min_nodes:
        report.add(
            "too_few_nodes",
            Severity.ERROR,
            f"found {len(nodes)} nodes; minimum is {config.min_nodes}",
        )
    if report.real_branch_count is not None:
        if report.real_branch_count < config.min_real_branches:
            report.add(
                "too_few_real_branches",
                Severity.ERROR,
                f"found {report.real_branch_count} real branches; minimum is "
                f"{config.min_real_branches}",
            )
        if report.real_branch_count > config.max_real_branches:
            report.add(
                "too_many_real_branches",
                Severity.ERROR,
                f"found {report.real_branch_count} real branches; maximum is "
                f"{config.max_real_branches}",
            )
    return report


def _without_admission_filters(config: SwcValidationConfig) -> SwcValidationConfig:
    return replace(
        config,
        require_3d=False,
        min_nodes=1,
        min_real_branches=0,
        max_real_branches=max(config.max_real_branches, 2**31 - 1),
    )


def clean_swc(
    morphology: SwcMorphology,
    config: SwcValidationConfig | None = None,
    *,
    repair_zero_length_edges: bool = True,
) -> CleaningResult:
    """Conservatively clean geometry, preserving a complete audit trail.

    Structural errors and ambiguous duplicate coordinates are fatal. The only automatic
    mutation is collapsing a zero-length child into its parent; the child's descendants
    are reparented to preserve connectivity. Dataset admission filters are evaluated only
    after cleaning so a repair cannot accidentally bypass them.
    """

    config = config or SwcValidationConfig()
    structural_config = _without_admission_filters(config)
    before = validate_swc(morphology, structural_config)
    fatal_before = [
        issue
        for issue in before.errors
        if issue.code != "ambiguous_duplicate_coordinate"
    ]
    if fatal_before or any(
        issue.code == "ambiguous_duplicate_coordinate" for issue in before.errors
    ):
        return CleaningResult(None, before, None, [])
    if not repair_zero_length_edges and any(
        issue.code == "zero_length_edge" for issue in before.issues
    ):
        before.add(
            "zero_length_repair_disabled",
            Severity.ERROR,
            "zero-length edges were found but repair is disabled",
        )
        return CleaningResult(None, before, None, [])

    mutable = {node.node_id: node for node in morphology.nodes}
    order = [node.node_id for node in morphology.nodes]
    actions: list[CleaningAction] = []
    tolerance_sq = config.coordinate_tolerance_um**2
    changed = True
    while changed:
        changed = False
        for node_id in list(order):
            node = mutable.get(node_id)
            if node is None or node.parent_id == -1:
                continue
            parent = mutable[node.parent_id]
            if _distance_squared(node, parent) > tolerance_sq:
                continue
            for child_id, child in list(mutable.items()):
                if child.parent_id == node_id:
                    mutable[child_id] = replace(child, parent_id=parent.node_id)
            del mutable[node_id]
            order.remove(node_id)
            actions.append(
                CleaningAction(
                    "collapse_zero_length_edge",
                    node_id,
                    f"removed node {node_id}; reparented its children to {parent.node_id}",
                )
            )
            changed = True
            break

    cleaned = SwcMorphology(
        tuple(mutable[node_id] for node_id in order),
        source=morphology.source,
        comments=morphology.comments,
    )
    after = validate_swc(cleaned, config)
    if not after.valid:
        return CleaningResult(None, before, after, actions)
    return CleaningResult(cleaned, before, after, actions)
