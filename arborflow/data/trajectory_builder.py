"""Legal leaf-pruning trajectories and their exact growth reversal."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from arborflow.structures.branch import BezierPrimitive
from arborflow.structures.embedded_tree import BezierTree
from arborflow.structures.tree_events import (
    EventType,
    GrowthReplayState,
    GrowthTrajectory,
    PruneActionKind,
    PruningStep,
    PruningStrategy,
    PruningTrajectory,
    TreeEvent,
)
from arborflow.structures.tree_invariants import validate_bezier_tree


@dataclass(frozen=True)
class PruningConfig:
    trajectories_per_neuron: int = 8
    uniform_probability: float = 0.40
    deep_probability: float = 0.30
    short_probability: float = 0.20
    long_probability: float = 0.10

    def __post_init__(self) -> None:
        if self.trajectories_per_neuron < 1:
            raise ValueError("trajectories_per_neuron must be positive")
        probabilities = self.strategy_probabilities
        if any(value < 0 for value in probabilities.values()):
            raise ValueError("pruning strategy probabilities cannot be negative")
        if not np.isclose(sum(probabilities.values()), 1.0):
            raise ValueError("pruning strategy probabilities must sum to one")

    @property
    def strategy_probabilities(self) -> dict[PruningStrategy, float]:
        return {
            PruningStrategy.UNIFORM: self.uniform_probability,
            PruningStrategy.DEEP: self.deep_probability,
            PruningStrategy.SHORT: self.short_probability,
            PruningStrategy.LONG: self.long_probability,
        }


@dataclass(frozen=True)
class TrajectoryValidationReport:
    valid: bool
    errors: tuple[str, ...]
    final_state: GrowthReplayState | None = None


@dataclass(frozen=True)
class _PruneAction:
    kind: PruneActionKind
    parent_id: int | None
    removed_ids: tuple[int, ...]


def _control_polygon_length(primitive: BezierPrimitive) -> float:
    points = primitive.control_points
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _validate_partial_state(tree: BezierTree, present: set[int]) -> list[str]:
    primitives = tree.by_id()
    errors: list[str] = []
    if not present <= set(primitives):
        errors.append("partial state contains unknown primitives")
        return errors
    for primitive_id in present:
        primitive = primitives[primitive_id]
        if primitive.parent_id is not None and primitive.parent_id not in present:
            errors.append(f"orphan primitive {primitive_id}")
        present_children = tuple(
            child_id for child_id in primitive.children_ids if child_id in present
        )
        if present_children and present_children != primitive.children_ids:
            errors.append(f"partial child event at primitive {primitive_id}")
    present_roots = tuple(
        primitive_id for primitive_id in tree.root_primitive_ids if primitive_id in present
    )
    if present_roots and present_roots != tree.root_primitive_ids:
        errors.append("partial root event")
    return errors


def _eligible_actions(tree: BezierTree, present: set[int]) -> list[_PruneAction]:
    primitives = tree.by_id()

    def is_leaf(primitive_id: int) -> bool:
        return not any(child_id in present for child_id in primitives[primitive_id].children_ids)

    actions: list[_PruneAction] = []
    owners: list[tuple[int | None, tuple[int, ...]]] = [
        (None, tree.root_primitive_ids)
    ]
    owners.extend(
        (primitive_id, primitive.children_ids)
        for primitive_id, primitive in sorted(primitives.items())
        if primitive_id in present
    )
    for owner_id, target_children in owners:
        present_children = tuple(child_id for child_id in target_children if child_id in present)
        if present_children != target_children or not present_children:
            continue
        if not all(is_leaf(child_id) for child_id in present_children):
            continue
        if len(present_children) == 1:
            actions.append(
                _PruneAction(
                    PruneActionKind.TERMINAL_BRANCH,
                    owner_id,
                    present_children,
                )
            )
        elif len(present_children) == 2:
            actions.append(
                _PruneAction(
                    PruneActionKind.TERMINAL_SIBLING_PAIR,
                    owner_id,
                    present_children,
                )
            )
        else:
            raise ValueError("trajectory construction requires a binary primitive tree")
    return sorted(
        actions,
        key=lambda action: (
            -1 if action.parent_id is None else action.parent_id,
            action.removed_ids,
        ),
    )


def _choose_action(
    tree: BezierTree,
    actions: list[_PruneAction],
    strategy: PruningStrategy,
    generator: np.random.Generator,
) -> _PruneAction:
    if not actions:
        raise ValueError("no legal pruning action remains")
    primitives = tree.by_id()
    if strategy is PruningStrategy.UNIFORM:
        weights = np.ones(len(actions), dtype=np.float64)
    elif strategy is PruningStrategy.DEEP:
        weights = np.asarray(
            [
                1.0 + max(primitives[item].depth for item in action.removed_ids)
                for action in actions
            ],
            dtype=np.float64,
        )
    else:
        lengths = np.asarray(
            [
                np.mean(
                    [
                        _control_polygon_length(primitives[item])
                        for item in action.removed_ids
                        if not primitives[item].virtual
                    ]
                    or [1e-6]
                )
                for action in actions
            ],
            dtype=np.float64,
        )
        if strategy is PruningStrategy.SHORT:
            weights = 1.0 / np.maximum(lengths, 1e-6)
        else:
            weights = np.maximum(lengths, 1e-6)
    probabilities = weights / weights.sum()
    return actions[int(generator.choice(len(actions), p=probabilities))]


def build_pruning_trajectory(
    tree: BezierTree,
    *,
    seed: int,
    config: PruningConfig | None = None,
) -> PruningTrajectory:
    """Repeatedly remove one continuation leaf or one terminal sibling pair."""

    config = config or PruningConfig()
    invariant_report = validate_bezier_tree(tree, require_binary=True)
    if not invariant_report.valid:
        raise ValueError(
            "pruning requires a valid binary tree: " + "; ".join(invariant_report.errors)
        )
    generator = np.random.default_rng(seed)
    strategy_values = tuple(config.strategy_probabilities)
    strategy_probabilities = tuple(
        config.strategy_probabilities[strategy] for strategy in strategy_values
    )
    present = set(tree.by_id())
    initial_ids = tuple(sorted(present))
    steps: list[PruningStep] = []
    while present:
        state_errors = _validate_partial_state(tree, present)
        if state_errors:
            raise ValueError("invalid pruning state: " + "; ".join(state_errors))
        strategy = strategy_values[
            int(generator.choice(len(strategy_values), p=strategy_probabilities))
        ]
        action = _choose_action(tree, _eligible_actions(tree, present), strategy, generator)
        present.difference_update(action.removed_ids)
        steps.append(
            PruningStep(
                step_index=len(steps),
                action_kind=action.kind,
                strategy=strategy,
                parent_branch_id=action.parent_id,
                removed_branch_ids=action.removed_ids,
                remaining_branch_count=len(present),
            )
        )
    return PruningTrajectory(seed, initial_ids, tuple(steps))


def build_pruning_trajectories(
    tree: BezierTree,
    *,
    base_seed: int,
    config: PruningConfig | None = None,
) -> tuple[PruningTrajectory, ...]:
    config = config or PruningConfig()
    seed_sequence = np.random.SeedSequence(base_seed)
    attempt_count = max(config.trajectories_per_neuron * 20, config.trajectories_per_neuron)
    child_sequences = seed_sequence.spawn(attempt_count)
    unique: list[PruningTrajectory] = []
    duplicates: list[PruningTrajectory] = []
    signatures: set[tuple[tuple[int, ...], ...]] = set()
    for sequence in child_sequences:
        seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
        trajectory = build_pruning_trajectory(tree, seed=seed, config=config)
        signature = tuple(step.removed_branch_ids for step in trajectory.steps)
        if signature in signatures:
            duplicates.append(trajectory)
        else:
            signatures.add(signature)
            unique.append(trajectory)
        if len(unique) >= config.trajectories_per_neuron:
            break
    selected = unique + duplicates[: config.trajectories_per_neuron - len(unique)]
    return tuple(selected)


def reverse_to_growth(
    tree: BezierTree, pruning: PruningTrajectory
) -> GrowthTrajectory:
    """Reverse deletion chunks into EXTEND/SPLIT events plus terminal STOP events."""

    primitives = tree.by_id()
    raw_events: list[TreeEvent] = []
    for step in reversed(pruning.steps):
        new_branches = tuple(primitives[branch_id] for branch_id in step.removed_branch_ids)
        event_type = (
            EventType.EXTEND if len(new_branches) == 1 else EventType.SPLIT
        )
        raw_events.append(
            TreeEvent(event_type, step.parent_branch_id, new_branches, 0.0)
        )
        for branch in new_branches:
            if not branch.children_ids:
                raw_events.append(TreeEvent(EventType.STOP, branch.primitive_id, (), 0.0))
    denominator = len(raw_events) + 1
    events = tuple(
        replace(event, event_time=(index + 1) / denominator)
        for index, event in enumerate(raw_events)
    )
    return GrowthTrajectory(pruning.seed, events)


def resample_event_times(
    trajectory: GrowthTrajectory,
    *,
    seed: int,
    maximum_jitter: float = 0.25,
) -> GrowthTrajectory:
    """Resample continuous event times while retaining the legal event order."""

    if not 0.0 <= maximum_jitter < 1.0:
        raise ValueError("maximum_jitter must be in [0, 1)")
    if not trajectory.events:
        return trajectory
    generator = np.random.default_rng(seed)
    jitter = generator.uniform(0.0, maximum_jitter, size=len(trajectory.events))
    denominator = len(trajectory.events) + 1
    events = tuple(
        replace(event, event_time=(index + 1 + float(jitter[index])) / denominator)
        for index, event in enumerate(trajectory.events)
    )
    return GrowthTrajectory(trajectory.source_pruning_seed, events)


def branch_birth_times(trajectory: GrowthTrajectory) -> dict[int, float]:
    """Return the event time at which each primitive first appears."""

    return {
        branch.primitive_id: event.event_time
        for event in trajectory.events
        for branch in event.new_branches
    }


def replay_growth_trajectory(
    tree: BezierTree, trajectory: GrowthTrajectory
) -> TrajectoryValidationReport:
    """Structurally replay events without performing Milestone 3 geometry interpolation."""

    target = tree.by_id()
    present: set[int] = set()
    active: set[int] = set()
    stopped: set[int] = set()
    errors: list[str] = []
    last_time = -1.0
    for event_index, event in enumerate(trajectory.events):
        if not last_time < event.event_time < 1.0:
            errors.append(f"event {event_index} has non-increasing or invalid time")
        last_time = event.event_time
        if event.event_type is EventType.STOP:
            branch_id = event.parent_branch_id
            if event.new_branches:
                errors.append(f"STOP event {event_index} creates branches")
            if branch_id is None or branch_id not in active:
                errors.append(f"STOP event {event_index} targets a non-active leaf")
                continue
            if target[branch_id].children_ids:
                errors.append(f"STOP event {event_index} targets a non-terminal branch")
                continue
            active.remove(branch_id)
            stopped.add(branch_id)
            continue

        expected_count = 1 if event.event_type is EventType.EXTEND else 2
        if len(event.new_branches) != expected_count:
            errors.append(f"event {event_index} has the wrong number of new branches")
            continue
        new_ids = event.new_branch_ids
        if any(branch_id in present for branch_id in new_ids):
            errors.append(f"event {event_index} recreates an existing branch")
            continue
        if event.parent_branch_id is None:
            if present:
                errors.append(f"root event {event_index} occurs after growth started")
            expected_children = tree.root_primitive_ids
        else:
            if event.parent_branch_id not in active:
                errors.append(f"event {event_index} targets a non-active parent")
                continue
            expected_children = target[event.parent_branch_id].children_ids
            active.remove(event.parent_branch_id)
        if new_ids != expected_children:
            errors.append(f"event {event_index} does not match target child ordering")
            continue
        present.update(new_ids)
        active.update(new_ids)

    expected_terminal = {
        primitive_id for primitive_id, primitive in target.items() if not primitive.children_ids
    }
    if present != set(target):
        errors.append("growth replay did not recover every target branch")
    if stopped != expected_terminal:
        errors.append("growth replay did not emit exactly one STOP per target terminal")
    if active:
        errors.append("growth replay ended with active leaves")
    final_state = GrowthReplayState(
        tuple(sorted(present)), tuple(sorted(active)), tuple(sorted(stopped))
    )
    return TrajectoryValidationReport(not errors, tuple(errors), final_state)


def validate_pruning_trajectory(
    tree: BezierTree, pruning: PruningTrajectory
) -> TrajectoryValidationReport:
    present = set(pruning.initial_branch_ids)
    errors: list[str] = []
    for step_index, step in enumerate(pruning.steps):
        if step.step_index != step_index:
            errors.append(f"step {step_index} has a mismatched index")
        legal = {
            (action.parent_id, action.removed_ids, action.kind)
            for action in _eligible_actions(tree, present)
        }
        candidate = (step.parent_branch_id, step.removed_branch_ids, step.action_kind)
        if candidate not in legal:
            errors.append(f"step {step_index} is not a legal leaf-pruning action")
            break
        present.difference_update(step.removed_branch_ids)
        if len(present) != step.remaining_branch_count:
            errors.append(f"step {step_index} has the wrong remaining count")
        errors.extend(
            f"step {step_index}: {message}" for message in _validate_partial_state(tree, present)
        )
    if present:
        errors.append("pruning trajectory did not reach the root anchor")
    growth_report = replay_growth_trajectory(tree, reverse_to_growth(tree, pruning))
    errors.extend(f"growth reversal: {message}" for message in growth_report.errors)
    return TrajectoryValidationReport(not errors, tuple(errors), growth_report.final_state)
