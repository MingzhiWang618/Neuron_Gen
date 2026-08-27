"""Dependency-free visualization of leaf-pruning trajectories."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

import numpy as np

from arborflow.data.bezier_fitting import evaluate_cubic
from arborflow.structures.embedded_tree import BezierTree
from arborflow.structures.tree_events import PruningTrajectory


def _selected_frames(step_count: int, max_frames: int) -> tuple[int, ...]:
    if max_frames < 2:
        raise ValueError("max_frames must be at least two")
    if step_count + 1 <= max_frames:
        return tuple(range(step_count + 1))
    return tuple(
        sorted(
            set(
                int(round(value))
                for value in np.linspace(0, step_count, max_frames)
            )
        )
    )


def render_pruning_trajectory_svg(
    tree: BezierTree,
    trajectory: PruningTrajectory,
    path: str | os.PathLike[str],
    *,
    max_frames: int = 12,
    title: str = "Leaf-pruning destruction trajectory",
) -> None:
    """Render selected XY snapshots from the full tree to the root anchor."""

    primitives = tree.by_id()
    samples = {
        primitive_id: evaluate_cubic(
            primitive.control_points, np.linspace(0.0, 1.0, 17)
        )
        for primitive_id, primitive in primitives.items()
    }
    all_points = [np.asarray(tree.root.position, dtype=np.float64)]
    all_points.extend(point for curve in samples.values() for point in curve)
    point_array = np.asarray(all_points)
    x_min, y_min = point_array[:, :2].min(axis=0)
    x_max, y_max = point_array[:, :2].max(axis=0)
    x_span = max(float(x_max - x_min), 1e-9)
    y_span = max(float(y_max - y_min), 1e-9)

    frame_indices = _selected_frames(len(trajectory.steps), max_frames)
    columns = min(4, len(frame_indices))
    rows = int(np.ceil(len(frame_indices) / columns))
    panel_width, panel_height = 280, 250
    width = columns * panel_width
    height = 50 + rows * panel_height
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    present = set(trajectory.initial_branch_ids)
    states: dict[int, set[int]] = {0: set(present)}
    selected = set(frame_indices)
    for step_number, step in enumerate(trajectory.steps, start=1):
        present.difference_update(step.removed_branch_ids)
        if step_number in selected:
            states[step_number] = set(present)

    scale = min(220.0 / x_span, 180.0 / y_span)
    for frame_position, step_number in enumerate(frame_indices):
        row, column = divmod(frame_position, columns)
        left = column * panel_width
        top = 50 + row * panel_height

        def project(point: np.ndarray) -> tuple[float, float]:
            x = left + 30 + (float(point[0]) - x_min) * scale
            y = top + 210 - (float(point[1]) - y_min) * scale
            return x, y

        elements.append(
            f'<rect x="{left + 10}" y="{top + 5}" width="{panel_width - 20}" '
            f'height="{panel_height - 15}" fill="#fafafa" stroke="#d1d5db"/>'
        )
        frame_state = states[step_number]
        for primitive_id in sorted(frame_state):
            primitive = primitives[primitive_id]
            coordinates = " ".join(
                f"{x:.2f},{y:.2f}" for x, y in map(project, samples[primitive_id])
            )
            color = "#f59e0b" if primitive.virtual else "#2563eb"
            dash = ' stroke-dasharray="4 3"' if primitive.virtual else ""
            elements.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                f'stroke-width="1.4"{dash}/>'
            )
        root_x, root_y = project(np.asarray(tree.root.position))
        elements.append(
            f'<circle cx="{root_x:.2f}" cy="{root_y:.2f}" r="3" fill="#111827"/>'
        )
        label = f"step {step_number}/{len(trajectory.steps)} · {len(frame_state)} primitives"
        elements.append(
            f'<text x="{left + panel_width / 2}" y="{top + 232}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
    elements.append("</svg>")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(elements) + "\n")
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

