"""Dependency-free visualization of continuous oracle growth replay."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

import numpy as np

from arborflow.data.bezier_fitting import evaluate_cubic
from arborflow.flow.oracle_replay import OracleReplay


def render_oracle_replay_svg(
    replay: OracleReplay,
    path: str | os.PathLike[str],
    *,
    max_frames: int = 12,
    title: str = "Oracle birth–death geometry replay",
) -> None:
    """Render uniformly spaced XY snapshots from the root anchor to the data tree."""

    if max_frames < 2:
        raise ValueError("max_frames must be at least two")
    times = np.linspace(0.0, 1.0, max_frames)
    states = [replay.state_at(float(global_time)) for global_time in times]
    target_points = [np.asarray(replay.tree.root.position, dtype=np.float64)]
    for primitive in replay.tree.primitives:
        target_points.extend(
            evaluate_cubic(primitive.control_points, np.linspace(0.0, 1.0, 17))
        )
    point_array = np.asarray(target_points)
    x_min, y_min = point_array[:, :2].min(axis=0)
    x_max, y_max = point_array[:, :2].max(axis=0)
    x_span = max(float(x_max - x_min), 1e-9)
    y_span = max(float(y_max - y_min), 1e-9)
    columns = min(4, max_frames)
    rows = int(np.ceil(max_frames / columns))
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
    scale = min(220.0 / x_span, 180.0 / y_span)
    for frame_index, state in enumerate(states):
        row, column = divmod(frame_index, columns)
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
        for branch_index, branch in enumerate(state.branches):
            curve = evaluate_cubic(branch.control_points, np.linspace(0.0, 1.0, 17))
            coordinates = " ".join(
                f"{x:.2f},{y:.2f}" for x, y in map(project, curve)
            )
            if branch.virtual:
                color, dash = "#f59e0b", ' stroke-dasharray="4 3"'
            elif state.active_leaf_mask[branch_index]:
                color, dash = "#dc2626", ""
            elif state.stopped_mask[branch_index]:
                color, dash = "#059669", ""
            else:
                color, dash = "#2563eb", ""
            elements.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                f'stroke-width="1.4"{dash}/>'
            )
        root_x, root_y = project(np.asarray(replay.tree.root.position, dtype=np.float64))
        elements.append(
            f'<circle cx="{root_x:.2f}" cy="{root_y:.2f}" r="3" fill="#111827"/>'
        )
        active_count = int(np.count_nonzero(state.active_leaf_mask))
        label = (
            f"t={state.global_time:.2f} · {len(state.branches)} primitives · "
            f"{active_count} active"
        )
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
