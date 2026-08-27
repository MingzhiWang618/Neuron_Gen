"""Render orthographic SWC comparisons as standalone SVG."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

from arborflow.data.swc_io import SwcMorphology


def _edges(morphology: SwcMorphology) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    by_id = morphology.by_id()
    return [
        (by_id[node.parent_id].position, node.position)
        for node in morphology
        if node.parent_id != -1
    ]


def render_swc_comparison_svg(
    original: SwcMorphology,
    reconstructed: SwcMorphology,
    path: str | os.PathLike[str],
    *,
    title: str = "SWC vs Bézier reconstruction",
) -> None:
    """Draw XY, XZ, and YZ projections without a plotting dependency."""

    width, height = 1200, 420
    panel_width, margin, top, plot_height = 344, 28, 58, 320
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    all_positions = [
        node.position
        for morphology in (original, reconstructed)
        for node in morphology
    ]
    original_edges = _edges(original)
    reconstructed_edges = _edges(reconstructed)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    for panel, (axis_x, axis_y, label) in enumerate(projections):
        left = panel * 400 + margin
        xs = [position[axis_x] for position in all_positions]
        ys = [position[axis_y] for position in all_positions]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_max - y_min, 1e-9)
        scale = min((panel_width - 2 * margin) / x_span, (plot_height - 2 * margin) / y_span)

        def project(point: tuple[float, ...]) -> tuple[float, float]:
            x = left + margin + (point[axis_x] - x_min) * scale
            y = top + plot_height - margin - (point[axis_y] - y_min) * scale
            return x, y

        elements.append(
            f'<rect x="{left}" y="{top}" width="{panel_width}" height="{plot_height}" '
            'fill="none" stroke="#d1d5db"/>'
        )
        elements.append(
            f'<text x="{left + panel_width / 2}" y="{top + 20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">{label}</text>'
        )
        for edge_set, color, opacity, dash in (
            (original_edges, "#2563eb", 0.85, ""),
            (reconstructed_edges, "#dc2626", 0.72, ' stroke-dasharray="4 2"'),
        ):
            for start, end in edge_set:
                x1, y1 = project(start)
                x2, y2 = project(end)
                elements.append(
                    f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                    f'stroke="{color}" stroke-opacity="{opacity}" stroke-width="1.25"{dash}/>'
                )
    elements.extend(
        (
            '<line x1="410" y1="404" x2="442" y2="404" stroke="#2563eb" stroke-width="2"/>',
            '<text x="448" y="409" font-family="sans-serif" font-size="13">original</text>',
            '<line x1="570" y1="404" x2="602" y2="404" stroke="#dc2626" '
            'stroke-width="2" stroke-dasharray="4 2"/>',
            '<text x="608" y="409" font-family="sans-serif" font-size="13">reconstruction</text>',
            "</svg>",
        )
    )
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
