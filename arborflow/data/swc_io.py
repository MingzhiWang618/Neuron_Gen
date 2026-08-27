"""Strict, dependency-light SWC reading and writing.

Parsing is deliberately separated from validation. This lets callers retain malformed
records long enough to produce a useful validation report (for example duplicate IDs).
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


class SwcParseError(ValueError):
    """Raised when an SWC row cannot be parsed losslessly."""


@dataclass(frozen=True, slots=True)
class SwcNode:
    """One seven-column SWC sample."""

    node_id: int
    swc_type: int
    x: float
    y: float
    z: float
    radius: float
    parent_id: int

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class SwcMorphology:
    """An ordered collection of SWC samples plus source metadata."""

    nodes: tuple[SwcNode, ...]
    source: str | None = None
    comments: tuple[str, ...] = ()

    def __iter__(self) -> Iterator[SwcNode]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def by_id(self) -> dict[int, SwcNode]:
        """Return an ID lookup, rejecting ambiguous duplicate IDs."""

        result: dict[int, SwcNode] = {}
        for node in self.nodes:
            if node.node_id in result:
                raise ValueError(f"duplicate node ID {node.node_id}")
            result[node.node_id] = node
        return result


def _parse_int(token: str, *, field: str, source: str, line_number: int) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise SwcParseError(
            f"{source}:{line_number}: {field} must be an integer, got {token!r}"
        ) from exc
    return value


def _parse_float(token: str, *, field: str, source: str, line_number: int) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise SwcParseError(
            f"{source}:{line_number}: {field} must be numeric, got {token!r}"
        ) from exc


def parse_swc_lines(lines: Iterable[str], *, source: str = "<memory>") -> SwcMorphology:
    """Parse SWC text while retaining comments and input order.

    Inline comments beginning with ``#`` are accepted. Non-comment rows must contain
    exactly the seven standard SWC columns.
    """

    nodes: list[SwcNode] = []
    comments: list[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comments.append(stripped[1:].strip())
            continue
        data, _, inline_comment = stripped.partition("#")
        fields = data.split()
        if len(fields) != 7:
            raise SwcParseError(
                f"{source}:{line_number}: expected 7 columns, found {len(fields)}"
            )
        if inline_comment.strip():
            comments.append(f"line {line_number}: {inline_comment.strip()}")
        nodes.append(
            SwcNode(
                node_id=_parse_int(
                    fields[0], field="node ID", source=source, line_number=line_number
                ),
                swc_type=_parse_int(
                    fields[1], field="SWC type", source=source, line_number=line_number
                ),
                x=_parse_float(fields[2], field="x", source=source, line_number=line_number),
                y=_parse_float(fields[3], field="y", source=source, line_number=line_number),
                z=_parse_float(fields[4], field="z", source=source, line_number=line_number),
                radius=_parse_float(
                    fields[5], field="radius", source=source, line_number=line_number
                ),
                parent_id=_parse_int(
                    fields[6], field="parent ID", source=source, line_number=line_number
                ),
            )
        )
    if not nodes:
        raise SwcParseError(f"{source}: no SWC samples found")
    return SwcMorphology(tuple(nodes), source=source, comments=tuple(comments))


def read_swc(path: str | os.PathLike[str]) -> SwcMorphology:
    """Read an SWC file using UTF-8 text."""

    swc_path = Path(path)
    with swc_path.open("r", encoding="utf-8-sig") as handle:
        return parse_swc_lines(handle, source=str(swc_path))


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.9g}"


def write_swc(
    morphology: SwcMorphology,
    path: str | os.PathLike[str],
    *,
    include_comments: bool = True,
) -> None:
    """Atomically write a morphology in deterministic SWC format."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if include_comments:
        lines.extend(f"# {comment}" for comment in morphology.comments)
    for node in morphology.nodes:
        lines.append(
            " ".join(
                (
                    str(node.node_id),
                    str(node.swc_type),
                    _format_float(node.x),
                    _format_float(node.y),
                    _format_float(node.z),
                    _format_float(node.radius),
                    str(node.parent_id),
                )
            )
        )
    payload = "\n".join(lines) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

