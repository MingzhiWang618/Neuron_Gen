"""Dataset-level SWC cleaning command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arborflow.data.swc_io import SwcParseError, read_swc, write_swc
from arborflow.data.swc_validation import SwcValidationConfig, clean_swc


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and conservatively clean SWC files")
    parser.add_argument("input", type=Path, help="an SWC file or a directory searched recursively")
    parser.add_argument("--output", type=Path, required=True, help="output dataset directory")
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--min-real-branches", type=int, default=5)
    parser.add_argument("--max-real-branches", type=int, default=1000)
    parser.add_argument("--coordinate-tolerance-um", type=float, default=1e-6)
    parser.add_argument("--require-3d", action="store_true")
    parser.add_argument("--no-repair-zero-length", action="store_true")
    return parser


def _discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            candidate for candidate in path.rglob("*") if candidate.suffix.lower() == ".swc"
        )
    raise FileNotFoundError(path)


def run(args: argparse.Namespace) -> int:
    config = SwcValidationConfig(
        require_3d=args.require_3d,
        min_nodes=args.min_nodes,
        min_real_branches=args.min_real_branches,
        max_real_branches=args.max_real_branches,
        coordinate_tolerance_um=args.coordinate_tolerance_um,
    )
    files = _discover(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for index, source in enumerate(files):
        relative = source.name if args.input.is_file() else str(source.relative_to(args.input))
        safe_stem = Path(relative).with_suffix("").as_posix().replace("/", "__")
        record: dict[str, object] = {"source": str(source), "relative_source": relative}
        try:
            morphology = read_swc(source)
            result = clean_swc(
                morphology,
                config,
                repair_zero_length_edges=not args.no_repair_zero_length,
            )
            record.update(result.to_dict())
            if result.success and result.morphology is not None:
                output_swc = args.output / "swcs" / f"{safe_stem}.swc"
                write_swc(result.morphology, output_swc)
                record["output"] = str(output_swc)
        except (OSError, SwcParseError, ValueError) as exc:
            record.update({"success": False, "parse_error": str(exc)})
        log_path = args.output / "logs" / f"{safe_stem}.json"
        _atomic_json(log_path, record)
        record["log"] = str(log_path)
        manifest.append(record)
        status = "OK" if record.get("success") else "FAIL"
        print(f"[{index + 1}/{len(files)}] {status} {relative}")
    manifest_path = args.output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest),
        encoding="utf-8",
    )
    succeeded = sum(bool(record.get("success")) for record in manifest)
    print(f"Processed {len(files)} files: {succeeded} succeeded, {len(files) - succeeded} failed")
    return 0 if succeeded == len(files) and files else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
