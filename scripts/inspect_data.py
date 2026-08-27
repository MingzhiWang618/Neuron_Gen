#!/usr/bin/env python3
"""Run and inspect the complete Milestone 1 representation round-trip."""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arborflow.data.milestone1 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

