#!/usr/bin/env python3
"""Build legal pruning and reversed growth trajectories for one SWC."""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arborflow.data.trajectory_pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

