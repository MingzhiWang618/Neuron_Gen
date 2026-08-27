#!/usr/bin/env python3
"""Validate and clean an SWC dataset.

The small path bootstrap keeps the task-book command usable from a fresh checkout;
installed environments use the ``arborflow-prepare-data`` console entry point instead.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arborflow.data.prepare import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
