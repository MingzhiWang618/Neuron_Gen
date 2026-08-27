from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from arborflow.data.swc_io import write_swc
from tests.fixtures import balanced_morphology


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TrajectoryCliTests(unittest.TestCase):
    def test_cli_writes_reproducible_json_and_svg_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "balanced.swc"
            output = root / "trajectories"
            write_swc(balanced_morphology(), source)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(REPOSITORY_ROOT / "scripts" / "build_trajectories.py"),
                    str(source),
                    "--output",
                    str(output),
                    "--num-trajectories",
                    "4",
                    "--seed",
                    "7",
                    "--min-nodes",
                    "1",
                    "--min-real-branches",
                    "0",
                    "--allow-planar",
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["success"])
            self.assertEqual(summary["trajectory_count"], 4)
            self.assertGreater(summary["unique_pruning_orders"], 1)
            for index in range(4):
                record = json.loads(
                    (output / f"trajectory_{index:03d}.json").read_text(encoding="utf-8")
                )
                self.assertTrue(record["valid"])
                self.assertTrue((output / f"trajectory_{index:03d}.svg").is_file())


if __name__ == "__main__":
    unittest.main()
