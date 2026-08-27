from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from arborflow.data.swc_io import read_swc, write_swc
from arborflow.structures.tree_invariants import critical_topology_signature
from tests.fixtures import balanced_morphology


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class OracleCliTests(unittest.TestCase):
    def test_cli_replays_multiple_trajectories_to_topology_exact_swcs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "balanced.swc"
            output = root / "oracle"
            write_swc(balanced_morphology(), source)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(REPOSITORY_ROOT / "scripts" / "replay_oracle.py"),
                    str(source),
                    "--output",
                    str(output),
                    "--num-trajectories",
                    "3",
                    "--seed",
                    "13",
                    "--min-nodes",
                    "1",
                    "--min-real-branches",
                    "0",
                    "--allow-planar",
                    "--max-visualization-frames",
                    "4",
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["success"])
            self.assertEqual(summary["valid_replay_count"], 3)
            self.assertEqual(summary["topology_exact_count"], 3)
            self.assertEqual(summary["swc_topology_exact_count"], 3)
            self.assertEqual(summary["dynamic_indices_stable_count"], 3)
            self.assertLessEqual(summary["max_oracle_control_error_um"], 1e-12)
            expected_signature = critical_topology_signature(read_swc(source))
            for index in range(3):
                record = json.loads(
                    (output / f"oracle_{index:03d}.json").read_text(encoding="utf-8")
                )
                self.assertTrue(record["valid"])
                self.assertEqual(
                    critical_topology_signature(read_swc(output / f"oracle_{index:03d}.swc")),
                    expected_signature,
                )
                self.assertTrue((output / f"oracle_{index:03d}.svg").is_file())


if __name__ == "__main__":
    unittest.main()
