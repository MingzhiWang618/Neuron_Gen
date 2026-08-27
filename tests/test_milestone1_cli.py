from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from arborflow.data.swc_io import write_swc
from tests.fixtures import branching_morphology


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class Milestone1CliTests(unittest.TestCase):
    def test_end_to_end_artifacts_and_acceptance_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.swc"
            output = root / "artifacts"
            write_swc(branching_morphology(), source)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(REPOSITORY_ROOT / "scripts" / "inspect_data.py"),
                    str(source),
                    "--output",
                    str(output),
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
            report = json.loads((output / "milestone1.json").read_text(encoding="utf-8"))
            self.assertTrue(report["success"])
            self.assertTrue(report["exact_roundtrip"])
            self.assertTrue(report["binary_roundtrip"])
            self.assertTrue(report["topology_preserved"])
            self.assertTrue(report["bifurcations_preserved"])
            self.assertEqual(
                report["source_bifurcations"], report["reconstruction_bifurcations"]
            )
            self.assertIn(report["morphio_readable"], (None, True))
            self.assertEqual(
                report["milestone1_acceptance_complete"],
                report["morphio_readable"] is True,
            )
            self.assertEqual(report["source_real_branches"], 4)
            self.assertTrue((output / "bezier_reconstruction.swc").is_file())
            self.assertTrue((output / "original_vs_reconstruction.svg").is_file())


if __name__ == "__main__":
    unittest.main()
