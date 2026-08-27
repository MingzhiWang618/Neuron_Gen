from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PrepareCliTests(unittest.TestCase):
    def test_direct_script_writes_cleaned_swc_and_audit_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "cell.swc"
            source.parent.mkdir()
            source.write_text(
                "1 1 0 0 0 2 -1\n"
                "2 3 0 0 0 1 1\n"
                "3 3 1 0 0 1 2\n"
                "4 3 2 1 0 1 3\n"
                "5 3 2 -1 0 1 3\n",
                encoding="utf-8",
            )
            output = root / "cleaned"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(REPOSITORY_ROOT / "scripts" / "prepare_data.py"),
                    str(source.parent),
                    "--output",
                    str(output),
                    "--min-nodes",
                    "1",
                    "--min-real-branches",
                    "0",
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue((output / "swcs" / "cell.swc").is_file())
            log = json.loads((output / "logs" / "cell.json").read_text(encoding="utf-8"))
            self.assertTrue(log["success"])
            self.assertEqual(log["actions"][0]["node_id"], 2)
            cleaned_lines = (output / "swcs" / "cell.swc").read_text(
                encoding="utf-8"
            )
            self.assertIn("3 3 1 0 0 1 1", cleaned_lines)


if __name__ == "__main__":
    unittest.main()
