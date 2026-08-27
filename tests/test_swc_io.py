from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arborflow.data.swc_io import SwcParseError, parse_swc_lines, read_swc, write_swc


VALID = """# fixture
1 1 0 0 0 4 -1
2 3 1 0 0 1 1
3 3 2 1 0.5 0.8 2
"""


class SwcIoTests(unittest.TestCase):
    def test_parse_retains_types_and_comments(self) -> None:
        morphology = parse_swc_lines(VALID.splitlines(), source="fixture.swc")
        self.assertEqual(len(morphology), 3)
        self.assertEqual(morphology.nodes[1].swc_type, 3)
        self.assertEqual(morphology.nodes[2].parent_id, 2)
        self.assertEqual(morphology.comments, ("fixture",))

    def test_invalid_column_count_has_location(self) -> None:
        with self.assertRaisesRegex(SwcParseError, r"fixture\.swc:2: expected 7"):
            parse_swc_lines(["# header", "1 1 0 0"], source="fixture.swc")

    def test_write_read_roundtrip(self) -> None:
        morphology = parse_swc_lines(VALID.splitlines())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "roundtrip.swc"
            write_swc(morphology, path)
            recovered = read_swc(path)
        self.assertEqual(recovered.nodes, morphology.nodes)


if __name__ == "__main__":
    unittest.main()

