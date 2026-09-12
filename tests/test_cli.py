import subprocess
import sys
import unittest
from importlib.metadata import version


class CliTest(unittest.TestCase):
    def test_scaffold_entrypoint(self):
        for args, expected in (
            ([], "usage:"),
            (["--help"], "usage:"),
            (["--version"], f"catrecap {version('catrecap')}"),
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-m", "catrecap", *args],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
