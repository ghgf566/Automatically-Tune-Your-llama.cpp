"""Process memory sampling uses only synthetic/current test processes."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from process_memory import sample_process_tree


class ProcessMemoryTests(unittest.TestCase):
    def test_current_process_tree_has_a_resident_sample(self):
        sample = sample_process_tree(os.getpid())
        self.assertEqual(sample["status"], "observed", sample)
        self.assertIn(sample["metric"], ("working_set", "rss"))
        self.assertGreater(sample["resident_bytes"], 0)
        self.assertGreaterEqual(sample["process_count"], 1)
        self.assertIn(sample["coverage_status"], ("complete", "partial"))

    def test_missing_process_is_unknown(self):
        sample = sample_process_tree(4_000_000_000)
        self.assertEqual(sample["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
