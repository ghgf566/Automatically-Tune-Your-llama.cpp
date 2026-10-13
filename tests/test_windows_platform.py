"""Windows-only platform memory API smoke test (no model or GPU)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from environment_observation import system_ram


@unittest.skipUnless(sys.platform == "win32", "Windows-only memory API")
class WindowsPlatformTests(unittest.TestCase):
    def test_globalmemorystatusex_reports_valid_system_ram(self):
        ram = system_ram()
        self.assertEqual(ram["status"], "observed", ram.get("reason"))
        self.assertGreater(ram["total_mib"], 0)
        self.assertGreaterEqual(ram["available_mib"], 0)
        self.assertLessEqual(ram["available_mib"], ram["total_mib"])


if __name__ == "__main__":
    unittest.main()
