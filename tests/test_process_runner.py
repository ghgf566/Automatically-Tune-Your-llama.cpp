"""Bounded subprocess behavior is tested using Python only, not llama."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from process_runner import run_bounded


class ProcessRunnerTests(unittest.TestCase):
    def test_command_exit_zero_without_shell(self):
        result = run_bounded([sys.executable, "-c", "print('bounded ok')"],
                             10, capture_bytes=512, disk_limit_bytes=2048)
        self.assertEqual(result["status"], "success", result["stderr"])
        self.assertEqual(result["returncode"], 0)
        self.assertIn("bounded ok", result["stdout"])
        self.assertFalse(result["truncated"])

    def test_failed_command_has_nonzero_exit(self):
        result = run_bounded([sys.executable, "-c", "raise SystemExit(7)"], 10)
        self.assertEqual(result["status"], "nonzero_exit")
        self.assertEqual(result["returncode"], 7)

    def test_timeout_stops_child(self):
        result = run_bounded([sys.executable, "-c", "import time;time.sleep(30)"],
                             0.2, capture_bytes=512, disk_limit_bytes=2048)
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["returncode"])

    def test_output_limit_stops_excessive_output(self):
        result = run_bounded(
            [sys.executable, "-c", "import sys;sys.stdout.write('x'*65536)"],
            10, capture_bytes=1024, disk_limit_bytes=2048)
        self.assertEqual(result["status"], "output_limit")
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["stdout"]), 1024)

    def test_missing_executable_is_safe_error(self):
        result = run_bounded(["this-program-does-not-exist-atyl-92831"], 5)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["returncode"])
        self.assertEqual(result["stderr"], "FileNotFoundError")

    def test_explicit_cancel_event_stops_owned_child(self):
        from threading import Event
        cancelled = Event()
        cancelled.set()
        result = run_bounded(
            [sys.executable, "-c", "import time;time.sleep(30)"], 10,
            cancel_event=cancelled)
        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(result["returncode"])

    def test_timeout_kills_descendant_process_before_file_write(self):
        import json
        import time
        with TemporaryDirectory() as d:
            target = Path(d) / "descendant-survived.txt"
            grandchild = ("import time,pathlib;time.sleep(0.7);"
                          "pathlib.Path(" + repr(str(target)) + ").write_text('orphan')")
            parent = ("import subprocess,sys,time;"
                      "subprocess.Popen([sys.executable,'-c'," +
                      repr(grandchild) + "]);time.sleep(30)")
            result = run_bounded(
                [sys.executable, "-c", parent], 0.25,
                capture_bytes=512, disk_limit_bytes=2048)
            self.assertEqual(result["status"], "timeout")
            time.sleep(0.85)
            self.assertFalse(target.exists(),
                             "timeout left a running descendant that escaped cleanup")

    def test_invalid_limits_rejected_before_execution(self):
        with self.assertRaises(ValueError):
            run_bounded([sys.executable], 10, capture_bytes=2048, disk_limit_bytes=512)


if __name__ == "__main__":
    unittest.main()
