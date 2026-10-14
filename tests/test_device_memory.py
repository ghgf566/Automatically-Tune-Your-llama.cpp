"""NVIDIA device memory observation tests use synthetic driver output."""
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from device_memory import DeviceMemoryMonitor, parse_nvidia_smi, sample_nvidia_smi


class DeviceMemoryTests(unittest.TestCase):
    def test_parser_keeps_device_identity_and_memory_units(self):
        rows = parse_nvidia_smi(
            "GPU-abc-123, NVIDIA Example, 8192, 7000, 1000\n"
            "GPU-def-456, NVIDIA Other, 4096, 3000, 900\n")
        self.assertEqual([item["uuid"] for item in rows],
                         ["GPU-abc-123", "GPU-def-456"])
        self.assertEqual(rows[0]["total_mib"], 8192)

    def test_parser_rejects_n_a_duplicate_and_inconsistent_rows(self):
        self.assertIsNone(parse_nvidia_smi("GPU-a, GPU, 8192, N/A, 10\n"))
        duplicate = "GPU-a, GPU, 8192, 7000, 1000\nGPU-a, GPU, 8192, 7000, 1000\n"
        self.assertIsNone(parse_nvidia_smi(duplicate))
        self.assertIsNone(parse_nvidia_smi("GPU-a, GPU, 8192, 9000, 10\n"))

    def test_query_is_bounded_and_unrecognized_output_is_unknown(self):
        with patch("device_memory.shutil.which", return_value="nvidia-smi.exe"), \
                patch("device_memory.run_bounded", return_value={
                    "status": "success", "truncated": False, "stdout": "bad", "stderr": ""}) as run:
            devices, reason = sample_nvidia_smi()
        self.assertIsNone(devices)
        self.assertEqual(reason, "unrecognized_or_incomplete_output")
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi.exe")
        self.assertEqual(run.call_args.args[0][-1], "--format=csv,noheader,nounits")

    def test_monitor_reports_system_wide_minimum_free_and_maximum_used(self):
        observations = [
            {"uuid": "GPU-a-1", "name": "NVIDIA Example", "total_mib": 8192,
             "free_mib": 7000, "used_mib": 1000},
            {"uuid": "GPU-a-1", "name": "NVIDIA Example", "total_mib": 8192,
             "free_mib": 6500, "used_mib": 1500},
        ]
        second_seen = threading.Event()

        def sampler():
            item = observations.pop(0)
            if not observations:
                second_seen.set()
            return [item], None

        monitor = DeviceMemoryMonitor(sampler=sampler, interval_seconds=1)
        monitor.start()
        self.assertTrue(second_seen.wait(2))
        report = monitor.stop_and_report()
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["provider"], "nvidia_smi")
        self.assertEqual(report["process_attribution"], "system_wide")
        self.assertEqual(report["devices"][0]["minimum_free_memory_mib"], 6500)
        self.assertEqual(report["devices"][0]["maximum_used_memory_mib"], 1500)


if __name__ == "__main__":
    unittest.main()
