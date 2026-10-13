"""Regression evidence reduced from user-supplied real Windows fit-daily JSON.

The two short output rows are unchanged (no local path / private metadata).
No model is loaded and no GPU calls occur in these unit tests.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fit_memory_parser import parse_memory_estimate


REAL_WINDOWS_11574_STDOUT = "Vulkan0 4136 101 525 \r\nHost 9111 304 44 \r\n"


class FitMemoryParserTests(unittest.TestCase):
    def test_real_windows_vulkan_and_host_breakdown(self):
        got = parse_memory_estimate(REAL_WINDOWS_11574_STDOUT, "Vulkan0")
        self.assertIsNotNone(got)
        self.assertEqual(got["status"], "parsed")
        self.assertEqual(got["unit"], "MiB")
        self.assertEqual(got["devices"], [{
            "device_id": "Vulkan0",
            "model_mib": 4136,
            "context_mib": 101,
            "compute_mib": 525,
            "estimated_total_mib": 4762,
        }])
        self.assertEqual(got["host"], {
            "model_mib": 9111, "context_mib": 304, "compute_mib": 44,
            "estimated_total_mib": 9459,
        })
        self.assertEqual(got["estimated_total_mib"], 14221)
        self.assertFalse(got["model_load_tested"])
        self.assertFalse(got["inference_tested"])

    def test_cpu_only_valid_host_row(self):
        got = parse_memory_estimate("Host 100 20 30\n", "none")
        self.assertEqual(got["devices"], [])
        self.assertEqual(got["estimated_total_mib"], 150)

    def test_reject_ambiguous_or_future_output(self):
        invalid = [
            "",
            " ",
            "Vulkan0 4136 101 525\n",
            "Vulkan0 4136 101 525\nHost 9111 304\n",
            "Vulkan0 4136 101 525\nHost 9111 304 44\nHost 1 2 3\n",
            "Vulkan0 -1 101 525\nHost 1 2 3\n",
            "Vulkan0 1.5 101 525\nHost 1 2 3\n",
            "Vulkan0 4136 101 525\nHost 9111 304 44\nother log line\n",
            "Vulkan0 0 0 0\nHost 0 0 0\n",
        ]
        for data in invalid[:-1]:
            with self.subTest(data=data):
                self.assertIsNone(parse_memory_estimate(data, "Vulkan0"))
        # Zero usage remains syntactically valid and is not proof of inference.
        zero = parse_memory_estimate(invalid[-1], "Vulkan0")
        self.assertEqual(zero["estimated_total_mib"], 0)

    def test_reject_different_device_or_cpu_only_with_gpu_row(self):
        self.assertIsNone(parse_memory_estimate(REAL_WINDOWS_11574_STDOUT, "CUDA0"))
        self.assertIsNone(parse_memory_estimate(REAL_WINDOWS_11574_STDOUT, "none"))
        self.assertIsNone(parse_memory_estimate("Host 1 2 3\n", "Vulkan0"))

    def test_refuse_unrealistic_number(self):
        value = 2**53 + 1
        self.assertIsNone(parse_memory_estimate(f"Host {value} 1 2\n", "none"))


if __name__ == "__main__":
    unittest.main()
