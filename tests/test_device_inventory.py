"""Validate the fail-closed GGML hardware inventory contract.

This suite uses synthetic data; it never launches installed llama.cpp.
"""
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from device_inventory import parse_list_devices, observation, unknown


class DeviceSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / "schemas/device-inventory.schema.json").read_text("utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_synthetic_device_result_conforms_to_schema(self):
        observed = parse_list_devices(
            "Available devices:\n  CUDA0: NVIDIA Example GPU (8192 MiB, 4096 MiB free)\n")
        self.validator.validate(observed)
        self.assertEqual(observed["devices"][0]["id"], "CUDA0")
        self.assertEqual(observed["devices"][0]["free_memory_mib"], 4096)

    def test_none_and_unknown_are_not_merged(self):
        empty = parse_list_devices("Available devices:\n  (none)\n")
        missing = parse_list_devices("unsupported help output\n")
        self.validator.validate(empty)
        self.validator.validate(missing)
        self.assertEqual(empty["status"], "none")
        self.assertEqual(missing["status"], "unknown")
        self.assertEqual(missing["devices"], None)

    def test_timeout_does_not_claim_zero_gpus(self):
        result = observation({"status": "timeout", "stdout": "", "stderr": ""})
        self.validator.validate(result)
        self.assertIsNone(result["non_cpu_device_count"])

    def test_reject_unknown_with_devices_array(self):
        result = unknown("example_error")
        result["devices"] = []
        with self.assertRaises(ValidationError):
            self.validator.validate(result)

    def test_reject_positive_count_for_none(self):
        none = parse_list_devices("Available devices:\n  (none)\n")
        none["non_cpu_device_count"] = 1
        with self.assertRaises(ValidationError):
            self.validator.validate(none)


if __name__ == "__main__":
    unittest.main()
