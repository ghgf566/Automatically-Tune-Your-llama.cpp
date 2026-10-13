"""Offline migration regressions use a reduced real Windows example."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from parse_fit_record import RecordParseError, main, upgrade


def archive():
    return {
        "schema_version": "0.2.0", "kind": "fit_estimate_plan",
        "status": "estimate_attempted",
        "engine_sha256": "a" * 64,
        "model": {"filename": "synthetic.gguf", "file_size_bytes": 4096,
                  "architecture": "qwen35", "gguf_version": 3},
        "candidate": {"device_id": "Vulkan0", "context_tokens": 4096,
                      "gpu_layers": 18, "fit_adjustment": "off",
                      "output_mode": "estimated_memory"},
        "argv": ["llama.exe", "fit-params", "--offline", "--model", "synthetic.gguf",
                 "--device", "Vulkan0", "--ctx-size", "4096",
                 "--n-gpu-layers", "18", "--fit", "off", "--fit-print", "on"],
        "estimate": {
            "status": "completed", "returncode": 0,
            "stdout_excerpt": "Vulkan0 4136 101 525 \r\nHost 9111 304 44 \r\n",
            "stderr_excerpt": "example only",
            "output_truncated_or_unknown": True,
            "estimate_unit": "MiB (as reported by upstream, if recognizable)",
            "parsed_numeric_memory": None,
            "model_load_tested": False, "inference_tested": False,
        },
        "environment": {
            "scenario": "daily", "scenario_is_user_declared": True,
            "background_note": "Wallpaper Engine running",
            "capture_status": "captured_best_effort",
            "before": {"timestamp_utc": "2026-10-13T18:29:46+00:00",
                       "ram": {"status": "observed", "reason": None,
                               "total_mib": 32700, "available_mib": 22634},
                       "gpu": {"status": "observed", "reason": None,
                               "device_id": "Vulkan0", "backend_id": "Vulkan",
                               "description": "synthetic GPU",
                               "total_mib": 8274, "available_mib": 7497}},
            "after": {"timestamp_utc": "2026-10-13T18:29:48+00:00",
                      "ram": {"status": "observed", "reason": None,
                              "total_mib": 32700, "available_mib": 22601},
                      "gpu": {"status": "observed", "reason": None,
                              "device_id": "Vulkan0", "backend_id": "Vulkan",
                              "description": "synthetic GPU",
                              "total_mib": 8274, "available_mib": 7497}},
            "available_memory_change": {
                "system_ram_available_delta_mib": -33,
                "gpu_available_delta_mib": 0,
                "peak_gpu_usage_mib": None, "peak_ram_usage_mib": None,
                "attribution": "unknown_background_load_and_driver_effects",
                "model_load_status": "not_tested",
            },
        },
        "limitations": ["synthetic"],
    }


class ExistingEstimateMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads((ROOT / "schemas/fit-estimate-plan.schema.json").read_text("utf-8"))
        Draft202012Validator.check_schema(schema)
        cls.validator = Draft202012Validator(schema)

    def test_archive_real_memory_rows_and_schema(self):
        original = archive()
        upgraded = upgrade(original)
        self.validator.validate(upgraded)
        self.assertEqual(upgraded["schema_version"], "0.3.0")
        self.assertEqual(upgraded["estimate"]["parsed_numeric_memory"]["estimated_total_mib"], 14221)
        self.assertEqual(upgraded["environment"], original["environment"])
        self.assertEqual(original["schema_version"], "0.2.0")
        self.assertIsNone(original["estimate"]["parsed_numeric_memory"])

    def test_reject_failure_wrong_device_and_missing_rows(self):
        for action in ("error", "device", "missing_host"):
            obj = archive()
            if action == "error":
                obj["estimate"]["status"] = "nonzero_exit"
            elif action == "device":
                obj["candidate"]["device_id"] = "CUDA0"
            else:
                obj["estimate"]["stdout_excerpt"] = "Vulkan0 4136 101 525\n"
            with self.subTest(action=action), self.assertRaises(RecordParseError):
                upgrade(obj)

    def test_new_truncated_output_is_rejected(self):
        obj = archive()
        obj["schema_version"] = "0.3.0"
        with self.assertRaisesRegex(RecordParseError, "truncated"):
            upgrade(obj)

    def test_offline_command_does_not_touch_model_or_engine(self):
        obj = archive()
        with TemporaryDirectory() as folder:
            source = Path(folder) / "original.json"
            target = Path(folder) / "enriched.json"
            source.write_text(json.dumps(obj), encoding="utf-8")
            with patch("sys.argv", ["parse_fit_record.py", "--input", str(source),
                                    "--output", str(target)]):
                self.assertEqual(main(), 0)
            self.validator.validate(json.loads(target.read_text(encoding="utf-8")))
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), obj)

    def test_refuse_to_overwrite_original(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "original.json"
            source.write_text(json.dumps(archive()), encoding="utf-8")
            with patch("sys.argv", ["parse_fit_record.py", "--input", str(source),
                                    "--output", str(source)]):
                self.assertEqual(main(), 1)
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")),
                             archive())


if __name__ == "__main__":
    unittest.main()
