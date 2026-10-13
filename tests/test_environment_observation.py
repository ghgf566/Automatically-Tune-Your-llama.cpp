"""One-shot environmental observations, all synthetic and mocked."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import environment_observation as env


class EnvironmentObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(
            (ROOT / "schemas/environment-observation.schema.json").read_text("utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_linux_ram_parsing_from_proc_meminfo(self):
        data = "MemTotal:       16384000 kB\nMemFree:   1000 kB\nMemAvailable:  8192000 kB\n"
        with patch.object(env.platform, "system", return_value="Linux"):
            with patch("builtins.open", mock_open(read_data=data)):
                result = env.system_ram()
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["total_mib"], 16000)
        self.assertEqual(result["available_mib"], 8000)
        self.assertFalse("MemFree" in result)

    def test_ram_unknown_for_unsupported_platform(self):
        with patch.object(env.platform, "system", return_value="Darwin"):
            output = env.system_ram()
        self.assertEqual(output["status"], "unknown")
        self.assertIsNone(output["available_mib"])

    def test_gpu_headroom_parses_selected_device(self):
        class Proc:
            def wait(self, timeout):
                return 0
        def launch(argv, **kwargs):
            self.assertEqual(argv, ["C:/fake/llama.exe", "serve", "--list-devices"])
            kwargs["stdout"].write(
                b"Available devices:\n  Vulkan0: Synthetic GPU (8192 MiB, 7300 MiB free)\n")
            return Proc()
        with patch.object(env.subprocess, "Popen", side_effect=launch) as popen:
            gpu = env.ggml_headroom("C:/fake/llama.exe", "Vulkan0")
        self.assertEqual(gpu["status"], "observed")
        self.assertEqual(gpu["backend_id"], "Vulkan")
        self.assertEqual(gpu["available_mib"], 7300)
        popen.assert_called_once()
        self.assertNotIn("shell", popen.call_args.kwargs)

    def test_missing_fresh_device_does_not_infer_no_gpu(self):
        class Proc:
            def wait(self, timeout):
                return 0
        def launch(argv, **kwargs):
            kwargs["stdout"].write(
                b"Available devices:\n  CUDA0: Synthetic (8192 MiB, 7300 MiB free)\n")
            return Proc()
        with patch.object(env.subprocess, "Popen", side_effect=launch):
            got = env.ggml_headroom("llama.exe", "Vulkan0")
        self.assertEqual(got["status"], "unknown")
        self.assertIsNone(got["available_mib"])

    def test_probe_nonzero_retains_unknown(self):
        class Proc:
            def wait(self, timeout):
                return 1
        with patch.object(env.subprocess, "Popen", return_value=Proc()):
            value = env.ggml_headroom("llama.exe", "Vulkan0")
        self.assertEqual(value["reason"], "device_probe_nonzero_exit")

    def test_cpu_only_skips_gpu_probe(self):
        with patch.object(env.subprocess, "Popen") as popen:
            result = env.ggml_headroom("llama.exe", "none")
        popen.assert_not_called()
        self.assertEqual(result["status"], "not_requested")
        self.assertIsNone(result["available_mib"])

    def test_availability_difference_is_not_peak_usage(self):
        before = {"ram": {"status": "observed", "available_mib": 10000},
                  "gpu": {"status": "observed", "available_mib": 7000}}
        after = {"ram": {"status": "observed", "available_mib": 9700},
                 "gpu": {"status": "observed", "available_mib": 7100}}
        changes = env.summarize(before, after)
        self.assertEqual(changes["system_ram_available_delta_mib"], -300)
        self.assertEqual(changes["gpu_available_delta_mib"], 100)
        self.assertIsNone(changes["peak_gpu_usage_mib"])
        self.assertEqual(changes["model_load_status"], "not_tested")

    def test_unknown_availability_delta_stays_null(self):
        a = {"ram": {"status": "unknown", "available_mib": None},
             "gpu": {"status": "not_requested", "available_mib": None}}
        b = {"ram": {"status": "observed", "available_mib": 1000},
             "gpu": {"status": "not_requested", "available_mib": None}}
        result = env.summarize(a, b)
        self.assertIsNone(result["system_ram_available_delta_mib"])
        self.assertIsNone(result["gpu_available_delta_mib"])

    def test_llama_override_env_not_passed_to_probe(self):
        with patch.dict(env.os.environ, {"LLAMA_ARG_MODEL": "bad.gguf",
                                          "HF_TOKEN": "secret",
                                          "ATYL_TEST_SAFE": "ok"}):
            clean = env.clean_llama_env()
        self.assertNotIn("LLAMA_ARG_MODEL", clean)
        self.assertNotIn("HF_TOKEN", clean)
        self.assertEqual(clean["ATYL_TEST_SAFE"], "ok")

    def test_standalone_environment_schema_sample(self):
        before = {
            "timestamp_utc": "2026-10-14T01:00:00+00:00",
            "ram": {"status": "observed", "reason": None,
                    "total_mib": 16000, "available_mib": 8000},
            "gpu": {"status": "unknown", "reason": "device_probe_timeout",
                    "device_id": "Vulkan0", "backend_id": None, "description": None,
                    "total_mib": None, "available_mib": None},
        }
        result = {
            "scenario": "daily", "scenario_is_user_declared": True,
            "background_note": "Wallpaper Engine running",
            "capture_status": "captured_best_effort",
            "before": before, "after": before,
            "available_memory_change": env.summarize(before, before),
        }
        self.validator.validate(result)


if __name__ == "__main__":
    unittest.main()
