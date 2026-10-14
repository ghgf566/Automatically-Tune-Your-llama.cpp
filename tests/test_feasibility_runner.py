"""Feasibility plan tests use synthetic GGUF metadata and a fake engine file."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from plan_speculative_search import make_preview
from run_feasibility import (
    FeasibilityError, REPORT_SCHEMA, REPORT_VALIDATOR, build_plan,
)
from tests.test_gguf_inspection import synthetic_gguf
from inspect_gguf import inspect


class FeasibilityRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Draft202012Validator.check_schema(REPORT_SCHEMA)

    def _fixture(self, temp):
        target = Path(temp) / "target.gguf"
        target.write_bytes(synthetic_gguf([
            ("general.architecture", 8, "qwen3moe"),
            ("qwen3moe.nextn_predict_layers", 4, 2),
            ("tokenizer.ggml.model", 8, "gpt2"),
            ("tokenizer.ggml.tokens", 9, ["a", "b", "c"]),
        ], tensor=False))
        engine = Path(temp) / "llama.exe"
        engine.write_bytes(b"synthetic executable identity; never launched")
        engine_sha = hashlib.sha256(engine.read_bytes()).hexdigest()
        profile = copy.deepcopy(json.loads(
            (ROOT / "examples/workload-profile.speculative.example.json").read_text("utf-8")))
        profile["engine"]["binary"] = str(engine)
        profile["engine"]["binary_sha256"] = engine_sha
        profile["model"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        profile["model"]["architecture"] = "qwen3moe"
        profile["candidate_parameters"].update({"--n-gpu-layers": 0, "--device": "none"})
        profile["speculative_search"]["modes"] = ["off", "embedded_mtp"]
        profile["speculative_search"]["draft_n_max_candidates"] = [2]
        profile["speculative_search"]["external_artifacts"] = []
        profile["speculative_search"]["draft_placements"] = []
        observation = inspect(target)
        flags = [
            "--model", "--ctx-size", "--parallel", "--host", "--port",
            "--fit", "--ui", "--no-ui", "--no-webui", "--spec-type",
            "--spec-draft-n-max", "--n-gpu-layers", "--device", "--threads",
            "--cache-type-k", "--cache-type-v",
        ]
        runtime = {
            "schema_version": "1.1.0", "kind": "runtime_help_observation",
            "binary": {"sha256": engine_sha},
            "identity": {"binary_sha256": engine_sha},
            "entrypoint": {"kind": "integrated", "command": "serve"},
            "registry_commit": profile["registry_upstream_commit"],
            "command_results": {
                "version": {"status": "success"},
                "help": {"status": "success", "flag_names_observed": flags},
            },
        }
        preview = make_preview(profile["speculative_search"], observation, runtime,
                               profile=profile)
        return profile, preview, observation, runtime, target, engine

    def test_off_candidate_plan_is_local_bounded_and_non_executable(self):
        with TemporaryDirectory() as temp:
            profile, preview, observation, runtime, target, engine = self._fixture(temp)
            candidate = next(item for item in preview["candidates"] if item["mode"] == "off")
            plan = build_plan(profile, preview, observation, runtime,
                              candidate["candidate_id"], target, binary=engine)
            argv = plan.pop("_argv")
        REPORT_VALIDATOR.validate(plan)
        self.assertEqual(plan["status"], "prepared_not_executed")
        self.assertEqual(plan["host"], "127.0.0.1")
        self.assertEqual(plan["mode"], "off")
        self.assertIn("--no-ui", plan["launched_argv_redacted"])
        self.assertIn("--fit", argv)
        self.assertIn("off", argv)
        self.assertIn("127.0.0.1", argv)
        self.assertNotIn(str(target), plan["launched_argv_redacted"])

    def test_profile_and_preview_identity_must_match(self):
        with TemporaryDirectory() as temp:
            profile, preview, observation, runtime, target, engine = self._fixture(temp)
            profile["profile_id"] = "different"
            candidate = next(item for item in preview["candidates"] if item["mode"] == "off")
            with self.assertRaisesRegex(FeasibilityError, "different Profile"):
                build_plan(profile, preview, observation, runtime,
                           candidate["candidate_id"], target, binary=engine)

    def test_unobserved_safe_bind_flags_block_before_plan(self):
        with TemporaryDirectory() as temp:
            profile, preview, observation, runtime, target, engine = self._fixture(temp)
            runtime["command_results"]["help"]["flag_names_observed"].remove("--no-ui")
            runtime["command_results"]["help"]["flag_names_observed"].remove("--no-webui")
            candidate = next(item for item in preview["candidates"] if item["mode"] == "off")
            with self.assertRaisesRegex(FeasibilityError, "UI-disable flag"):
                build_plan(profile, preview, observation, runtime,
                           candidate["candidate_id"], target, binary=engine)

    def test_execute_waits_for_fake_local_health_and_reports_load_only(self):
        from run_feasibility import execute_plan

        with TemporaryDirectory() as temp:
            profile, preview, observation, runtime, target, engine = self._fixture(temp)
            candidate = next(item for item in preview["candidates"] if item["mode"] == "off")
            plan = build_plan(profile, preview, observation, runtime,
                              candidate["candidate_id"], target, binary=engine,
                              timeout_seconds=10)
            script = (
                "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
                "import json\n"
                "class H(BaseHTTPRequestHandler):\n"
                " def do_GET(self):\n"
                "  self.send_response(200);self.send_header('Content-Type','application/json');"
                "self.end_headers();self.wfile.write(b'{\\\"status\\\":\\\"ok\\\"}')\n"
                " def log_message(self,*args): pass\n"
                f"HTTPServer(('127.0.0.1',{plan['port']}),H).serve_forever()"
            )
            plan["_argv"] = [sys.executable, "-u", "-c", script]
            result = execute_plan(plan)
        self.assertEqual(result["status"], "load_verified", result["load_result"])
        self.assertTrue(result["load_result"]["ready_endpoint_verified"])
        self.assertEqual(result["memory_observation"]["host_process_memory_sampling_status"],
                         "observed")
        self.assertGreater(result["memory_observation"]["host_process_memory_sample_count"], 0)
        self.assertGreater(result["memory_observation"]["host_peak_process_mib"], 0)
        self.assertIsNone(result["memory_observation"]["device_peak_process_mib"])
        self.assertEqual(result["memory_observation"]["device_memory_observation"]["reason"],
                         "gpu_offload_not_requested")
        REPORT_VALIDATOR.validate(result)

    def test_gpu_memory_observation_is_optional_and_schema_valid(self):
        from run_feasibility import execute_plan

        class FakeMonitor:
            def start(self):
                pass

            def stop_and_report(self):
                return {
                    "status": "observed", "reason": None, "provider": "nvidia_smi",
                    "process_attribution": "system_wide", "sample_count": 1,
                    "sample_interval_ms": 1000,
                    "devices": [{
                        "uuid": "GPU-abc-123", "name": "NVIDIA Example",
                        "total_memory_mib": 8192, "minimum_free_memory_mib": 7000,
                        "maximum_used_memory_mib": 1000, "sample_count": 1,
                    }],
                }

        with TemporaryDirectory() as temp:
            profile, preview, observation, runtime, target, engine = self._fixture(temp)
            candidate = next(item for item in preview["candidates"] if item["mode"] == "off")
            plan = build_plan(profile, preview, observation, runtime,
                              candidate["candidate_id"], target, binary=engine,
                              timeout_seconds=10)
            script = (
                "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
                "class H(BaseHTTPRequestHandler):\n"
                " def do_GET(self):\n"
                "  self.send_response(200);self.end_headers();self.wfile.write(b'{\"status\":\"ok\"}')\n"
                " def log_message(self,*args): pass\n"
                f"HTTPServer(('127.0.0.1',{plan['port']}),H).serve_forever()"
            )
            plan["_argv"] = [sys.executable, "-u", "-c", script, "--gpu-layers", "1"]
            with patch("run_feasibility.DeviceMemoryMonitor", FakeMonitor):
                result = execute_plan(plan)

        self.assertEqual(result["memory_observation"]["device_memory_observation"]["status"],
                         "observed")
        self.assertEqual(result["memory_observation"]["device_memory_observation"]["devices"][0][
            "minimum_free_memory_mib"], 7000)
        self.assertIsNone(result["memory_observation"]["device_peak_process_mib"])
        REPORT_VALIDATOR.validate(result)


if __name__ == "__main__":
    unittest.main()
