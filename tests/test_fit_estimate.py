"""Controlled fit estimate planning and subprocess mocking; no model inference."""
import json
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from jsonschema import Draft202012Validator

from engine_identity import sha256_file
from fit_estimate import FitPlanError, build_plan, execute_plan


def write_fake_gguf(path, *, shards=1):
    def gs(s):
        value = s.encode("utf-8")
        return struct.pack("<Q", len(value)) + value
    pairs = [("general.architecture", 8, "qwen35"), ("qwen35.block_count", 4, 65)]
    if shards > 1:
        pairs.append(("split.count", 4, shards))
    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs))
    for key, kind, value in pairs:
        blob += gs(key) + struct.pack("<I", kind)
        blob += gs(value) if kind == 8 else struct.pack("<I", value)
    path.write_bytes(blob)


class FitEstimateTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "synthetic model.gguf"
        write_fake_gguf(self.model)
        self.binary = self.root / "llama.exe"
        self.binary.write_bytes(b"fake llama binary")
        self.sha = sha256_file(self.binary)
        self.fit_snapshot = self.root / "fit.json"
        self.device_snapshot = self.root / "devices.json"
        fit = {
            "schema_version": "1.1.0", "kind": "runtime_help_observation",
            "binary": {"sha256": self.sha},
            "identity": {"binary_sha256": self.sha},
            "entrypoint": {"kind": "integrated", "command": "fit-params"},
            "command_results": {
                "version": {"status": "success"},
                "help": {"status": "success", "flag_names_observed": [
                    "--model", "--fit-print", "--fit", "--ctx-size",
                    "--n-gpu-layers", "--device", "--offline",
                ]},
            },
        }
        device = {
            "schema_version": "1.1.0", "kind": "runtime_help_observation",
            "binary": {"sha256": self.sha},
            "identity": {"binary_sha256": self.sha},
            "entrypoint": {"kind": "integrated", "command": "serve"},
            "command_results": {
                "version": {"status": "success"},
                "help": {"status": "success", "flag_names_observed": ["--list-devices"]},
                "devices": {"status": "success"},
            },
            "device_inventory": {"status": "observed", "devices": [
                {"id": "Vulkan0", "description": "Synthetic", "total_memory_mib": 8000,
                 "free_memory_mib": 7500},
            ]},
        }
        schema = json.loads((ROOT / "schemas/fit-estimate-plan.schema.json").read_text("utf-8"))
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)
        self.fit_snapshot.write_text(json.dumps(fit), "utf-8")
        self.device_snapshot.write_text(json.dumps(device), "utf-8")

    def make(self, **kwargs):
        return build_plan(self.model, self.fit_snapshot, binary=self.binary, **kwargs)

    def test_default_is_cpu_preview_and_no_engine_launch(self):
        with patch("fit_estimate.subprocess.Popen") as process:
            out = self.make()
            process.assert_not_called()
        self.assertEqual(out["status"], "prepared_not_executed")
        self.validator.validate(out)
        self.assertIsNone(out["estimate"])
        self.assertEqual(out["candidate"]["device_id"], "none")
        self.assertEqual(out["argv"][1:3], ["fit-params", "--offline"])
        self.assertEqual(out["argv"][-4:], ["--fit", "off", "--fit-print", "on"])

    def test_gpu_plan_requires_matching_snapshot_and_explicit_layers(self):
        result = self.make(device="Vulkan0", gpu_layers=18,
                           devices_snapshot=self.device_snapshot)
        self.assertIn("Vulkan0", result["argv"])
        self.assertEqual(result["candidate"]["gpu_layers"], 18)
        with self.assertRaisesRegex(FitPlanError, "device snapshot"):
            self.make(device="Vulkan0", gpu_layers=18)
        with self.assertRaisesRegex(FitPlanError, "positive"):
            self.make(device="Vulkan0", devices_snapshot=self.device_snapshot)
        with self.assertRaisesRegex(FitPlanError, "CPU-only"):
            self.make(device="none", gpu_layers=18)

    def test_reject_stale_binary_and_missing_required_flags(self):
        self.binary.write_bytes(b"changed binary")
        with self.assertRaisesRegex(FitPlanError, "another engine"):
            self.make()
        self.binary.write_bytes(b"fake llama binary")
        obj = json.loads(self.fit_snapshot.read_text("utf-8"))
        obj["command_results"]["help"]["flag_names_observed"].remove("--offline")
        self.fit_snapshot.write_text(json.dumps(obj), "utf-8")
        with self.assertRaisesRegex(FitPlanError, "missing required flags"):
            self.make()

    def test_reject_unobserved_device(self):
        with self.assertRaisesRegex(FitPlanError, "not present"):
            self.make(device="CUDA0", gpu_layers=18,
                      devices_snapshot=self.device_snapshot)
        obj = json.loads(self.device_snapshot.read_text("utf-8"))
        obj["device_inventory"]["status"] = "unknown"
        obj["device_inventory"]["devices"] = None
        self.device_snapshot.write_text(json.dumps(obj), "utf-8")
        with self.assertRaisesRegex(FitPlanError, "no confirmed"):
            self.make(device="Vulkan0", gpu_layers=18,
                      devices_snapshot=self.device_snapshot)

    def test_context_and_layer_bounds(self):
        for context in (0, 127, 262145):
            with self.assertRaises(FitPlanError):
                self.make(context=context)
        for layers in (-1, 4097):
            with self.assertRaises(FitPlanError):
                self.make(gpu_layers=layers)
        with self.assertRaises(FitPlanError):
            self.make(device="Vulkan0,CUDA0", gpu_layers=18)

    def test_reject_multishard_model_until_validation(self):
        write_fake_gguf(self.model, shards=2)
        with self.assertRaisesRegex(FitPlanError, "multi-part"):
            self.make()

    def test_explicit_execution_uses_argv_without_shell(self):
        plan = self.make()
        class Dummy:
            def wait(self, timeout=None):
                return 0
        with patch("fit_estimate.subprocess.Popen", return_value=Dummy()) as launch:
            out = execute_plan(plan, timeout_seconds=10)
        self.assertEqual(out["status"], "completed")
        # execute_plan only returns an estimate; main() wraps it in environment
        # snapshots before creating an estimate_attempted full plan.
        self.assertEqual(out["status"], "completed")
        self.assertFalse(out["inference_tested"])
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0], plan["argv"])
        self.assertNotIn("shell", launch.call_args.kwargs)
        self.assertNotIn("LLAMA_ARG_DEVICE", launch.call_args.kwargs["env"])

    def test_fingerprint_is_part_of_new_model_descriptor(self):
        plan = self.make()
        identity = plan["model"]["fingerprint"]
        self.assertEqual(identity["method"], "metadata_prefix_and_stat_v1")
        self.assertEqual(len(identity["metadata_sha256"]), 64)
        self.assertFalse(identity["full_file_hashed"])
        self.assertIsNone(identity["full_file_sha256"])
        self.validator.validate(plan)

    def test_mutation_with_same_file_size_is_rejected_before_model_run(self):
        plan = self.make()
        raw = self.model.read_bytes()
        changed = raw.replace(b"qwen35", b"qwen36")
        self.assertEqual(len(raw), len(changed))
        self.model.write_bytes(changed)
        with patch("fit_estimate.subprocess.Popen") as process:
            with self.assertRaisesRegex(FitPlanError, "model fingerprint changed"):
                execute_plan(plan)
            process.assert_not_called()

    def test_reject_execution_if_model_or_binary_changed(self):
        plan = self.make()
        self.model.write_bytes(self.model.read_bytes() + b"extra")
        with patch("fit_estimate.subprocess.Popen") as process:
            with self.assertRaisesRegex(FitPlanError, "model file size changed"):
                execute_plan(plan)
            process.assert_not_called()

    def test_reject_flag_shaped_device_id(self):
        with self.assertRaisesRegex(FitPlanError, "GGML device"):
            self.make(device="--version", gpu_layers=1)

    def test_output_path_never_overwrites_model(self):
        from fit_estimate import main
        with patch("sys.argv", ["fit_estimate.py",
                                "--model", str(self.model),
                                "--fit-snapshot", str(self.fit_snapshot),
                                "--binary", str(self.binary),
                                "--output", str(self.model)]):
            self.assertEqual(main(), 1)
        self.assertTrue(self.model.read_bytes().startswith(b"GGUF"))

    def test_real_stdout_is_parsed_during_opt_in_execution(self):
        plan = self.make(device="Vulkan0", gpu_layers=18,
                         devices_snapshot=self.device_snapshot)
        raw = b"Vulkan0 4136 101 525 \r\nHost 9111 304 44 \r\n"

        class Dummy:
            def wait(self, timeout=None):
                return 0

        def launch(argv, **kwargs):
            kwargs["stdout"].write(raw)
            kwargs["stderr"].write(b"estimate completed\n")
            return Dummy()

        with patch("fit_estimate.subprocess.Popen", side_effect=launch):
            outcome = execute_plan(plan, timeout_seconds=10)
        self.assertEqual(outcome["status"], "completed")
        self.assertFalse(outcome["output_truncated_or_unknown"])
        self.assertEqual(outcome["parsed_numeric_memory"]["devices"][0]["estimated_total_mib"], 4762)
        self.assertEqual(outcome["parsed_numeric_memory"]["host"]["estimated_total_mib"], 9459)
        self.assertEqual(outcome["parsed_numeric_memory"]["estimated_total_mib"], 14221)

    def test_nonzero_output_is_not_treated_as_numeric_memory(self):
        plan = self.make()
        class Dummy:
            def wait(self, timeout=None):
                return 1
        def launch(argv, **kwargs):
            kwargs["stdout"].write(b"Host 100 20 30\n")
            return Dummy()
        with patch("fit_estimate.subprocess.Popen", side_effect=launch):
            out = execute_plan(plan)
        self.assertEqual(out["status"], "nonzero_exit")
        self.assertIsNone(out["parsed_numeric_memory"])

    def test_process_launch_error_returns_schema_valid_record(self):
        plan = self.make()
        with patch("fit_estimate.subprocess.Popen", side_effect=OSError("mock failure")):
            out = execute_plan(plan)
        self.assertEqual(out["status"], "process_error")
        self.assertIsNone(out["parsed_numeric_memory"])
        self.assertEqual(out["stderr_excerpt"], "OSError")

    def test_scenario_is_user_declared_even_in_preview(self):
        plan = self.make(scenario="reduced_background",
                         background_note="Wallpaper Engine manually paused")
        self.validator.validate(plan)
        self.assertEqual(plan["environment"]["scenario"], "reduced_background")
        self.assertTrue(plan["environment"]["scenario_is_user_declared"])
        self.assertEqual(plan["environment"]["capture_status"], "not_sampled")
        self.assertIsNone(plan["environment"]["before"])
        with self.assertRaises(FitPlanError):
            self.make(scenario="autoclose_wallpaper")
        with self.assertRaises(FitPlanError):
            self.make(background_note="X" * 301)

    def test_execute_main_brackets_estimate_with_memory_snapshots(self):
        from fit_estimate import main
        from environment_observation import summarize
        ram = {"status": "observed", "reason": None,
               "total_mib": 16000, "available_mib": 12000}
        gpu = {"status": "observed", "reason": None, "device_id": "Vulkan0",
               "backend_id": "Vulkan", "description": "Synthetic",
               "total_mib": 8000, "available_mib": 7500}
        before = {"timestamp_utc": "2026-10-14T01:00:00+00:00",
                  "ram": ram, "gpu": gpu}
        after = {"timestamp_utc": "2026-10-14T01:01:00+00:00",
                 "ram": dict(ram, available_mib=11900),
                 "gpu": dict(gpu, available_mib=7450)}
        estimate = {"status": "completed", "returncode": 0,
                    "stdout_excerpt": "synthetic upstream memory estimate",
                    "stderr_excerpt": "", "output_truncated_or_unknown": True,
                    "estimate_unit": "MiB (as reported by upstream, if recognizable)",
                    "parsed_numeric_memory": None, "model_load_tested": False,
                    "inference_tested": False}
        destination = self.root / "daily.json"
        argv = ["fit_estimate.py", "--model", str(self.model),
                "--fit-snapshot", str(self.fit_snapshot),
                "--device-snapshot", str(self.device_snapshot),
                "--device", "Vulkan0", "--gpu-layers", "18",
                "--binary", str(self.binary), "--scenario", "daily",
                "--background-note", "Wallpaper Engine running",
                "--execute", "--output", str(destination)]
        with patch("sys.argv", argv), \
             patch("fit_estimate.observe_environment", side_effect=[before, after]) as observe, \
             patch("fit_estimate.execute_plan", return_value=estimate) as execute:
            self.assertEqual(main(), 0)
        self.assertEqual(observe.call_count, 2)
        execute.assert_called_once()
        output = json.loads(destination.read_text("utf-8"))
        self.validator.validate(output)
        self.assertEqual(output["environment"]["capture_status"], "captured_best_effort")
        self.assertEqual(output["environment"]["available_memory_change"]["gpu_available_delta_mib"], -50)
        self.assertEqual(output["environment"]["available_memory_change"]["system_ram_available_delta_mib"], -100)
        self.assertIsNone(output["environment"]["available_memory_change"]["peak_gpu_usage_mib"])
        self.assertEqual(output["environment"]["background_note"], "Wallpaper Engine running")
        self.assertEqual(output["status"], "estimate_attempted")

    def test_corrupted_snapshot_fails_closed(self):
        self.fit_snapshot.write_text('{"binary": null}', "utf-8")
        with self.assertRaises(FitPlanError):
            self.make()


if __name__ == "__main__":
    unittest.main()
