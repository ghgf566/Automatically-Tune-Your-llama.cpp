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
            "binary": {"sha256": self.sha},
            "identity": {"binary_sha256": self.sha},
            "entrypoint": {"kind": "integrated", "command": "serve"},
            "command_results": {"devices": {"status": "success"}},
            "device_inventory": {"status": "observed", "devices": [
                {"id": "Vulkan0", "description": "Synthetic", "total_memory_mib": 8000,
                 "free_memory_mib": 7500},
            ]},
        }
        self.fit_snapshot.write_text(json.dumps(fit), "utf-8")
        self.device_snapshot.write_text(json.dumps(device), "utf-8")

    def make(self, **kwargs):
        return build_plan(self.model, self.fit_snapshot, binary=self.binary, **kwargs)

    def test_default_is_cpu_preview_and_no_engine_launch(self):
        with patch("fit_estimate.subprocess.Popen") as process:
            out = self.make()
            process.assert_not_called()
        self.assertEqual(out["status"], "prepared_not_executed")
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
        self.assertFalse(out["inference_tested"])
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0], plan["argv"])
        self.assertNotIn("shell", launch.call_args.kwargs)
        self.assertNotIn("LLAMA_ARG_DEVICE", launch.call_args.kwargs["env"])

    def test_reject_execution_if_model_or_binary_changed(self):
        plan = self.make()
        self.model.write_bytes(self.model.read_bytes() + b"extra")
        with patch("fit_estimate.subprocess.Popen") as process:
            with self.assertRaisesRegex(FitPlanError, "model file size changed"):
                execute_plan(plan)
            process.assert_not_called()

    def test_corrupted_snapshot_fails_closed(self):
        self.fit_snapshot.write_text('{"binary": null}', "utf-8")
        with self.assertRaises(FitPlanError):
            self.make()


if __name__ == "__main__":
    unittest.main()
