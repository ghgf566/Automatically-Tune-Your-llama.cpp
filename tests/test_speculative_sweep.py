"""Bounds and pairing rules for the explicit speculative sweep."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_speculative_sweep as sweep
from run_speculative_benchmark import BenchmarkError


def _base_plan():
    common = ["llama", "serve", "--port", "32000", "--spec-type"]
    off = common + ["none"]
    mtp = common + ["draft-mtp", "--spec-draft-n-max", "4"]
    return {
        "schema_version": "0.1.0", "kind": "speculative_server_benchmark",
        "status": "prepared_not_executed", "engine_sha256": "a" * 64,
        "engine_version_excerpt": "version", "engine_commit": "commit",
        "model": {"filename": "model.gguf", "file_size_bytes": 100,
                  "architecture": "test", "nextn_predict_layers": 1,
                  "sha256": "b" * 64},
        "configuration": {"context_tokens": 4096, "concurrency": 1,
                          "device_id": "CUDA0", "gpu_layers": 16,
                          "n_cpu_ffn": None,
                          "cache_type_k": "q4_0", "cache_type_v": "q4_0",
                          "kv_offload": True,
                          "batch_size": 512, "ubatch_size": 128,
                          "fit": "on", "device_fit_target_mib": 1024,
                          "host_reserve_mib": 4096, "draft_n_max": 4,
                          "mode_order": "off_first"},
        "workload": {"workload_id": "short_prompt", "prompt_sha256": "c" * 64},
        "variants": [
            {"mode": "off", "port": 32000, "argv_redacted": off},
            {"mode": "embedded_mtp", "port": 32000, "argv_redacted": mtp},
        ],
        "runs": [], "summary": None,
        "_variants": [
            {"mode": "off", "port": 32000, "argv": off.copy(), "argv_redacted": off.copy()},
            {"mode": "embedded_mtp", "port": 32000, "argv": mtp.copy(), "argv_redacted": mtp.copy()},
        ],
        "_binary": "llama.exe", "_model_path": "model.gguf",
        "_max_run_seconds": 100, "_prompt": "synthetic prompt",
        "_seed": 42,
    }


class SweepValidationTests(unittest.TestCase):
    def test_candidates_are_strictly_increasing_and_bounded(self):
        self.assertEqual(sweep.parse_draft_candidates("1, 2, 4"), [1, 2, 4])
        for value in ("", "1,1", "2,1", "0", "1,2,3,4,5", "65"):
            with self.subTest(value=value), self.assertRaises(BenchmarkError):
                sweep.parse_draft_candidates(value)

    def test_workloads_reject_unknown_and_duplicates(self):
        self.assertEqual(sweep.parse_workloads("short_prompt,code_generation"),
                         ["short_prompt", "code_generation"])
        for value in ("missing", "short_prompt,short_prompt"):
            with self.subTest(value=value), self.assertRaises(BenchmarkError):
                sweep.parse_workloads(value)

    def test_case_plan_binds_workload_n_order_and_port(self):
        plan = sweep._case_plan(_base_plan(), "code_generation", 2,
                                "mtp_first", 32123)
        self.assertEqual(plan["configuration"]["draft_n_max"], 2)
        self.assertEqual(plan["workload"]["workload_id"], "code_generation")
        self.assertEqual(plan["_variants"][0]["mode"], "embedded_mtp")
        self.assertEqual(plan["_variants"][0]["argv"][-1], "2")
        self.assertEqual(plan["_variants"][0]["argv_redacted"][-1], "2")
        self.assertEqual(plan["_variants"][0]["port"], 32123)
        self.assertEqual(plan["_variants"][1]["mode"], "off")

    def test_sweep_plan_caps_case_count_and_alternates_mode_order(self):
        with patch.object(sweep, "build_plan", return_value=_base_plan()), \
                patch.object(sweep, "_port", side_effect=[32001, 32002, 32003, 32004]):
            plan = sweep.build_sweep_plan(
                model_path="model.gguf", draft_n_candidates=[1, 2],
                workloads=["short_prompt", "code_generation"],
            )
        self.assertEqual(plan["sweep"]["comparison_count"], 4)
        self.assertEqual([case["mode_order"] for case in plan["_cases"]],
                         ["off_first", "mtp_first", "mtp_first", "off_first"])
        public_plan = {key: value for key, value in plan.items() if not key.startswith("_")}
        sweep.validate_sweep_report(public_plan)
        with patch.object(sweep, "build_plan", return_value=_base_plan()), \
                patch.object(sweep, "MAX_SWEEP_CASES", 8):
            with self.assertRaisesRegex(BenchmarkError, "maximum is 8"):
                sweep.build_sweep_plan(
                    model_path="model.gguf", draft_n_candidates=[1, 2, 4],
                    workloads=list(sweep.WORKLOADS),
                )

    def test_sweep_plumbs_attention_first_and_kv_offload_configuration(self):
        with patch.object(sweep, "build_plan", return_value=_base_plan()) as build, \
                patch.object(sweep, "_port", side_effect=[32001]):
            plan = sweep.build_sweep_plan(
                model_path="model.gguf", draft_n_candidates=[1],
                workloads=["short_prompt"], gpu_layers="all", n_cpu_ffn=65,
                cache_type_k="q4_0", cache_type_v="q4_0",
            )
        self.assertEqual(build.call_args.kwargs["gpu_layers"], "all")
        self.assertEqual(build.call_args.kwargs["n_cpu_ffn"], 65)
        self.assertEqual(build.call_args.kwargs["cache_type_k"], "q4_0")
        self.assertEqual(build.call_args.kwargs["cache_type_v"], "q4_0")
        sweep.validate_sweep_report(
            {key: value for key, value in plan.items() if not key.startswith("_")}
        )

    def test_execute_sweep_returns_updated_case_records(self):
        with patch.object(sweep, "build_plan", return_value=_base_plan()), \
                patch.object(sweep, "_port", side_effect=[32001]):
            plan = sweep.build_sweep_plan(
                model_path="model.gguf", draft_n_candidates=[1],
                workloads=["short_prompt"], max_total_seconds=10,
            )
        result = {
            "status": "completed",
            "summary": {"comparison_status": "comparable"},
            "runs": [],
        }
        with patch.object(sweep, "execute_plan", return_value=result):
            report = sweep.execute_sweep(plan)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["cases"][0]["status"], "completed")
        self.assertEqual(report["cases"][0]["comparison_status"], "comparable")


if __name__ == "__main__":
    unittest.main()
