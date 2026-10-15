"""Bounds and command contract for the CPU MoE/GPU cache sweep."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_moe_cache_sweep as sweep
from run_speculative_benchmark import BenchmarkError


class MoeCacheSweepTests(unittest.TestCase):
    def test_cache_candidates_require_zero_baseline_and_increasing_order(self):
        self.assertEqual(sweep.parse_cache_candidates("0, 2048, 4096"), [0, 2048, 4096])
        for value in ("", "1,2", "0,0", "2048,0", "0,-1", "0,1,2,3,4,5,6"):
            with self.subTest(value=value), self.assertRaises(BenchmarkError):
                sweep.parse_cache_candidates(value)

    def test_report_statistics_use_two_middle_values(self):
        stats = sweep._stats([1, 4])
        self.assertEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["sample_stddev"], 2.1213203436)
        self.assertAlmostEqual(stats["coefficient_of_variation_pct"], 84.85281374)
        self.assertIsNone(sweep._stats([1])["sample_stddev"])

    def test_plan_binds_cpu_moe_cache_q4_and_disables_prompt_cache(self):
        fake_model = {
            "source": {"filename": "model.gguf", "file_size_bytes": 100},
            "general": {"architecture": "qwen35moe"},
            "architecture_metadata": {"expert_count": 256},
            "split": {"multiple_parts_reported": False},
        }
        required_help = "--model --ctx-size --parallel --device --gpu-layers --cpu-moe " \
            "--moe-cache-mib --cache-type-k --cache-type-v --cache-ram --fit --fit-target " \
            "--spec-type --no-ui --host --port --batch-size --ubatch-size"
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.write_bytes(b"x")
            with patch.object(sweep, "inspect", return_value=fake_model), \
                    patch.object(sweep, "discover_binary", return_value=Path("llama.exe")), \
                    patch.object(sweep, "resolve_entrypoint", return_value="serve"), \
                    patch.object(sweep, "command_prefix", return_value=["llama.exe", "serve"]), \
                    patch.object(sweep, "_bounded", side_effect=["version", required_help,
                                                                  "CUDA0 : NVIDIA"]), \
                    patch.object(sweep, "sha256_file", return_value="a" * 64), \
                    patch.object(sweep, "_port", return_value=32000):
                plan = sweep.build_sweep_plan(
                    model_path=model_path, candidates_mib=[0, 1024, 2048],
                )
        self.assertEqual(
            sorted(plan["sweep"]["execution_order_cache_mib"]), [0, 1024, 2048]
        )
        self.assertEqual(
            plan["sweep"]["execution_order_cache_mib"],
            sweep.random.Random(42).sample([0, 1024, 2048], 3),
        )
        self.assertEqual(plan["configuration"]["cache_type_k"], "q4_0")
        self.assertEqual(plan["configuration"]["cache_type_v"], "q4_0")
        for case in plan["_cases"]:
            argv = case["_argv"]
            self.assertIn("--cpu-moe", argv)
            self.assertIn("--moe-cache-mib", argv)
            self.assertIn("--cache-ram", argv)
            self.assertEqual(argv[argv.index("--cache-ram") + 1], "0")
            self.assertEqual(argv[argv.index("--fit") + 1], "off")
            self.assertIn("--spec-type", argv)
            self.assertEqual(argv[argv.index("--spec-type") + 1], "none")
        public_plan = {key: value for key, value in plan.items() if not key.startswith("_")}
        public_plan["cases"] = [
            {key: value for key, value in case.items() if not key.startswith("_")}
            for case in plan["_cases"]
        ]
        sweep.validate_report(public_plan)

    def test_non_moe_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "dense.gguf"
            model_path.write_bytes(b"x")
            with patch.object(sweep, "inspect", return_value={
                    "architecture_metadata": {"expert_count": 0},
                    "split": {"multiple_parts_reported": False}}):
                with self.assertRaisesRegex(BenchmarkError, "not identify a MoE"):
                    sweep.build_sweep_plan(model_path=model_path)

    def test_execute_revalidates_paths_as_path_objects_and_builds_report(self):
        fake_model = {
            "source": {"filename": "model.gguf", "file_size_bytes": 100},
            "general": {"architecture": "qwen35moe"},
            "architecture_metadata": {"expert_count": 256},
            "split": {"multiple_parts_reported": False},
        }
        required_help = "--model --ctx-size --parallel --device --gpu-layers --cpu-moe " \
            "--moe-cache-mib --cache-type-k --cache-type-v --cache-ram --fit --fit-target " \
            "--spec-type --no-ui --host --port --batch-size --ubatch-size"
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.write_bytes(b"x")
            with patch.object(sweep, "inspect", return_value=fake_model), \
                    patch.object(sweep, "discover_binary", return_value=Path("llama.exe")), \
                    patch.object(sweep, "resolve_entrypoint", return_value="serve"), \
                    patch.object(sweep, "command_prefix", return_value=["llama.exe", "serve"]), \
                    patch.object(sweep, "_bounded", side_effect=["version", required_help,
                                                                  "CUDA0 : NVIDIA"]), \
                    patch.object(sweep, "sha256_file", return_value="a" * 64), \
                    patch.object(sweep, "_port", return_value=32000):
                plan = sweep.build_sweep_plan(
                    model_path=model_path, candidates_mib=[0, 1024, 2048],
                    repetitions=1, warmup_requests=0,
                )
            run = {
                "status": "completed", "requests": [{
                    "warmup": False, "decode_tokens_per_second": 4.0, "latency_ms": 250.0,
                }], "effective_gpu_layers_observed": 0,
                "effective_gpu_layers_total_observed": 41,
                "memory": {"device_reserve_status": "observed"},
                "server_log_excerpt": "loaded",
            }
            seen_order = []
            def fake_run(argv, **kwargs):
                seen_order.append(int(argv[argv.index("--moe-cache-mib") + 1]))
                return run

            with patch.object(sweep, "sha256_file", return_value="a" * 64) as digest, \
                    patch.object(sweep, "_run_scenario", side_effect=fake_run):
                report = sweep.execute_sweep(plan)
            self.assertEqual(digest.call_count, 2 * 3)
        self.assertEqual(seen_order, report["sweep"]["execution_order_cache_mib"])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["cases"][0]["measurement"]["decode_tokens_per_second"]["mean"], 4.0)
        sweep.validate_report(report)


if __name__ == "__main__":
    unittest.main()
