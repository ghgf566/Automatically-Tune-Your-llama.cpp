"""Scenario comparison uses synthetic JSON; never requires llama or GPU."""
import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compare_scenarios import compare, ComparisonError


def record(scenario, ram, gpu):
    def sample(r, g):
        return {"ram": {"status": "observed", "available_mib": r},
                "gpu": {"status": "observed", "available_mib": g, "device_id": "Vulkan0"}}
    return {
        "kind": "fit_estimate_plan", "status": "estimate_attempted",
        "engine_sha256": "a" * 64,
        "model": {"filename": "synthetic.gguf", "file_size_bytes": 999},
        "candidate": {"device_id": "Vulkan0", "context_tokens": 4096, "gpu_layers": 18,
                      "fit_adjustment": "off", "output_mode": "estimated_memory"},
        "estimate": {"status": "completed"},
        "environment": {
            "scenario": scenario, "capture_status": "captured_best_effort",
            "background_note": scenario, "before": sample(ram, gpu),
            "after": sample(ram - 20, gpu - 10),
        },
    }


class ScenarioComparisonTests(unittest.TestCase):
    def test_compares_headroom_with_explicit_source(self):
        daily = record("daily", 10000, 7000)
        reduced = record("reduced_background", 11000, 7600)
        result = compare(daily, reduced)
        self.assertEqual(result["before_estimate"]["gpu"]["reduced_minus_daily_available_mib"], 600)
        self.assertEqual(result["after_estimate"]["ram"]["reduced_minus_daily_available_mib"], 1000)
        self.assertEqual(result["estimate_outcomes"]["daily"], "completed")
        self.assertFalse("peak_gpu_usage_mib" in result)

    def test_mismatched_candidate_is_rejected(self):
        a, b = record("daily", 10000, 7000), record("reduced_background", 10000, 7000)
        b["candidate"]["gpu_layers"] = 24
        with self.assertRaisesRegex(ComparisonError, "candidate differs"):
            compare(a, b)

    def test_mismatched_engine_and_model_are_rejected(self):
        a, b = record("daily", 10000, 7000), record("reduced_background", 10000, 7000)
        b["engine_sha256"] = "b" * 64
        with self.assertRaisesRegex(ComparisonError, "runtime binary"):
            compare(a, b)
        b["engine_sha256"] = "a" * 64
        b["model"]["file_size_bytes"] = 2
        with self.assertRaisesRegex(ComparisonError, "model descriptors"):
            compare(a, b)

    def test_different_scenario_labels_rejected(self):
        a,b = record("daily", 10000, 7000),record("daily", 11000, 7000)
        with self.assertRaisesRegex(ComparisonError, "matching scenario"):
            compare(a,b)

    def test_unknown_gpu_does_not_turn_into_zero(self):
        a, b = record("daily", 10000, 7000), record("reduced_background", 11000, 8000)
        b["environment"]["before"]["gpu"] = {"status": "unknown", "available_mib": None}
        result = compare(a,b)
        self.assertIsNone(result["before_estimate"]["gpu"]["reduced_minus_daily_available_mib"])
        self.assertIsNone(result["before_estimate"]["gpu"]["reduced_background_available_mib"])

    def test_false_observation_status_rejected(self):
        a, b = record("daily", 10000, 7000), record("reduced_background", 11000, 8000)
        a["environment"]["capture_status"] = "not_sampled"
        with self.assertRaisesRegex(ComparisonError, "before/after"):
            compare(a,b)


if __name__ == "__main__":
    unittest.main()
