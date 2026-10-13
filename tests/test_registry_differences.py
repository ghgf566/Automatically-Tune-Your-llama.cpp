"""Reduced, version-pinned CLI evidence interpreted without overclaiming capability."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from registry_differences import interpret_differences, inspect_fit_reference

FIXTURE = json.loads((ROOT / "tests/fixtures/windows-cli-evidence.json").read_text("utf-8"))
REFERENCE = json.loads((ROOT / "data/fit-capability-reference.json").read_text("utf-8"))
INVENTORY = json.loads((ROOT / "data/cli-inventory.json").read_text("utf-8"))


class DifferenceInterpretationTest(unittest.TestCase):
    def test_upstream_and_curated_reference_stay_pinned(self):
        self.assertEqual(REFERENCE["upstream"]["commit"], INVENTORY["upstream"]["commit"])
        self.assertEqual(REFERENCE["scope"], "curated_fit_params_only")
        self.assertEqual(len({item["flag"] for item in REFERENCE["entries"]}),
                         len(REFERENCE["entries"]))

    def test_bench_rpc_is_build_conditional_not_declared_unsupported(self):
        result = interpret_differences("llama-bench", {
            "documented_not_observed_in_help": ["--rpc"],
            "unregistered_flags_observed_in_help": ["--offline", "--version"],
        })
        self.assertEqual(result["missing_documented"][0]["classification"],
                         "conditional_in_upstream")
        self.assertEqual(result["missing_documented"][0]["condition"], "llama_supports_rpc()")
        self.assertEqual(result["missing_documented"][0]["support"], "not_confirmed")
        self.assertEqual({x["flag"] for x in result["unregistered_observed"]},
                         {"--offline", "--version"})
        self.assertTrue(all(x["classification"] == "not_in_pinned_inventory"
                            for x in result["unregistered_observed"]))

    def test_other_absences_are_unconfirmed(self):
        got = interpret_differences("llama-server", {
            "documented_not_observed_in_help": ["--some-flag"],
            "unregistered_flags_observed_in_help": [],
        })
        self.assertEqual(got["missing_documented"][0]["classification"],
                         "not_observed_in_help")
        self.assertEqual(got["missing_documented"][0]["support"], "not_confirmed")

    def test_failed_help_remains_unknown(self):
        out = interpret_differences("llama-bench", {
            "documented_not_observed_in_help": None,
            "unregistered_flags_observed_in_help": None,
        })
        self.assertEqual(out["status"], "unknown")
        self.assertIsNone(out["missing_documented"])

    def test_fit_reference_matches_real_observed_subset(self):
        flags = FIXTURE["command_evidence"]["fit-params"]["observed_flags"]
        got = inspect_fit_reference(flags, REFERENCE, INVENTORY["upstream"]["commit"])
        self.assertEqual(got["status"], "curated_reference_checked")
        self.assertIn("--fit-print", got["observed"])
        self.assertIn("--list-devices", got["observed"])
        self.assertIn("--tensor-split", got["not_observed"])
        self.assertNotIn("--some-unregistered-flag", got["observed"])

    def test_fit_reference_never_invents_evidence(self):
        bad = inspect_fit_reference(None, REFERENCE, INVENTORY["upstream"]["commit"])
        self.assertEqual(bad["status"], "unknown")
        different = inspect_fit_reference(
            ["--fit-print"], REFERENCE, "f" * 40)
        self.assertEqual(different["status"], "reference_version_mismatch")
        self.assertIsNone(different["observed"])

    def test_fit_print_is_an_estimate_mode_not_fitted_arguments(self):
        entry = next(x for x in REFERENCE["entries"] if x["flag"] == "--fit-print")
        self.assertEqual(entry["role"], "estimated_memory_report")
        self.assertIn("instead of", entry["semantics"])


if __name__ == "__main__":
    unittest.main()
