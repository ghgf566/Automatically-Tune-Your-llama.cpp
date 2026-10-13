"""Regression tests based on *reduced* real Windows llama.app CLI observations."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from engine_identity import observed_identity, parse_reported_build
from probe_runtime import flags_from_help
from device_inventory import parse_list_devices

FIXTURE = json.loads((ROOT / "tests/fixtures/windows-cli-evidence.json").read_text("utf-8"))


class RealWorldEvidenceTests(unittest.TestCase):
    def test_parse_exact_build_11574_version(self):
        version = parse_reported_build(FIXTURE["version_output"])
        self.assertEqual(version["engine_version_label"], "0.6.0-dev")
        self.assertEqual(version["engine_build_number"], 11574)
        self.assertEqual(version["engine_build_commit"], "8bb3cf0d2")
        self.assertEqual(version["engine_build_commit_form"], "abbreviated")

    def test_keep_registry_commit_separate(self):
        result = observed_identity(
            "a" * 64, "bench", "77c4540ce4667f4577753fe034a3a8042f510b8f",
            {"status": "success", "stdout": FIXTURE["version_output"], "stderr": ""}, True)
        self.assertEqual(result["engine_build_commit"], "8bb3cf0d2")
        self.assertEqual(result["registry_upstream_commit"],
                         "77c4540ce4667f4577753fe034a3a8042f510b8f")
        self.assertNotEqual(result["engine_build_commit"], result["registry_upstream_commit"])

    def test_unknown_version_does_not_invent_commit(self):
        for version in ("llama build 777", "version: invalid (commit maybe)", "",
                        "version: 1.0 (build 10, commit abc)\n"
                        "version: 1.0 (build 11, commit abcdef0123)"):
            with self.subTest(version=version):
                self.assertIsNone(parse_reported_build(version)["engine_build_commit"])

    def test_incomplete_prefix_from_real_help_is_not_a_flag(self):
        names = FIXTURE["command_evidence"]["serve"]
        raw = "\n".join("  " + f + " example" for f in names["observed_flags"])
        raw += "\n  " + names["malformed_legacy_token"] + " label\n"
        parsed = flags_from_help(raw)
        self.assertNotIn("--spec-ngram-", parsed)
        self.assertTrue(set(names["observed_flags"]).issubset(parsed))

    def test_real_bench_and_fit_flag_evidence_is_present(self):
        bench = FIXTURE["command_evidence"]["bench"]
        fit = FIXTURE["command_evidence"]["fit-params"]
        self.assertNotIn("--rpc", bench["observed_flags"])
        self.assertIn("--offline", bench["observed_flags"])
        self.assertIn("--fit-print", fit["observed_flags"])
        self.assertIn("--list-devices", fit["observed_flags"])

    def test_synthetic_ggml_format_matches_observed_shape(self):
        result = parse_list_devices(FIXTURE["synthetic_device_output"])
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["devices"][0]["id"], "Vulkan0")


if __name__ == "__main__":
    unittest.main()
