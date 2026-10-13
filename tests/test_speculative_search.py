"""Speculative search planning is pure metadata work, not a model runner."""
import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from plan_speculative_search import (
    SEARCH_VALIDATOR, PREVIEW_VALIDATOR, SpeculativePlanError, make_preview, main)
from validate_contracts import validate


def search_config():
    return json.loads((ROOT / "examples/speculative-search.example.json").read_text("utf-8"))


def model_observation(nextn=2):
    arch = {"nextn_predict_layers": nextn} if nextn is not None else {}
    return {
        "kind": "gguf_metadata_observation",
        "source": {"filename": "example.gguf", "file_size_bytes": 4000},
        "architecture_metadata": arch,
        "validation": {"metadata_structure_parsed": True,
                       "model_load_tested": False},
    }


def serve_runtime(flags=None, command="serve"):
    if flags is None:
        flags = ["--spec-type", "--spec-draft-n-max", "--spec-draft-model",
                 "--spec-draft-ngl", "--spec-draft-device"]
    sha = "a" * 64
    return {
        "schema_version": "1.1.0", "kind": "runtime_help_observation",
        "binary": {"sha256": sha}, "identity": {"binary_sha256": sha},
        "entrypoint": {"kind": "integrated", "command": command},
        "command_results": {
            "version": {"status": "success"},
            "help": {"status": "success", "flag_names_observed": flags},
        },
    }


class SpeculativePlanningTest(unittest.TestCase):
    def test_profile_result_examples_with_shared_local_schema(self):
        p = validate(ROOT / "schemas/workload-profile.schema.json",
                     ROOT / "examples/workload-profile.example.json")
        r = validate(ROOT / "schemas/run-record.schema.json",
                     ROOT / "examples/run-record.example.json")
        self.assertEqual(p["speculative_search"]["draft_n_max_candidates"], [1, 2, 4, 8])
        self.assertEqual(r["speculative_observation"]["measurement_status"], "not_started")

    def test_embedded_mtp_uses_each_requested_draft_n_without_inferring_head_limit(self):
        plan = make_preview(search_config(), model_observation(nextn=2), serve_runtime())
        PREVIEW_VALIDATOR.validate(plan)
        self.assertEqual(len(plan["candidates"]), 5)
        off, *mtp = plan["candidates"]
        self.assertEqual(off["mode"], "off")
        self.assertIsNone(off["draft_n_max"])
        self.assertEqual(off["readiness"], "baseline_only")
        self.assertEqual([x["draft_n_max"] for x in mtp], [1, 2, 4, 8])
        self.assertTrue(all(x["readiness"] == "requires_runtime_mode_and_load_test"
                            for x in mtp))
        self.assertTrue(all(x["executable"] is False for x in plan["candidates"]))
        self.assertEqual(mtp[-1]["draft_n_max"], 8)

    def test_unknown_mtp_metadata_is_not_proof_of_support_or_absence(self):
        plan = make_preview(search_config(), model_observation(nextn=None), serve_runtime())
        self.assertEqual(plan["candidates"][1]["readiness"], "blocked_missing_evidence")
        self.assertIn("embedded_nextn_not_observed_or_invalid",
                      plan["candidates"][1]["reasons"])

    def test_unobserved_engine_flags_blocks_enabled_mode(self):
        plan = make_preview(search_config(), model_observation(), serve_runtime(["--help"]))
        self.assertEqual(plan["candidates"][1]["readiness"], "blocked_missing_evidence")
        self.assertTrue(any("runtime_flags_not_observed" in reason
                            for reason in plan["candidates"][1]["reasons"]))

    def test_bench_snapshot_must_not_prove_server_spec_support(self):
        plan = make_preview(search_config(), model_observation(),
                            serve_runtime(command="bench"))
        self.assertIn("serve_runtime_snapshot_required",
                      plan["candidates"][1]["reasons"])

    def test_external_mtp_is_not_claimed_verified_even_with_draft_hash(self):
        config = search_config()
        config["modes"] = ["off", "external_mtp"]
        config["external_artifacts"] = [{
            "mode": "external_mtp", "artifact_id": "draft1", "sha256": "f" * 64,
            "pairing_evidence": None,
        }]
        config["draft_placements"] = [{"device_id": "Vulkan0", "gpu_layers": 10}]
        preview = make_preview(config, model_observation(nextn=None), serve_runtime())
        self.assertEqual(len(preview["candidates"]), 5)
        draft = preview["candidates"][1]
        self.assertEqual(draft["draft_artifact_id"], "draft1")
        self.assertEqual(draft["draft_device_id"], "Vulkan0")
        self.assertEqual(draft["readiness"], "requires_pairing_and_load_test")
        self.assertFalse(draft["executable"])

    def test_external_missing_hash_is_blocked(self):
        config = search_config()
        config["modes"] = ["off", "external_draft_simple"]
        config["external_artifacts"] = [{
            "mode": "external_draft_simple", "artifact_id": "draft1",
            "sha256": None, "pairing_evidence": "user assertion",
        }]
        draft = make_preview(config, model_observation(), serve_runtime())["candidates"][1]
        self.assertIn("external_draft_fingerprint_missing", draft["reasons"])

    def test_missing_off_and_external_artifact_rejected(self):
        config = search_config()
        config["modes"] = ["embedded_mtp"]
        with self.assertRaisesRegex(SpeculativePlanError, "off baseline"):
            make_preview(config, model_observation(), serve_runtime())
        config["modes"] = ["off", "external_mtp"]
        with self.assertRaisesRegex(SpeculativePlanError, "requires an explicitly"):
            make_preview(config, model_observation(), serve_runtime())

    def test_candidate_budget_is_hard_limit_not_silent_truncation(self):
        config = search_config()
        config["max_candidates"] = 3
        with self.assertRaisesRegex(SpeculativePlanError, "max_candidates"):
            make_preview(config, model_observation(), serve_runtime())

    def test_schema_rejects_invalid_token_length_and_unknown_mode(self):
        config = search_config()
        config["draft_n_max_candidates"] = [1, 2, 1000]
        with self.assertRaises(ValidationError):
            SEARCH_VALIDATOR.validate(config)
        config["draft_n_max_candidates"] = [1, 2]
        config["modes"] = ["off", "magic_mtp"]
        with self.assertRaises(ValidationError):
            SEARCH_VALIDATOR.validate(config)

    def test_actual_cli_output_is_preview_only_and_does_not_overwrite_sources(self):
        with TemporaryDirectory() as d:
            folder = Path(d)
            s, m, r, out = [folder / x for x in
                           ("search.json", "model.json", "runtime.json", "preview.json")]
            s.write_text(json.dumps(search_config()), encoding="utf-8")
            m.write_text(json.dumps(model_observation()), encoding="utf-8")
            r.write_text(json.dumps(serve_runtime()), encoding="utf-8")
            argv = ["plan_speculative_search.py", "--search", str(s),
                    "--model-observation", str(m), "--runtime-snapshot", str(r),
                    "--output", str(out)]
            with patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            payload = json.loads(out.read_text("utf-8"))
            self.assertFalse(payload["candidates"][1]["executable"])
            with patch("sys.argv", argv):
                self.assertEqual(main(), 1)
            self.assertTrue(s.read_text("utf-8").startswith('{"schema_version"'))

    def test_run_result_crossfield_guards(self):
        base = json.loads((ROOT / "examples/run-record.example.json").read_text("utf-8"))
        from validate_contracts import check_speculative_result_consistency
        result = copy.deepcopy(base["speculative_observation"])
        result.update({"mode": "embedded_mtp", "requested_draft_n_max": 8,
                       "drafted_tokens": 6, "accepted_tokens": 7,
                       "measurement_status": "completed"})
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            check_speculative_result_consistency(result, "not_started")
        result.update({"accepted_tokens": 3, "acceptance_rate": .25})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            check_speculative_result_consistency(result, "not_started")
        result.update({"accepted_tokens": None, "drafted_tokens": None,
                       "acceptance_rate": None,
                       "load_verification": "not_tested"})
        with self.assertRaisesRegex(ValueError, "actual load verification"):
            check_speculative_result_consistency(result, "success")


if __name__ == "__main__":
    unittest.main()
