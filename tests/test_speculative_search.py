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
        "schema_version": "0.1.0",
        "kind": "gguf_metadata_observation",
        "source": {"filename": "example.gguf", "file_size_bytes": 4000},
        "gguf_version": 3,
        "tensor_count_in_this_file": 0,
        "metadata_key_count": 2,
        "general": {"architecture": "illustrative-dense", "name": "synthetic",
                    "file_type_id": 30, "quantization_version": 2},
        "architecture_metadata": arch,
        "tensor_type_id_counts": {},
        "split": {"part_index": None, "part_count": None,
                  "total_tensor_count_reported": None,
                  "multiple_parts_reported": False, "all_parts_inspected": False},
        "metadata_bytes_scanned": 128,
        "validation": {"metadata_structure_parsed": True,
                       "tensor_descriptors_parsed": True,
                       "tensor_data_read": False,
                       "tensor_payload_integrity_verified": False,
                       "model_load_tested": False,
                       "model_backend_compatibility": "not_tested"},
        "limitations": ["Synthetic research fixture; no model read"],
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
        old = validate(ROOT / "schemas/workload-profile.schema.json",
                       ROOT / "examples/workload-profile.example.json")
        p = validate(ROOT / "schemas/workload-profile.schema.json",
                     ROOT / "examples/workload-profile.speculative.example.json")
        r = validate(ROOT / "schemas/run-record.schema.json",
                     ROOT / "examples/run-record.example.json")
        self.assertNotIn("speculative_search", old)
        self.assertEqual(p["workload"]["mode"], "server")
        self.assertEqual(p["speculative_search"]["draft_n_max_candidates"], [1, 2, 4, 8])
        self.assertEqual(r["speculative_observation"]["measurement_status"], "not_started")

    def test_non_server_speculative_profiles_fail_closed(self):
        from validate_contracts import check_speculative_search_consistency
        with self.assertRaisesRegex(ValueError, "server workload"):
            check_speculative_search_consistency(
                search_config(), {}, {}, {"mode": "bench", "request_kind": "completion"})
        configuration = search_config()
        configuration["draft_placements"] = [{"device_id": "none", "gpu_layers": 4}]
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            check_speculative_search_consistency(
                configuration, {}, {}, {"mode": "server", "request_kind": "chat"})

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

    def test_engine_help_draft_aliases_resolve_to_canonical_names(self):
        flags = ["--spec-type", "--spec-draft-n-max", "--model-draft",
                 "-ngld", "-devd"]
        cfg = search_config()
        cfg["modes"] = ["off", "external_mtp"]
        cfg["external_artifacts"] = [{
            "mode": "external_mtp", "artifact_id": "head1",
            "sha256": "f" * 64, "pairing_evidence": None}]
        cfg["draft_placements"] = [{"device_id": "Vulkan0", "gpu_layers": 1}]
        result = make_preview(cfg, model_observation(), serve_runtime(flags))
        self.assertEqual(result["candidates"][1]["readiness"],
                         "requires_pairing_and_load_test")

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
            from shutil import copyfile
            copyfile(ROOT / "examples/workload-profile.speculative.example.json", s)
            argv = ["plan_speculative_search.py", "--profile", str(s),
                    "--model-observation", str(m), "--runtime-snapshot", str(r),
                    "--output", str(out)]
            with patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            payload = json.loads(out.read_text("utf-8"))
            self.assertFalse(payload["candidates"][1]["executable"])
            with patch("sys.argv", argv):
                self.assertEqual(main(), 1)
            self.assertEqual(json.loads(s.read_text("utf-8"))["profile_id"],
                             "illustrative-mtp-server-01")

    def test_full_profile_rejects_incomplete_gguf_record(self):
        profile = json.loads(
            (ROOT / "examples/workload-profile.speculative.example.json").read_text("utf-8"))
        incomplete = model_observation()
        del incomplete["tensor_type_id_counts"]
        with self.assertRaisesRegex(SpeculativePlanError, "GGUF observation"):
            make_preview(profile["speculative_search"], incomplete,
                         serve_runtime(), profile=profile)

    def test_complete_profile_is_bound_and_missing_model_digest_blocks_mtp(self):
        profile = json.loads(
            (ROOT / "examples/workload-profile.speculative.example.json").read_text("utf-8"))
        preview = make_preview(profile["speculative_search"], model_observation(),
                               serve_runtime(), profile=profile)
        self.assertEqual(preview["profile_id"], profile["profile_id"])
        self.assertEqual(preview["model_identity_status"], "unverified")
        self.assertTrue(all(len(c["candidate_id"]) == 64 for c in preview["candidates"]))
        self.assertIn("target_model_full_hash_evidence_missing",
                      preview["candidates"][1]["reasons"])
        repeated = make_preview(profile["speculative_search"], model_observation(),
                                serve_runtime(), profile=profile)
        self.assertEqual([c["candidate_id"] for c in preview["candidates"]],
                         [c["candidate_id"] for c in repeated["candidates"]])
        altered = copy.deepcopy(profile)
        altered["workload"]["context_tokens"] = 8192
        altered["constraints"]["minimum_context_tokens"] = 4096
        different = make_preview(altered["speculative_search"], model_observation(),
                                 serve_runtime(), profile=altered)
        self.assertNotEqual(preview["candidates"][1]["candidate_id"],
                            different["candidates"][1]["candidate_id"])

    def test_profile_model_identity_evidence_can_be_recorded_but_not_execute(self):
        profile = json.loads(
            (ROOT / "examples/workload-profile.speculative.example.json").read_text("utf-8"))
        observed = model_observation()
        evidence = {
            "kind": "fit_estimate_plan",
            "engine_sha256": "a" * 64,
            "model": {
                "filename": observed["source"]["filename"],
                "file_size_bytes": observed["source"]["file_size_bytes"],
                "fingerprint": {
                    "full_file_hashed": True,
                    "full_file_sha256": profile["model"]["sha256"],
                }
            },
        }
        p = make_preview(profile["speculative_search"], observed, serve_runtime(),
                         profile=profile, fit_plan=evidence)
        self.assertEqual(p["model_identity_status"], "recorded_full_digest_match")
        self.assertEqual(p["candidates"][1]["readiness"], "requires_runtime_mode_and_load_test")
        self.assertFalse(p["candidates"][1]["executable"])
        evidence["model"]["fingerprint"]["full_file_sha256"] = "f" * 64
        with self.assertRaisesRegex(SpeculativePlanError, "digest"):
            make_preview(profile["speculative_search"], observed, serve_runtime(),
                         profile=profile, fit_plan=evidence)

    def test_locked_draft_alias_is_detected_during_full_profile_validation(self):
        from validate_contracts import validate_profile_document
        base = json.loads(
            (ROOT / "examples/workload-profile.speculative.example.json").read_text("utf-8"))
        for alias in ("-ngld", "--n-gpu-layers-draft", "--spec-draft-p-min", "--draft-n", "-md"):
            profile = copy.deepcopy(base)
            profile["user_locked_parameters"][alias] = 1
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, "speculative controls"):
                validate_profile_document(profile)

    def test_full_profile_rejects_runtime_binary_and_architecture_disagreement(self):
        profile = json.loads(
            (ROOT / "examples/workload-profile.speculative.example.json").read_text("utf-8"))
        profile["engine"]["binary_sha256"] = "f" * 64
        with self.assertRaisesRegex(SpeculativePlanError, "engine SHA"):
            make_preview(profile["speculative_search"], model_observation(),
                         serve_runtime(), profile=profile)
        profile["engine"]["binary_sha256"] = None
        observed = model_observation()
        observed["general"] = {"architecture": "other"}
        with self.assertRaisesRegex(SpeculativePlanError, "architecture"):
            make_preview(profile["speculative_search"], observed,
                         serve_runtime(), profile=profile)

    def test_measured_result_must_retain_candidate_id(self):
        from validate_contracts import validate
        obj = json.loads((ROOT / "examples/run-record.example.json").read_text("utf-8"))
        obj["speculative_observation"]["mode"] = "embedded_mtp"
        obj["speculative_observation"]["requested_draft_n_max"] = 2
        obj["speculative_observation"]["measurement_status"] = "completed"
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate_id"):
                validate(ROOT / "schemas/run-record.schema.json", path)
            obj["candidate_id"] = "a" * 64
            path.write_text(json.dumps(obj), encoding="utf-8")
            validate(ROOT / "schemas/run-record.schema.json", path)

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
