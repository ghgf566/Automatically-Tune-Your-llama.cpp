"""ATYL workload and result schema tests (illustrative fixtures only)."""
import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_contracts import validate


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pfile = ROOT / "examples/workload-profile.example.json"
        cls.rfile = ROOT / "examples/run-record.example.json"
        cls.p_schema = json.loads((ROOT / "schemas/workload-profile.schema.json").read_text("utf-8"))
        cls.r_schema = json.loads((ROOT / "schemas/run-record.schema.json").read_text("utf-8"))

    def test_schema_is_valid(self):
        Draft202012Validator.check_schema(self.p_schema)
        Draft202012Validator.check_schema(self.r_schema)

    def test_illustrative_examples(self):
        self.assertEqual(validate(ROOT / "schemas/workload-profile.schema.json", self.pfile)["schema_version"], "1.1.0")
        self.assertEqual(validate(ROOT / "schemas/run-record.schema.json", self.rfile)["status"], "not_started")

    def test_quality_and_workload_are_explicit(self):
        data = json.loads(self.pfile.read_text("utf-8"))
        self.assertFalse(data["constraints"]["allow_quality_changes"])
        self.assertFalse(data["constraints"]["allow_service_semantic_changes"])
        self.assertEqual(data["workload"]["context_tokens"], 4096)

    def test_unknown_schema_property_fails(self):
        data = json.loads(self.pfile.read_text("utf-8"))
        data["unexpected"] = "do not silently ignore"
        with self.assertRaises(ValidationError):
            Draft202012Validator(self.p_schema).validate(data)

    def test_missing_launched_argv_fails(self):
        data = json.loads(self.rfile.read_text("utf-8"))
        data["configuration"].pop("launched_argv")
        with self.assertRaises(ValidationError):
            Draft202012Validator(self.r_schema).validate(data)

    def validate_mutated_profile(self, mutate):
        from tempfile import TemporaryDirectory
        data = json.loads(self.pfile.read_text("utf-8"))
        mutate(data)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return validate(ROOT / "schemas/workload-profile.schema.json", path)

    def test_locked_parameter_conflict_fails(self):
        with self.assertRaisesRegex(ValueError, "user-locked"):
            self.validate_mutated_profile(
                lambda data: data["candidate_parameters"].update({"--cache-type-k": "q8_0"}))

    def test_alias_locked_parameter_conflict_fails(self):
        with self.assertRaisesRegex(ValueError, "user-locked"):
            self.validate_mutated_profile(
                lambda data: data["candidate_parameters"].update({"-ctk": "q8_0"}))

    def test_minimum_context_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "minimum_context"):
            self.validate_mutated_profile(
                lambda data: data["constraints"].update({"minimum_context_tokens": 8192}))

    def test_required_concurrency_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "required_concurrency"):
            self.validate_mutated_profile(
                lambda data: data["constraints"].update({"required_concurrency": 2}))

    def test_unknown_binary_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "expected official"):
            self.validate_mutated_profile(
                lambda data: data["engine"].update({"binary": "unrecognized.exe"}))

    def test_official_integrated_profile_has_command(self):
        data = self.validate_mutated_profile(lambda document: None)
        self.assertEqual(data["engine"]["binary"], "llama")
        self.assertEqual(data["engine"]["command"], "bench")

    def test_official_missing_command_rejected(self):
        with self.assertRaisesRegex(ValueError, "engine.command"):
            self.validate_mutated_profile(
                lambda doc: doc["engine"].pop("command"))

    def test_official_subcommand_workload_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "bench workload"):
            self.validate_mutated_profile(
                lambda doc: doc["engine"].update({"command": "serve"}))

    def test_legacy_profile_still_accepted(self):
        def convert(doc):
            doc["engine"].update({"binary": "llama-bench", "installation_source": "external"})
            doc["engine"].pop("command")
        self.assertEqual(self.validate_mutated_profile(convert)["engine"]["binary"],
                         "llama-bench")

    def test_legacy_mismatched_command_rejected(self):
        def convert(doc):
            doc["engine"].update({"binary": "llama-bench", "command": "serve"})
        with self.assertRaisesRegex(ValueError, "requires command bench"):
            self.validate_mutated_profile(convert)


    def test_engine_commit_not_inferred_from_registry(self):
        profile = json.loads(self.pfile.read_text("utf-8"))
        result = json.loads(self.rfile.read_text("utf-8"))
        self.assertEqual(profile["registry_upstream_commit"],
                         result["registry_upstream_commit"])
        self.assertIsNone(profile["engine"]["build_commit"])
        self.assertIsNone(profile["engine"]["binary_sha256"])
        self.assertIsNone(result["engine_build_commit"])
        self.assertIsNone(result["engine_binary_sha256"])

    def test_reject_mismatched_registry_snapshot(self):
        with self.assertRaisesRegex(ValueError, "registry_upstream_commit"):
            self.validate_mutated_profile(
                lambda doc: doc.update({"registry_upstream_commit": "f" * 40}))

    def test_result_requires_verified_binary_for_success(self):
        from tempfile import TemporaryDirectory
        record = json.loads(self.rfile.read_text("utf-8"))
        record["status"] = "success"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "successful run"):
                validate(ROOT / "schemas/run-record.schema.json", path)

    def test_build_sha256_format_is_checked(self):
        with self.assertRaises(ValidationError):
            self.validate_mutated_profile(
                lambda doc: doc["engine"].update({"binary_sha256": "not-a-sha"}))



if __name__ == "__main__":
    unittest.main()
