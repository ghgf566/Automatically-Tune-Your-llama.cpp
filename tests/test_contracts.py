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
        self.assertEqual(validate(ROOT / "schemas/workload-profile.schema.json", self.pfile)["schema_version"], "1.0.0")
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

    def test_locked_parameter_conflict_fails(self):
        data = json.loads(self.pfile.read_text("utf-8"))
        data["candidate_parameters"]["--cache-type-k"] = "q8_0"
        with self.assertRaisesRegex(ValueError, "user-locked"):
            locks = set(data["user_locked_parameters"])
            candidates = set(data["candidate_parameters"])
            overlaps = locks & candidates
            if overlaps:
                raise ValueError("candidate overrides user-locked parameters: " + str(overlaps))


if __name__ == "__main__":
    unittest.main()
