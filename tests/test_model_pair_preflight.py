"""External draft preflight tests; all GGUFs are synthetic and never loaded."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from tests.test_gguf_inspection import synthetic_gguf
from verify_model_pair import PairingError, REPORT_SCHEMA, verify_pair


class ModelPairPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Draft202012Validator.check_schema(REPORT_SCHEMA)
        cls.validator = Draft202012Validator(REPORT_SCHEMA)

    def _fixture(self, temp, *, draft_fields=None):
        fields = [
            ("general.architecture", 8, "qwen3moe"),
            ("tokenizer.ggml.model", 8, "gpt2"),
            ("tokenizer.ggml.tokens", 9, ["alpha", "beta", "gamma"]),
        ]
        target = Path(temp) / "target.gguf"
        draft = Path(temp) / "draft.gguf"
        target.write_bytes(synthetic_gguf(fields, tensor=False))
        draft.write_bytes(synthetic_gguf(draft_fields or fields, tensor=False))
        profile = json.loads(
            (ROOT / "examples/workload-profile.speculative.example.json").read_text("utf-8"))
        profile = copy.deepcopy(profile)
        profile["model"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        profile["model"]["architecture"] = "qwen3moe"
        profile["speculative_search"]["modes"] = ["off", "external_mtp"]
        profile["speculative_search"]["external_artifacts"] = [{
            "mode": "external_mtp", "artifact_id": "draft_a",
            "sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
            "pairing_evidence": None,
        }]
        return profile, target, draft

    def test_matching_full_digests_architecture_and_tokenizer_remain_nonexecutable(self):
        with TemporaryDirectory() as temp:
            profile, target, draft = self._fixture(temp)
            report = verify_pair(profile, "draft_a", target, draft)
        self.validator.validate(report)
        self.assertEqual(report["status"], "requires_runtime_load_verification")
        self.assertTrue(report["checks"]["full_file_digests_match_profile"])
        self.assertTrue(report["checks"]["architecture_match"])
        self.assertTrue(report["checks"]["tokenizer_metadata_byte_equivalent"])
        self.assertFalse(report["executable"])

    def test_tokenizer_metadata_mismatch_blocks_pair(self):
        with TemporaryDirectory() as temp:
            fields = [
                ("general.architecture", 8, "qwen3moe"),
                ("tokenizer.ggml.model", 8, "gpt2"),
                ("tokenizer.ggml.tokens", 9, ["alpha", "different", "gamma"]),
            ]
            profile, target, draft = self._fixture(temp, draft_fields=fields)
            report = verify_pair(profile, "draft_a", target, draft)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["checks"]["tokenizer_metadata_byte_equivalent"])
        self.assertFalse(report["executable"])

    def test_architecture_mismatch_blocks_pair(self):
        with TemporaryDirectory() as temp:
            fields = [
                ("general.architecture", 8, "qwen2"),
                ("tokenizer.ggml.model", 8, "gpt2"),
                ("tokenizer.ggml.tokens", 9, ["alpha", "beta", "gamma"]),
            ]
            profile, target, draft = self._fixture(temp, draft_fields=fields)
            report = verify_pair(profile, "draft_a", target, draft)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["checks"]["architecture_match"])

    def test_stale_target_digest_is_rejected(self):
        with TemporaryDirectory() as temp:
            profile, target, draft = self._fixture(temp)
            profile["model"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(PairingError, "on-disk model SHA-256"):
                verify_pair(profile, "draft_a", target, draft)

    def test_multi_part_draft_is_rejected_until_bundle_validator_exists(self):
        with TemporaryDirectory() as temp:
            profile, target, draft = self._fixture(temp)
            fields = [
                ("general.architecture", 8, "qwen3moe"),
                ("tokenizer.ggml.model", 8, "gpt2"),
                ("tokenizer.ggml.tokens", 9, ["alpha", "beta", "gamma"]),
                ("split.no", 4, 0),
                ("split.count", 4, 2),
            ]
            draft.write_bytes(synthetic_gguf(fields, tensor=False))
            profile["speculative_search"]["external_artifacts"][0]["sha256"] = (
                hashlib.sha256(draft.read_bytes()).hexdigest())
            with self.assertRaisesRegex(PairingError, "multi-part GGUF"):
                verify_pair(profile, "draft_a", target, draft)


if __name__ == "__main__":
    unittest.main()
