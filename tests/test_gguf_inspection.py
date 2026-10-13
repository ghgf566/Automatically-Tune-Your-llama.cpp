"""Synthetic minimal GGUF containers; never load real model weights."""
import json
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from inspect_gguf import GGUFMetadataError, inspect

SCHEMA = json.loads((ROOT / "schemas/gguf-observation.schema.json").read_text("utf-8"))


def gstr(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv(key, kind, value):
    prefix = gstr(key) + struct.pack("<I", kind)
    if kind == 8:
        return prefix + gstr(value)
    if kind == 4:
        return prefix + struct.pack("<I", value)
    if kind == 10:
        return prefix + struct.pack("<Q", value)
    if kind == 9:
        return prefix + struct.pack("<IQ", 8, len(value)) + b"".join(gstr(s) for s in value)
    raise AssertionError("unsupported test fixture kind")


def synthetic_gguf(fields=None, *, tensor=True, version=3):
    fields = fields if fields is not None else [
        ("general.architecture", 8, "qwen3moe"),
        ("general.name", 8, "Synthetic Example"),
        ("general.file_type", 4, 15),
        ("general.quantization_version", 4, 2),
        ("qwen3moe.block_count", 4, 48),
        ("qwen3moe.expert_count", 4, 128),
        ("qwen3moe.expert_used_count", 4, 8),
        ("tokenizer.ggml.tokens", 9, ["a", "b", "private"]),
    ]
    chunks = [b"GGUF", struct.pack("<IQQ", version, int(tensor), len(fields))]
    chunks.extend(kv(*f) for f in fields)
    if tensor:
        chunks += [gstr("blk.0.attn_q.weight"), struct.pack("<IQQIQ", 2, 32, 64, 8, 0)]
    data = b"".join(chunks)
    if tensor:
        data += b"\0" * (-len(data) % 32) + b"\0" * 32
    return data


class GGUFInspectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Draft202012Validator.check_schema(SCHEMA)
        cls.validator = Draft202012Validator(SCHEMA)

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "example.gguf"

    def write(self, data):
        self.path.write_bytes(data)
        return self.path

    def test_read_only_minimal_model_fields(self):
        out = inspect(self.write(synthetic_gguf()))
        self.validator.validate(out)
        self.assertEqual(out["gguf_version"], 3)
        self.assertEqual(out["general"]["architecture"], "qwen3moe")
        self.assertEqual(out["general"]["file_type_id"], 15)
        self.assertEqual(out["architecture_metadata"]["block_count"], 48)
        self.assertEqual(out["architecture_metadata"]["expert_count"], 128)
        self.assertEqual(out["tensor_type_id_counts"], {"8": 1})
        self.assertEqual(out["tensor_count_in_this_file"], 1)
        self.assertNotIn("tokenizer", json.dumps(out))
        self.assertFalse(out["validation"]["tensor_data_read"])
        self.assertFalse(out["validation"]["model_load_tested"])
        self.assertNotIn(self.folder.name, json.dumps(out))

    def test_attention_kv_and_rope_metadata_are_retained_without_inference(self):
        fields = [
            ("general.architecture", 8, "qwen35"),
            ("qwen35.attention.head_count_kv", 4, 8),
            ("qwen35.attention.key_length", 4, 128),
            ("qwen35.attention.value_length", 4, 128),
            ("qwen35.attention.sliding_window", 4, 4096),
            ("qwen35.rope.dimension_count", 4, 128),
            ("qwen35.expert_count", 4, 128),
        ]
        out = inspect(self.write(synthetic_gguf(fields, tensor=False)))
        self.validator.validate(out)
        props = out["architecture_metadata"]
        self.assertEqual(props["attention.head_count_kv"], 8)
        self.assertEqual(props["attention.key_length"], 128)
        self.assertEqual(props["attention.value_length"], 128)
        self.assertEqual(props["rope.dimension_count"], 128)
        self.assertNotIn("kv_cache_bytes", out)
        self.assertFalse(out["validation"]["model_load_tested"])

    def test_synthetic_split_reports_incomplete_multifile(self):
        fields = [
            ("general.architecture", 8, "llama"),
            ("split.no", 4, 0),
            ("split.count", 4, 3),
            ("split.tensors.count", 10, 300),
        ]
        out = inspect(self.write(synthetic_gguf(fields, tensor=False, version=2)))
        self.validator.validate(out)
        self.assertEqual(out["split"]["part_count"], 3)
        self.assertTrue(out["split"]["multiple_parts_reported"])
        self.assertFalse(out["split"]["all_parts_inspected"])
        self.assertEqual(out["gguf_version"], 2)

    def test_reject_non_gguf_and_future_version(self):
        for raw in (b"not gguf", synthetic_gguf(version=99)):
            with self.subTest(raw=raw[:25]):
                with self.assertRaises(GGUFMetadataError):
                    inspect(self.write(raw))

    def test_bounded_string_and_truncated_data(self):
        huge_name = b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 999999999)
        for raw in (huge_name, synthetic_gguf()[:55]):
            with self.subTest(size=len(raw)):
                with self.assertRaises(GGUFMetadataError):
                    inspect(self.write(raw))

    def test_reject_array_count_and_unknown_value_type(self):
        token = gstr("tokenizer.ggml.tokens")
        for payload in (token + struct.pack("<IIQ", 9, 8, 4_000_000),
                        gstr("a") + struct.pack("<I", 199)):
            raw = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + payload
            with self.assertRaises(GGUFMetadataError):
                inspect(self.write(raw))

    def test_reject_invalid_alignment(self):
        raw = synthetic_gguf([("general.alignment", 4, 3)], tensor=True)
        with self.assertRaisesRegex(GGUFMetadataError, "alignment"):
            inspect(self.write(raw))

    def test_duplicate_ignored_keys_rejected(self):
        fields = [("ignored.key", 8, "one"), ("ignored.key", 8, "two")]
        with self.assertRaisesRegex(GGUFMetadataError, "duplicate"):
            inspect(self.write(synthetic_gguf(fields, tensor=False)))

    def test_tensor_descriptor_needs_valid_dimensions(self):
        raw = bytearray(synthetic_gguf([], tensor=True))
        # After GGUF header and tensor name, the 4-byte n_dims field follows.
        n_dims_at = 24 + len(gstr("blk.0.attn_q.weight"))
        struct.pack_into("<I", raw, n_dims_at, 0)
        with self.assertRaisesRegex(GGUFMetadataError, "dimensions"):
            inspect(self.write(raw))

    def test_limit_rejects_implausible_counts(self):
        for tensors, keys in ((999999999, 0), (0, 999999999)):
            raw = b"GGUF" + struct.pack("<IQQ", 3, tensors, keys)
            with self.assertRaisesRegex(GGUFMetadataError, "implausible"):
                inspect(self.write(raw))


if __name__ == "__main__":
    unittest.main()
