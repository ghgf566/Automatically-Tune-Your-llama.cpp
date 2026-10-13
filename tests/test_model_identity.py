"""GGUF identity tests: fast scan vs explicit full-file digest."""
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from model_identity import fingerprint, matches_existing, ModelIdentityError


class ModelFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "synthetic.gguf"
        self.path.write_bytes(b"GGUF" + b"metadata" + b"\x00" * 128)

    def test_fast_identity_is_metadata_only(self):
        a = fingerprint(self.path, 12)
        self.assertEqual(len(a["metadata_sha256"]), 64)
        self.assertIsNone(a["full_file_sha256"])
        self.assertFalse(a["full_file_hashed"])
        self.assertFalse(a["weight_integrity_verified"])
        self.assertTrue(matches_existing(self.path, a))

    def test_same_size_header_change_is_detected(self):
        a = fingerprint(self.path, 12)
        raw = self.path.read_bytes()
        self.path.write_bytes(b"GGUX" + raw[4:])
        self.assertFalse(matches_existing(self.path, a))

    def test_opt_in_full_file_fingerprint_can_detect_tail_change(self):
        a = fingerprint(self.path, 12, full_sha256=True)
        self.assertTrue(a["full_file_hashed"])
        self.assertEqual(len(a["full_file_sha256"]), 64)
        self.assertTrue(matches_existing(self.path, a))
        stat = self.path.stat()
        raw = bytearray(self.path.read_bytes())
        raw[-1] = 7
        self.path.write_bytes(raw)
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(matches_existing(self.path, a))

    def test_reject_invalid_header_span(self):
        for span in (0, 999999999):
            with self.assertRaises(ModelIdentityError):
                fingerprint(self.path, span)


if __name__ == "__main__":
    unittest.main()
