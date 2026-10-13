"""Output is never destructive by default; no live engine used."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from safe_output import UnsafeOutputError, write_text


class SafeOutputTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_new_file_is_atomic_and_exact(self):
        target = self.root / "nested" / "new.json"
        write_text(target, '{"ok":true}\n')
        self.assertEqual(target.read_text("utf-8"), '{"ok":true}\n')
        self.assertFalse(list(target.parent.glob(".atyl-*.tmp")))

    def test_existing_output_not_overwritten_without_explicit_opt_in(self):
        target = self.root / "saved.json"
        target.write_text("original", "utf-8")
        with self.assertRaisesRegex(UnsafeOutputError, "output exists"):
            write_text(target, "changed")
        self.assertEqual(target.read_text("utf-8"), "original")
        write_text(target, "replacement", overwrite=True)
        self.assertEqual(target.read_text("utf-8"), "replacement")

    def test_protected_inputs_are_never_overwritten_even_when_overwrite_true(self):
        source = self.root / "model.gguf"
        source.write_bytes(b"GGUF fake binary data")
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite):
                with self.assertRaisesRegex(UnsafeOutputError, "protected"):
                    write_text(source, "bad", protected=[source], overwrite=overwrite)
                self.assertEqual(source.read_bytes(), b"GGUF fake binary data")

    def test_parent_resolved_alias_is_protected(self):
        src = self.root / "model.gguf"
        src.write_text("DATA", "utf-8")
        alias = self.root / "sub" / ".." / "model.gguf"
        with self.assertRaisesRegex(UnsafeOutputError, "protected"):
            write_text(alias, "BAD", protected=[src])
        self.assertEqual(src.read_text("utf-8"), "DATA")

    def test_existing_symlink_does_not_overwrite_protected_target(self):
        src = self.root / "data.json"
        src.write_text("protected", "utf-8")
        alias = self.root / "alias.json"
        try:
            alias.symlink_to(src)
        except (OSError, NotImplementedError):
            self.skipTest("symlink permissions unavailable")
        with self.assertRaises(UnsafeOutputError):
            write_text(alias, "bad", protected=[src], overwrite=True)
        self.assertEqual(src.read_text("utf-8"), "protected")


if __name__ == "__main__":
    unittest.main()
