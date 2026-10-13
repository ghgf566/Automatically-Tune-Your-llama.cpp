"""Runtime CLI probe tests use mocks only; they never run a real executable."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import probe_runtime as probe


class RuntimeProbeTest(unittest.TestCase):
    def setUp(self):
        self.inventory = json.loads((ROOT / "data/cli-inventory.json").read_text(encoding="utf-8"))
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.binary = Path(self.tmp.name) / "llama-server.exe"
        self.binary.write_bytes(b"fake-binary-for-testing-only")

    def test_parses_only_option_headings(self):
        output = ("build: llama.cpp\n"
                  "   -h, --help, --usage    print usage\n"
                  "   -ngl, --gpu-layers, --n-gpu-layers N   GPU layers\n"
                  "  --flash-attn [on|off|auto]  flash attention\n"
                  "  description mentions --fake but isn't an option line\n")
        flags = probe.flags_from_help(output)
        self.assertTrue({"--help", "--usage", "--gpu-layers", "--n-gpu-layers", "-ngl",
                         "--flash-attn"}.issubset(flags))
        self.assertNotIn("--fake", flags)

    def test_compares_help_against_pinned_inventory(self):
        commands = []
        def fake_run(path, flag, timeout):
            commands.append(flag)
            if flag == "--help":
                return {"status": "success", "returncode": 0,
                        "stdout": "  -h, --help\n  -ngl, --gpu-layers N\n  --new-flag N\n",
                        "stderr": ""}
            return {"status": "success", "returncode": 0,
                    "stdout": "llama.cpp synthetic version\n", "stderr": ""}
        with patch.object(probe, "run_cli", side_effect=fake_run):
            snapshot = probe.probe_binary(self.binary, self.inventory)
        self.assertEqual(commands, ["--version", "--help"])
        self.assertEqual(snapshot["registry_commit"], self.inventory["upstream"]["commit"])
        self.assertEqual(snapshot["binary"]["filename"], "llama-server.exe")
        self.assertEqual(snapshot["command_results"]["devices"]["status"], "not_requested")
        self.assertIn("--gpu-layers", snapshot["comparison"]["documented_observed_in_help"])
        self.assertIn("--new-flag", snapshot["comparison"]["unregistered_flags_observed_in_help"])

    def test_device_query_requires_opt_in(self):
        calls = []
        def fake_run(path, flag, timeout):
            calls.append(flag)
            return {"status": "success", "returncode": 0, "stdout": "fake\n", "stderr": ""}
        with patch.object(probe, "run_cli", side_effect=fake_run):
            probe.probe_binary(self.binary, self.inventory, include_devices=True)
        self.assertEqual(calls, ["--version", "--help", "--list-devices"])

    def test_timeout_is_not_unsupported(self):
        with patch.object(probe, "run_cli", return_value={
            "status": "timeout", "returncode": None, "stdout": "", "stderr": ""}):
            result = probe.probe_binary(self.binary, self.inventory)
        self.assertEqual(result["command_results"]["help"]["status"], "timeout")
        self.assertEqual(result["command_results"]["help"]["documented_flag_presence"],
                         "unavailable")

    def test_unknown_executable_fails_before_execution(self):
        other = Path(self.tmp.name) / "suspicious.exe"
        other.write_bytes(b"fake")
        with patch.object(probe, "run_cli") as command:
            with self.assertRaisesRegex(ValueError, "Expected"):
                probe.probe_binary(other, self.inventory)
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
