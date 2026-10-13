"""Mocked tests: never run any real llama process."""
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
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.binary = Path(self.temp.name) / "llama.exe"
        self.binary.write_bytes(b"mock-llama-executable")

    def test_official_binary_discovery_uses_path(self):
        with patch.object(probe.shutil, "which", side_effect=lambda name: str(self.binary) if name == "llama" else None):
            self.assertEqual(probe.discover_binary(), self.binary.resolve())

    def test_missing_official_binary_has_clear_error(self):
        with patch.object(probe.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "not found in PATH"):
                probe.discover_binary()

    def test_routes_official_commands_and_version(self):
        self.assertEqual(probe.determine_route(self.binary, "serve"),
                         ("integrated", "serve", "llama-server"))
        self.assertEqual(probe.determine_route(self.binary, "bench"),
                         ("integrated", "bench", "llama-bench"))
        self.assertEqual(probe.argv_for("integrated", self.binary, "serve", "version"),
                         [str(self.binary), "version"])
        self.assertEqual(probe.argv_for("integrated", self.binary, "serve", "--help"),
                         [str(self.binary), "serve", "--help"])

    def test_parses_only_flag_headings(self):
        output = ("  -h, --help, --usage   print help\n"
                  "  -ngl, --gpu-layers N   offload\n"
                  "  --flash-attn [on|off|auto]   attention\n"
                  "  description mentions --fake but this is prose\n")
        flags = probe.flags_from_help(output)
        self.assertTrue({"-h", "--help", "--usage", "-ngl", "--gpu-layers", "--flash-attn"}.issubset(flags))
        self.assertNotIn("--fake", flags)

    def test_integrated_serve_probes_without_loading(self):
        commands = []
        def fake_run(args, timeout):
            commands.append(args[1:])
            if "--help" in args:
                return {"status": "success", "returncode": 0,
                        "stdout": "  -ngl, --gpu-layers N\n  --new-flag N\n", "stderr": ""}
            return {"status": "success", "returncode": 0, "stdout": "b12345", "stderr": ""}
        with patch.object(probe, "run_cli", side_effect=fake_run):
            result = probe.probe_binary(self.binary, self.inventory)
        self.assertEqual(commands, [["version"], ["serve", "--help"]])
        self.assertEqual(result["entrypoint"], {"kind": "integrated", "command": "serve"})
        self.assertEqual(result["program"], "llama-server")
        self.assertEqual(result["command_results"]["devices"]["status"], "not_requested")
        self.assertIn("--gpu-layers", result["comparison"]["documented_observed_in_help"])
        self.assertIn("--new-flag", result["comparison"]["unregistered_flags_observed_in_help"])

    def test_integrated_bench_uses_bench_registry(self):
        commands = []
        def fake_run(args, timeout):
            commands.append(args[1:])
            if "--help" in args:
                return {"status": "success", "returncode": 0, "stdout": "  -t, --threads <n>\n", "stderr": ""}
            return {"status": "success", "returncode": 0, "stdout": "b12345", "stderr": ""}
        with patch.object(probe, "run_cli", side_effect=fake_run):
            result = probe.probe_binary(self.binary, self.inventory, command="bench")
        self.assertEqual(commands, [["version"], ["bench", "--help"]])
        self.assertEqual(result["program"], "llama-bench")

    def test_devices_is_opt_in(self):
        commands = []
        def fake_run(args, timeout):
            commands.append(args[1:])
            return {"status": "success", "returncode": 0, "stdout": "  --help\n", "stderr": ""}
        with patch.object(probe, "run_cli", side_effect=fake_run):
            probe.probe_binary(self.binary, self.inventory, include_devices=True)
        self.assertEqual(commands, [["version"], ["serve", "--help"], ["serve", "--list-devices"]])

    def test_failed_help_is_unknown_not_unsupported(self):
        with patch.object(probe, "run_cli", return_value={
                "status": "timeout", "returncode": None, "stdout": "", "stderr": ""}):
            result = probe.probe_binary(self.binary, self.inventory)
        self.assertIsNone(result["comparison"]["documented_not_observed_in_help"])

    def test_empty_help_is_unknown_not_unsupported(self):
        with patch.object(probe, "run_cli", return_value={
                "status": "nonzero_exit", "returncode": 1, "stdout": "", "stderr": "failed"}):
            result = probe.probe_binary(self.binary, self.inventory)
        self.assertIsNone(result["comparison"]["documented_not_observed_in_help"])

    def test_legacy_supported_only_when_explicit_and_matched(self):
        old = Path(self.temp.name) / "llama-bench.exe"
        old.write_bytes(b"mock-legacy-binary")
        with self.assertRaisesRegex(ValueError, "requires command bench"):
            probe.determine_route(old, "serve")
        self.assertEqual(probe.determine_route(old, "bench"),
                         ("legacy", "bench", "llama-bench"))
        self.assertEqual(probe.argv_for("legacy", old, "bench", "--help"), [str(old), "--help"])

    def test_unknown_executable_fails_before_running(self):
        old = Path(self.temp.name) / "suspicious.exe"
        old.write_bytes(b"test")
        with patch.object(probe, "run_cli") as cmd:
            with self.assertRaisesRegex(ValueError, "Expected official"):
                probe.probe_binary(old, self.inventory)
            cmd.assert_not_called()


if __name__ == "__main__":
    unittest.main()
