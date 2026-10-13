"""Only mocks and dummy files are used; never run real inference."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import engine_adapter as adapter
import engine_identity as identity
import probe_runtime as probe


class CommandAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bin = Path(self.tmp.name) / "llama.exe"
        self.bin.write_bytes(b"synthetic-executable")
        self.registry = json.loads((ROOT / "data/cli-inventory.json").read_text("utf-8"))

    def test_path_discovery_only_official_llama(self):
        with patch.object(adapter.shutil, "which", side_effect=lambda x: str(self.bin) if x == "llama" else None):
            self.assertEqual(adapter.discover_binary(), self.bin.resolve())
        with patch.object(adapter.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "not found in PATH"):
                adapter.discover_binary()

    def test_integrated_command_mapping(self):
        entry = adapter.resolve_entrypoint(self.bin, "serve")
        self.assertEqual(entry.registry_program, "llama-server")
        self.assertEqual(adapter.probe_argv(entry, "version"), [str(self.bin), "version"])
        self.assertEqual(adapter.probe_argv(entry, "--help"),
                         [str(self.bin), "serve", "--help"])
        self.assertEqual(adapter.command_prefix(entry), [str(self.bin), "serve"])
        bench = adapter.resolve_entrypoint(self.bin, "bench")
        self.assertEqual(bench.registry_program, "llama-bench")

    def test_fit_params_is_unregistered_yet(self):
        route = adapter.resolve_entrypoint(self.bin, "fit-params")
        self.assertIsNone(route.registry_program)
        self.assertEqual(adapter.probe_argv(route, "--help"),
                         [str(self.bin), "fit-params", "--help"])
        with self.assertRaisesRegex(ValueError, "not validated"):
            adapter.probe_argv(route, "--list-devices")

    def test_legacy_and_windows_path_profile(self):
        route = adapter.route_from_profile({"binary": r"C:\tools\llama-bench.exe"})
        self.assertEqual(route.command, "bench")
        self.assertEqual(route.kind, "legacy")
        with self.assertRaisesRegex(ValueError, "requires command"):
            adapter.resolve_entrypoint("llama-server.exe", "bench")

    def test_unknown_binary_and_unsafe_action_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "expected official"):
            adapter.resolve_entrypoint("random.exe", "serve")
        entry = adapter.resolve_entrypoint(self.bin, "serve")
        for arg in ["update", "serve", "--model", "--help; whoami"]:
            with self.assertRaises(ValueError):
                adapter.probe_argv(entry, arg)

    def test_version_and_registry_identity_are_separate(self):
        digest = "a" * 64
        sample = {"status": "success", "stdout": "llama build 777", "stderr": ""}
        found = identity.observed_identity(digest, "serve", "b" * 40, sample, True)
        self.assertEqual(found["registry_upstream_commit"], "b" * 40)
        self.assertEqual(found["engine_version_output"], "llama build 777")
        self.assertIsNone(found["engine_build_commit"])
        self.assertTrue(found["cacheable"])
        self.assertNotEqual(
            found["cache_key"],
            identity.capability_cache_key("c" * 64, "serve", "b" * 40, "llama build 777"))
        self.assertNotEqual(
            found["cache_key"],
            identity.capability_cache_key(digest, "bench", "b" * 40, "llama build 777"))
        self.assertNotEqual(
            found["cache_key"],
            identity.capability_cache_key(digest, "serve", "d" * 40, "llama build 777"))

    def test_integrated_read_only_probe(self):
        calls = []
        def fake(args, timeout):
            calls.append(args[1:])
            if "--help" in args:
                return {"status": "success", "returncode": 0,
                        "stdout": "  -ngl, --gpu-layers N\n  --brand-new N\n", "stderr": ""}
            return {"status": "success", "returncode": 0, "stdout": "version 777", "stderr": ""}
        with patch.object(probe, "run_cli", side_effect=fake):
            got = probe.probe_binary(self.bin, self.registry)
        self.assertEqual(calls, [["version"], ["serve", "--help"]])
        self.assertEqual(got["program"], "llama-server")
        self.assertIn("--gpu-layers", got["comparison"]["documented_observed_in_help"])
        self.assertIn("--brand-new", got["comparison"]["unregistered_flags_observed_in_help"])
        self.assertTrue(got["identity"]["cacheable"])
        self.assertEqual(got["binary"]["sha256"], got["identity"]["binary_sha256"])

    def test_fit_params_help_has_no_pinned_registry(self):
        calls = []
        def fake(args, timeout):
            calls.append(args[1:])
            return {"status": "success", "returncode": 0,
                    "stdout": "  -m, --model <file>\n", "stderr": ""}
        with patch.object(probe, "run_cli", side_effect=fake):
            found = probe.probe_binary(self.bin, self.registry, command="fit-params")
        self.assertEqual(calls, [["version"], ["fit-params", "--help"]])
        self.assertIsNone(found["comparison"]["documented_not_observed_in_help"])
        self.assertEqual(found["command_results"]["help"]["documented_flag_presence"],
                         "no_pinned_registry")

    def test_devices_are_opt_in_and_fit_rejected_before_execution(self):
        calls = []
        def fake(args, timeout):
            calls.append(args[1:])
            return {"status": "success", "returncode": 0,
                    "stdout": "  --help\n", "stderr": ""}
        with patch.object(probe, "run_cli", side_effect=fake):
            probe.probe_binary(self.bin, self.registry, include_devices=True)
        self.assertEqual(calls, [["version"], ["serve", "--help"], ["serve", "--list-devices"]])
        with patch.object(probe, "run_cli") as call:
            with self.assertRaises(ValueError):
                probe.probe_binary(self.bin, self.registry, command="fit-params",
                                   include_devices=True)
            call.assert_not_called()

    def test_timeout_yields_uncacheable_unknown(self):
        with patch.object(probe, "run_cli", return_value={
            "status": "timeout", "returncode": None, "stdout": "", "stderr": ""}):
            found = probe.probe_binary(self.bin, self.registry)
        self.assertIsNone(found["comparison"]["documented_not_observed_in_help"])
        self.assertFalse(found["identity"]["cacheable"])

    def test_help_parser_ignores_prose(self):
        raw = "  -h, --help\n  -ngl, --gpu-layers N\n  words mention --wrong option\n"
        self.assertEqual(probe.flags_from_help(raw), {"-h", "--help", "-ngl", "--gpu-layers"})


if __name__ == "__main__":
    unittest.main()
