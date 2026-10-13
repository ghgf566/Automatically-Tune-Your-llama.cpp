"""Runtime snapshot validation regression cases; no real executables."""
import sys
import unittest
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from runtime_snapshot import check_snapshot, SnapshotValidationError


def snapshot():
    sha = "a" * 64
    return {
        "schema_version": "1.1.0", "kind": "runtime_help_observation",
        "binary": {"sha256": sha}, "identity": {"binary_sha256": sha},
        "entrypoint": {"kind": "integrated", "command": "serve"},
        "command_results": {
            "version": {"status": "success"},
            "help": {"status": "success", "flag_names_observed": ["--list-devices"]},
            "devices": {"status": "success"},
        },
        "device_inventory": {
            "status": "observed", "devices": [
                {"id": "Vulkan0", "total_memory_mib": 8192,
                 "free_memory_mib": 7000}
            ],
        },
    }


class RuntimeSnapshotTests(unittest.TestCase):
    def test_valid_observation(self):
        self.assertEqual(check_snapshot(snapshot())["kind"], "runtime_help_observation")

    def test_reject_malformed_nested_structures(self):
        bad = [
            lambda s: s.update(binary=None),
            lambda s: s["command_results"].update(help=None),
            lambda s: s["command_results"]["help"].update(flag_names_observed=[2]),
            lambda s: s["command_results"]["help"].update(flag_names_observed=["--help", "--help"]),
            lambda s: s["entrypoint"].update(command="update"),
            lambda s: s.update(schema_version="0.1"),
        ]
        for change in bad:
            s = snapshot()
            change(s)
            with self.subTest(change=str(change)), self.assertRaises(SnapshotValidationError):
                check_snapshot(s)

    def test_engine_identity_must_match(self):
        s = snapshot()
        s["identity"]["binary_sha256"] = "b" * 64
        with self.assertRaisesRegex(SnapshotValidationError, "disagrees"):
            check_snapshot(s)

    def test_device_status_must_not_invent_arrays(self):
        s = snapshot()
        s["device_inventory"] = {"status": "unknown", "devices": []}
        with self.assertRaisesRegex(SnapshotValidationError, "must be null"):
            check_snapshot(s)
        s["device_inventory"] = {"status": "observed", "devices": None}
        with self.assertRaises(SnapshotValidationError):
            check_snapshot(s)

    def test_device_free_memory_cannot_exceed_total(self):
        s = snapshot()
        s["device_inventory"]["devices"][0]["free_memory_mib"] = 9000
        with self.assertRaisesRegex(SnapshotValidationError, "free than total"):
            check_snapshot(s)


if __name__ == "__main__":
    unittest.main()
