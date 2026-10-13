"""Regression tests for ATYL's documented CLI inventory and tuning registry."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import generate_cli_registry as reg


class CLIRegistryTest(unittest.TestCase):
    def test_server_aliases_and_escaped_pipes(self):
        tick = chr(96)
        source = ("### Common params\n| Argument | Explanation |\n| " + tick +
                  r"-fa, --flash-attn [on\|off\|auto]" + tick +
                  " | Flash<br/>(default: auto)<br/>(env: LLAMA_ARG_FLASH_ATTN) |\n"
                  "<!-- HELP_END -->\n")
        entry = reg.server_entries(source)[0]
        self.assertEqual(entry["name"], "--flash-attn")
        self.assertEqual(entry["aliases"], ["-fa"])
        self.assertEqual(entry["value_syntax"], "[on|off|auto]")
        self.assertEqual(entry["documented_default"], "auto")
        self.assertEqual(entry["environment_variable"], "LLAMA_ARG_FLASH_ATTN")

    def test_deprecated_and_removed(self):
        tick = chr(96)
        raw = ("### Server-specific params\n"
               "| " + tick + "--old N" + tick + " | the argument has been removed |\n"
               "| " + tick + "--other N" + tick + " | deprecated |\n"
               "<!-- HELP_END -->\n")
        self.assertEqual([e["status"] for e in reg.server_entries(raw)],
                         ["removed", "deprecated"])

    def test_bench_wrapped_default(self):
        raw = "\n".join(["usage: llama-bench [options]", "", "options:",
                         "  -h, --help",
                         "test parameters:",
                         "  -ot --override-tensor <tensor name pattern>=<buffer type>;...",
                         "                                            (default: disabled)",
                         chr(96) * 3])
        rows = reg.bench_entries(raw)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["name"], "--override-tensor")
        self.assertEqual(rows[1]["aliases"], ["-ot"])
        self.assertEqual(rows[1]["documented_default"], "disabled")

    def test_inventory_coverage(self):
        inventory = json.loads((ROOT / "data/cli-inventory.json").read_text(encoding="utf-8"))
        knowledge = json.loads((ROOT / "data/tuning-knowledge.json").read_text(encoding="utf-8"))
        self.assertEqual(inventory["upstream"]["commit"], reg.PIN)
        self.assertEqual(knowledge["upstream_commit"], reg.PIN)
        entries = inventory["entries"]
        self.assertEqual(len(entries), 302)
        self.assertEqual(sum(row["program"] == "llama-server" for row in entries), 257)
        self.assertEqual(sum(row["program"] == "llama-bench" for row in entries), 45)
        self.assertEqual(len({row["id"] for row in entries}), len(entries))
        valid = {n for row in entries if row["program"] == "llama-server"
                 for n in [row["name"], *row["aliases"]]}
        self.assertTrue(all(row["flag"] in valid for row in knowledge["entries"]))


if __name__ == "__main__":
    unittest.main()
