#!/usr/bin/env python3
"""Observe the installed llama command without loading a model or starting a server."""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from engine_adapter import ROUTES, discover_binary, resolve_entrypoint, probe_argv
from engine_identity import observed_identity, sha256_file
from device_inventory import observation as observe_devices, unknown as unknown_devices
from registry_differences import interpret_differences, inspect_fit_reference
from safe_output import write_text
from runtime_snapshot import check_snapshot

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "cli-inventory.json"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
START_FLAGS = re.compile(r"^-{1,2}[A-Za-z][\w-]*(?:(?:,\s*|\s+)-{1,2}[A-Za-z][\w-]*)*")
FLAG = re.compile(r"-{1,2}[A-Za-z][\w-]*")


def flags_from_help(output):
    found = set()
    for line in ANSI.sub("", output).splitlines():
        match = START_FLAGS.match(line.lstrip())
        if match:
            found.update(f for f in FLAG.findall(match.group(0)) if not f.endswith("-"))
    return found


def commands_from_help_all(output):
    """Parse the upstream app router's 'Available commands:' help-all section."""
    in_commands = False
    found = set()
    for line in ANSI.sub("", output).splitlines():
        if line.strip() == "Available commands:":
            in_commands = True
            continue
        if not in_commands:
            continue
        if not line.strip():
            if found:
                break
            continue
        match = re.match(r"^\s{2,}([a-z][a-z0-9-]*)\s{2,}\S", line)
        if not match:
            # If help format changes, return only verifiably observed commands.
            break
        found.add(match.group(1))
    return found


def run_cli(argv, timeout_seconds):
    """Run an explicit argv, never through a shell."""
    try:
        completed = subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            errors="replace", timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": None, "stdout": "", "stderr": ""}
    except OSError as exc:
        return {"status": "error", "returncode": None, "stdout": "",
                "stderr": type(exc).__name__ + ": " + str(exc)[:500]}
    return {"status": "success" if completed.returncode == 0 else "nonzero_exit",
            "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def compact_result(result, limit=4096):
    return {"status": result["status"], "returncode": result["returncode"],
            "summary": (result["stdout"] or result["stderr"])[:limit]}


def probe_binary(binary, registry, *, command="serve", include_devices=False,
                 timeout_seconds=10):
    """Help/version only. Includes no inferred runtime capability or model results."""
    if not 1 <= timeout_seconds <= 120:
        raise ValueError("timeout must be between 1 and 120 seconds")
    path = discover_binary(binary)
    entry = resolve_entrypoint(path, command)
    # Validate all optional commands before executing anything.
    if include_devices:
        probe_argv(entry, "--list-devices")
    version = run_cli(probe_argv(entry, "version"), timeout_seconds)
    catalog_result = (run_cli(probe_argv(entry, "help all"), timeout_seconds)
                      if entry.kind == "integrated" else None)
    help_result = run_cli(probe_argv(entry, "--help"), timeout_seconds)

    raw_help = help_result["stdout"] + "\n" + help_result["stderr"]
    parsed = (flags_from_help(raw_help)
              if help_result["status"] in ("success", "nonzero_exit") else set())
    observed = bool(parsed)
    rows = [e for e in registry["entries"] if e["program"] == entry.registry_program]
    if observed and rows:
        alias_map = {f: e["name"] for e in rows for f in [e["name"], *e["aliases"]]}
        matched = sorted({alias_map[f] for f in parsed if f in alias_map})
        comparison = {
            "documented_observed_in_help": matched,
            "documented_not_observed_in_help": sorted(
                {e["name"] for e in rows} - set(matched)),
            "unregistered_flags_observed_in_help": sorted(parsed - alias_map.keys()),
        }
        presence = "observed_in_help"
    else:
        # fit-params has no pinned inventory; failed help cannot mean unsupported.
        comparison = {key: None for key in (
            "documented_observed_in_help", "documented_not_observed_in_help",
            "unregistered_flags_observed_in_help")}
        presence = ("no_pinned_registry" if observed and not rows else "unavailable")

    diagnostics = interpret_differences(entry.registry_program, comparison)
    fit_reference = None
    if entry.command == "fit-params":
        curated = json.loads((ROOT / "data/fit-capability-reference.json").read_text("utf-8"))
        fit_reference = inspect_fit_reference(
            parsed if observed else None, curated, registry["upstream"]["commit"])

    binary_sha = sha256_file(path)
    provenance = observed_identity(binary_sha, entry.command,
                                   registry["upstream"]["commit"], version, observed)
    available_commands = None
    catalog_status = "not_applicable"
    if catalog_result is not None:
        catalog_status = catalog_result["status"]
        if catalog_result["status"] in ("success", "nonzero_exit"):
            available = commands_from_help_all(
                catalog_result["stdout"] + "\n" + catalog_result["stderr"])
            if available:
                available_commands = sorted(available)
    commands = {
        "version": compact_result(version),
        "catalog": {
            "status": catalog_status,
            "commands_observed_in_help_all": available_commands,
            "requested_command_observed": (
                entry.command in available_commands if available_commands is not None
                else None),
        },
        "help": {"status": help_result["status"], "returncode": help_result["returncode"],
                 "documented_flag_presence": presence,
                 "flag_names_observed": sorted(parsed) if observed else None},
        "devices": {"status": "not_requested"},
    }
    hardware = unknown_devices("not_requested")
    if include_devices:
        # The pinned upstream supports --list-devices for serve and bench,
        # but never execute it against an unknown build without seeing its flag.
        if "--list-devices" not in parsed:
            commands["devices"] = {"status": "skipped_unverified_flag"}
            hardware = unknown_devices("list_devices_not_observed_in_help")
        else:
            device_output = run_cli(probe_argv(entry, "--list-devices"), timeout_seconds)
            commands["devices"] = compact_result(device_output)
            hardware = observe_devices(device_output)
    return {
        "schema_version": "1.1.0", "kind": "runtime_help_observation",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "program": entry.registry_program,
        "entrypoint": {"kind": entry.kind, "command": entry.command},
        "binary": {"filename": path.name, "sha256": binary_sha},
        "registry_commit": registry["upstream"]["commit"],  # Backward-compatible alias.
        "identity": provenance,
        "command_results": commands,
        "device_inventory": hardware,
        "comparison": comparison,
        "diagnostics": diagnostics,
        "fit_reference": fit_reference,
        "limitations": [
            "No build commit is inferred from the documentation registry SHA.",
            "Help presence does not guarantee accepted values or backend/model support.",
            "An absent or failed --help output is unknown, not proof of unsupported flags.",
            "Fit-params has a curated reference only, not a complete CLI inventory.",
            "Not listed in help all is not by itself proof of unsupported commands.",
            "Run only trusted binaries; --list-devices may initialize GPU drivers.",
            "The --list-devices listing excludes CPU and does not prove NPU support.",
            "Device detection is not proof that a model can execute on that backend.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, help="Trusted executable; PATH llama by default")
    parser.add_argument("--command", choices=tuple(ROUTES), default="serve")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing output JSON")
    parser.add_argument("--include-devices", action="store_true")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        result = probe_binary(args.binary, registry, command=args.command,
                              include_devices=args.include_devices, timeout_seconds=args.timeout)
        text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, text, protected=(
                args.registry, discover_binary(args.binary)),
                overwrite=args.overwrite)
            print("Created capability snapshot:", args.output)
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print("Probe failed:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
