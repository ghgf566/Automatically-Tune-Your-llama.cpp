#!/usr/bin/env python3
"""Read-only llama.cpp CLI capability probe. Explicitly runs a trusted local binary."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "cli-inventory.json"
PROGRAMS = {"llama-server", "llama-bench"}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
START_FLAGS = re.compile(
    r"^-{1,2}[A-Za-z][\w-]*(?:(?:,\s*|\s+)-{1,2}[A-Za-z][\w-]*)*"
)
FLAG = re.compile(r"-{1,2}[A-Za-z][\w-]*")


def flags_from_help(output):
    """Conservatively detect flags *listed at the beginning* of help lines."""
    found = set()
    for line in ANSI.sub("", output).splitlines():
        match = START_FLAGS.match(line.lstrip())
        if match:
            found.update(FLAG.findall(match.group(0)))
    return found


def sha256_file(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            checksum.update(chunk)
    return checksum.hexdigest()


def run_cli(binary, flag, timeout_seconds):
    """Never use a shell, stdin, model loading, network calls or server start."""
    try:
        done = subprocess.run(
            [str(binary), flag], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, errors="replace",
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": None, "stdout": "", "stderr": ""}
    except OSError as error:
        return {"status": "error", "returncode": None, "stdout": "",
                "stderr": type(error).__name__ + ": " + str(error)[:500]}
    return {
        "status": "success" if done.returncode == 0 else "nonzero_exit",
        "returncode": done.returncode,
        "stdout": done.stdout,
        "stderr": done.stderr,
    }


def compact_result(command_result, limit=4096):
    """Do not place complete raw CLI output into shareable snapshots."""
    return {
        "status": command_result["status"],
        "returncode": command_result["returncode"],
        "summary": (command_result["stdout"] or command_result["stderr"])[:limit],
    }


def probe_binary(binary, registry, *, include_devices=False, timeout_seconds=10):
    """Observe help/version; optional device enumeration may initialize drivers."""
    path = Path(binary).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Binary path is not a regular file: " + str(path))
    if timeout_seconds <= 0 or timeout_seconds > 120:
        raise ValueError("Timeout must be between 1 and 120 seconds")
    program = path.name.lower().removesuffix(".exe")
    if program not in PROGRAMS:
        raise ValueError("Expected llama-server[.exe] or llama-bench[.exe] (not a wrapper)")
    relevant = [item for item in registry["entries"] if item["program"] == program]
    if not relevant:
        raise ValueError("Registry has no entries for " + program)
    reported = {}
    version = run_cli(path, "--version", timeout_seconds)
    help_result = run_cli(path, "--help", timeout_seconds)
    reported["version"] = compact_result(version)
    reported["help"] = {"status": help_result["status"],
                        "returncode": help_result["returncode"]}
    known_aliases = {alias: item["name"]
                     for item in relevant
                     for alias in [item["name"], *item["aliases"]]}
    # A matching flag in --help indicates documented presence only; it does
    # not prove that the installed build/model/backend accepts every value.
    help_text = help_result["stdout"] + "\n" + help_result["stderr"]
    listed = flags_from_help(help_text)
    documented = sorted({known_aliases[flag] for flag in listed
                         if flag in known_aliases})
    reported["help"]["documented_flag_presence"] = (
        "observed_in_help" if help_result["status"] in ("success", "nonzero_exit")
        else "unavailable"
    )
    if help_result["status"] in ("success", "nonzero_exit"):
        comparison = {
            "documented_observed_in_help": documented,
            "documented_not_observed_in_help": sorted(
                {item["name"] for item in relevant} - set(documented)),
            "unregistered_flags_observed_in_help": sorted(listed - set(known_aliases)),
        }
    else:
        # A failed/expired help call produces NO usable capability evidence.
        comparison = {
            "documented_observed_in_help": None,
            "documented_not_observed_in_help": None,
            "unregistered_flags_observed_in_help": None,
        }
    if include_devices:
        devices = run_cli(path, "--list-devices", timeout_seconds)
        reported["devices"] = compact_result(devices)
    else:
        reported["devices"] = {"status": "not_requested"}
    return {
        "schema_version": "1.0.0",
        "kind": "runtime_help_observation",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "program": program,
        "binary": {"filename": path.name, "sha256": sha256_file(path)},
        "registry_commit": registry["upstream"]["commit"],
        "command_results": reported,
        "comparison": comparison,
        "limitations": [
            "Presence in --help does not prove model/backend compatibility.",
            "Not observed in --help does not prove a flag is unsupported.",
            "Only trusted binaries explicitly selected by the user should be probed.",
            "No model was loaded; device enumeration is opt-in and may initialize drivers.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path,
                        help="Path to a trusted llama-server or llama-bench executable")
    parser.add_argument("--registry", type=Path, default=REGISTRY,
                        help="ATYL pinned CLI inventory")
    parser.add_argument("--output", type=Path, help="Write JSON snapshot instead of stdout")
    parser.add_argument("--include-devices", action="store_true",
                        help="Also invoke --list-devices (may initialize GPU drivers)")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        snapshot = probe_binary(args.binary, registry, include_devices=args.include_devices,
                                timeout_seconds=args.timeout)
        rendered = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print("Created capability snapshot:", args.output)
        else:
            print(rendered, end="")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("Probe failed:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
