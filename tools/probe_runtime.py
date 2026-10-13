#!/usr/bin/env python3
"""Read-only llama.cpp CLI probe; official installed 'llama' is the primary entry point.

No model load, server start, benchmark or installation occurs. --list-devices is opt-in.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "cli-inventory.json"

# Names below are confirmed in llama.cpp's pinned app/llama.cpp command router.
ROUTES = {"serve": "llama-server", "bench": "llama-bench"}
LEGACY = {"llama-server": "serve", "llama-bench": "bench"}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
START_FLAGS = re.compile(r"^-{1,2}[A-Za-z][\w-]*(?:(?:,\s*|\s+)-{1,2}[A-Za-z][\w-]*)*")
FLAG = re.compile(r"-{1,2}[A-Za-z][\w-]*")


def discover_binary(binary=None):
    """Prefer the official llama executable on PATH. Never install or update it."""
    if binary is None:
        found = shutil.which("llama") or shutil.which("llama.exe")
        if not found:
            raise ValueError("Official 'llama' not found in PATH; install from https://llama.app")
        path = Path(found).resolve()
    else:
        supplied = Path(binary).expanduser()
        if supplied.is_file():
            path = supplied.resolve()
        elif supplied.name.lower() in {"llama", "llama.exe"}:
            found = shutil.which(str(supplied))
            if not found:
                raise ValueError("Unable to find 'llama' in PATH")
            path = Path(found).resolve()
        else:
            raise ValueError("Binary path is not a file: " + str(supplied))
    if not path.is_file():
        raise ValueError("Executable path is not a file: " + str(path))
    return path


def determine_route(path, command="serve"):
    """Return (kind, subcommand, registry_program); legacy is explicit compatibility."""
    name = path.name.lower().removesuffix(".exe")
    if command not in ROUTES:
        raise ValueError("Supported capability commands are serve and bench")
    if name == "llama":
        return "integrated", command, ROUTES[command]
    if name in LEGACY:
        expected = LEGACY[name]
        if command != expected:
            raise ValueError("The legacy binary " + name + " requires command " + expected)
        return "legacy", expected, name
    raise ValueError("Expected official llama[.exe] or a supported legacy llama-* binary")


def argv_for(kind, binary, command, action):
    executable = str(binary)
    if kind == "integrated":
        if action == "version":
            return [executable, "version"]
        return [executable, command, action]
    return [executable, action]


def flags_from_help(output):
    """Find option *headings*, not flags merely mentioned in prose."""
    found = set()
    for line in ANSI.sub("", output).splitlines():
        match = START_FLAGS.match(line.lstrip())
        if match:
            found.update(FLAG.findall(match.group(0)))
    return found


def sha256_file(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def run_cli(argv, timeout_seconds):
    """Executes only vetted read-only flags, without a shell or stdin."""
    try:
        done = subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, errors="replace", timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": None, "stdout": "", "stderr": ""}
    except OSError as error:
        return {"status": "error", "returncode": None, "stdout": "",
                "stderr": type(error).__name__ + ": " + str(error)[:500]}
    return {"status": "success" if done.returncode == 0 else "nonzero_exit",
            "returncode": done.returncode,
            "stdout": done.stdout, "stderr": done.stderr}


def compact_result(result, limit=4096):
    return {"status": result["status"], "returncode": result["returncode"],
            "summary": (result["stdout"] or result["stderr"])[:limit]}


def probe_binary(binary, registry, *, command="serve", include_devices=False,
                 timeout_seconds=10):
    """Probe a trusted installed binary; no implicit update and no model execution."""
    path = discover_binary(binary)
    if not 1 <= timeout_seconds <= 120:
        raise ValueError("Timeout must be between 1 and 120 seconds")
    kind, subcommand, program = determine_route(path, command)
    relevant = [row for row in registry["entries"] if row["program"] == program]
    if not relevant:
        raise ValueError("Registry has no entries for " + program)

    version = run_cli(argv_for(kind, path, subcommand, "version"), timeout_seconds)
    help_result = run_cli(argv_for(kind, path, subcommand, "--help"), timeout_seconds)
    aliases = {alias: row["name"] for row in relevant
               for alias in [row["name"], *row["aliases"]]}
    help_good = help_result["status"] in ("success", "nonzero_exit")
    # A nonzero exit may emit valid help, but an empty help output is inconclusive.
    raw = help_result["stdout"] + "\n" + help_result["stderr"]
    parsed = flags_from_help(raw) if help_good else set()
    observable = help_good and bool(parsed)
    if observable:
        matched = sorted({aliases[f] for f in parsed if f in aliases})
        comparison = {
            "documented_observed_in_help": matched,
            "documented_not_observed_in_help": sorted(
                {e["name"] for e in relevant} - set(matched)),
            "unregistered_flags_observed_in_help": sorted(parsed - aliases.keys()),
        }
    else:
        comparison = {
            "documented_observed_in_help": None,
            "documented_not_observed_in_help": None,
            "unregistered_flags_observed_in_help": None,
        }

    reported = {
        "version": compact_result(version),
        "help": {"status": help_result["status"], "returncode": help_result["returncode"],
                 "documented_flag_presence": "observed_in_help" if observable else "unavailable"},
        "devices": {"status": "not_requested"},
    }
    if include_devices:
        reported["devices"] = compact_result(
            run_cli(argv_for(kind, path, subcommand, "--list-devices"), timeout_seconds))
    return {
        "schema_version": "1.0.0", "kind": "runtime_help_observation",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "program": program, "entrypoint": {"kind": kind, "command": subcommand},
        "binary": {"filename": path.name, "sha256": sha256_file(path)},
        "registry_commit": registry["upstream"]["commit"],
        "command_results": reported, "comparison": comparison,
        "limitations": [
            "Presence in --help does not prove model/backend or value compatibility.",
            "Not observed in --help does not prove a flag is unsupported.",
            "Only explicitly selected or PATH-discovered trusted binaries should be probed.",
            "This does not load a model; opt-in device enumeration can initialize drivers.",
            "A llama update changes binary identity; cached results must be invalidated.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=None,
                        help="Trusted explicit executable; default official llama on PATH")
    parser.add_argument("--command", choices=tuple(ROUTES), default="serve",
                        help="Official integrated CLI subcommand, default serve")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output", type=Path, help="Write JSON instead of stdout")
    parser.add_argument("--include-devices", action="store_true",
                        help="Opt-in --list-devices; may initialize GPU drivers")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        snapshot = probe_binary(args.binary, registry, command=args.command,
                                include_devices=args.include_devices,
                                timeout_seconds=args.timeout)
        result = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding="utf-8")
            print("Created capability snapshot:", args.output)
        else:
            print(result, end="")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("Probe failed:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
