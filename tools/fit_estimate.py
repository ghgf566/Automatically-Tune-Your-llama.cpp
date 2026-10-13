#!/usr/bin/env python3
"""Plan an explicit, opt-in offline llama fit-params MEMORY ESTIMATE.

A plan does not load a model or run llama. --execute is required for a real
memory estimate; it does not execute fitted arguments or launch a server.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from engine_adapter import command_prefix, discover_binary, resolve_entrypoint
from engine_identity import sha256_file
from inspect_gguf import GGUFMetadataError, inspect
from environment_observation import (clean_llama_env, observe as observe_environment,
                                     summarize as summarize_environment)

REQUIRED_FIT_FLAGS = frozenset({
    "--model", "--fit-print", "--fit", "--ctx-size", "--n-gpu-layers",
    "--device", "--offline",
})
MAX_STDIO_BYTES = 16384
MIN_CONTEXT = 128
MAX_CONTEXT = 262144
MAX_GPU_LAYERS = 4096
MAX_TIMEOUT_SECONDS = 1800


class FitPlanError(ValueError):
    pass


def _snapshot(path):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise FitPlanError("cannot read runtime capability snapshot") from error
    if not isinstance(doc, dict):
        raise FitPlanError("runtime capability snapshot is not an object")
    return doc


def _same_engine(snapshot, sha):
    fields = [snapshot.get(key) for key in ("binary", "identity", "entrypoint")]
    if not all(isinstance(value, dict) for value in fields):
        return False
    binary, identity, entrypoint = fields
    return (binary.get("sha256") == sha
            and identity.get("binary_sha256") == sha
            and entrypoint.get("kind") == "integrated")


def _validate_fit_snapshot(doc, sha):
    if not _same_engine(doc, sha) or doc.get("entrypoint", {}).get("command") != "fit-params":
        raise FitPlanError("fit snapshot belongs to another engine/build or command")
    version = doc.get("command_results", {}).get("version", {})
    help_observation = doc.get("command_results", {}).get("help", {})
    observed = help_observation.get("flag_names_observed")
    if (version.get("status") != "success" or help_observation.get("status") != "success"
            or not isinstance(observed, list)):
        raise FitPlanError("fit snapshot lacks successful version/help evidence")
    missing = REQUIRED_FIT_FLAGS - set(observed)
    if missing:
        raise FitPlanError("fit snapshot missing required flags: " + ", ".join(sorted(missing)))


def _validate_device_snapshot(doc, sha, device_id):
    if not _same_engine(doc, sha):
        raise FitPlanError("device snapshot belongs to another engine/build")
    result = doc.get("device_inventory", {})
    if (doc.get("command_results", {}).get("devices", {}).get("status") != "success"
            or result.get("status") != "observed"):
        raise FitPlanError("device snapshot has no confirmed GGML device listing")
    devices = result.get("devices")
    if not isinstance(devices, list) or not all(isinstance(item, dict) for item in devices):
        raise FitPlanError("device snapshot has invalid device structure")
    if device_id not in {item.get("id") for item in devices}:
        raise FitPlanError("requested device ID not present in verified listing")


def build_plan(model, fit_snapshot, *, device="none", devices_snapshot=None,
               context=4096, gpu_layers=0, binary=None, scenario="daily", background_note=None):
    """Fail closed on stale engine snapshots; preview a single fixed configuration."""
    if scenario not in ("daily", "reduced_background"):
        raise FitPlanError("scenario must be daily or reduced_background")
    if background_note is not None and len(background_note) > 300:
        raise FitPlanError("background note must have at most 300 characters")
    if not (MIN_CONTEXT <= context <= MAX_CONTEXT):
        raise FitPlanError("context must be between 128 and 262144")
    if not (0 <= gpu_layers <= MAX_GPU_LAYERS):
        raise FitPlanError("gpu layers must be between 0 and 4096")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", device):
        raise FitPlanError("select one specific GGML device ID or 'none'")
    if device == "none" and gpu_layers != 0:
        raise FitPlanError("CPU-only device none requires --gpu-layers 0")
    if device != "none" and gpu_layers == 0:
        raise FitPlanError("GPU device requires a positive explicit --gpu-layers")
    model_path = Path(model).expanduser().resolve()
    # Fresh read prevents accepting metadata from an unrelated or stale JSON.
    model_info = inspect(model_path)
    if model_info["split"]["multiple_parts_reported"]:
        raise FitPlanError("multi-part GGUF requires a shard-completeness validator first")
    path = discover_binary(binary)
    entry = resolve_entrypoint(path, "fit-params")
    if entry.kind != "integrated":
        raise FitPlanError("use official integrated llama for the first controlled estimator")
    binary_sha = sha256_file(path)
    fit_doc = _snapshot(fit_snapshot)
    _validate_fit_snapshot(fit_doc, binary_sha)
    if device != "none":
        if not devices_snapshot:
            raise FitPlanError("a recent device snapshot is required for GPU execution")
        _validate_device_snapshot(_snapshot(devices_snapshot), binary_sha, device)
    argv = command_prefix(entry) + [
        "--offline", "--model", str(model_path),
        "--device", device,
        "--ctx-size", str(context),
        "--n-gpu-layers", str(gpu_layers),
        "--fit", "off", "--fit-print", "on",
    ]
    return {
        "schema_version": "0.2.0", "kind": "fit_estimate_plan",
        "status": "prepared_not_executed",
        "engine_sha256": binary_sha,
        "model": {
            "filename": model_info["source"]["filename"],
            "file_size_bytes": model_info["source"]["file_size_bytes"],
            "architecture": model_info["general"]["architecture"],
            "gguf_version": model_info["gguf_version"],
        },
        "candidate": {
            "device_id": device, "context_tokens": context, "gpu_layers": gpu_layers,
            "fit_adjustment": "off", "output_mode": "estimated_memory",
        },
        "argv": argv,
        "estimate": None,
        "environment": {
            "scenario": scenario,
            "scenario_is_user_declared": True,
            "background_note": background_note,
            "capture_status": "not_sampled",
            "before": None, "after": None, "available_memory_change": None,
        },
        "limitations": [
            "Plan generation never reads tensor payloads or launches llama.",
            "Executing runs llama fit-params for a requested offline memory estimate.",
            "Runtime snapshots are observations tied to a binary, not a proof of compatibility.",
            "GGML estimates are not a validated model load, speed benchmark or guaranteed VRAM fit.",
            "Before/after headroom is not a measure of peak model/process memory use.",
            "Background-load labels are declared by the user, never verified from processes.",
            "Command argv includes local file paths; redact before sharing.",
        ],
    }


def execute_plan(plan, *, timeout_seconds=600):
    """Execute exactly the prepared argv with bounded output, no shell or server."""
    if not (1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS):
        raise FitPlanError("timeout must be in [1, 1800] seconds")
    if plan["status"] != "prepared_not_executed" or plan["kind"] != "fit_estimate_plan":
        raise FitPlanError("only an unexecuted fit estimate plan can run")
    argv = plan["argv"]
    if (len(argv) != 15 or argv[1:3] != ["fit-params", "--offline"]
            or argv[-4:] != ["--fit", "off", "--fit-print", "on"]
            or argv[3] != "--model" or argv[5] != "--device"
            or argv[7] != "--ctx-size" or argv[9] != "--n-gpu-layers"):
        raise FitPlanError("unexpected fit estimate command shape")
    if sha256_file(Path(argv[0])) != plan["engine_sha256"]:
        raise FitPlanError("engine changed since preview; create a new fit plan")
    if Path(argv[4]).stat().st_size != plan["model"]["file_size_bytes"]:
        raise FitPlanError("model file size changed since preview; create a new fit plan")
    # Upstream may emit verbose logs; send to temporary disk files, not
    # unbounded subprocess.PIPE memory. Do not invoke a shell.
    try:
        with tempfile.TemporaryFile(mode="w+b") as out, tempfile.TemporaryFile(mode="w+b") as err:
            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                env=clean_llama_env(),
            )
            try:
                rc = process.wait(timeout=timeout_seconds)
                status = "completed" if rc == 0 else "nonzero_exit"
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                rc, status = None, "timeout"
            out.seek(0)
            err.seek(0)
            stdout = out.read(MAX_STDIO_BYTES).decode("utf-8", "replace")
            stderr = err.read(MAX_STDIO_BYTES).decode("utf-8", "replace")
    except OSError as exc:
        return {
            "status": "process_error", "returncode": None,
            "stdout_excerpt": "", "stderr_excerpt": type(exc).__name__,
            "output_truncated_or_unknown": True,
            "estimate_unit": "MiB (as reported by upstream, if recognizable)",
            "parsed_numeric_memory": None,
            "model_load_tested": False, "inference_tested": False,
        }
    return {
        "status": status, "returncode": rc,
        "stdout_excerpt": stdout, "stderr_excerpt": stderr,
        "output_truncated_or_unknown": True,
        "estimate_unit": "MiB (as reported by upstream, if recognizable)",
        "parsed_numeric_memory": None,
        "model_load_tested": False, "inference_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fit-snapshot", type=Path, required=True)
    parser.add_argument("--device-snapshot", type=Path)
    parser.add_argument("--device", default="none")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--scenario", choices=("daily", "reduced_background"), default="daily")
    parser.add_argument("--background-note", help="Optional manual note (e.g. Wallpaper Engine running)")
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly run llama fit-params (may read model files and initialize GPU)")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        plan = build_plan(args.model, args.fit_snapshot, device=args.device,
                          devices_snapshot=args.device_snapshot, context=args.context,
                          gpu_layers=args.gpu_layers, binary=args.binary,
                          scenario=args.scenario, background_note=args.background_note)
        # Reject destructive destinations BEFORE optional subprocess execution.
        if args.output:
            destination = args.output.expanduser().resolve()
            protected = {Path(args.model).expanduser().resolve(),
                         Path(args.fit_snapshot).expanduser().resolve(),
                         Path(plan["argv"][0]).resolve()}
            if args.device_snapshot:
                protected.add(Path(args.device_snapshot).expanduser().resolve())
            if destination in protected:
                raise FitPlanError("output must not overwrite model, snapshot or engine")
        if args.execute:
            environment = plan["environment"]
            environment["before"] = observe_environment(
                plan["argv"][0], plan["candidate"]["device_id"])
            try:
                plan["estimate"] = execute_plan(plan, timeout_seconds=args.timeout)
                plan["status"] = "estimate_attempted"
            finally:
                environment["after"] = observe_environment(
                    plan["argv"][0], plan["candidate"]["device_id"])
                environment["available_memory_change"] = summarize_environment(
                    environment["before"], environment["after"])
                environment["capture_status"] = "captured_best_effort"
        text = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print("Created fit estimate plan:", args.output)
        else:
            print(text, end="")
    except (OSError, ValueError) as exc:
        print("Fit estimate plan rejected:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
