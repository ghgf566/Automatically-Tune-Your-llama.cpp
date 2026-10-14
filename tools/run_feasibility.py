#!/usr/bin/env python3
"""Prepare or explicitly run one bounded llama-server model-load check.

Without --execute this only writes a plan. With --execute it starts an isolated
server bound to 127.0.0.1, polls GET /health, and always stops the owned process
tree immediately after readiness or failure. It never sends inference requests.
"""
import argparse
import http.client
import json
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from candidate_identity import canonical_digest
from engine_adapter import command_prefix, discover_binary, resolve_entrypoint
from engine_identity import sha256_file
from environment_observation import clean_llama_env, system_ram
from inspect_gguf import inspect
from model_identity import fingerprint
from plan_speculative_search import PREVIEW_VALIDATOR
from process_runner import run_until_ready
from process_memory import MIB, sample_process_tree
from device_memory import DeviceMemoryMonitor
from runtime_snapshot import SnapshotValidationError, check_snapshot
from safe_output import write_text
from validate_contracts import (
    ROOT, _registry, canonical_cli_name, load_json, validate_profile_document,
)
from verify_model_pair import PairingError, _verified_model, verify_pair

SPEC_SCHEMA = load_json(ROOT / "schemas/speculative-search.schema.json")
GGUF_VALIDATOR = Draft202012Validator(load_json(ROOT / "schemas/gguf-observation.schema.json"))
REPORT_SCHEMA = load_json(ROOT / "schemas/feasibility-run.schema.json")
Draft202012Validator.check_schema(REPORT_SCHEMA)
REPORT_VALIDATOR = Draft202012Validator(REPORT_SCHEMA, registry=_registry())
INVENTORY = load_json(ROOT / "data/cli-inventory.json")
ALLOWED_PROFILE_FLAGS = {
    "--gpu-layers", "--device", "--threads", "--threads-batch",
    "--batch-size", "--ubatch-size", "--cache-type-k", "--cache-type-v",
    "--flash-attn", "--split-mode", "--tensor-split", "--main-gpu",
    "--numa", "--mlock", "--mmap", "--kv-offload", "--op-offload",
}
REQUIRED_FLAGS = {"--model", "--ctx-size", "--parallel", "--host", "--port",
                  "--fit", "--ui", "--spec-type"}
MAX_TIMEOUT_SECONDS = 1800
OOM_TEXT = re.compile(r"out[ -]of[ -]memory|oom|allocation failed|failed to allocate", re.I)


class FeasibilityError(ValueError):
    pass


def _serve_flag_rows():
    return [item for item in INVENTORY["entries"] if item["program"] == "llama-server"]


def _canonical(flag):
    return canonical_cli_name("llama-server", flag, INVENTORY)


def _observed_flags(runtime):
    result = runtime["command_results"]["help"]
    raw = result.get("flag_names_observed")
    if result.get("status") != "success" or not isinstance(raw, list):
        raise FeasibilityError("runtime snapshot lacks successful serve Help evidence")
    return set(raw), {_canonical(item) for item in raw}


def _flag_row(name):
    matches = [item for item in _serve_flag_rows() if item["name"] == name]
    if len(matches) != 1:
        raise FeasibilityError("CLI registry does not define " + name)
    return matches[0]


def _text_value(value):
    if isinstance(value, bool) or value is None:
        raise FeasibilityError("profile parameter values must be strings or numbers")
    if isinstance(value, (str, int, float)):
        result = str(value)
    elif isinstance(value, list) and all(
            isinstance(item, (str, int, float)) and not isinstance(item, bool)
            for item in value):
        result = ",".join(str(item) for item in value)
    else:
        raise FeasibilityError("unsupported profile CLI parameter value")
    if not result or any(ord(char) < 32 for char in result):
        raise FeasibilityError("empty or control-character CLI value is not allowed")
    return result


def _profile_argv(profile, runtime_flags, context, concurrency):
    values = {}
    for source in (profile["user_locked_parameters"], profile["candidate_parameters"]):
        for flag, value in source.items():
            canonical = _canonical(flag)
            if canonical in values:
                raise FeasibilityError("duplicate canonical Profile parameter: " + canonical)
            values[canonical] = value

    if "--fit" in values:
        raise FeasibilityError("runner owns --fit and always sets it to off")
    for field, expected in (("--ctx-size", context), ("--parallel", concurrency)):
        if field in values:
            if str(values.pop(field)) != str(expected):
                raise FeasibilityError(field + " must match the fixed workload Profile")

    # The registry canonical name is --gpu-layers; --n-gpu-layers remains an
    # accepted spelling in Profiles but is normalized by _canonical above.
    if "--gpu-layers" not in values:
        raise FeasibilityError("Profile must explicitly set --gpu-layers for load feasibility")
    n_gpu_layers = values["--gpu-layers"]
    try:
        n_gpu_layers = int(n_gpu_layers)
    except (TypeError, ValueError) as error:
        raise FeasibilityError("--gpu-layers must be an explicit integer") from error
    if n_gpu_layers < 0:
        raise FeasibilityError("--gpu-layers must not be negative")
    if n_gpu_layers > 0 and "--device" not in values:
        raise FeasibilityError("GPU offload requires an explicit --device in the Profile")
    if n_gpu_layers == 0 and "--device" not in values:
        values["--device"] = "none"

    argv = []
    for canonical in sorted(values):
        if canonical not in ALLOWED_PROFILE_FLAGS:
            raise FeasibilityError("Profile flag is outside the load-check allowlist: " + canonical)
        if canonical not in runtime_flags:
            raise FeasibilityError("installed serve Help did not expose " + canonical)
        row = _flag_row(canonical)
        value = _text_value(values[canonical])
        if row["arity"] == "switch":
            raise FeasibilityError("switch-valued Profile parameters need an explicit value form")
        argv.extend((canonical, value))
    return argv


def _reserve_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _health_check(port):
    """Only accept the pinned server's ready response; never use proxy env vars."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/health", headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read(4096)
        if response.status != 200:
            return False
        try:
            return json.loads(body.decode("utf-8")) == {"status": "ok"}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def build_plan(profile, preview, model_observation, runtime_snapshot, candidate_id,
               target_path, *, draft_path=None, binary=None, timeout_seconds=600):
    """Revalidate all recorded identities, then prepare one bounded server argv."""
    if not (1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS):
        raise FeasibilityError("timeout must be between 1 and 1800 seconds")
    try:
        validate_profile_document(profile)
        GGUF_VALIDATOR.validate(model_observation)
        PREVIEW_VALIDATOR.validate(preview)
        runtime = check_snapshot(runtime_snapshot)
    except (KeyError, ValidationError, ValueError, SnapshotValidationError) as error:
        raise FeasibilityError("invalid Profile, model, preview or runtime evidence: " + str(error)) from error
    if profile["workload"]["mode"] != "server" or profile["engine"].get("command") != "serve":
        raise FeasibilityError("feasibility load checks require a server workload and serve engine")
    if preview["profile_sha256"] != canonical_digest(profile):
        raise FeasibilityError("candidate preview belongs to a different Profile")
    if preview["model_observation_sha256"] != canonical_digest(model_observation):
        raise FeasibilityError("candidate preview belongs to different GGUF metadata")
    if preview["engine_binary_sha256"] != runtime["binary"]["sha256"]:
        raise FeasibilityError("candidate preview and runtime snapshot use different engines")
    if runtime["entrypoint"]["command"] != "serve":
        raise FeasibilityError("runtime snapshot is not for serve")

    path = discover_binary(binary or profile["engine"]["binary"])
    entry = resolve_entrypoint(path, "serve")
    binary_sha = sha256_file(path)
    if binary_sha != runtime["binary"]["sha256"]:
        raise FeasibilityError("on-disk engine SHA differs from the runtime snapshot")
    if (profile["engine"].get("binary_sha256") is not None
            and profile["engine"]["binary_sha256"].lower() != binary_sha.lower()):
        raise FeasibilityError("Profile engine SHA differs from the on-disk engine")

    selected = [item for item in preview["candidates"] if item["candidate_id"] == candidate_id]
    if len(selected) != 1:
        raise FeasibilityError("candidate ID must select exactly one preview candidate")
    candidate = selected[0]
    if candidate["readiness"] == "blocked_missing_evidence":
        raise FeasibilityError("blocked candidate cannot enter load feasibility")
    mode = candidate["mode"]
    if mode not in profile["speculative_search"]["modes"]:
        raise FeasibilityError("candidate mode is not declared in the Profile")

    model_info = _verified_model(target_path, profile["model"]["sha256"])
    if model_info["architecture"] != profile["model"]["architecture"]:
        raise FeasibilityError("on-disk target architecture differs from the workload Profile")
    if model_info["metadata_sha256"] != model_observation["metadata_sha256"]:
        raise FeasibilityError("on-disk target metadata differs from the preview observation")

    pair_report = None
    draft_full_path = None
    if mode.startswith("external_"):
        if not draft_path:
            raise FeasibilityError("external candidate requires an explicitly selected draft GGUF")
        pair_report = verify_pair(profile, candidate["draft_artifact_id"],
                                  target_path, draft_path)
        if pair_report["status"] != "requires_runtime_load_verification":
            raise FeasibilityError("target/draft metadata preflight is blocked")
        draft_full_path = str(Path(draft_path).expanduser().resolve())
        if candidate["draft_device_id"] is None or candidate["draft_gpu_layers"] is None:
            raise FeasibilityError("external candidate must declare an explicit draft device and layer placement")
    elif draft_path is not None:
        raise FeasibilityError("draft file is only accepted for an external speculative candidate")
    elif mode == "embedded_mtp":
        nextn = model_observation["architecture_metadata"].get("nextn_predict_layers")
        if not isinstance(nextn, int) or isinstance(nextn, bool) or nextn <= 0:
            raise FeasibilityError("embedded MTP candidate lacks positive NextN metadata")

    raw_flags, observed_flags = _observed_flags(runtime)
    missing = REQUIRED_FLAGS - observed_flags
    if missing:
        raise FeasibilityError("installed serve Help did not expose required flags: "
                               + ", ".join(sorted(missing)))
    no_ui_aliases = {"--no-ui", "--no-webui"}
    if not (raw_flags & no_ui_aliases):
        raise FeasibilityError("installed serve Help did not expose an explicit UI-disable flag")
    spec_row = _flag_row("--spec-type")
    spec_values = set((spec_row.get("value_syntax") or "").split(","))
    mode_value = {
        "off": "none", "embedded_mtp": "draft-mtp",
        "external_mtp": "draft-mtp", "external_draft_simple": "draft-simple",
    }[mode]
    if mode_value not in spec_values:
        raise FeasibilityError("pinned CLI registry does not list this speculative mode")
    if mode != "off":
        for flag in ("--spec-draft-n-max",):
            if flag not in observed_flags:
                raise FeasibilityError("installed serve Help did not expose " + flag)
    if mode.startswith("external_"):
        for flag in ("--spec-draft-model", "--spec-draft-device", "--spec-draft-ngl"):
            if flag not in observed_flags:
                raise FeasibilityError("installed serve Help did not expose " + flag)

    context = profile["workload"]["context_tokens"]
    concurrency = profile["workload"]["concurrency"]
    param_argv = _profile_argv(profile, observed_flags, context, concurrency)
    if "--ctx-size" not in observed_flags or "--parallel" not in observed_flags:
        raise FeasibilityError("installed serve Help lacks fixed workload flags")
    port = _reserve_local_port()
    argv = command_prefix(entry) + [
        "--model", str(Path(target_path).expanduser().resolve()),
        "--ctx-size", str(context), "--parallel", str(concurrency),
        "--fit", "off", "--no-ui", "--host", "127.0.0.1", "--port", str(port),
        *param_argv,
    ]
    if mode_value == "none":
        argv.extend(("--spec-type", "none"))
    else:
        argv.extend(("--spec-type", mode_value,
                     "--spec-draft-n-max", str(candidate["draft_n_max"])))
    if mode.startswith("external_"):
        argv.extend(("--spec-draft-model", draft_full_path,
                     "--spec-draft-device", candidate["draft_device_id"],
                     "--spec-draft-ngl", str(candidate["draft_gpu_layers"])))

    redact = {str(path): "<llama-executable>",
              str(Path(target_path).expanduser().resolve()): "<target-model>"}
    if draft_full_path:
        redact[draft_full_path] = "<draft-model>"
    display_argv = [redact.get(item, item) for item in argv]
    return {
        "schema_version": "0.2.0", "kind": "feasibility_load_check",
        "status": "prepared_not_executed", "candidate_id": candidate_id,
        "profile_id": profile["profile_id"],
        "profile_sha256": canonical_digest(profile),
        "engine_binary_sha256": binary_sha,
        "mode": mode, "requested_draft_n_max": candidate["draft_n_max"],
        "artifact_id": candidate["draft_artifact_id"],
        "target": model_info,
        "draft": pair_report["draft"] if pair_report else None,
        "pair_preflight": pair_report,
        "host": "127.0.0.1", "port": port,
        "timeout_seconds": timeout_seconds,
        "launched_argv_redacted": display_argv,
        "load_result": None,
        "memory_observation": None,
        "limitations": [
            "Plan creation never launches llama; --execute is required for a real model load.",
            "Execution starts a local server, waits for /health readiness, then stops its owned process tree.",
            "No inference request, throughput benchmark, acceptance counter, or model-quality check is performed.",
            "Host availability is system-wide; the load report separately records a 500 ms sampled process-tree Working Set/RSS maximum, which can miss spikes and double-count shared pages.",
            "NVIDIA device free/used memory is sampled system-wide once per second when GPU offload is requested; it is not process-attributed, not mapped to a llama device ID, and is not a process VRAM peak.",
            "External server features and model bundles outside single-file GGUF inputs are not accepted.",
        ],
        "_argv": argv,
    }


def execute_plan(plan, *, cancel_event=None):
    if plan["status"] != "prepared_not_executed":
        raise FeasibilityError("only an unexecuted load-check plan can run")
    before = system_ram()
    samples = []
    process_samples = []
    process_metric = None
    process_count_max = 0
    process_coverage = []

    def sample_memory(root_pid):
        nonlocal process_metric, process_count_max
        try:
            sample = sample_process_tree(root_pid)
        except Exception:
            # Resource sampling is advisory and must not suppress load outcome.
            return
        if sample["status"] == "observed":
            process_samples.append(sample["resident_bytes"])
            process_metric = sample["metric"]
            process_count_max = max(process_count_max, sample["process_count"])
            process_coverage.append(sample["coverage_status"])

    def ready_check():
        sample = system_ram()
        if sample["status"] == "observed":
            samples.append(sample["available_mib"])
        return _health_check(plan["port"])

    gpu_offload_requested = False
    for flag in ("--gpu-layers", "--spec-draft-ngl"):
        if flag in plan["_argv"]:
            gpu_offload_requested |= int(plan["_argv"][plan["_argv"].index(flag) + 1]) > 0
    device_monitor = DeviceMemoryMonitor() if gpu_offload_requested else None
    if device_monitor:
        device_monitor.start()
    started = time.monotonic()
    try:
        result = run_until_ready(
            plan["_argv"], plan["timeout_seconds"], ready_check,
            capture_bytes=65536, output_limit_bytes=8 * 1024 * 1024,
            env=clean_llama_env(), poll_interval=0.5, cancel_event=cancel_event,
            sample_callback=sample_memory,
        )
    finally:
        device_observation = (device_monitor.stop_and_report() if device_monitor else {
            "status": "unknown", "reason": "gpu_offload_not_requested",
            "provider": None, "process_attribution": "not_available",
            "sample_count": 0, "sample_interval_ms": 1000, "devices": [],
        })
    elapsed = round(time.monotonic() - started, 3)
    after = system_ram()
    status_map = {
        "ready": "load_verified", "nonzero_exit": "load_failed",
        "exited_before_ready": "load_failed", "timeout": "timeout",
        "output_limit": "output_limit", "cancelled": "cancelled",
        "error": "process_error",
    }
    stdout, stderr = result["stdout"], result["stderr"]
    likely_oom = bool(OOM_TEXT.search(stdout + "\n" + stderr))
    plan["status"] = status_map[result["status"]]
    plan["load_result"] = {
        "status": plan["status"],
        "failure_classification": "likely_out_of_memory" if likely_oom else (
            "model_load_or_runtime_error" if plan["status"] == "load_failed" else None),
        "ready_endpoint_verified": result["status"] == "ready",
        "returncode_after_cleanup": result["returncode"],
        "elapsed_seconds": elapsed,
        "stdout_excerpt": stdout,
        "stderr_excerpt": stderr,
        "output_truncated_or_unknown": result["truncated"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    min_available = min(samples) if samples else None
    plan["memory_observation"] = {
        "host_available_before_mib": before["available_mib"],
        "host_available_min_during_startup_mib": min_available,
        "host_available_after_cleanup_mib": after["available_mib"],
        "host_minimum_delta_from_before_mib": (
            before["available_mib"] - min_available
            if before["status"] == "observed" and min_available is not None else None),
        "host_peak_process_mib": max(process_samples) // MIB if process_samples else None,
        "host_process_memory_metric": process_metric,
        "host_process_memory_sampling_status": "observed" if process_samples else "unknown",
        "host_process_memory_sample_count": len(process_samples),
        "host_process_memory_sample_interval_ms": 500,
        "host_process_tree_peak_process_count": process_count_max if process_samples else None,
        "host_process_tree_sampling_coverage": (
            "partial" if "partial" in process_coverage else "complete"
            if process_coverage else "unknown"),
        "device_available_during_startup_mib": None,
        "device_peak_process_mib": None,
        "device_memory_observation": device_observation,
        "attribution": "owned_process_tree_sampled" if process_samples else "system_wide_host_memory_only",
    }
    plan.pop("_argv", None)
    REPORT_VALIDATOR.validate(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--model-observation", type=Path, required=True)
    parser.add_argument("--runtime-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly load the selected model candidate in local llama serve")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plan = None
    try:
        plan = build_plan(
            load_json(args.profile), load_json(args.preview),
            load_json(args.model_observation), load_json(args.runtime_snapshot),
            args.candidate_id, args.target_model, draft_path=args.draft_model,
            binary=args.binary, timeout_seconds=args.timeout,
        )
        if args.execute:
            execute_plan(plan)
        else:
            plan.pop("_argv", None)
            REPORT_VALIDATOR.validate(plan)
        payload = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, payload,
                       protected=(args.profile, args.preview, args.model_observation,
                                  args.runtime_snapshot, args.target_model,
                                  args.draft_model, args.binary),
                       overwrite=args.overwrite)
            print("Created feasibility load check:", args.output)
        else:
            print(payload, end="")
        return 0 if not args.execute or plan["status"] == "load_verified" else 2
    except (OSError, KeyError, PairingError, FeasibilityError,
            SnapshotValidationError, ValidationError, ValueError) as error:
        print("Feasibility load check rejected:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
