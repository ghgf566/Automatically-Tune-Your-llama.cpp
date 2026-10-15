#!/usr/bin/env python3
"""Prepare or explicitly run a bounded CPU-MoE/GPU expert-cache sweep."""
import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

from engine_adapter import command_prefix, discover_binary, resolve_entrypoint
from engine_identity import sha256_file
from inspect_gguf import inspect
from run_speculative_benchmark import (
    BenchmarkError, _bounded, _port, _run_scenario,
)
from safe_output import write_text


ROOT = Path(__file__).resolve().parents[1]
MAX_CANDIDATES = 6
MAX_TOTAL_SECONDS = 21600
SCHEMA = json.loads((ROOT / "schemas/moe-cache-sweep.schema.json").read_text("utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)
PROMPT = (
    "Explain how to compare two inference settings fairly. Keep the same model, "
    "prompt, sampling, context, backend, and request count; report throughput, "
    "latency, and memory use. State what evidence is needed before claiming one "
    "setting is faster."
)


def parse_cache_candidates(value):
    try:
        candidates = [int(item.strip()) for item in value.split(",")]
    except (AttributeError, ValueError) as error:
        raise BenchmarkError("MoE cache candidates must be comma-separated MiB integers") from error
    if (not candidates or len(candidates) > MAX_CANDIDATES
            or candidates != sorted(set(candidates))
            or any(item < 0 or item > 16384 for item in candidates)
            or 0 not in candidates):
        raise BenchmarkError(
            f"provide 1-{MAX_CANDIDATES} increasing cache sizes in [0, 16384] MiB, including 0")
    return candidates


def build_sweep_plan(*, model_path, binary=None, candidates_mib=(0, 2048, 4096),
                     device="CUDA0", gpu_layers=0, cache_type_k="q4_0",
                     cache_type_v="q4_0", context=4096, fit_target_mib=512,
                     host_reserve_mib=4096, batch_size=512, ubatch_size=128,
                     repetitions=3, warmup_requests=1, n_predict=128, seed=42,
                     timeout_seconds=1800, max_total_seconds=MAX_TOTAL_SECONDS):
    candidates = list(candidates_mib)
    parsed = parse_cache_candidates(",".join(str(item) for item in candidates))
    if parsed != candidates:
        raise BenchmarkError("cache candidates must be increasing integers")
    if not 0 <= gpu_layers <= 4096:
        raise BenchmarkError("GPU layers must be in [0, 4096]")
    if not 1 <= context <= 262144:
        raise BenchmarkError("context must be in [1, 262144]")
    if not 1 <= fit_target_mib <= 2**20 or not 0 <= host_reserve_mib <= 2**20:
        raise BenchmarkError("memory reserves are outside supported bounds")
    if not 1 <= repetitions <= 30 or not 0 <= warmup_requests <= 3:
        raise BenchmarkError("repetitions must be in [1, 30] and warmups in [0, 3]")
    if not 16 <= n_predict <= 4096:
        raise BenchmarkError("generation length must be in [16, 4096]")
    if not 1 <= timeout_seconds <= 1800 or not 1 <= max_total_seconds <= MAX_TOTAL_SECONDS:
        raise BenchmarkError("timeout is outside supported bounds")
    cache_types = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
    if cache_type_k not in cache_types or cache_type_v not in cache_types:
        raise BenchmarkError("unsupported KV cache type")

    model = Path(model_path).expanduser().resolve(strict=True)
    observed = inspect(model)
    architecture = observed["architecture_metadata"]
    if not isinstance(architecture.get("expert_count"), int) or architecture["expert_count"] <= 0:
        raise BenchmarkError("selected model metadata does not identify a MoE model")
    if observed["split"]["multiple_parts_reported"]:
        raise BenchmarkError("multi-part GGUFs are not supported by this sweep")
    engine = discover_binary(binary)
    entry = resolve_entrypoint(engine, "serve")
    version = _bounded([str(engine), "--version"])
    help_text = _bounded(command_prefix(entry) + ["--help"], timeout=30)
    required = ("--model", "--ctx-size", "--parallel", "--device", "--gpu-layers",
                "--cpu-moe", "--moe-cache-mib", "--cache-type-k", "--cache-type-v",
                "--cache-ram", "--fit", "--fit-target", "--spec-type", "--no-ui",
                "--host", "--port", "--batch-size", "--ubatch-size")
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise BenchmarkError("serve Help lacks required flags: " + ", ".join(missing))
    device_text = _bounded(command_prefix(entry) + ["--list-devices"], timeout=30)
    if not re.search(r"(?m)^\s*" + re.escape(device) + r"\s*:", device_text):
        raise BenchmarkError("requested device is absent from a fresh --list-devices result")

    engine_sha = sha256_file(engine)
    model_sha = sha256_file(model)
    prompt_sha = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    cases = []
    for cache_mib in candidates:
        port = _port()
        argv = command_prefix(entry) + [
            "--offline", "--model", str(model), "--ctx-size", str(context),
            "--parallel", "1", "--device", device, "--gpu-layers", str(gpu_layers),
            "--cpu-moe", "--moe-cache-mib", str(cache_mib),
            "--cache-type-k", cache_type_k, "--cache-type-v", cache_type_v,
            "--cache-ram", "0", "--batch-size", str(batch_size),
            "--ubatch-size", str(ubatch_size), "--fit", "off",
            "--no-ui", "--host", "127.0.0.1", "--port", str(port),
            "--verbosity", "5", "--spec-type", "none",
        ]
        redacted = ["<llama-executable>" if item == str(engine) else
                    "<model>" if item == str(model) else item for item in argv]
        cases.append({
            "case_id": f"moe-cache-{cache_mib}-mib", "moe_cache_mib": cache_mib,
            "status": "prepared_not_executed", "measurement": None,
            "argv_redacted": redacted, "_argv": argv, "_port": port,
        })
    return {
        "schema_version": "0.1.0", "kind": "moe_gpu_expert_cache_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_executed", "engine_sha256": engine_sha,
        "engine_version_excerpt": version.strip()[:500],
        "model": {"filename": observed["source"]["filename"],
                  "file_size_bytes": observed["source"]["file_size_bytes"],
                  "architecture": observed["general"]["architecture"],
                  "expert_count": architecture["expert_count"],
                  "sha256": model_sha},
        "configuration": {
            "device_id": device, "gpu_layers": gpu_layers, "cpu_moe": True,
            "cache_type_k": cache_type_k, "cache_type_v": cache_type_v,
            "context_tokens": context, "concurrency": 1, "fit": "off",
            "prompt_cache_ram_mib": 0, "device_reserve_mib": fit_target_mib,
            "host_reserve_mib": host_reserve_mib, "batch_size": batch_size,
            "ubatch_size": ubatch_size,
        },
        "workload": {"workload_id": "moe_cache_synthetic", "prompt_sha256": prompt_sha,
                     "n_predict": n_predict, "warmup_requests": warmup_requests,
                     "repetitions_per_case": repetitions, "seed": seed},
        "sweep": {"cache_candidates_mib": candidates, "case_count": len(cases),
                  "execution_order_cache_mib": random.Random(seed).sample(candidates, len(candidates)),
                  "timeout_seconds_per_case": timeout_seconds,
                  "max_total_seconds": max_total_seconds},
        "cases": [{key: value for key, value in case.items() if not key.startswith("_")}
                  for case in cases],
        "summary": None,
        "limitations": [
            "This compares CPU-resident MoE weights with a bounded GPU expert cache; it does not place all experts in VRAM.",
            "GPU layer count is fixed across cases; the default 0 keeps ordinary model layers on CPU.",
            "Synthetic concurrency-one workload is not representative user traffic.",
            "GPU process-attributed memory may be unavailable on Windows WDDM; device totals are system-wide.",
            "No MTP/speculative decoding or output quality comparison is performed.",
        ],
        "_cases": cases, "_binary": str(engine), "_model_path": str(model),
        "_prompt": PROMPT, "_seed": seed, "_repetitions": repetitions,
        "_warmups": warmup_requests, "_n_predict": n_predict,
        "_host_reserve_mib": host_reserve_mib, "_device_reserve_mib": fit_target_mib,
        "_timeout_seconds": timeout_seconds, "_max_total_seconds": max_total_seconds,
    }


def execute_sweep(plan):
    if plan["status"] != "prepared_not_executed":
        raise BenchmarkError("only a prepared sweep can execute")
    plan["status"] = "running"
    started = time.monotonic()
    deadline = started + plan["_max_total_seconds"]
    stop_reason = None
    cases_by_cache = {case["moe_cache_mib"]: case for case in plan["_cases"]}
    execution_order = plan["sweep"]["execution_order_cache_mib"]
    for index, cache_mib in enumerate(execution_order, start=1):
        case = cases_by_cache[cache_mib]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            case["status"] = "budget_exhausted"
            stop_reason = "total_time_budget_exhausted"
            break
        if (sha256_file(Path(plan["_binary"])) != plan["engine_sha256"]
                or sha256_file(Path(plan["_model_path"])) != plan["model"]["sha256"]):
            case["status"] = "input_identity_changed"
            stop_reason = "input_identity_changed"
            break
        print(f"Starting MoE cache case {index}/{len(plan['_cases'])}: {case['case_id']}",
              file=sys.stderr, flush=True)
        run = _run_scenario(
            case["_argv"], mode="off", port=case["_port"],
            timeout_seconds=plan["_timeout_seconds"],
            repetitions=plan["_repetitions"], warmup_requests=plan["_warmups"],
            n_predict=plan["_n_predict"], seed=plan["_seed"],
            host_reserve_mib=plan["_host_reserve_mib"],
            device_reserve_mib=plan["_device_reserve_mib"],
            prompt=plan["_prompt"], overall_deadline=deadline,
        )
        rows = [row for row in run.get("requests", []) if not row.get("warmup")]
        speeds = [row["decode_tokens_per_second"] for row in rows]
        latencies = [row["latency_ms"] for row in rows]
        case["measurement"] = {
            "status": run["status"], "request_count": len(rows),
            "decode_tokens_per_second": _stats(speeds), "latency_ms": _stats(latencies),
            "effective_gpu_layers_observed": run.get("effective_gpu_layers_observed"),
            "effective_gpu_layers_total_observed": run.get("effective_gpu_layers_total_observed"),
            "placement_evidence": run.get("placement_evidence", []),
            "memory": run.get("memory"), "requests": run.get("requests"),
            "server_log_excerpt": run.get("server_log_excerpt"),
        }
        case["status"] = "completed" if run["status"] == "completed" else run["status"]
        if run["status"] != "completed":
            stop_reason = "case_failed_or_reserve_guard"
            break
        print(f"Finished MoE cache case {index}/{len(plan['_cases'])}: "
              f"{case['case_id']} status={case['status']}", file=sys.stderr, flush=True)
    if stop_reason:
        for case in plan["_cases"]:
            if case["status"] == "prepared_not_executed":
                case["status"] = "not_run"
    successful = sum(case["status"] == "completed" for case in plan["_cases"])
    plan["summary"] = {
        "case_count": len(plan["_cases"]), "completed_case_count": successful,
        "incomplete_case_count": len(plan["_cases"]) - successful,
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "winner_selection": "not_performed; review variation and memory trade-offs",
    }
    plan["status"] = "completed" if successful == len(plan["_cases"]) else "incomplete"
    plan["cases"] = [{key: value for key, value in case.items() if not key.startswith("_")}
                     for case in plan["_cases"]]
    for key in [name for name in plan if name.startswith("_")]:
        plan.pop(key, None)
    return plan


def _stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None,
                "sample_stddev": None, "coefficient_of_variation_pct": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    mean = sum(values) / len(values)
    sample_stddev = statistics.stdev(values) if len(values) > 1 else None
    return {
        "count": len(values), "mean": mean, "median": median,
        "p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
        "sample_stddev": sample_stddev,
        "coefficient_of_variation_pct": (
            sample_stddev / mean * 100 if sample_stddev is not None and mean else None
        ),
    }


def validate_report(report):
    errors = sorted(VALIDATOR.iter_errors(report),
                    key=lambda error: tuple(str(item) for item in error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise BenchmarkError(f"invalid MoE cache report at {location}: {error.message}")
    if len(report["cases"]) != report["sweep"]["case_count"]:
        raise BenchmarkError("case count does not match the requested cache sweep")
    if set(report["sweep"]["execution_order_cache_mib"]) != set(
            report["sweep"]["cache_candidates_mib"]):
        raise BenchmarkError("execution order does not match the requested cache candidates")
    if len({case["case_id"] for case in report["cases"]}) != len(report["cases"]):
        raise BenchmarkError("duplicate case IDs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--cache-candidates-mib", default="0,2048,4096")
    parser.add_argument("--device", default="CUDA0")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--device-reserve-mib", type=int, default=512)
    parser.add_argument("--host-reserve-mib", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-total-seconds", type=int, default=MAX_TOTAL_SECONDS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_sweep_plan(
            model_path=args.model, binary=args.binary,
            candidates_mib=parse_cache_candidates(args.cache_candidates_mib),
            device=args.device, gpu_layers=args.gpu_layers,
            cache_type_k=args.cache_type_k, cache_type_v=args.cache_type_v,
            context=args.context, fit_target_mib=args.device_reserve_mib,
            host_reserve_mib=args.host_reserve_mib, batch_size=args.batch_size,
            ubatch_size=args.ubatch_size, repetitions=args.repetitions,
            warmup_requests=args.warmup_requests, n_predict=args.n_predict,
            seed=args.seed, timeout_seconds=args.timeout,
            max_total_seconds=args.max_total_seconds,
        )
        if args.execute:
            plan = execute_sweep(plan)
        else:
            for key in [name for name in plan if name.startswith("_")]:
                plan.pop(key, None)
            for case in plan["cases"]:
                for key in [name for name in case if name.startswith("_")]:
                    case.pop(key, None)
        validate_report(plan)
        payload = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, payload, protected=(args.model, args.binary),
                       overwrite=args.overwrite)
            print("Created MoE cache sweep:", args.output)
        else:
            print(payload, end="")
        return 0 if plan["status"] in ("prepared_not_executed", "completed") else 2
    except (OSError, ValueError, BenchmarkError) as error:
        print("MoE cache sweep rejected:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
