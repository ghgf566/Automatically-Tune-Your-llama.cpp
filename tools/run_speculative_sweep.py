#!/usr/bin/env python3
"""Prepare or explicitly run a bounded, paired Draft N/workload sweep."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

from jsonschema import Draft202012Validator

from run_speculative_benchmark import (
    BenchmarkError,
    WORKLOADS,
    _port,
    build_plan,
    execute_plan,
)
from safe_output import write_text


MAX_SWEEP_CASES = 12
MAX_DRAFT_CANDIDATES = 4
MAX_TOTAL_SECONDS = 21600
ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCHEMA = json.loads((ROOT / "schemas/speculative-sweep.schema.json").read_text("utf-8"))
SWEEP_VALIDATOR = Draft202012Validator(SWEEP_SCHEMA)


def parse_draft_candidates(value):
    try:
        candidates = [int(item.strip()) for item in value.split(",")]
    except (AttributeError, ValueError) as error:
        raise BenchmarkError("Draft N candidates must be comma-separated integers") from error
    if (not candidates or len(candidates) > MAX_DRAFT_CANDIDATES
            or any(not 1 <= item <= 64 for item in candidates)
            or candidates != sorted(set(candidates))):
        raise BenchmarkError(
            f"provide 1-{MAX_DRAFT_CANDIDATES} strictly increasing Draft N values in [1, 64]")
    return candidates


def parse_workloads(value):
    workloads = [item.strip() for item in value.split(",")]
    if (not workloads or len(workloads) != len(set(workloads))
            or any(item not in WORKLOADS for item in workloads)):
        raise BenchmarkError("workloads must be distinct built-in workload names")
    return workloads


def _case_plan(base, workload_id, draft_n_max, mode_order, port):
    plan = copy.deepcopy(base)
    workload = WORKLOADS[workload_id]
    prompt = workload["prompt"]
    plan["workload"].update({
        "workload_id": workload_id,
        "description": workload["description"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    })
    plan["configuration"]["draft_n_max"] = draft_n_max
    plan["configuration"]["mode_order"] = mode_order
    plan["_prompt"] = prompt
    mode_by_name = {item["mode"]: item for item in plan["_variants"]}
    mtp = mode_by_name["embedded_mtp"]
    for argv in (mtp["argv"], mtp["argv_redacted"]):
        n_index = argv.index("--spec-draft-n-max") + 1
        argv[n_index] = str(draft_n_max)
    for item in plan["_variants"]:
        item["port"] = port
        item["argv"][item["argv"].index("--port") + 1] = str(port)
        item["argv_redacted"][item["argv_redacted"].index("--port") + 1] = str(port)
    if mode_order == "mtp_first":
        plan["_variants"].reverse()
    plan["variants"] = [
        {"mode": item["mode"], "port": port,
         "argv_redacted": item["argv_redacted"]}
        for item in plan["_variants"]
    ]
    return plan


def build_sweep_plan(*, model_path, binary=None, draft_n_candidates=(1, 2, 4),
                     workloads=("short_prompt", "long_prompt", "code_generation"),
                     context=4096, device="CUDA0", gpu_layers=16, n_cpu_ffn=None,
                     cache_type_k="f16", cache_type_v="f16",
                     fit_target_mib=512, host_reserve_mib=4096,
                     batch_size=512, ubatch_size=128, repetitions=3,
                     warmup_requests=1, n_predict=128, seed=42,
                     timeout_seconds=1800, case_timeout_seconds=3600,
                     max_total_seconds=MAX_TOTAL_SECONDS):
    candidates = list(draft_n_candidates)
    selected_workloads = list(workloads)
    if candidates != sorted(set(candidates)) or not candidates:
        raise BenchmarkError("Draft N candidates must be unique and increasing")
    if len(candidates) > MAX_DRAFT_CANDIDATES or any(not 1 <= n <= 64 for n in candidates):
        raise BenchmarkError("Draft N sweep exceeds its candidate bound")
    if (not selected_workloads or len(set(selected_workloads)) != len(selected_workloads)
            or any(item not in WORKLOADS for item in selected_workloads)):
        raise BenchmarkError("select distinct supported workloads")
    case_count = len(candidates) * len(selected_workloads)
    if case_count > MAX_SWEEP_CASES:
        raise BenchmarkError(f"sweep has {case_count} comparisons; maximum is {MAX_SWEEP_CASES}")
    if not 1 <= case_timeout_seconds <= 21600:
        raise BenchmarkError("per-comparison budget must be in [1, 21600] seconds")
    if not 1 <= max_total_seconds <= MAX_TOTAL_SECONDS:
        raise BenchmarkError(f"total budget must be in [1, {MAX_TOTAL_SECONDS}] seconds")

    base = build_plan(
        model_path=model_path, binary=binary, context=context, device=device,
        gpu_layers=gpu_layers, n_cpu_ffn=n_cpu_ffn, cache_type_k=cache_type_k,
        cache_type_v=cache_type_v, fit_target_mib=fit_target_mib,
        host_reserve_mib=host_reserve_mib, batch_size=batch_size,
        ubatch_size=ubatch_size, draft_n_max=candidates[0],
        repetitions=repetitions, warmup_requests=warmup_requests,
        n_predict=n_predict, seed=seed, timeout_seconds=timeout_seconds,
        max_run_seconds=case_timeout_seconds, workload_id=selected_workloads[0],
    )
    cases = []
    for workload_index, workload_id in enumerate(selected_workloads):
        for candidate_index, draft_n in enumerate(candidates):
            mode_order = ("off_first" if (workload_index + candidate_index) % 2 == 0
                          else "mtp_first")
            port = _port()
            subplan = _case_plan(base, workload_id, draft_n, mode_order, port)
            case_id = f"{workload_id}-draft-n-{draft_n}"
            cases.append({
                "case_id": case_id,
                "workload_id": workload_id,
                "requested_draft_n_max": draft_n,
                "mode_order": mode_order,
                "status": "prepared_not_executed",
                "comparison_status": None,
                "benchmark": {key: value for key, value in subplan.items()
                              if not key.startswith("_")},
                "_plan": subplan,
            })
    return {
        "schema_version": "0.1.0",
        "kind": "speculative_bounded_sweep",
        "status": "prepared_not_executed",
        "engine_sha256": base["engine_sha256"],
        "engine_version_excerpt": base["engine_version_excerpt"],
        "engine_commit": base["engine_commit"],
        "model": base["model"],
        "configuration": {key: value for key, value in base["configuration"].items()
                          if key not in ("draft_n_max", "mode_order")},
        "sweep": {
            "requested_draft_n_max_candidates": candidates,
            "workloads": [
                {"workload_id": item,
                 "description": WORKLOADS[item]["description"],
                 "prompt_sha256": hashlib.sha256(
                     WORKLOADS[item]["prompt"].encode("utf-8")).hexdigest()}
                for item in selected_workloads
            ],
            "comparison_count": case_count,
            "repetitions_per_mode": repetitions,
            "warmup_requests_per_mode": warmup_requests,
            "per_comparison_timeout_seconds": case_timeout_seconds,
            "total_timeout_seconds": max_total_seconds,
            "pairing": "each workload and Draft N has its own off baseline",
            "order_policy": "mode order alternates across adjacent workload/N pairs",
        },
        "cases": [{key: value for key, value in item.items() if not key.startswith("_")}
                  for item in cases],
        "summary": None,
        "limitations": [
            "All workloads are bundled synthetic completion prompts; none represents measured user traffic.",
            "Each requested Draft N is compared with its own off baseline to limit order and thermal bias.",
            "Effective Draft N remains unknown unless the engine exposes independent runtime evidence.",
            "The sweep holds model, context, target placement, device, batch, fit reserve, and sampling fixed.",
            "No quality equivalence, multimodal workload, long-context service, or concurrent serving is evaluated.",
            "WDDM may withhold per-process VRAM; system-wide free memory is not a process peak.",
        ],
        "_cases": cases,
        "_binary": base["_binary"],
        "_model_path": base["_model_path"],
        "_max_total_seconds": max_total_seconds,
        "_case_timeout_seconds": case_timeout_seconds,
    }


def execute_sweep(plan):
    if plan["status"] != "prepared_not_executed":
        raise BenchmarkError("only a prepared sweep can execute")
    plan["status"] = "running"
    started = time.monotonic()
    deadline = started + plan["_max_total_seconds"]
    stop_reason = None
    case_count = len(plan["_cases"])
    for case_index, case in enumerate(plan["_cases"], start=1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop_reason = "total_time_budget_exhausted"
            case["status"] = "budget_exhausted"
            case["comparison_status"] = "not_run"
            continue
        print(
            f"Starting sweep case {case_index}/{case_count}: {case['case_id']} "
            f"({int(remaining)}s total budget remains)",
            file=sys.stderr, flush=True,
        )
        subplan = case["_plan"]
        subplan["_max_run_seconds"] = max(1, min(
            plan["_case_timeout_seconds"], int(remaining)))
        try:
            result = execute_plan(subplan)
        except Exception as error:
            case["status"] = "runner_error"
            case["comparison_status"] = "execution_failed"
            case["error"] = f"{type(error).__name__}: {error}"
            stop_reason = "case_execution_failed"
            break
        case["benchmark"] = result
        case["status"] = result["status"]
        case["comparison_status"] = result.get("summary", {}).get("comparison_status")
        print(
            f"Finished sweep case {case_index}/{case_count}: {case['case_id']} "
            f"status={case['status']} comparison={case['comparison_status']}",
            file=sys.stderr, flush=True,
        )
        if any(run.get("status") == "reserve_exceeded" for run in result.get("runs", [])):
            stop_reason = "memory_reserve_exceeded"
            break
        if any(run.get("status") == "input_identity_changed" for run in result.get("runs", [])):
            stop_reason = "input_identity_changed"
            break
    if stop_reason:
        for case in plan["_cases"]:
            if case["status"] == "prepared_not_executed":
                case["status"] = "not_run"
                case["comparison_status"] = stop_reason
    successful = sum(item["comparison_status"] == "comparable" for item in plan["_cases"])
    plan["summary"] = {
        "comparison_count": len(plan["_cases"]),
        "completed_comparable_count": successful,
        "incomplete_or_not_comparable_count": len(plan["_cases"]) - successful,
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "winner_selection": "not_performed; review per-case measurements and trade-offs",
    }
    plan["status"] = ("completed" if successful == len(plan["_cases"])
                      else "incomplete_or_not_comparable")
    plan["cases"] = [
        {key: value for key, value in case.items() if not key.startswith("_")}
        for case in plan["_cases"]
    ]
    for key in [name for name in plan if name.startswith("_")]:
        plan.pop(key, None)
    for case in plan["cases"]:
        benchmark = case.get("benchmark")
        if isinstance(benchmark, dict):
            for key in [name for name in benchmark if name.startswith("_")]:
                benchmark.pop(key, None)
    return plan


def validate_sweep_report(report):
    errors = sorted(SWEEP_VALIDATOR.iter_errors(report),
                    key=lambda error: tuple(str(item) for item in error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise BenchmarkError(f"invalid sweep report at {location}: {error.message}")
    expected = (len(report["sweep"]["requested_draft_n_max_candidates"])
                * len(report["sweep"]["workloads"]))
    if report["sweep"]["comparison_count"] != expected or len(report["cases"]) != expected:
        raise BenchmarkError("sweep case count does not match requested workload/N combinations")
    case_ids = [case["case_id"] for case in report["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise BenchmarkError("sweep report contains duplicate case IDs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--draft-n-candidates", default="1,2,4")
    parser.add_argument("--workloads", default="short_prompt,long_prompt,code_generation")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--device", default="CUDA0")
    parser.add_argument("--gpu-layers", default="16",
                        help="GPU layer count or 'all'")
    parser.add_argument("--n-cpu-ffn", type=int,
                        help="keep dense FFN weights in the first N layers on CPU")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--fit-target-mib", type=int, default=512)
    parser.add_argument("--host-reserve-mib", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Maximum server startup time per mode")
    parser.add_argument("--case-timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-total-seconds", type=int, default=MAX_TOTAL_SECONDS)
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly load models and run every bounded comparison")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_sweep_plan(
            model_path=args.model, binary=args.binary,
            draft_n_candidates=parse_draft_candidates(args.draft_n_candidates),
            workloads=parse_workloads(args.workloads), context=args.context,
            device=args.device,
            gpu_layers=(args.gpu_layers if args.gpu_layers == "all" else int(args.gpu_layers)),
            n_cpu_ffn=args.n_cpu_ffn,
            cache_type_k=args.cache_type_k, cache_type_v=args.cache_type_v,
            fit_target_mib=args.fit_target_mib,
            host_reserve_mib=args.host_reserve_mib, batch_size=args.batch_size,
            ubatch_size=args.ubatch_size, repetitions=args.repetitions,
            warmup_requests=args.warmup_requests, n_predict=args.n_predict,
            seed=args.seed, timeout_seconds=args.timeout,
            case_timeout_seconds=args.case_timeout_seconds,
            max_total_seconds=args.max_total_seconds,
        )
        if args.execute:
            plan = execute_sweep(plan)
        else:
            for key in [name for name in plan if name.startswith("_")]:
                plan.pop(key, None)
            for case in plan["cases"]:
                if isinstance(case.get("benchmark"), dict):
                    for key in [name for name in case["benchmark"] if name.startswith("_")]:
                        case["benchmark"].pop(key, None)
        validate_sweep_report(plan)
        payload = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, payload, protected=(args.model, args.binary),
                       overwrite=args.overwrite)
            print("Created speculative sweep:", args.output)
        else:
            print(payload, end="")
        return 0 if plan["status"] in ("prepared_not_executed", "completed") else 2
    except (OSError, ValueError, BenchmarkError) as error:
        print("Speculative sweep rejected:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
