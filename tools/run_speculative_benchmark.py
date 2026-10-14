#!/usr/bin/env python3
"""Plan or explicitly run a bounded off-vs-embedded-MTP server benchmark.

Both modes use the same model, fixed context/placement, fit reserve, prompt,
sampling and request count. The benchmark owns and stops every server process.
"""
import argparse
import hashlib
import http.client
import json
import math
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from device_memory import sample_nvidia_process_memory, sample_nvidia_smi
from engine_adapter import command_prefix, discover_binary, resolve_entrypoint
from engine_identity import sha256_file
from environment_observation import clean_llama_env, system_ram
from inspect_gguf import inspect
from process_lifecycle import isolation_options, stop_owned_tree
from process_memory import MIB, sample_process_tree
from process_runner import run_bounded
from safe_output import write_text

ROOT = Path(__file__).resolve().parents[1]
MAX_STDIO_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MTP_FALLBACK = re.compile(
    r"failed to measure the memory of the extra model, fitting without it", re.I
)
OFFLOAD_LAYERS = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers? to GPU", re.I)
EXTRA_FIT_MEASUREMENT = re.compile(
    r"getting device memory data for the extra model at a context size of\s+(\d+)", re.I
)
PROMPT = (
    "You are comparing two city transportation plans. Give a concise, factual "
    "analysis with headings for capacity, reliability, cost, and risks. "
    "Assume Plan A adds frequent buses on existing roads. Plan B builds a new "
    "rail line with fewer stops. Both plans have the same first-year budget, "
    "but Plan B has higher long-term maintenance costs. Explain which evidence "
    "would be needed before choosing either plan, distinguish assumptions from "
    "known facts, and finish with a short decision checklist. Do not invent "
    "ridership figures, prices, or local conditions."
)


class BenchmarkError(ValueError):
    pass


def _bounded(argv, timeout=20, capture_bytes=MAX_OUTPUT_BYTES):
    result = run_bounded(argv, timeout, capture_bytes=capture_bytes,
                         disk_limit_bytes=MAX_OUTPUT_BYTES,
                         env=clean_llama_env())
    if result["status"] != "success" or result["truncated"]:
        raise BenchmarkError("runtime capability command failed: " + result["status"])
    return result["stdout"] + "\n" + result["stderr"]


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _health(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read(4096)
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


class _Capture:
    def __init__(self, stream):
        self.data = bytearray()
        self.truncated = False
        self.pending = bytearray()
        self.offload_observations = []
        self.extra_fit_contexts = []
        self.fallback_detected = False
        self._thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self._thread.start()

    def _scan_line(self, raw):
        line = raw.decode("utf-8", "replace")
        match = OFFLOAD_LAYERS.search(line)
        if match:
            self.offload_observations.append({
                "offloaded_layers": int(match.group(1)),
                "total_layers": int(match.group(2)),
                "line": line.strip()[:500],
            })
        match = EXTRA_FIT_MEASUREMENT.search(line)
        if match:
            self.extra_fit_contexts.append(int(match.group(1)))
        if MTP_FALLBACK.search(line):
            self.fallback_detected = True

    def _scan(self, chunk):
        lines = self.pending + chunk
        pieces = lines.split(b"\n")
        self.pending = bytearray(pieces.pop())
        if len(self.pending) > MAX_STDIO_BYTES:
            del self.pending[:-MAX_STDIO_BYTES]
            self.truncated = True
        for line in pieces:
            self._scan_line(line)

    def _read(self, stream):
        while True:
            # read1 returns currently available pipe bytes instead of waiting
            # for a full 8 KiB block; startup fit warnings must be visible
            # before the benchmark decides whether MTP was jointly measured.
            chunk = stream.read1(8192) if hasattr(stream, "read1") else stream.read(8192)
            if not chunk:
                if self.pending:
                    self._scan_line(self.pending)
                    self.pending.clear()
                return
            self._scan(chunk)
            overflow = len(self.data) + len(chunk) - MAX_STDIO_BYTES
            if overflow > 0:
                del self.data[:min(overflow, len(self.data))]
                self.truncated = True
                if len(chunk) > MAX_STDIO_BYTES:
                    chunk = chunk[-MAX_STDIO_BYTES:]
            self.data.extend(chunk)

    def text(self):
        self._thread.join(timeout=3)
        return bytes(self.data).decode("utf-8", "replace")


class _MemoryMonitor:
    def __init__(self, proc, *, host_reserve_mib, device_reserve_mib):
        self.proc = proc
        self.host_reserve = host_reserve_mib
        self.device_reserve = device_reserve_mib
        self.stop_event = threading.Event()
        self.violation = threading.Event()
        self.lock = threading.Lock()
        self.host_samples = []
        self.process_samples = []
        self.gpu_samples = []
        self.process_gpu_samples = []
        self.process_gpu_failures = []
        self._thread = threading.Thread(target=self._run, name="atyl-bench-memory",
                                        daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ram = system_ram()
            if ram.get("status") == "observed":
                with self.lock:
                    self.host_samples.append(ram["available_mib"])
                if ram["available_mib"] < self.host_reserve:
                    self.violation.set()
            process = sample_process_tree(self.proc.pid)
            if process.get("status") == "observed":
                with self.lock:
                    self.process_samples.append(process["resident_bytes"])
            devices, device_error = sample_nvidia_smi()
            if devices:
                with self.lock:
                    self.gpu_samples.append(devices)
                if any(device["free_mib"] < self.device_reserve for device in devices):
                    self.violation.set()
            else:
                with self.lock:
                    self.process_gpu_failures.append(device_error or "sample_unavailable")
            owned, process_error = sample_nvidia_process_memory(self.proc.pid)
            if owned:
                with self.lock:
                    self.process_gpu_samples.append(owned)
            else:
                with self.lock:
                    self.process_gpu_failures.append(process_error or "process_sample_unavailable")
            if self.violation.is_set() and self.proc.poll() is None:
                stop_owned_tree(self.proc)
            if self.stop_event.wait(1):
                break

    def stop_and_report(self):
        self.stop_event.set()
        self._thread.join(timeout=8)
        with self.lock:
            host = list(self.host_samples)
            process = list(self.process_samples)
            gpu = list(self.gpu_samples)
            process_gpu = list(self.process_gpu_samples)
            process_failures = list(self.process_gpu_failures)
        gpu_by_uuid = {}
        for sample in gpu:
            for device in sample:
                row = gpu_by_uuid.setdefault(device["uuid"], {
                    "uuid": device["uuid"], "name": device["name"],
                    "total_memory_mib": device["total_mib"],
                    "minimum_free_memory_mib": device["free_mib"],
                    "maximum_used_memory_mib": device["used_mib"],
                    "sample_count": 0,
                })
                row["minimum_free_memory_mib"] = min(
                    row["minimum_free_memory_mib"], device["free_mib"])
                row["maximum_used_memory_mib"] = max(
                    row["maximum_used_memory_mib"], device["used_mib"])
                row["sample_count"] += 1
        owned_by_uuid = {}
        for sample in process_gpu:
            for uuid, memory in sample.items():
                owned_by_uuid[uuid] = max(owned_by_uuid.get(uuid, 0), memory)
        process_status = "observed" if process_gpu else "unknown"
        reserve_status = ("exceeded" if self.violation.is_set() else
                          "observed" if host and gpu else "unknown")
        return {
            "host_available_min_mib": min(host) if host else None,
            "host_process_peak_mib": max(process) // MIB if process else None,
            "host_process_sample_count": len(process),
            "device_sample_count": len(gpu),
            "devices": [gpu_by_uuid[key] for key in sorted(gpu_by_uuid)],
            "device_process_status": process_status,
            "device_process_reason": process_failures[0] if process_failures else None,
            "device_process_peak_mib_by_uuid": owned_by_uuid,
            "device_process_sample_count": len(process_gpu),
            "device_reserve_status": reserve_status,
            "reserve_violation": self.violation.is_set(),
        }


def _request(port, *, n_predict, seed):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    body = json.dumps({
        "prompt": PROMPT,
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": False,
        "stream": True,
    }).encode("utf-8")
    started = time.perf_counter()
    first_token_at = None
    token_text = []
    timings = None
    try:
        connection.request("POST", "/completion", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            excerpt = response.read(4096).decode("utf-8", "replace")
            raise BenchmarkError(f"completion returned HTTP {response.status}: {excerpt}")
        while True:
            line = response.readline()
            if not line:
                break
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            piece = event.get("content")
            if isinstance(piece, str) and piece:
                token_text.append(piece)
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            if isinstance(event.get("timings"), dict):
                timings = event["timings"]
        ended = time.perf_counter()
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
        raise BenchmarkError("completion stream failed: " + str(error)) from error
    finally:
        connection.close()
    if timings is None:
        raise BenchmarkError("server stream omitted timing metrics")
    return {
        "latency_ms": round((ended - started) * 1000, 3),
        "ttft_ms": round((first_token_at - started) * 1000, 3)
        if first_token_at is not None else None,
        "completion_characters": sum(len(item) for item in token_text),
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "predicted_tokens": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "drafted_tokens": timings.get("draft_n", 0),
        "accepted_tokens": timings.get("draft_n_accepted", 0),
        "timings": timings,
    }


def _run_scenario(argv, *, mode, port, timeout_seconds, repetitions,
                  warmup_requests, n_predict, seed, host_reserve_mib,
                  device_reserve_mib):
    started_utc = datetime.now(timezone.utc).isoformat()
    before_ram = system_ram()
    before_gpu, before_gpu_error = sample_nvidia_smi()
    if before_ram.get("status") != "observed" or before_ram["available_mib"] < host_reserve_mib:
        return {"mode": mode, "status": "reserve_rejected",
                "reason": "host_headroom_below_reserve", "requests": [],
                "memory": {"host_before_mib": before_ram.get("available_mib"),
                           "device_before": before_gpu or [],
                           "device_probe_error": before_gpu_error}}
    if not before_gpu or any(item["free_mib"] < device_reserve_mib for item in before_gpu):
        return {"mode": mode, "status": "reserve_rejected",
                "reason": "device_headroom_below_reserve", "requests": [],
                "memory": {"host_before_mib": before_ram["available_mib"],
                           "device_before": before_gpu or [],
                           "device_probe_error": before_gpu_error}}

    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=clean_llama_env(), **isolation_options(),
    )
    stdout = _Capture(proc.stdout)
    stderr = _Capture(proc.stderr)
    monitor = _MemoryMonitor(proc, host_reserve_mib=host_reserve_mib,
                             device_reserve_mib=device_reserve_mib)
    monitor.start()
    deadline = time.monotonic() + timeout_seconds
    ready = False
    status = "load_failed"
    while time.monotonic() < deadline:
        if monitor.violation.is_set():
            status = "reserve_exceeded"
            break
        if proc.poll() is not None:
            status = "load_failed"
            break
        if _health(port):
            ready = True
            status = "ready"
            break
        time.sleep(0.5)
    else:
        status = "timeout"
        stop_owned_tree(proc)

    log_text = stdout.text() + "\n" + stderr.text()
    fallback = (stdout.fallback_detected or stderr.fallback_detected or
                bool(MTP_FALLBACK.search(log_text)))
    layers = (stdout.offload_observations + stderr.offload_observations)
    extra_fit_contexts = stdout.extra_fit_contexts + stderr.extra_fit_contexts
    joint_fit_status = (
        "fallback_target_only" if mode == "embedded_mtp" and fallback else
        "combined_fit_ready" if mode == "embedded_mtp" and ready and extra_fit_contexts else
        "unknown" if mode == "embedded_mtp" else "target_only_fit"
    )
    requests = []
    request_errors = []
    if ready and joint_fit_status not in ("fallback_target_only", "unknown"):
        try:
            for index in range(warmup_requests + repetitions):
                if monitor.violation.is_set() or proc.poll() is not None:
                    raise BenchmarkError("server stopped by reserve guard or process exit")
                item = _request(port, n_predict=n_predict, seed=seed)
                item["warmup"] = index < warmup_requests
                item["request_index"] = index
                requests.append(item)
        except (BenchmarkError, OSError) as error:
            request_errors.append(str(error))
            status = "benchmark_failed"
    stop_owned_tree(proc)
    memory = monitor.stop_and_report()
    status_code = proc.returncode
    log_text = stdout.text() + "\n" + stderr.text()
    fallback = (fallback or stdout.fallback_detected or stderr.fallback_detected or
                bool(MTP_FALLBACK.search(log_text)))
    if mode == "embedded_mtp" and fallback:
        joint_fit_status = "fallback_target_only"
        status = "combined_fit_rejected"
    if memory["reserve_violation"]:
        status = "reserve_exceeded"
    completed = [item for item in requests if not item["warmup"]]
    if status == "ready" and len(completed) == repetitions:
        status = "completed"
    return {
        "mode": mode,
        "status": status,
        "joint_fit_status": joint_fit_status,
        "extra_model_fallback_detected": fallback,
        "effective_gpu_layers_observed": layers[-1]["offloaded_layers"] if layers else None,
        "effective_gpu_layers_total_observed": layers[-1]["total_layers"] if layers else None,
        "effective_gpu_layers_evidence": layers[-1]["line"] if layers else None,
        "effective_gpu_layers_observation_status": "observed" if layers else "unknown",
        "joint_fit_extra_model_contexts_observed": extra_fit_contexts,
        "server_logs_truncated": stdout.truncated or stderr.truncated,
        "started_at_utc": started_utc,
        "server_ready": ready,
        "server_returncode_after_cleanup": status_code,
        "requests": requests,
        "request_errors": request_errors,
        "memory": {
            "host_before_mib": before_ram["available_mib"],
            "device_before": before_gpu,
            **memory,
        },
        "server_log_excerpt": log_text[-MAX_STDIO_BYTES:],
    }


def _summarize_scenario(scenario):
    rows = [row for row in scenario.get("requests", []) if not row.get("warmup")]
    drafted = sum(int(row.get("drafted_tokens") or 0) for row in rows)
    accepted = sum(int(row.get("accepted_tokens") or 0) for row in rows)
    return {
        "measurement_count": len(rows),
        "latency_ms": _summary([row["latency_ms"] for row in rows]),
        "ttft_ms": _summary([row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]),
        "prompt_tokens_per_second": _summary([
            row["prompt_tokens_per_second"] for row in rows
            if isinstance(row["prompt_tokens_per_second"], (int, float))]),
        "decode_tokens_per_second": _summary([
            row["decode_tokens_per_second"] for row in rows
            if isinstance(row["decode_tokens_per_second"], (int, float))]),
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


def _effective_gpu_layers_match(off_layers, mtp_layers):
    """Return None unless both effective layer counts were actually observed."""
    if not isinstance(off_layers, int) or not isinstance(mtp_layers, int):
        return None
    return off_layers == mtp_layers


def build_plan(*, model_path, binary=None, context=4096, device="Vulkan0",
               gpu_layers=16, fit_target_mib=1024, host_reserve_mib=4096,
               batch_size=512, ubatch_size=128, draft_n_max=4,
               repetitions=5, warmup_requests=1, n_predict=128, seed=42,
               timeout_seconds=1800):
    if not (128 <= context <= 262144):
        raise BenchmarkError("context must be between 128 and 262144")
    if not (0 <= gpu_layers <= 4096) or gpu_layers == 0:
        raise BenchmarkError("GPU benchmark requires an explicit layer count in [1, 4096]")
    if not (1 <= fit_target_mib <= 2**20 and 0 <= host_reserve_mib <= 2**20):
        raise BenchmarkError("memory reserves must be between 0 and 1048576 MiB")
    if not (1 <= repetitions <= 30 and 0 <= warmup_requests <= 3):
        raise BenchmarkError("repetitions must be in [1, 30] and warmups in [0, 3]")
    if not (16 <= n_predict <= 4096 and 1 <= draft_n_max <= 64):
        raise BenchmarkError("generation length or requested draft maximum is out of bounds")
    if not (1 <= timeout_seconds <= 1800):
        raise BenchmarkError("timeout must be in [1, 1800] seconds")
    model = Path(model_path).expanduser().resolve(strict=True)
    observed = inspect(model)
    if observed["split"]["multiple_parts_reported"]:
        raise BenchmarkError("multi-part GGUFs are not supported by this benchmark")
    nextn = observed["architecture_metadata"].get("nextn_predict_layers")
    if not isinstance(nextn, int) or isinstance(nextn, bool) or nextn <= 0:
        raise BenchmarkError("selected model has no positive embedded NextN metadata")
    engine = discover_binary(binary)
    entry = resolve_entrypoint(engine, "serve")
    engine_sha = sha256_file(engine)
    version = _bounded([str(engine), "--version"])
    help_text = _bounded(command_prefix(entry) + ["--help"], timeout=30)
    required = ("--model", "--ctx-size", "--parallel", "--fit", "--fit-target",
                "--spec-type", "--spec-draft-n-max", "--gpu-layers", "--device",
                "--no-ui", "--host", "--port", "--batch-size", "--ubatch-size",
                "--verbosity")
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise BenchmarkError("serve Help lacks required flags: " + ", ".join(missing))
    if "draft-mtp" not in help_text:
        raise BenchmarkError("serve Help does not advertise draft-mtp")
    device_text = _bounded(command_prefix(entry) + ["--list-devices"], timeout=30)
    if not re.search(r"(?m)^\s*" + re.escape(device) + r"\s*:", device_text):
        raise BenchmarkError("requested device is absent from a fresh --list-devices result")
    port = _port()
    prompt_sha = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    common = [
        "--offline", "--model", str(model), "--ctx-size", str(context),
        "--parallel", "1", "--device", device, "--gpu-layers", str(gpu_layers),
        "--batch-size", str(batch_size), "--ubatch-size", str(ubatch_size),
        "--fit", "on", "--fit-target", str(fit_target_mib),
        "--no-ui", "--host", "127.0.0.1", "--port", str(port),
        # Current llama.cpp prints effective offload counts and joint-fit trace
        # evidence only at trace/debug verbosity. Capture remains memory-bounded.
        "--verbosity", "5",
    ]
    variants = []
    for mode, spec, extra in (
        ("off", "none", []),
        ("embedded_mtp", "draft-mtp", ["--spec-draft-n-max", str(draft_n_max)]),
    ):
        argv = command_prefix(entry) + common + ["--spec-type", spec, *extra]
        redacted = ["<llama-executable>" if item == str(engine) else
                    "<model>" if item == str(model) else item for item in argv]
        variants.append({"mode": mode, "port": port, "argv": argv,
                         "argv_redacted": redacted})
    return {
        "schema_version": "0.1.0", "kind": "speculative_server_benchmark",
        "status": "prepared_not_executed", "engine_sha256": engine_sha,
        "engine_version_excerpt": version.strip()[:500],
        "engine_commit": re.search(r"commit\s+([0-9a-f]{8,40})", version, re.I).group(1)
        if re.search(r"commit\s+([0-9a-f]{8,40})", version, re.I) else None,
        "model": {"filename": observed["source"]["filename"],
                  "file_size_bytes": observed["source"]["file_size_bytes"],
                  "architecture": observed["general"]["architecture"],
                  "nextn_predict_layers": nextn,
                  "sha256": sha256_file(model)},
        "configuration": {
            "context_tokens": context, "concurrency": 1, "device_id": device,
            "gpu_layers": gpu_layers, "batch_size": batch_size,
            "ubatch_size": ubatch_size, "fit": "on",
            "device_fit_target_mib": fit_target_mib,
            "host_reserve_mib": host_reserve_mib,
            "draft_n_max": draft_n_max,
        },
        "workload": {
            "kind": "synthetic_fixed_completion", "prompt_sha256": prompt_sha,
            "prompt": PROMPT, "sampling": {"temperature": 0, "top_k": 1,
                                             "seed": seed, "cache_prompt": False},
            "n_predict": n_predict, "warmup_requests": warmup_requests,
            "repetitions_per_mode": repetitions,
        },
        "variants": [{"mode": item["mode"], "port": item["port"],
                      "argv_redacted": item["argv_redacted"]} for item in variants],
        "runs": [], "summary": None,
        "limitations": [
            "This is a synthetic single-prompt, concurrency-one server workload, not representative user traffic.",
            "The server fit path is joint only when the selected speculative extra model is measured; fallback-to-target-only logs invalidate MTP results.",
            "WDDM per-process VRAM may be unavailable; device reserve enforcement therefore uses conservative system-wide free memory.",
            "Host Working Set and NVIDIA device sampling are periodic and may miss short peaks.",
            "The fixed GPU layer count, context, batch, device and fit target are held equal across modes.",
            "No quality equivalence, long-context behavior, multimodal behavior or concurrent serving is evaluated.",
        ],
        "_variants": variants,
        "_binary": str(engine),
        "_model_path": str(model),
        "_timeout_seconds": timeout_seconds,
        "_repetitions": repetitions,
        "_warmups": warmup_requests,
        "_n_predict": n_predict,
        "_seed": seed,
        "_host_reserve_mib": host_reserve_mib,
        "_device_reserve_mib": fit_target_mib,
    }


def execute_plan(plan):
    if plan["status"] != "prepared_not_executed":
        raise BenchmarkError("only a prepared benchmark can execute")
    binary = Path(plan["_binary"])
    model = Path(plan["_model_path"])
    if sha256_file(binary) != plan["engine_sha256"]:
        raise BenchmarkError("engine changed since benchmark plan was prepared")
    if model.stat().st_size != plan["model"]["file_size_bytes"]:
        raise BenchmarkError("model size changed since benchmark plan was prepared")
    if sha256_file(model) != plan["model"]["sha256"]:
        raise BenchmarkError("model content changed since benchmark plan was prepared")
    plan["status"] = "running"
    overall_started = time.monotonic()
    runs = []
    for variant in plan["_variants"]:
        run = _run_scenario(
            variant["argv"], mode=variant["mode"], port=variant["port"],
            timeout_seconds=plan["_timeout_seconds"],
            repetitions=plan["_repetitions"], warmup_requests=plan["_warmups"],
            n_predict=plan["_n_predict"], seed=plan["_seed"],
            host_reserve_mib=plan["_host_reserve_mib"],
            device_reserve_mib=plan["_device_reserve_mib"],
        )
        run["summary"] = _summarize_scenario(run)
        runs.append(run)
    plan["runs"] = runs
    off, mtp = runs
    layer_match = _effective_gpu_layers_match(
        off.get("effective_gpu_layers_observed"),
        mtp.get("effective_gpu_layers_observed"),
    )
    equal_layers = layer_match is True
    runs_ok = all(run["status"] == "completed" for run in runs)
    memory_ok = all(run.get("memory", {}).get("device_reserve_status") == "observed"
                    for run in runs)
    fit_ok = mtp.get("joint_fit_status") == "combined_fit_ready"
    limitation = None
    if layer_match is None:
        limitation = "effective_gpu_layers_unobserved; requested values match but the comparison is not certified fair"
    elif layer_match is False:
        limitation = "effective_gpu_layers_differ; the comparison is not certified fair"
    elif not fit_ok:
        limitation = "joint_fit_path_unverified_or_fell_back_to_target_only"
    elif not memory_ok:
        limitation = "device_memory_reserve_telemetry_incomplete"
    elif not runs_ok:
        limitation = "one_or_more_benchmark_runs_did_not_complete"
    plan["summary"] = {
        "comparison_status": "comparable" if runs_ok and fit_ok and equal_layers and memory_ok else "incomplete_or_not_comparable",
        "memory_reserve_telemetry_complete": memory_ok,
        "effective_gpu_layers_match": layer_match,
        "requested_gpu_layers": plan["configuration"]["gpu_layers"],
        "requested_gpu_layers_match": True,
        "comparison_limitation": limitation,
        "off_decode_tokens_per_second": off["summary"]["decode_tokens_per_second"],
        "mtp_decode_tokens_per_second": mtp["summary"]["decode_tokens_per_second"],
        "decode_speedup": (
            mtp["summary"]["decode_tokens_per_second"]["mean"] /
            off["summary"]["decode_tokens_per_second"]["mean"]
            if off["summary"]["decode_tokens_per_second"]["mean"] and
               mtp["summary"]["decode_tokens_per_second"]["mean"] else None),
        "off_latency_ms": off["summary"]["latency_ms"],
        "mtp_latency_ms": mtp["summary"]["latency_ms"],
        "off_ttft_ms": off["summary"]["ttft_ms"],
        "mtp_ttft_ms": mtp["summary"]["ttft_ms"],
        "mtp_acceptance_rate": mtp["summary"]["acceptance_rate"],
        "elapsed_seconds": round(time.monotonic() - overall_started, 3),
    }
    plan["status"] = ("completed" if runs_ok and fit_ok and equal_layers and memory_ok else
                      "incomplete_or_not_comparable")
    for key in [name for name in plan if name.startswith("_")]:
        plan.pop(key, None)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--device", default="Vulkan0")
    parser.add_argument("--gpu-layers", type=int, default=16)
    parser.add_argument("--fit-target-mib", type=int, default=1024)
    parser.add_argument("--host-reserve-mib", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--draft-n-max", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly load both model modes and run the HTTP benchmark")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plan = None
    try:
        plan = build_plan(
            model_path=args.model, binary=args.binary, context=args.context,
            device=args.device, gpu_layers=args.gpu_layers,
            fit_target_mib=args.fit_target_mib,
            host_reserve_mib=args.host_reserve_mib,
            batch_size=args.batch_size, ubatch_size=args.ubatch_size,
            draft_n_max=args.draft_n_max, repetitions=args.repetitions,
            warmup_requests=args.warmup_requests, n_predict=args.n_predict,
            seed=args.seed, timeout_seconds=args.timeout,
        )
        if args.execute:
            plan = execute_plan(plan)
        else:
            for key in [name for name in plan if name.startswith("_")]:
                plan.pop(key, None)
        payload = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, payload, protected=(args.model, args.binary),
                       overwrite=args.overwrite)
            print("Created speculative server benchmark:", args.output)
        else:
            print(payload, end="")
        return 0 if plan["status"] in ("prepared_not_executed", "completed") else 2
    except (OSError, ValueError, BenchmarkError) as error:
        print("Speculative benchmark rejected:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
