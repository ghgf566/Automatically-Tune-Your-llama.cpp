"""Best-effort, system-wide NVIDIA memory sampling for bounded load checks.

This intentionally does not claim process attribution: on Windows WDDM,
``nvidia-smi`` commonly reports per-process memory as N/A. The device-level
free/used values are useful headroom observations, not model VRAM peaks.
"""
import csv
import io
import shutil
import threading
import time

from process_runner import run_bounded

MAX_OUTPUT_BYTES = 64 * 1024
SAMPLE_TIMEOUT_SECONDS = 2
SAMPLE_INTERVAL_SECONDS = 1


def _unknown(reason):
    return {"status": "unknown", "reason": reason, "provider": None,
            "process_attribution": "not_available", "sample_count": 0,
            "sample_interval_ms": SAMPLE_INTERVAL_SECONDS * 1000, "devices": []}


def parse_nvidia_smi(stdout):
    """Parse the strict CSV form requested by ``--query-gpu``.

    Invalid or incomplete output is unknown. We never infer GPU identity from
    device order or a llama.cpp backend name such as ``Vulkan0``.
    """
    devices = []
    seen = set()
    try:
        rows = csv.reader(io.StringIO(stdout, newline=""), skipinitialspace=True)
        for row in rows:
            if len(row) != 5:
                return None
            uuid, name, total, free, used = (value.strip() for value in row)
            if (not uuid.startswith("GPU-") or not name or uuid in seen
                    or not all(value.isdecimal() for value in (total, free, used))):
                return None
            total, free, used = int(total), int(free), int(used)
            if total <= 0 or free > total or used > total:
                return None
            devices.append({"uuid": uuid, "name": name, "total_mib": total,
                            "free_mib": free, "used_mib": used})
            seen.add(uuid)
    except (csv.Error, ValueError):
        return None
    return devices or None


def sample_nvidia_smi(executable=None):
    """Take one bounded system-wide NVIDIA device memory snapshot."""
    executable = executable or shutil.which("nvidia-smi")
    if not executable:
        return None, "provider_unavailable"
    result = run_bounded(
        [executable, "--query-gpu=uuid,name,memory.total,memory.free,memory.used",
         "--format=csv,noheader,nounits"],
        SAMPLE_TIMEOUT_SECONDS, capture_bytes=MAX_OUTPUT_BYTES,
        disk_limit_bytes=MAX_OUTPUT_BYTES,
    )
    if result["status"] != "success" or result["truncated"]:
        return None, "query_" + result["status"]
    devices = parse_nvidia_smi(result["stdout"])
    if devices is None:
        return None, "unrecognized_or_incomplete_output"
    return devices, None


class DeviceMemoryMonitor:
    """Sample NVIDIA device headroom off the readiness polling thread."""
    def __init__(self, sampler=sample_nvidia_smi, interval_seconds=SAMPLE_INTERVAL_SECONDS):
        if interval_seconds < 1:
            raise ValueError("device memory interval must be at least one second")
        self._sampler = sampler
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples = []
        self._failures = []
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("device memory monitor already started")
        self._thread = threading.Thread(target=self._run, name="atyl-device-memory",
                                       daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                devices, reason = self._sampler()
            except Exception:
                devices, reason = None, "sampler_error"
            with self._lock:
                if devices:
                    self._samples.append(devices)
                else:
                    self._failures.append(reason or "sample_unavailable")
            if self._stop.wait(self._interval):
                break

    def stop_and_report(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=SAMPLE_TIMEOUT_SECONDS + self._interval + 1)
        with self._lock:
            samples = list(self._samples)
            failures = list(self._failures)
        if not samples:
            reason = failures[0] if failures else "no_samples"
            report = _unknown(reason)
            report["sample_count"] = len(samples)
            return report

        per_device = {}
        expected_device_set = None
        for sample in samples:
            current_device_set = {device["uuid"] for device in sample}
            if expected_device_set is None:
                expected_device_set = current_device_set
            elif current_device_set != expected_device_set:
                failures.append("device_set_changed_during_sampling")
            for device in sample:
                item = per_device.setdefault(device["uuid"], {
                    "uuid": device["uuid"], "name": device["name"],
                    "total_memory_mib": device["total_mib"],
                    "minimum_free_memory_mib": device["free_mib"],
                    "maximum_used_memory_mib": device["used_mib"],
                    "sample_count": 0,
                })
                if (item["name"] != device["name"]
                        or item["total_memory_mib"] != device["total_mib"]):
                    failures.append("device_identity_changed")
                    continue
                item["minimum_free_memory_mib"] = min(
                    item["minimum_free_memory_mib"], device["free_mib"])
                item["maximum_used_memory_mib"] = max(
                    item["maximum_used_memory_mib"], device["used_mib"])
                item["sample_count"] += 1
        return {
            "status": "partial" if failures else "observed",
            "reason": failures[0] if failures else None,
            "provider": "nvidia_smi",
            "process_attribution": "system_wide",
            "sample_count": len(samples),
            "sample_interval_ms": int(self._interval * 1000),
            "devices": [per_device[key] for key in sorted(per_device)],
        }
