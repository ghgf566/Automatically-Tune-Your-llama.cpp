"""One-shot, best-effort RAM and GGML device headroom observations.

Numbers are availability snapshots, NOT peak process usage, committed VRAM,
proof of model loading, or a source of automatic GPU-layer recommendations.
"""
import ctypes
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone

from device_inventory import observation as parse_observation
from process_runner import run_bounded

MIB = 1024 * 1024
DEVICE_PROBE_TIMEOUT = 15
MAX_DEVICE_OUTPUT = 64 * 1024


def clean_llama_env():
    """Discard CLI override environment variables without modifying the parent."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith("LLAMA_ARG_")
            and k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}


def _unavailable(reason):
    return {"status": "unknown", "reason": reason,
            "total_mib": None, "available_mib": None}


def system_ram():
    """Get a system-level snapshot; no process list and no privileged APIs."""
    host = platform.system()
    try:
        if host == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            memory = MemoryStatus()
            memory.dwLength = ctypes.sizeof(memory)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                return _unavailable("GlobalMemoryStatusEx_failed")
            total, available = memory.ullTotalPhys, memory.ullAvailPhys
        elif host == "Linux":
            data = {}
            with open("/proc/meminfo", encoding="ascii") as file:
                for line in file:
                    parts = line.split()
                    if len(parts) >= 3 and parts[2] == "kB":
                        data[parts[0].rstrip(":")] = int(parts[1]) * 1024
            total = data["MemTotal"]
            available = data["MemAvailable"]
        else:
            return _unavailable("unsupported_platform")
        if total <= 0 or not 0 <= available <= total:
            return _unavailable("inconsistent_ram_data")
        return {"status": "observed", "reason": None,
                "total_mib": total // MIB,
                "available_mib": available // MIB}
    except (OSError, ValueError, KeyError, AttributeError):
        return _unavailable("ram_query_failed")


def ggml_headroom(binary, device_id):
    """Read one known GGML device from explicit, bounded serve --list-devices.

    Use only after the caller validates the binary SHA and previous successful
    device probe, and only within an explicitly requested estimate execution.
    """
    if device_id == "none":
        return {"status": "not_requested", "reason": "cpu_only",
                "device_id": None, "backend_id": None, "description": None,
                "total_mib": None, "available_mib": None}
    result = {"status": "unknown", "reason": "unverified_device",
              "device_id": device_id, "backend_id": None, "description": None,
              "total_mib": None, "available_mib": None}
    observed = run_bounded(
        [str(binary), "serve", "--list-devices"], DEVICE_PROBE_TIMEOUT,
        capture_bytes=MAX_DEVICE_OUTPUT, disk_limit_bytes=1024 * 1024,
        env=clean_llama_env(),
    )
    if observed["status"] != "success" or observed["truncated"]:
        result["reason"] = "device_probe_" + observed["status"]
        return result
    raw_stdout, raw_stderr = observed["stdout"], observed["stderr"]

    inventory = parse_observation({"status": "success",
                                   "stdout": raw_stdout, "stderr": raw_stderr})
    if inventory["status"] != "observed":
        result["reason"] = "device_list_" + inventory["status"]
        return result
    selected = [dev for dev in inventory["devices"] if dev["id"] == device_id]
    if len(selected) != 1:
        result["reason"] = "device_not_in_fresh_listing"
        return result
    device = selected[0]
    return {
        "status": "observed", "reason": None, "device_id": device_id,
        "backend_id": re.sub(r"[0-9]+$", "", device_id) or None,
        "description": device["description"],
        "total_mib": device["total_memory_mib"],
        "available_mib": device["free_memory_mib"],
    }


def observe(binary, device_id):
    """The observation is taken immediately before/after estimate execution."""
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ram": system_ram(), "gpu": ggml_headroom(binary, device_id)}


def _delta(before, after, category):
    a = before[category]
    b = after[category]
    if a["status"] == b["status"] == "observed":
        return b["available_mib"] - a["available_mib"]
    return None


def summarize(before, after):
    """A difference in free RAM/VRAM is not the peak memory used by a model."""
    return {
        "system_ram_available_delta_mib": _delta(before, after, "ram"),
        "gpu_available_delta_mib": _delta(before, after, "gpu"),
        "peak_gpu_usage_mib": None,
        "peak_ram_usage_mib": None,
        "attribution": "unknown_background_load_and_driver_effects",
        "model_load_status": "not_tested",
    }
