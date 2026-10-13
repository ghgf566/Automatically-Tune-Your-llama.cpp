"""Strict structured parser for pinned llama.cpp 'fit-params --fit-print on'.

The upstream common/fit.cpp::common_fit_print prints exactly four whitespace-
separated columns per device: name, model MiB, context MiB, compute MiB.
This parser does not claim that the estimate is a successful model load.
"""
import re

ROW = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_.-]*)\s+"
    r"(?P<model>[0-9]+)\s+(?P<context>[0-9]+)\s+(?P<compute>[0-9]+)"
)
MAX_DEVICE_ROWS = 32
MAX_MEMORY_MIB = 2**53  # Keep exact integer semantics in JSON consumers.


def parse_memory_estimate(stdout, selected_device):
    """Return None for unknown/unexpected output, never partial numeric results.

    Device selection is a caller-verified single device ID or 'none'. A parser
    success means only format recognition of the upstream MiB report.
    """
    if not isinstance(stdout, str) or not isinstance(selected_device, str):
        return None
    rows = stdout.splitlines()
    if not 1 <= len(rows) <= MAX_DEVICE_ROWS + 1:
        return None
    observed = {}
    for raw in rows:
        matched = ROW.fullmatch(raw.strip())
        if matched is None:
            return None
        name = matched["name"]
        if name in observed:
            return None
        numbers = [int(matched[key]) for key in ("model", "context", "compute")]
        if any(value > MAX_MEMORY_MIB for value in numbers):
            return None
        observed[name] = {
            "model_mib": numbers[0],
            "context_mib": numbers[1],
            "compute_mib": numbers[2],
            "estimated_total_mib": sum(numbers),
        }
    host = observed.pop("Host", None)
    if host is None:
        return None
    expected = set() if selected_device == "none" else {selected_device}
    if set(observed) != expected:
        return None
    return {
        "status": "parsed",
        "unit": "MiB",
        "devices": [
            {"device_id": name, **values}
            for name, values in sorted(observed.items())
        ],
        "host": host,
        "estimated_total_mib": (
            host["estimated_total_mib"] +
            sum(entry["estimated_total_mib"] for entry in observed.values())
        ),
        "model_load_tested": False,
        "inference_tested": False,
    }
