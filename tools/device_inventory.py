"""Parse the pinned llama.cpp GGML --list-devices text without making capability claims.

Upstream: common/arg.cpp::common_print_available_devices at commit
77c4540ce4667f4577753fe034a3a8042f510b8f.
"""
import re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ROW = re.compile(
    r"^  (?P<device_id>[A-Za-z][A-Za-z0-9_.-]*): (?P<description>.+) "
    r"\((?P<total>[0-9]+) MiB, (?P<free>[0-9]+) MiB free\)$"
)


def unknown(reason):
    return {
        "status": "unknown", "reason": reason, "devices": None,
        "non_cpu_device_count": None, "cpu_devices_enumerated": False,
        "model_backend_compatibility": "not_tested",
    }


def parse_list_devices(stdout, stderr=""):
    """Only call on a zero-exit result. Log lines before header are permitted.

    An invalid listing is unknown rather than an empty or partially observed list.
    GGML intentionally excludes CPU from this command.
    """
    source = stdout if "Available devices:" in ANSI.sub("", stdout) else stderr
    lines = ANSI.sub("", source).splitlines()
    headers = [i for i, line in enumerate(lines)
               if line.strip() == "Available devices:"]
    if len(headers) != 1:
        return unknown("missing_or_ambiguous_header")
    following = [line.rstrip() for line in lines[headers[0] + 1:] if line.strip()]
    if not following:
        return unknown("missing_device_rows")
    if following[0] == "  (none)":
        if len(following) != 1:
            return unknown("unexpected_data_after_none")
        return {
            "status": "none", "reason": None, "devices": [],
            "non_cpu_device_count": 0, "cpu_devices_enumerated": False,
            "model_backend_compatibility": "not_tested",
        }
    devices = []
    ids = set()
    for line in following:
        match = ROW.fullmatch(line)
        if not match:
            # Do not turn newer upstream formats into false negatives.
            return unknown("unexpected_device_row_format")
        item = match.groupdict()
        total = int(item["total"])
        free = int(item["free"])
        if item["device_id"] in ids or free > total:
            return unknown("duplicate_id_or_inconsistent_memory")
        ids.add(item["device_id"])
        devices.append({
            "id": item["device_id"],
            "description": item["description"],
            "total_memory_mib": total, "free_memory_mib": free,
        })
    return {
        "status": "observed", "reason": None, "devices": devices,
        "non_cpu_device_count": len(devices), "cpu_devices_enumerated": False,
        "model_backend_compatibility": "not_tested",
    }


def observation(command_result):
    """Do not parse nonzero-exit, timeout or failed commands as valid hardware."""
    status = command_result["status"]
    if status != "success":
        return unknown("command_" + status)
    return parse_list_devices(command_result["stdout"], command_result["stderr"])
