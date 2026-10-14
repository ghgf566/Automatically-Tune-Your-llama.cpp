"""Sample resident memory for an owned process tree using OS interfaces.

Values are sampled working-set/RSS totals, not an instantaneous or guaranteed
true peak. Sampling can miss short spikes and shared pages may be counted in
more than one process working set.
"""
import ctypes
import os
import platform
from ctypes import wintypes

MIB = 1024 * 1024


def _linux_process_tree(root_pid):
    parent_children = {}
    resident = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdecimal():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="ascii") as file:
                stat = file.read()
            # comm is parenthesized and may itself contain spaces/parentheses.
            fields = stat[stat.rfind(")") + 2:].split()
            pid = int(entry)
            parent_pid = int(fields[1])
            rss_pages = int(fields[21])
        except (OSError, ValueError, IndexError):
            continue
        parent_children.setdefault(parent_pid, []).append(pid)
        resident[pid] = max(0, rss_pages * page_size)
    if root_pid not in resident:
        return None
    selected = {root_pid}
    pending = [root_pid]
    while pending:
        for child in parent_children.get(pending.pop(), []):
            if child not in selected:
                selected.add(child)
                pending.append(child)
    return sum(resident.get(pid, 0) for pid in selected), len(selected), len(selected)


class _ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
    ]


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _windows_process_tree(root_pid):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCountersEx), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid = ctypes.c_void_p(-1).value
    if snapshot == invalid:
        return None
    parents = {}
    try:
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            return None
        while True:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            entry.dwSize = ctypes.sizeof(entry)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    if root_pid not in parents:
        return None

    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid in parents.items():
            if parent_pid in selected and pid not in selected:
                selected.add(pid)
                changed = True

    total = 0
    observed = 0
    root_observed = False
    for pid in selected:
        handle = kernel32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
        if not handle:
            continue
        try:
            counters = _ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                total += counters.WorkingSetSize
                observed += 1
                root_observed = root_observed or pid == root_pid
        finally:
            kernel32.CloseHandle(handle)
    if root_observed:
        return total, observed, len(selected)
    return None


def sample_process_tree(root_pid):
    """Return one sampled process-tree resident-memory total in bytes."""
    host = platform.system()
    try:
        if host == "Windows":
            result = _windows_process_tree(int(root_pid))
            metric = "working_set"
        elif host == "Linux":
            result = _linux_process_tree(int(root_pid))
            metric = "rss"
        else:
            return {"status": "unknown", "reason": "unsupported_platform"}
    except (OSError, ValueError, TypeError, AttributeError):
        result = None
        metric = "working_set" if host == "Windows" else "rss"
    if result is None:
        return {"status": "unknown", "reason": "process_tree_unavailable"}
    total_bytes, process_count, discovered_count = result
    return {"status": "observed", "reason": None, "resident_bytes": total_bytes,
            "process_count": process_count, "discovered_process_count": discovered_count,
            "coverage_status": "complete" if process_count == discovered_count else "partial",
            "metric": metric}


def mib_floor(byte_count):
    return byte_count // MIB
