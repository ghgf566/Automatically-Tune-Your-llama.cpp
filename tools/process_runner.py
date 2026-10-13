"""Bounded process execution shared by llama help, device probes and fit estimates.

Stores stdout/stderr in temporary files to avoid unbounded Python RAM use.
Monitors temporary file sizes and kills a noisy process before disk growth
exceeds the configured cap. No shell, no stdin, no automatic installation.
"""
import os
import subprocess
import tempfile
import time


def run_bounded(argv, timeout_seconds, *, capture_bytes=65536,
                disk_limit_bytes=8 * 1024 * 1024, env=None):
    """Status: success, nonzero_exit, timeout, output_limit, error."""
    if not argv or timeout_seconds <= 0 or capture_bytes < 1 or disk_limit_bytes < capture_bytes:
        raise ValueError("invalid bounded-process settings")
    try:
        with tempfile.TemporaryFile(mode="w+b") as out, tempfile.TemporaryFile(mode="w+b") as err:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env,
            )
            deadline = time.monotonic() + timeout_seconds
            status, code = None, None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    break
                try:
                    code = proc.wait(timeout=min(0.5, remaining))
                    status = "success" if code == 0 else "nonzero_exit"
                    break
                except subprocess.TimeoutExpired:
                    if (os.fstat(out.fileno()).st_size > disk_limit_bytes
                            or os.fstat(err.fileno()).st_size > disk_limit_bytes):
                        status = "output_limit"
                        break
            if status in ("timeout", "output_limit"):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                proc.wait()
            out.seek(0)
            err.seek(0)
            stdout_bytes = out.read(capture_bytes + 1)
            stderr_bytes = err.read(capture_bytes + 1)
            return {
                "status": status, "returncode": code,
                "stdout": stdout_bytes[:capture_bytes].decode("utf-8", "replace"),
                "stderr": stderr_bytes[:capture_bytes].decode("utf-8", "replace"),
                "truncated": (len(stdout_bytes) > capture_bytes
                              or len(stderr_bytes) > capture_bytes
                              or status == "output_limit"),
            }
    except OSError as exc:
        return {"status": "error", "returncode": None, "stdout": "",
                "stderr": type(exc).__name__, "truncated": True}
