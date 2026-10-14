"""Bounded process execution shared by llama help, device probes and fit estimates.

Captures only bounded stdout/stderr excerpts in memory and stops a noisy child
when either stream exceeds its output cap. No shell, no stdin, no automatic
installation.
"""
import os
import subprocess
import sys
import io
from threading import Event, Thread
import time

from process_lifecycle import isolation_options, stop_owned_tree


def _read_available(stream):
    """Read pipe bytes without waiting for a full requested chunk on Windows."""
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        try:
            file_descriptor = stream.fileno()
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            # Unit tests and adapters may supply in-memory file-like streams.
            return stream.read(65536)
        available = ctypes.c_ulong()
        handle = msvcrt.get_osfhandle(file_descriptor)
        peek_named_pipe = ctypes.windll.kernel32.PeekNamedPipe
        peek_named_pipe.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
            wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
        ]
        peek_named_pipe.restype = wintypes.BOOL
        ok = peek_named_pipe(
            handle, None, 0, None, ctypes.byref(available), None)
        if not ok:
            return b""
        if available.value == 0:
            time.sleep(0.01)
            return None
        return os.read(file_descriptor, min(available.value, 65536))
    return os.read(stream.fileno(), 65536)


def _capture_stream(stream, capture_bytes, output_limit_bytes, limit_event):
    """Drain one pipe while retaining only a bounded diagnostic excerpt."""
    captured = bytearray()
    total = 0
    try:
        while True:
            # BufferedReader.read(n) may wait for n bytes or EOF. Read from the
            # OS pipe in currently available chunks so a live noisy child cannot
            # hide behind a partially filled requested read size.
            chunk = _read_available(stream)
            if chunk is None:
                continue
            if not chunk:
                break
            total += len(chunk)
            remaining = capture_bytes - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
            if total > output_limit_bytes:
                limit_event.set()
                break
    finally:
        stream.close()
    return captured, total


def run_bounded(argv, timeout_seconds, *, capture_bytes=65536,
                disk_limit_bytes=8 * 1024 * 1024, env=None, cancel_event=None):
    """Status: success, nonzero_exit, timeout, output_limit, cancelled, error.

    ``disk_limit_bytes`` retains its old keyword for compatibility; it is now
    an in-memory output cap applied independently to stdout and stderr.
    """
    if not argv or timeout_seconds <= 0 or capture_bytes < 1 or disk_limit_bytes < capture_bytes:
        raise ValueError("invalid bounded-process settings")
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, **isolation_options(),
        )
        limit_event = Event()
        output = {"stdout": None, "stderr": None}

        def capture(name, stream):
            output[name] = _capture_stream(
                stream, capture_bytes, disk_limit_bytes, limit_event)

        readers = [
            Thread(target=capture, args=("stdout", proc.stdout), daemon=True),
            Thread(target=capture, args=("stderr", proc.stderr), daemon=True),
        ]
        for reader in readers:
            reader.start()

        deadline = time.monotonic() + timeout_seconds
        status, code = None, None
        try:
            while True:
                if limit_event.is_set():
                    status = "output_limit"
                    break
                if cancel_event is not None and cancel_event.is_set():
                    status = "cancelled"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    break
                try:
                    code = proc.wait(timeout=min(0.05, remaining))
                    status = "success" if code == 0 else "nonzero_exit"
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            # Ctrl+C or an unexpected exception must never orphan a child.
            stop_owned_tree(proc)
            for reader in readers:
                reader.join(timeout=5)
            raise

        if status in ("timeout", "output_limit", "cancelled"):
            stop_owned_tree(proc)
        for reader in readers:
            reader.join(timeout=0.2)
        # A descendant can inherit a pipe after the direct child exits. Do not
        # let that keep a reader thread (or this call) alive indefinitely.
        if any(reader.is_alive() for reader in readers):
            status = "output_limit"
            stop_owned_tree(proc)
            for reader in readers:
                reader.join(timeout=5)
        if limit_event.is_set():
            status = "output_limit"
            code = None
        if status in ("timeout", "output_limit", "cancelled"):
            code = None

        stdout_bytes = bytes(output["stdout"][0]) if output["stdout"] is not None else b""
        stderr_bytes = bytes(output["stderr"][0]) if output["stderr"] is not None else b""
        stdout_total = output["stdout"][1] if output["stdout"] is not None else 0
        stderr_total = output["stderr"][1] if output["stderr"] is not None else 0
        return {
            "status": status, "returncode": code,
            "stdout": stdout_bytes.decode("utf-8", "replace"),
            "stderr": stderr_bytes.decode("utf-8", "replace"),
            "truncated": (stdout_total > capture_bytes
                          or stderr_total > capture_bytes
                          or status == "output_limit"),
        }
    except OSError as exc:
        return {"status": "error", "returncode": None, "stdout": "",
                "stderr": type(exc).__name__, "truncated": True}


def run_until_ready(argv, timeout_seconds, ready_check, *, capture_bytes=65536,
                    output_limit_bytes=8 * 1024 * 1024, env=None,
                    poll_interval=0.1, cancel_event=None, sample_callback=None):
    """Run an owned service until a caller-supplied local readiness probe passes.

    The process tree is always stopped after readiness, timeout, output overflow
    or cancellation. ``ready_check`` must be a bounded local probe and return a
    boolean; no command is run through a shell.
    """
    if (not argv or timeout_seconds <= 0 or capture_bytes < 1
            or output_limit_bytes < capture_bytes or poll_interval <= 0
            or not callable(ready_check)):
        raise ValueError("invalid readiness-process settings")
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, **isolation_options(),
        )
        limit_event = Event()
        output = {"stdout": None, "stderr": None}

        def capture(name, stream):
            output[name] = _capture_stream(
                stream, capture_bytes, output_limit_bytes, limit_event)

        readers = [
            Thread(target=capture, args=("stdout", proc.stdout), daemon=True),
            Thread(target=capture, args=("stderr", proc.stderr), daemon=True),
        ]
        for reader in readers:
            reader.start()

        deadline = time.monotonic() + timeout_seconds
        status = "error"
        try:
            while True:
                if limit_event.is_set():
                    status = "output_limit"
                    break
                if cancel_event is not None and cancel_event.is_set():
                    status = "cancelled"
                    break
                code = proc.poll()
                if code is not None:
                    status = "nonzero_exit" if code else "exited_before_ready"
                    break
                if sample_callback is not None:
                    sample_callback(proc.pid)
                if ready_check():
                    # A readiness response is only accepted while the owned
                    # service process is still alive.
                    if proc.poll() is None:
                        status = "ready"
                        break
                    status = "exited_before_ready"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    break
                time.sleep(min(poll_interval, remaining))
        except BaseException:
            stop_owned_tree(proc)
            for reader in readers:
                reader.join(timeout=5)
            raise

        # This helper is for short feasibility probes, never a persistent server.
        stop_owned_tree(proc)
        for reader in readers:
            reader.join(timeout=0.2)
        if any(reader.is_alive() for reader in readers):
            status = "output_limit"
            stop_owned_tree(proc)
            for reader in readers:
                reader.join(timeout=5)
        if limit_event.is_set():
            status = "output_limit"

        stdout_bytes = bytes(output["stdout"][0]) if output["stdout"] is not None else b""
        stderr_bytes = bytes(output["stderr"][0]) if output["stderr"] is not None else b""
        return {
            "status": status, "returncode": proc.returncode,
            "stdout": stdout_bytes.decode("utf-8", "replace"),
            "stderr": stderr_bytes.decode("utf-8", "replace"),
            "truncated": (output["stdout"] is None or output["stderr"] is None
                          or output["stdout"][1] > capture_bytes
                          or output["stderr"][1] > capture_bytes
                          or status == "output_limit"),
        }
    except OSError as exc:
        return {"status": "error", "returncode": None, "stdout": "",
                "stderr": type(exc).__name__, "truncated": True}
