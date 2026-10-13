"""Owned-child lifecycle helpers for bounded llama processes.

POSIX: run each child in a fresh session and signal its process group.
Windows: use a new process group and taskkill /T /F for descendants.
This applies only to children launched by ATYL. It does not manage
external services, unrelated processes or an already detached daemon.
"""
import os
import signal
import subprocess


def isolation_options():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_owned_tree(proc):
    """Best-effort stop descendants, then reap the direct child.

    Never call this for an arbitrary PID. Process groups are established by
    isolation_options() at creation, and Windows taskkill scopes to proc.pid.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=False, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except (OSError, ProcessLookupError):
                pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if proc.poll() is None:
                proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
