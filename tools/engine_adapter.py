"""Pure command routing for official llama.app and opt-in legacy llama.cpp binaries.

The adapter never launches a process or installs/updates llama.cpp.
"""
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Optional
import shutil

ROUTES = {
    "serve": "llama-server",
    "bench": "llama-bench",
    "fit-params": None,  # No pinned argument registry yet.
}
LEGACY = {program: command for command, program in ROUTES.items() if program}
LEGACY["llama-fit-params"] = "fit-params"
READ_ONLY_ACTIONS = {"version", "--help", "--list-devices"}


@dataclass(frozen=True)
class EntryPoint:
    kind: str
    command: str
    registry_program: Optional[str]
    binary: str


def basename(binary):
    """Handle Windows paths while validating profiles on a Linux CI runner."""
    return PureWindowsPath(str(binary)).name.lower().removesuffix(".exe")


def resolve_entrypoint(binary, command):
    if command not in ROUTES:
        raise ValueError("unsupported llama command: " + str(command))
    name = basename(binary)
    if name == "llama":
        return EntryPoint("integrated", command, ROUTES[command], str(binary))
    if name in LEGACY:
        expected = LEGACY[name]
        if command != expected:
            raise ValueError("legacy binary " + name + " requires command " + expected)
        return EntryPoint("legacy", command, ROUTES[command], str(binary))
    raise ValueError("expected official llama[.exe] or supported legacy llama-* binary")


def route_from_profile(engine):
    """One routing authority shared by Profile validation and runtime probing."""
    name = basename(engine["binary"])
    command = engine.get("command")
    if name in LEGACY and command is None:
        command = LEGACY[name]
    if name == "llama" and command is None:
        raise ValueError("official llama binary requires engine.command")
    return resolve_entrypoint(engine["binary"], command)


def discover_binary(binary=None):
    """Default to PATH. Explicit legacy paths are never discovered automatically."""
    if binary is None:
        found = shutil.which("llama") or shutil.which("llama.exe")
        if found is None:
            raise ValueError("official 'llama' not found in PATH; see https://llama.app")
        path = Path(found).expanduser().resolve()
    else:
        requested = Path(binary).expanduser()
        if requested.is_file():
            path = requested.resolve()
        elif basename(binary) == "llama" and not requested.is_absolute():
            found = shutil.which(str(requested))
            if found is None:
                raise ValueError("unable to find official 'llama' in PATH")
            path = Path(found).resolve()
        else:
            raise ValueError("binary path is not a file: " + str(requested))
    if not path.is_file():
        raise ValueError("executable path is not a regular file: " + str(path))
    return path


def command_prefix(entrypoint):
    """The only place in ATYL that assembles the engine command prefix."""
    prefix = [entrypoint.binary]
    if entrypoint.kind == "integrated":
        prefix.append(entrypoint.command)
    return prefix


def probe_argv(entrypoint, action):
    """Permit only proven read-only CLI invocations, never a bare 'serve'."""
    if action not in READ_ONLY_ACTIONS:
        raise ValueError("not a read-only probe action: " + str(action))
    if action == "--list-devices" and entrypoint.command == "fit-params":
        raise ValueError("device listing is not validated for fit-params")
    if action == "version":
        if entrypoint.kind == "integrated":
            return [entrypoint.binary, "version"]
        return [entrypoint.binary, "--version"]
    return command_prefix(entrypoint) + [action]
