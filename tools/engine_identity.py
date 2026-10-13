"""Observed engine identity and cache keys, kept separate from CLI registry version."""
import hashlib
import json
import re


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for piece in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def capability_cache_key(binary_sha256, subcommand, registry_upstream_commit, version_output):
    """Same inputs => same key; any binary, route, registry or version change => new key."""
    payload = {
        "binary_sha256": binary_sha256,
        "subcommand": subcommand,
        "registry_upstream_commit": registry_upstream_commit,
        "version_output": version_output,
        "key_schema": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()



# This is the observed official Windows build-info line, not a guess derived
# from the pinned documentation commit. Reject unfamiliar formats.
BUILD_LINE = re.compile(
    r"^version:\s*(?P<version>[^\r\n()]+?)\s+"
    r"\(build\s+(?P<number>[0-9]+),\s*commit\s+"
    r"(?P<commit>[0-9a-fA-F]{7,40})\)\s*$",
    re.MULTILINE,
)


def parse_reported_build(version_text):
    """Return only unambiguous version evidence; abbreviated SHAs stay abbreviated."""
    matches = list(BUILD_LINE.finditer(version_text))
    if len(matches) != 1:
        return {"engine_version_label": None, "engine_build_number": None,
                "engine_build_commit": None, "engine_build_commit_form": "unknown"}
    match = matches[0]
    commit = match.group("commit").lower()
    return {
        "engine_version_label": match.group("version").strip(),
        "engine_build_number": int(match.group("number")),
        "engine_build_commit": commit,
        "engine_build_commit_form": "full" if len(commit) == 40 else "abbreviated",
    }

def observed_identity(binary_sha256, command, registry_upstream_commit, version_result,
                      help_observed):
    """Do not claim an engine build commit from a registry SHA or an unknown version."""
    successful_version = version_result["status"] == "success"
    version_text = (version_result["stdout"] or version_result["stderr"]).strip()
    known = successful_version and bool(version_text)
    return {
        "binary_sha256": binary_sha256,
        "engine_version_output": version_text[:4096] if known else None,
        # Only identify a commit printed by the executable itself.
        **parse_reported_build(version_text) if known else
          parse_reported_build(""),
        "registry_upstream_commit": registry_upstream_commit,
        "cache_key": capability_cache_key(binary_sha256, command, registry_upstream_commit,
                                          version_text if known else None),
        "cacheable": bool(known and help_observed),
    }
