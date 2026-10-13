"""Observed engine identity and cache keys, kept separate from CLI registry version."""
import hashlib
import json


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


def observed_identity(binary_sha256, command, registry_upstream_commit, version_result,
                      help_observed):
    """Do not claim an engine build commit from a registry SHA or an unknown version."""
    successful_version = version_result["status"] == "success"
    version_text = (version_result["stdout"] or version_result["stderr"]).strip()
    known = successful_version and bool(version_text)
    return {
        "binary_sha256": binary_sha256,
        "engine_version_output": version_text[:4096] if known else None,
        "engine_build_commit": None,  # Not reliably decoded from all builds yet.
        "registry_upstream_commit": registry_upstream_commit,
        "cache_key": capability_cache_key(binary_sha256, command, registry_upstream_commit,
                                          version_text if known else None),
        "cacheable": bool(known and help_observed),
    }
