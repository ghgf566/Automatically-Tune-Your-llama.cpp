"""Tiered, read-only GGUF identity: fast metadata prefix vs opt-in full SHA256.

A prefix hash + file stat detects many accidental changes but is NOT a
cryptographic identity for the weight payload. A full digest identifies bytes
but does not prove valid tensors, backend support or inference correctness.
"""
import hashlib
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


class ModelIdentityError(ValueError):
    pass


def fingerprint(path, metadata_span_bytes, *, full_sha256=False):
    path = Path(path).expanduser().resolve()
    first = path.stat()
    if not 0 < metadata_span_bytes <= first.st_size:
        raise ModelIdentityError("invalid GGUF metadata span for fingerprint")
    meta = hashlib.sha256()
    complete = hashlib.sha256() if full_sha256 else None
    remaining = metadata_span_bytes
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise ModelIdentityError("GGUF metadata changed during fingerprint")
            meta.update(chunk)
            if complete:
                complete.update(chunk)
            remaining -= len(chunk)
        if complete:
            for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                complete.update(chunk)
    final = path.stat()
    if (first.st_size != final.st_size
            or first.st_mtime_ns != final.st_mtime_ns):
        raise ModelIdentityError("model file changed during fingerprinting")
    return {
        "method": "metadata_prefix_and_stat_v1",
        "metadata_span_bytes": metadata_span_bytes,
        "metadata_sha256": meta.hexdigest(),
        "file_size_bytes": final.st_size,
        "modified_ns": final.st_mtime_ns,
        "full_file_sha256": complete.hexdigest() if complete else None,
        "full_file_hashed": bool(complete),
        "weight_integrity_verified": False,
    }


def matches_existing(path, saved):
    """Check the saved fingerprint, including full hash if previously requested."""
    if not isinstance(saved, dict):
        return False
    if saved.get("method") != "metadata_prefix_and_stat_v1":
        return False
    try:
        updated = fingerprint(
            path, saved["metadata_span_bytes"],
            full_sha256=saved.get("full_file_hashed") is True)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return updated == saved
