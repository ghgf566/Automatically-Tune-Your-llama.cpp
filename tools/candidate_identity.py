"""Stable, content-addressed ATYL candidate keys (not executable authorizations).

Only deterministic JSON-compatible configuration is hashed. Volatile timestamps,
paths, diagnostics and observed outcomes are never part of a candidate's key.
"""
import hashlib
import json


def canonical_digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def candidate_id(search, model, engine_sha256, spec_candidate, *, profile=None):
    config = {
        "schema": "atyl_candidate_identity_v1",
        "engine_binary_sha256": engine_sha256,
        "model": model,
        "spec": {key: spec_candidate.get(key) for key in (
            "mode", "draft_n_max", "draft_artifact_id",
            "draft_device_id", "draft_gpu_layers")},
    }
    if profile is None:
        config["search"] = search
    else:
        config["profile"] = {
            "profile_id": profile["profile_id"],
            "engine": profile["engine"],
            "model": profile["model"],
            "workload": profile["workload"],
            "objective": profile["objective"],
            "constraints": profile["constraints"],
            "candidate_parameters": profile["candidate_parameters"],
            "user_locked_parameters": profile["user_locked_parameters"],
            "speculative_search": profile["speculative_search"],
            "registry_upstream_commit": profile["registry_upstream_commit"],
        }
    return canonical_digest(config)
