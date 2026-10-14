#!/usr/bin/env python3
"""Recheck an explicit target/draft GGUF pair without loading either model.

This preflight verifies the exact local file digests, single-file status,
architecture equality and byte-equivalent tokenizer metadata. A passing report
still requires an opt-in runtime load test before a candidate can run.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from inspect_gguf import inspect
from model_identity import fingerprint
from safe_output import write_text
from validate_contracts import ROOT, load_json, validate_profile_document

REPORT_SCHEMA = load_json(ROOT / "schemas/speculative-pairing.schema.json")
REPORT_VALIDATOR = Draft202012Validator(REPORT_SCHEMA)


class PairingError(ValueError):
    pass


def _verified_model(path, expected_sha256):
    path = Path(path).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected_sha256 or "")):
        raise PairingError("profile must contain a full expected SHA-256 for each model")
    observation = inspect(path)
    if observation["split"]["multiple_parts_reported"]:
        raise PairingError("multi-part GGUF requires a shard-completeness validator first")
    parts = observation["split"]
    if parts["part_count"] not in (None, 1) or parts["part_index"] not in (None, 0):
        raise PairingError("model GGUF is not a verified single-file input")
    digest = fingerprint(path, observation["metadata_bytes_scanned"], full_sha256=True)
    if digest["metadata_sha256"] != observation["metadata_sha256"]:
        raise PairingError("model metadata changed between inspection and full hashing")
    actual = digest["full_file_sha256"]
    if actual is None or actual.lower() != expected_sha256.lower():
        raise PairingError("on-disk model SHA-256 differs from the Profile/Artifact")
    return {
        "filename": path.name,
        "file_size_bytes": observation["source"]["file_size_bytes"],
        "sha256": actual,
        "architecture": observation["general"]["architecture"],
        "metadata_sha256": observation["metadata_sha256"],
        "tokenizer_metadata": observation["tokenizer_metadata"],
        "split": observation["split"],
    }


def verify_pair(profile, artifact_id, target_path, draft_path):
    """Return a non-executable evidence report for the selected external draft."""
    try:
        validate_profile_document(profile)
        search = profile["speculative_search"]
    except (KeyError, ValidationError, ValueError) as error:
        raise PairingError("invalid speculative workload Profile: " + str(error)) from error
    artifacts = [item for item in search["external_artifacts"]
                 if item["artifact_id"] == artifact_id]
    if len(artifacts) != 1:
        raise PairingError("artifact ID must select exactly one external draft in the Profile")
    artifact = artifacts[0]
    target = _verified_model(target_path, profile["model"]["sha256"])
    draft = _verified_model(draft_path, artifact["sha256"])
    if target["architecture"] != profile["model"]["architecture"]:
        raise PairingError("on-disk target architecture differs from the workload Profile")

    architecture_match = (
        isinstance(target["architecture"], str)
        and target["architecture"] == draft["architecture"]
    )
    target_tokenizer = target["tokenizer_metadata"]
    draft_tokenizer = draft["tokenizer_metadata"]
    tokenizer_match = (
        target_tokenizer["status"] == "observed"
        and draft_tokenizer["status"] == "observed"
        and target_tokenizer["metadata_key_count"] == draft_tokenizer["metadata_key_count"]
        and target_tokenizer["metadata_sha256"] == draft_tokenizer["metadata_sha256"]
    )
    reasons = []
    if not architecture_match:
        reasons.append("model_architecture_mismatch_or_unknown")
    if not tokenizer_match:
        reasons.append("tokenizer_metadata_not_byte_equivalent")
    if reasons:
        status = "blocked"
    else:
        status = "requires_runtime_load_verification"
        reasons.append("metadata_match_does_not_prove_engine_pairing_or_load_success")
    report = {
        "schema_version": "0.1.0",
        "kind": "speculative_model_pair_preflight",
        "artifact_id": artifact_id,
        "mode": artifact["mode"],
        "target": target,
        "draft": draft,
        "checks": {
            "full_file_digests_match_profile": True,
            "both_inputs_are_single_file_gguf": True,
            "architecture_match": architecture_match,
            "tokenizer_metadata_byte_equivalent": tokenizer_match,
        },
        "status": status,
        "executable": False,
        "reasons": reasons,
        "limitations": [
            "GGUF metadata and SHA-256 checks do not prove tensor integrity, tokenizer behavior, target-draft compatibility, or backend support.",
            "Multi-part GGUF inputs are rejected until shard completeness can be verified.",
            "No model weights are loaded and no runtime, memory, generation, or benchmark check is performed.",
            "A passing preflight is historical evidence; re-run immediately before any future load operation.",
        ],
    }
    REPORT_VALIDATOR.validate(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = verify_pair(load_json(args.profile), args.artifact_id,
                             args.target_model, args.draft_model)
        payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, payload,
                       protected=(args.profile, args.target_model, args.draft_model),
                       overwrite=args.overwrite)
            print("Created model-pair preflight:", args.output)
        else:
            print(payload, end="")
    except (OSError, KeyError, PairingError, ValidationError, ValueError) as error:
        print("Model-pair preflight rejected:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
