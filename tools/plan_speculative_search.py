#!/usr/bin/env python3
"""Read-only speculative decoding search preview.

Enumerates bounded MTP/draft candidates; never launches llama, loads model
weights, reads other local models, guesses engine enum support or downloads
draft files. Preview output is NOT an executable configuration.
"""
import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from runtime_snapshot import check_snapshot, SnapshotValidationError
from safe_output import write_text
from validate_contracts import (
    ROOT, _registry, load_json, check_speculative_search_consistency,
)

SPEC_SCHEMA = load_json(ROOT / "schemas/speculative-search.schema.json")
SEARCH_VALIDATOR = Draft202012Validator(SPEC_SCHEMA, registry=_registry())
PREVIEW_VALIDATOR = Draft202012Validator(
    {**SPEC_SCHEMA, "$ref": "#/$defs/preview"}, registry=_registry()
)
REQUIRED_FLAGS = {"--spec-type", "--spec-draft-n-max"}


class SpeculativePlanError(ValueError):
    pass


def _candidate(mode, n, artifact=None, placement=None):
    return {
        "mode": mode,
        "draft_n_max": n,
        "draft_artifact_id": None if artifact is None else artifact["artifact_id"],
        "draft_device_id": None if placement is None else placement["device_id"],
        "draft_gpu_layers": None if placement is None else placement["gpu_layers"],
        "readiness": "blocked_missing_evidence",
        "reasons": [],
        "executable": False,
    }


def make_preview(search, target_observation, runtime_snapshot):
    """Candidates require actual model/engine evidence; none are runnable yet."""
    try:
        SEARCH_VALIDATOR.validate(search)
        check_speculative_search_consistency(
            search, {}, {}, {"request_kind": "completion"})
        runtime = check_snapshot(runtime_snapshot)
    except (ValidationError, SnapshotValidationError, ValueError) as exc:
        raise SpeculativePlanError("invalid search or runtime snapshot: " + str(exc)) from exc
    if not isinstance(target_observation, dict) or target_observation.get("kind") != "gguf_metadata_observation":
        raise SpeculativePlanError("requires real GGUF metadata observation structure")
    meta = target_observation.get("architecture_metadata")
    source = target_observation.get("source")
    validation = target_observation.get("validation")
    if (not isinstance(meta, dict) or not isinstance(source, dict)
            or not isinstance(validation, dict)
            or validation.get("metadata_structure_parsed") is not True
            or not isinstance(source.get("filename"), str)):
        raise SpeculativePlanError("incomplete GGUF metadata evidence")
    nextn = meta.get("nextn_predict_layers")
    embedded_observed = (isinstance(nextn, int) and
                         not isinstance(nextn, bool) and nextn > 0)
    help_result = runtime["command_results"]["help"]
    flags = help_result.get("flag_names_observed")
    flags = set(flags) if (help_result["status"] == "success" and
                          isinstance(flags, list)) else set()
    engine_serve = runtime["entrypoint"]["command"] == "serve"
    candidates = []
    if "off" in search["modes"]:
        base = _candidate("off", None)
        base["readiness"] = "baseline_only"
        base["reasons"] = ["baseline_without_speculative_flags"]
        candidates.append(base)

    for mode in search["modes"]:
        if mode == "off":
            continue
        artifacts = ([None] if not mode.startswith("external_") else
                     [item for item in search["external_artifacts"]
                      if item["mode"] == mode])
        placements = ([None] if not mode.startswith("external_") else
                      (search["draft_placements"] or [None]))
        for artifact in artifacts:
            for placement in placements:
                for n in search["draft_n_max_candidates"]:
                    candidate = _candidate(mode, n, artifact, placement)
                    missing = REQUIRED_FLAGS - flags
                    if not engine_serve:
                        candidate["reasons"].append("serve_runtime_snapshot_required")
                    if missing:
                        candidate["reasons"].append(
                            "runtime_flags_not_observed:" + ",".join(sorted(missing)))
                    if mode == "embedded_mtp" and not embedded_observed:
                        candidate["reasons"].append("embedded_nextn_not_observed_or_invalid")
                    if mode.startswith("external_"):
                        if "--spec-draft-model" not in flags:
                            candidate["reasons"].append("external_draft_flag_not_observed")
                        if artifact["sha256"] is None:
                            candidate["reasons"].append("external_draft_fingerprint_missing")
                        if placement is not None:
                            if placement["device_id"] != "none" and "--spec-draft-device" not in flags:
                                candidate["reasons"].append("draft_device_flag_not_observed")
                            if "--spec-draft-ngl" not in flags:
                                candidate["reasons"].append("draft_ngl_flag_not_observed")
                    if candidate["reasons"]:
                        candidate["readiness"] = "blocked_missing_evidence"
                    elif mode.startswith("external_"):
                        candidate["readiness"] = "requires_pairing_and_load_test"
                        candidate["reasons"] = [
                            "external_pairing_and_runtime_mode_unverified",
                            "actual_model_load_not_tested"]
                    else:
                        candidate["readiness"] = "requires_runtime_mode_and_load_test"
                        candidate["reasons"] = [
                            "engine_spec_type_value_unverified",
                            "actual_model_load_not_tested"]
                    candidates.append(candidate)
                    if len(candidates) > search["max_candidates"]:
                        raise SpeculativePlanError(
                            "candidate count exceeds max_candidates; narrow draft N or modes")
    output = {
        "schema_version": "0.1.0", "kind": "speculative_search_preview",
        "engine_binary_sha256": runtime["binary"]["sha256"],
        "source_model_filename": source["filename"],
        "candidates": candidates,
        "limitations": [
            "A draft-n-max candidate is only the requested upper bound, not necessarily generated/accepted tokens.",
            "Help flag presence cannot verify --spec-type values or backend/model MTP execution.",
            "GGUF nextn_predict_layers does not imply a maximum usable draft-n-max.",
            "External artifacts require independent pairing, file identity and model-load checks.",
            "No GPU/Host memory fit, target load, benchmark or inference is performed.",
            "All candidates are non-executable previews, including the off baseline.",
        ],
    }
    PREVIEW_VALIDATOR.validate(output)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", type=Path, required=True)
    ap.add_argument("--model-observation", type=Path, required=True)
    ap.add_argument("--runtime-snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    try:
        data = make_preview(load_json(args.search),
                            load_json(args.model_observation),
                            load_json(args.runtime_snapshot))
        payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, payload,
                       protected=(args.search, args.model_observation,
                                  args.runtime_snapshot),
                       overwrite=args.overwrite)
            print("Created speculative search preview:", args.output)
        else:
            print(payload, end="")
    except (OSError, ValueError, ValidationError) as exc:
        print("Speculative preview rejected:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
