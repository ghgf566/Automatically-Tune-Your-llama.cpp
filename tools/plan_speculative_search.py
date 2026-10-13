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
from candidate_identity import canonical_digest, candidate_id
from validate_contracts import validate_profile_document
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


def make_preview(search, target_observation, runtime_snapshot, *, profile=None, fit_plan=None):
    """Candidates require actual model/engine evidence; none are runnable yet."""
    try:
        SEARCH_VALIDATOR.validate(search)
        check_speculative_search_consistency(
            search, {}, {}, {"request_kind": "completion", "mode": "server"})
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
    if profile is not None:
        try:
            validate_profile_document(profile)
        except (ValidationError, ValueError, KeyError) as exc:
            raise SpeculativePlanError("invalid full workload profile: " + str(exc)) from exc
        if profile.get("speculative_search") != search:
            raise SpeculativePlanError("search must match profile speculative_search")
        if runtime["entrypoint"]["command"] != "serve" or runtime["entrypoint"]["kind"] != "integrated":
            raise SpeculativePlanError("profile speculative plan requires integrated serve snapshot")
        if profile["engine"].get("command") != "serve":
            raise SpeculativePlanError("profile requires serve subcommand")
        requested_engine_sha = profile["engine"].get("binary_sha256")
        if requested_engine_sha is not None and requested_engine_sha.lower() != runtime["binary"]["sha256"].lower():
            raise SpeculativePlanError("profile engine SHA differs from runtime snapshot")
        if (runtime.get("registry_commit") is not None
                and profile["registry_upstream_commit"] != runtime["registry_commit"]):
            raise SpeculativePlanError("profile registry revision disagrees with runtime snapshot")
        if (isinstance(target_observation.get("general"), dict)
                and target_observation["general"].get("architecture") is not None
                and target_observation["general"]["architecture"] != profile["model"]["architecture"]):
            raise SpeculativePlanError("profile and GGUF architecture disagree")
    identity_status = "legacy_unbound" if profile is None else "unverified"
    if fit_plan is not None:
        if profile is None:
            raise SpeculativePlanError("fit identity evidence requires a full workload profile")
        if not isinstance(fit_plan, dict) or fit_plan.get("kind") != "fit_estimate_plan":
            raise SpeculativePlanError("invalid fit plan identity evidence")
        observed_fit = fit_plan.get("model")
        if not isinstance(observed_fit, dict):
            raise SpeculativePlanError("fit model identity missing")
        fp = observed_fit.get("fingerprint")
        if not isinstance(fp, dict):
            raise SpeculativePlanError("fit model fingerprint missing")
        if (fit_plan.get("engine_sha256") != runtime["binary"]["sha256"]
                or observed_fit.get("filename") != source["filename"]
                or observed_fit.get("file_size_bytes") != source.get("file_size_bytes")
                or fp.get("full_file_hashed") is not True
                or not isinstance(fp.get("full_file_sha256"), str)
                or fp["full_file_sha256"].lower() != profile["model"]["sha256"].lower()):
            raise SpeculativePlanError("recorded full model digest, size, name or engine mismatch")
        # Historical evidence only; actual future execution must revalidate
        # this fingerprint directly against the on-disk model at launch.
        identity_status = "recorded_full_digest_match"
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
                    if profile is not None and identity_status == "unverified":
                        candidate["reasons"].append("target_model_full_hash_evidence_missing")
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
    # Bind each proposed configuration to the full context before any result
    # persistence, regardless of MTP readiness.
    for candidate in candidates:
        candidate["candidate_id"] = candidate_id(
            search, {"profile_model": profile["model"] if profile is not None else None,
                     "observation_sha256": canonical_digest(target_observation)},
            runtime["binary"]["sha256"], candidate, profile=profile)
    output = {
        "schema_version": "0.1.0", "kind": "speculative_search_preview",
        "profile_id": profile["profile_id"] if profile is not None else None,
        "profile_sha256": canonical_digest(profile) if profile is not None else None,
        "model_observation_sha256": canonical_digest(target_observation),
        "model_identity_status": identity_status,
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
            "Recorded model identity does not replace on-disk verification before execution.",
        ],
    }
    PREVIEW_VALIDATOR.validate(output)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, required=True,
                    help="Validated full workload profile (server + speculative_search)")
    ap.add_argument("--fit-plan", type=Path,
                    help="Optional prior fit plan with a recorded full-file SHA256")
    ap.add_argument("--model-observation", type=Path, required=True)
    ap.add_argument("--runtime-snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    try:
        profile = load_json(args.profile)
        data = make_preview(profile["speculative_search"],
                            load_json(args.model_observation),
                            load_json(args.runtime_snapshot),
                            profile=profile,
                            fit_plan=load_json(args.fit_plan) if args.fit_plan else None)
        payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, payload,
                       protected=(args.profile, args.model_observation,
                                  args.runtime_snapshot, args.fit_plan),
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
