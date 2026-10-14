#!/usr/bin/env python3
"""Validate ATYL's research JSON contracts. No llama.cpp process is launched."""
import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]

def _registry():
    """Resolve shared contracts locally, never via network or GitHub URLs."""
    spec = load_json(ROOT / "schemas" / "speculative-search.schema.json")
    return Registry().with_resource(spec["$id"], Resource.from_contents(spec))




def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))



from engine_adapter import route_from_profile


def canonical_cli_name(program, flag, inventory):
    """Canonicalize aliases in the selected logical program."""
    mapping = {}
    for entry in inventory["entries"]:
        if entry["program"] != program:
            continue
        for alias in [entry["name"], *entry["aliases"]]:
            existing = mapping.get(alias)
            if existing is not None and existing != entry["name"]:
                raise ValueError("ambiguous CLI alias in registry: " + alias)
            mapping[alias] = entry["name"]
    return mapping.get(flag, flag)


def check_profile_consistency(document):
    """Separate engine identity from registry revision and enforce user locks."""
    inventory = load_json(ROOT / "data" / "cli-inventory.json")
    if document["registry_upstream_commit"] != inventory["upstream"]["commit"]:
        raise ValueError("registry_upstream_commit differs from local pinned inventory")
    route = route_from_profile(document["engine"])
    program = route.registry_program
    if program is None:
        raise ValueError("no CLI inventory available for this profile command")
    work_mode = document["workload"]["mode"]
    if work_mode == "bench" and route.command != "bench":
        raise ValueError("bench workload requires the bench subcommand")
    if work_mode == "server" and route.command != "serve":
        raise ValueError("server workload requires the serve subcommand")
    locks = document["user_locked_parameters"]
    candidates = document["candidate_parameters"]
    canonical_locks = {canonical_cli_name(program, flag, inventory) for flag in locks}
    canonical_candidates = {canonical_cli_name(program, flag, inventory) for flag in candidates}
    overlap = sorted(canonical_locks.intersection(canonical_candidates))
    if overlap:
        raise ValueError("candidate overrides user-locked parameters: " + ", ".join(overlap))
    work = document["workload"]
    constraints = document["constraints"]
    if (constraints.get("minimum_context_tokens") is not None
            and work["context_tokens"] < constraints["minimum_context_tokens"]):
        raise ValueError("context_tokens falls below minimum_context_tokens")
    if (constraints.get("required_concurrency") is not None
            and work["concurrency"] < constraints["required_concurrency"]):
        raise ValueError("concurrency falls below required_concurrency")
    speculative = document.get("speculative_search")
    if speculative is not None:
        check_speculative_search_consistency(speculative, locks, candidates, work, inventory)



def speculative_aliases(inventory):
    """All pinned draft/spec aliases; never depend on a hand-maintained subset."""
    reserved = set()
    for entry in inventory["entries"]:
        if entry["program"] != "llama-server":
            continue
        name = entry["name"]
        if (name.startswith("--spec-") or name.startswith("--draft")
                or name == "--spec-default"):
            reserved.update([name, *entry["aliases"]])
    return reserved


def _is_speculative_flag(flag, reserved):
    return (flag in reserved or flag.startswith("--spec-")
            or flag.startswith("--draft-") or flag == "--draft")


def validate_profile_document(document):
    schema = load_json(ROOT / "schemas/workload-profile.schema.json")
    Draft202012Validator(schema, format_checker=FormatChecker(),
                         registry=_registry()).validate(document)
    check_profile_consistency(document)
    return document


def check_speculative_search_consistency(search, locks, candidates, workload, inventory=None):
    modes = search["modes"]
    if "off" not in modes:
        raise ValueError("speculative search requires explicit off baseline")
    if inventory is None:
        inventory = load_json(ROOT / "data/cli-inventory.json")
    reserved = speculative_aliases(inventory)
    if any(_is_speculative_flag(k, reserved) for k in [*locks, *candidates]):
        raise ValueError("speculative controls must not be duplicated in CLI parameters")
    if workload.get("request_kind") == "embedding" and modes != ["off"]:
        raise ValueError("embedding workload cannot tune speculative generation")
    if workload.get("mode") != "server" and modes != ["off"]:
        raise ValueError("enabled speculative search requires server workload for current runner")

    for placement in search["draft_placements"]:
        if placement["device_id"] == "none" and placement["gpu_layers"] != 0:
            raise ValueError("CPU-only draft placement requires zero GPU layers")
    artifacts = search["external_artifacts"]
    used = set()
    for item in artifacts:
        key = (item["mode"], item["artifact_id"])
        if key in used:
            raise ValueError("duplicate speculative external artifact")
        used.add(key)
    for mode in ("external_mtp", "external_draft_simple"):
        if mode in modes and not any(item["mode"] == mode for item in artifacts):
            raise ValueError(mode + " requires an explicitly named external artifact")
    if search["draft_placements"] and not any(
        mode.startswith("external_") for mode in modes
    ):
        raise ValueError("draft placement candidates require an external mode")


def check_speculative_result_consistency(observation, run_status):
    mode = observation["mode"]
    if mode == "off":
        if any(observation[k] is not None for k in
               ("requested_draft_n_max", "effective_draft_n_max",
                "draft_artifact_id", "drafted_tokens", "accepted_tokens", "acceptance_rate")):
            raise ValueError("off baseline cannot report speculative draft values")
    if mode.startswith("external_") and not observation["draft_artifact_id"]:
        raise ValueError("external speculative result must identify draft artifact")
    drafted = observation["drafted_tokens"]
    accepted = observation["accepted_tokens"]
    if drafted is not None and accepted is not None and accepted > drafted:
        raise ValueError("accepted draft tokens cannot exceed drafted tokens")
    rate = observation["acceptance_rate"]
    if rate is not None:
        if drafted in (None, 0) or accepted is None:
            raise ValueError(
                "acceptance rate requires known drafted and accepted counts, with drafted > 0")
        if abs(rate - accepted / drafted) > 0.005:
            raise ValueError("reported speculative acceptance rate inconsistent with counts")
    if observation["measurement_status"] == "not_started" and any(
            observation[x] is not None for x in
            ("drafted_tokens", "accepted_tokens", "acceptance_rate",
             "decode_tokens_per_second", "time_to_first_token_ms",
             "peak_host_mib", "peak_device_mib")):
        raise ValueError("not_started speculative experiment must not have measured values")
    requested = observation["requested_draft_n_max"]
    effective = observation["effective_draft_n_max"]
    if requested is not None and effective is not None and effective > requested:
        raise ValueError("effective draft N cannot exceed requested draft N maximum")
    if observation["measurement_status"] == "completed" and mode != "off" and requested is None:
        raise ValueError("completed speculative measurement requires requested draft N")
    if run_status == "success" and observation["load_verification"] != "passed":
        raise ValueError("successful speculative run requires actual load verification")
    if run_status == "success" and observation["measurement_status"] != "completed":
        raise ValueError("successful speculative run requires completed measurement")


def validate(schema_path, document_path):
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    document = load_json(document_path)
    Draft202012Validator(schema, format_checker=FormatChecker(),
                         registry=_registry()).validate(document)
    if Path(schema_path).name == "workload-profile.schema.json":
        check_profile_consistency(document)
    if Path(schema_path).name == "run-record.schema.json":
        # Verify the core execution identity before checking optional
        # speculative sub-records, so invalid runs fail on their root cause.
        if document["status"] == "success":
            if document["engine_binary_sha256"] is None or not document["configuration"]["launched_argv"]:
                raise ValueError("successful run must record engine binary hash and launched_argv")
        if "speculative_observation" in document:
            check_speculative_result_consistency(
                document["speculative_observation"], document["status"])
            if (document["speculative_observation"]["measurement_status"] == "completed"
                    and not document.get("candidate_id")):
                raise ValueError("completed speculative measurement requires candidate_id")
    return document


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("profile", "result"))
    p.add_argument("path", type=Path)
    args = p.parse_args()
    name = {"profile": "workload-profile", "result": "run-record"}[args.kind]
    try:
        validate(ROOT / "schemas" / (name + ".schema.json"), args.path)
    except (ValueError, OSError, KeyError) as exc:
        print("Invalid contract:", exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print("Schema validation failed:", exc, file=sys.stderr)
        return 1
    print("Valid", args.kind, args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
