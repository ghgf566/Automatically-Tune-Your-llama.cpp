#!/usr/bin/env python3
"""Validate ATYL's research JSON contracts. No llama.cpp process is launched."""
import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]


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

def validate(schema_path, document_path):
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    document = load_json(document_path)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    if Path(schema_path).name == "workload-profile.schema.json":
        check_profile_consistency(document)
    if Path(schema_path).name == "run-record.schema.json":
        if document["status"] == "success":
            if document["engine_binary_sha256"] is None or not document["configuration"]["launched_argv"]:
                raise ValueError("successful run must record engine binary hash and launched_argv")
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
