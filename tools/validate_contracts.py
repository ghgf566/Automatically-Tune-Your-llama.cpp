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



def canonical_cli_name(program, flag, inventory):
    """Resolve documented aliases within one binary. Unknown flags stay distinct."""
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
    """Enforce user locks and workload minima beyond JSON Schema."""
    inventory = load_json(ROOT / "data" / "cli-inventory.json")
    binary = Path(document["engine"]["binary"]).name.lower()
    program = binary.removesuffix(".exe")
    if program not in ("llama-server", "llama-bench"):
        # Fail closed for unknown binaries: we cannot safely normalize aliases.
        raise ValueError("cannot validate CLI aliases for executable: " + binary)
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
