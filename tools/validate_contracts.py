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


def validate(schema_path, document_path):
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    document = load_json(document_path)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    if Path(schema_path).name == "workload-profile.schema.json":
        locks = document["user_locked_parameters"]
        candidate = document["candidate_parameters"]
        duplicate = sorted(set(locks).intersection(candidate))
        if duplicate:
            raise ValueError("candidate overrides user-locked parameters: " + ", ".join(duplicate))
        work = document["workload"]
        constraints = document["constraints"]
        min_ctx = constraints.get("minimum_context_tokens")
        if min_ctx is not None and work["context_tokens"] < min_ctx:
            raise ValueError("context_tokens falls below minimum_context_tokens")
        required_slots = constraints.get("required_concurrency")
        if required_slots is not None and work["concurrency"] < required_slots:
            raise ValueError("concurrency falls below required_concurrency")
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
