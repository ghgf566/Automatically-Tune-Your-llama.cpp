#!/usr/bin/env python3
"""Compare declared daily vs reduced-background fit estimate observations.

This compares two user-run snapshots, NOT maximum memory consumption, timing,
inference performance, or proof that background processes were stopped.
"""
import argparse
import json
import sys
from pathlib import Path


class ComparisonError(ValueError):
    pass


def load(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComparisonError("unable to load estimate JSON") from exc
    if not isinstance(value, dict):
        raise ComparisonError("estimate JSON must be an object")
    return value


def _check(record, label):
    if (record.get("kind") != "fit_estimate_plan"
            or record.get("status") != "estimate_attempted"):
        raise ComparisonError(label + " must be an attempted fit estimate")
    env = record.get("environment")
    if not isinstance(env, dict) or env.get("scenario") != label:
        raise ComparisonError(label + " must have matching scenario label")
    if (env.get("capture_status") != "captured_best_effort"
            or not isinstance(env.get("before"), dict)
            or not isinstance(env.get("after"), dict)):
        raise ComparisonError(label + " lacks before/after environmental observations")
    if not isinstance(record.get("estimate"), dict):
        raise ComparisonError(label + " lacks estimate outcome")


def _available(record, phase, kind, device):
    entry = record["environment"][phase].get(kind)
    if not isinstance(entry, dict) or entry.get("status") != "observed":
        return None
    if kind == "gpu" and entry.get("device_id") != device:
        return None
    value = entry.get("available_mib")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _pair(daily, reduced, phase, kind, device):
    a = _available(daily, phase, kind, device)
    b = _available(reduced, phase, kind, device)
    return {
        "daily_available_mib": a,
        "reduced_background_available_mib": b,
        "reduced_minus_daily_available_mib": b - a if a is not None and b is not None else None,
    }


def compare(daily, reduced):
    _check(daily, "daily")
    _check(reduced, "reduced_background")
    if daily.get("engine_sha256") != reduced.get("engine_sha256"):
        raise ComparisonError("cannot compare different runtime binary SHA-256")
    if daily.get("model") != reduced.get("model"):
        raise ComparisonError("model descriptors differ between samples")
    if daily.get("candidate") != reduced.get("candidate"):
        raise ComparisonError("context, layer, or device candidate differs between samples")
    device = daily["candidate"]["device_id"]
    if not isinstance(device, str):
        raise ComparisonError("missing device ID")
    before = {kind: _pair(daily, reduced, "before", kind, device)
              for kind in ("ram", "gpu")}
    after = {kind: _pair(daily, reduced, "after", kind, device)
             for kind in ("ram", "gpu")}
    return {
        "schema_version": "0.1.0", "kind": "background_scenario_comparison",
        "same_runtime_sha256": True,
        "same_recorded_model_descriptor": True,
        "same_candidate": True,
        "candidate": daily["candidate"],
        "background_notes": {
            "daily": daily["environment"].get("background_note"),
            "reduced_background": reduced["environment"].get("background_note"),
        },
        "before_estimate": before, "after_estimate": after,
        "estimate_outcomes": {
            "daily": daily["estimate"]["status"],
            "reduced_background": reduced["estimate"]["status"],
        },
        "limitations": [
            "Scenario labels are user-declared; no process was stopped or verified.",
            "Availability differences cannot be attributed to a particular background app.",
            "Before/after availability is not process peak memory or actual model VRAM usage.",
            "Matching name/size is not proof of identical GGUF weights or auxiliary files.",
            "This is not a benchmark, successful model load or recommended configuration.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--reduced", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(load(args.daily), load(args.reduced))
        data = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            destination = args.output.expanduser().resolve()
            if destination in {args.daily.expanduser().resolve(),
                               args.reduced.expanduser().resolve()}:
                raise ComparisonError("comparison output cannot replace input result")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(data, encoding="utf-8")
            print("Created scenario comparison:", args.output)
        else:
            print(data, end="")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("Scenario comparison rejected:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
