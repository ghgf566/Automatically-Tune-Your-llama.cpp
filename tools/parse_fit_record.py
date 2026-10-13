#!/usr/bin/env python3
"""Re-parse an existing ATYL fit-estimate JSON WITHOUT starting llama or reading GGUF.

Historical v0.2 captures set output_truncated_or_unknown=true unconditionally;
this tool requires the entire recognized upstream row format before converting.
"""
import argparse
import json
import sys
from pathlib import Path

from fit_memory_parser import parse_memory_estimate
from safe_output import write_text

MAX_RECORD_BYTES = 2 * 1024 * 1024


class RecordParseError(ValueError):
    pass


def upgrade(record):
    if not isinstance(record, dict):
        raise RecordParseError("fit estimate record must be an object")
    if record.get("schema_version") not in {"0.2.0", "0.3.0"}:
        raise RecordParseError("supports only fit-estimate records v0.2 and v0.3")
    if (record.get("kind") != "fit_estimate_plan"
            or record.get("status") != "estimate_attempted"):
        raise RecordParseError("requires a previously executed fit estimate")
    candidate = record.get("candidate")
    estimate = record.get("estimate")
    argv = record.get("argv")
    if (not isinstance(candidate, dict) or not isinstance(estimate, dict)
            or not isinstance(argv, list) or len(argv) != 15):
        raise RecordParseError("missing expected candidate, estimate, or argv")
    device = candidate.get("device_id")
    if (not isinstance(device, str) or
            argv[1:3] != ["fit-params", "--offline"] or
            argv[5:7] != ["--device", device] or
            argv[-4:] != ["--fit", "off", "--fit-print", "on"]):
        raise RecordParseError("record is not a fixed-configuration memory estimate")
    if estimate.get("status") != "completed" or estimate.get("returncode") != 0:
        raise RecordParseError("cannot parse failed or incomplete estimates")
    stdout = estimate.get("stdout_excerpt")
    if not isinstance(stdout, str):
        raise RecordParseError("missing stored stdout")
    if record["schema_version"] == "0.3.0" and estimate.get("output_truncated_or_unknown"):
        raise RecordParseError("v0.3 output was truncated or capture uncertain")
    result = parse_memory_estimate(stdout, device)
    if result is None:
        raise RecordParseError("stored output does not match known upstream memory rows")
    # JSON roundtrip creates an independent copy of the input.
    updated = json.loads(json.dumps(record))
    updated["schema_version"] = "0.3.0"
    updated["estimate"]["parsed_numeric_memory"] = result
    return updated


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--input", type=Path, required=True,
                     help="Already saved fit-daily.json or another fit estimate")
    arg.add_argument("--output", type=Path, help="New JSON destination; no in-place editing")
    arg.add_argument("--overwrite", action="store_true", help="Explicitly replace existing output JSON")
    args = arg.parse_args()
    try:
        source = args.input.expanduser().resolve()
        if source.stat().st_size > MAX_RECORD_BYTES:
            raise RecordParseError("input record exceeds 2 MiB safety limit")
        doc = json.loads(source.read_text(encoding="utf-8"))
        result = json.dumps(upgrade(doc), indent=2, ensure_ascii=False) + "\n"
        if args.output:
            write_text(args.output, result, protected=(source,), overwrite=args.overwrite)
            print("Created structured estimate record:", args.output)
        else:
            print(result, end="")
    except (OSError, ValueError, IndexError, TypeError) as error:
        print("Existing estimate rejected:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
