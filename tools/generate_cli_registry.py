#!/usr/bin/env python3
"""Regenerate the documented llama.cpp CLI inventory from a pinned upstream SHA."""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

PIN = "77c4540ce4667f4577753fe034a3a8042f510b8f"
REPO = "ggml-org/llama.cpp"
SERVER, BENCH = "tools/server/README.md", "tools/llama-bench/README.md"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "cli-inventory.json"


def flag_info(s):
    names = re.findall(r"(?:^|[,\s]+)(-{1,2}[A-Za-z][\w-]*)", s)
    if not names:
        raise ValueError("unparsable CLI row: " + s)
    primary = next((n for n in names if n.startswith("--")), names[0])
    remainder = re.sub(r"(?:^|[,\s]+)-{1,2}[A-Za-z][\w-]*", " ", s).strip().replace(r"\|", "|")
    return {"name": primary, "aliases": [n for n in names if n != primary],
            "value_syntax": remainder or None}


def fields(desc):
    default = re.search(r"\(default:\s*([^)]*)\)", desc)
    env = re.search(r"\(env:\s*([A-Z_][A-Z0-9_]*)\)", desc)
    return {"documented_default": default.group(1) if default else None,
            "environment_variable": env.group(1) if env else None}


def server_entries(raw):
    result, group = [], None
    groups = {"Common params": "common", "Sampling params": "sampling",
              "Server-specific params": "server"}
    for i, line in enumerate(raw.split("\n")):
        heading = re.match(r"^### (Common params|Sampling params|Server-specific params)$", line)
        if heading:
            group = groups[heading.group(1)]
        if "<!-- HELP_END -->" in line:
            break
        if not group:
            continue
        row = re.match(r"^\|\s*\x60([^\x60]+)\x60\s*\|\s*(.*?)\s*\|\s*$", line)
        if not row:
            continue
        flag = flag_info(row.group(1))
        desc = re.sub(r"<br\s*/?>", "\n", row.group(2)).replace(r"\|", "|").strip()
        status = ("removed" if re.search("the argument has been removed", desc, re.I)
                  else "deprecated" if re.search("deprecated", desc, re.I)
                  else "documented")
        result.append({"id": "llama-server:" + flag["name"], "program": "llama-server",
                       "group": group, **flag,
                       "arity": "value_or_optional_value" if flag["value_syntax"] else "switch",
                       "value_type": "not_verified", **fields(desc), "status": status,
                       "description": desc, "source": {"path": SERVER, "line": i + 1}})
    return result


def bench_entries(raw):
    lines = raw.split("\n")
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("usage: llama-bench"))
    result, group = [], "options"
    for i in range(start + 2, len(lines)):
        line = lines[i]
        if line.startswith(chr(96) * 3):
            break
        if line.strip() == "test parameters:":
            group = "test_parameters"
            continue
        if re.match(r"^\s{2}-{1,2}[A-Za-z]", line):
            match = re.match(r"^\s+(-{1,2}[A-Za-z][\w-]*(?:(?:,\s*|\s+)-{1,2}[A-Za-z][\w-]*)*)(.*)$", line)
            if not match:
                raise ValueError("unparsed bench line: " + line)
            flag = flag_info(match.group(1))
            desc = match.group(2).strip()
            result.append({"id": "llama-bench:" + flag["name"],
                           "program": "llama-bench", "group": group, **flag,
                           "arity": "value_or_optional_value" if desc.startswith(("<", "[")) else "switch",
                           "value_type": "not_verified", **fields(desc),
                           "status": "documented", "description": desc,
                           "source": {"path": BENCH, "line": i + 1}})
        elif re.match(r"^\s{10,}\S", line) and result:
            result[-1]["description"] += "\n" + line.strip()
            result[-1].update(fields(result[-1]["description"]))
    return result


def build(server, bench):
    rows = server_entries(server) + bench_entries(bench)
    if len(set(row["id"] for row in rows)) != len(rows):
        raise ValueError("duplicate CLI names")
    return {"schema_version": "1.0.0",
            "upstream": {"repository": REPO, "commit": PIN,
                         "source_paths": [SERVER, BENCH]},
            "limits": "Documented flag inventory, not per-build capability. Null means undocumented/unknown. Type hints have not been verified.",
            "entries": rows}


def read_source(path, root):
    if root is not None:
        return (root / path).read_text(encoding="utf-8")
    url = "https://raw.githubusercontent.com/" + REPO + "/" + PIN + "/" + path
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Local pinned upstream checkout")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(read_source(SERVER, args.source_root),
                   read_source(BENCH, args.source_root))
    if args.check:
        stored = json.loads(args.output.read_text(encoding="utf-8"))
        if stored != result:
            print("Registry mismatch with pinned upstream", file=sys.stderr)
            return 1
        print("Verified", len(result["entries"]), "CLI rows")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("Wrote", len(result["entries"]), "CLI rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
