# CLI Capability Registry v1

This is a **pinned documentation inventory**, not a runtime hardware capability probe or an automatic tuner.

## Two separate layers

- [CLI Inventory](../../data/cli-inventory.json): 257 documented llama-server entries plus 45 llama-bench entries (including removed/deprecated flags). Each entry records executable, aliases, source line, and documented defaults/environment variable where identifiable.
- [Tuning Knowledge](../../data/tuning-knowledge.json): 32 curated *research hypotheses* for possible parameter exploration. A flag appearing here does not certify speedup, backend support, or quality equivalence.

Both layers are pinned to upstream llama.cpp commit `77c4540ce4667f4577753fe034a3a8042f510b8f`.

## Regenerate and verify

Use Python 3.9 or newer. This script fetches the **pinned** official Markdown files, not the moving master branch:

    python tools/generate_cli_registry.py --check
    python tools/generate_cli_registry.py

To work offline using an upstream checkout at the same pinned commit:

    python tools/generate_cli_registry.py --source-root PATH_TO_LLAMA_CPP --check

Run local tests after installing development dependencies:

    python -m pip install -r requirements-dev.txt
    python -m unittest discover -s tests -v

## Boundaries

- Documentation flags are **not** guaranteed to be present or work in every compiled executable. Runtime `--help`, model metadata and actual backend support must be checked separately.
- `value_type: not_verified` deliberately records unknown types. `arity` is only a best-effort hint derived from docs, not reliable input validation.
- Null `documented_default` or `environment_variable` means not confirmed by this parser, not that the executable has no such setting.
- Retired or deprecated options must never become tuning candidates.
- Candidate prerequisites in tuning knowledge are provisional and must be verified against the specific build, model and hardware.
- This research phase does not change llama.cpp files or attempt to fix upstream inference bugs.

## Next research contracts

- [Workload Profile JSON schema](../../schemas/workload-profile.schema.json)
- [Run Record JSON schema](../../schemas/run-record.schema.json)
- [Illustrative example profile](../../examples/workload-profile.example.json)
- [Illustrative example result](../../examples/run-record.example.json)
