# Stability hardening and cross-platform test status

This round is deliberately a reliability refactor: **no new model-load or benchmark worker was introduced**.

## What is now enforced

- **Safe output:** `tools/safe_output.py` is shared by CLI snapshot, GGUF metadata, fit estimate, historical estimate converter, and scenario comparison. Writes use sibling temp files and atomic publication. Default refuses an existing output; `--overwrite` opts in. The named input files (engine, GGUF, snapshots, registry and result files as applicable) cannot be used as the output, including path aliases/symlinks.
- **CLI completion:** a successful preview and completed estimator return exit code 0. Invalid input/writing returns 1. Opt-in fit estimation that timed out, failed, or exceeded the output cap still saves the diagnostic record but returns exit code **2**. The JSON distinguishes `status: estimate_attempted` (an attempt) from `estimate.status: completed|timeout|nonzero_exit|process_error|output_limit`.
- **Runtime snapshots:** new `schemas/runtime-capability.schema.json` and `tools/runtime_snapshot.py` validate structural fields and binary/identity hash consistency. Observations remain evidence of Help/device listing, not executable model support. Real records can contain additional upstream fields not represented in the mandatory minimum schema.
- **Shared subprocess:** `tools/process_runner.py` launches argv without shell, discards stdin, captures into temp files to keep Python RAM bounded, periodically checks output growth, and handles timeout, output limit and launch failures. The current 8 MiB file cap is polled; a very fast writer may momentarily exceed it. Full descendant-process cleanup and cancellation remain out of scope until the future actual inference runner.
- **Model identity:** by default an estimator plan records the GGUF metadata-prefix SHA-256, file size and mtime. Before `--execute` these are rechecked. These are *fast change indicators*, not proof of identical weight payloads. `--full-model-sha256` explicitly hashes all GGUF bytes for stronger identity at a substantial disk-I/O cost; it still cannot validate weight tensor correctness, model load, or sharded auxiliary inputs.
- **CI:** GitHub Actions now tests Ubuntu and Windows on Python 3.11 and 3.13. No GPU-backed CI hardware is assumed.

## Example (PowerShell)

```powershell
python tools/probe_runtime.py --command serve --output devices-capability.json --include-devices
# A repeat using the same output name must now be explicit:
python tools/probe_runtime.py --command serve --output devices-capability.json --include-devices --overwrite

# Run after preparing fit/device snapshots; example flags, not optimal choices:
python tools/fit_estimate.py --model "D:\AIModels\Author\Repo\model.gguf" `
  --fit-snapshot fit-capability-v2.json --device-snapshot devices-capability-v2.json `
  --device Vulkan0 --context 4096 --gpu-layers 18 `
  --scenario daily --output fit-daily-preview.json
```

`--execute` is still required to run the actual estimator; add `--full-model-sha256` **only if** complete GGUF hashing is needed. If desired, save a new output name per run instead of `--overwrite` to keep historical observations.

## Remaining work

- No real model loading, inference quality or throughput/latency verification yet.
- No robust process-tree/group termination for future serving workers, and no end-to-end user cancellation.
- No full multisplit GGUF or mmproj bundle identification; the fast fingerprint covers one selected GGUF.
- The pinned 302-row registry stays separate from detected runtime Help; actual support must be tested per build and backend.
- Periodic availability samples are not peaks, and observed GPU free memory is not guaranteed headroom.
- A future CLI/GUI should expose clear progress, workload permissions and privacy/redaction controls, and manage model/engine fingerprint caches explicitly.
