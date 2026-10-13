# Controlled fit-params memory estimate — preview first

This is an **opt-in single-candidate estimate** for the official integrated `llama.exe`; it does not tune parameters automatically or launch a server. Model metadata extraction is read-only, but executing the estimate makes `llama fit-params` read the selected GGUF and initialize its runtime devices. Preview is the default and **does not launch any llama process**.

## Preconditions

- A model GGUF in a real local path; `tools/inspect_gguf.py` parses its header fresh.
- A real `fit-capability-v2.json` snapshot from `python tools/probe_runtime.py --command fit-params --output fit-capability-v2.json`.
- For a Vulkan GPU estimate, a real `devices-capability-v2.json` from `python tools/probe_runtime.py --command serve --include-devices --output devices-capability-v2.json`.
- Both snapshot SHA-256 IDs must match the **currently installed executable**, or planning fails closed.
- No multi-shard GGUF aggregation is implemented yet. Declared multi-file GGUF input is rejected.

## Preview on the user's Vulkan0 device

Run in PowerShell from the ATYL repository. Adjust the model path to the real filename; `18` GPU layers and `4096` tokens are **illustrative inputs, not auto-selected optimal values**.

```powershell
python tools/fit_estimate.py --model "D:\model\Qwen3.8-27B-UD-IQ4_XS.gguf" `
  --fit-snapshot fit-capability-v2.json `
  --device-snapshot devices-capability-v2.json `
  --device Vulkan0 --context 4096 --gpu-layers 18 `
  --output fit-plan.json
```

`fit-plan.json` will have `status: prepared_not_executed` and `estimate: null`. The command includes `--offline`, `--fit off`, `--fit-print on`, the chosen context, GPU layer count, and the explicitly verified `Vulkan0` device ID. It **does not** try other configurations, calculate a VRAM fit ratio from file size, or change user settings.

## Explicitly execute the same kind of estimate

**Only after reviewing the preview**, repeat the command above with `--execute --output fit-estimate-result.json`. This is the first model-reading subprocess in ATYL; timeout defaults to 600 seconds and can be changed with `--timeout` (maximum 1800).

```powershell
python tools/fit_estimate.py --model "D:\model\Qwen3.8-27B-UD-IQ4_XS.gguf" `
  --fit-snapshot fit-capability-v2.json `
  --device-snapshot devices-capability-v2.json `
  --device Vulkan0 --context 4096 --gpu-layers 18 `
  --execute --timeout 600 --output fit-estimate-result.json
```

An execution uses an argument array (`shell=False` in effect), never PowerShell command concatenation, and filters `LLAMA_ARG_*` overrides and common HF credentials from the child environment. The program captures at most 16 KiB per output stream to the JSON (temporary output streams can be larger on disk), records exit status and does not auto-apply any returned suggestions. The complete estimate remains an **upstream message excerpt** (`estimate.stdout_excerpt`), not yet a validated numeric structured memory model.

**Caution:** Runtime model-reading and GGML initialization may use appreciable disk I/O, CPU, RAM and GPU resources even though this is not a benchmark or model inference. A successful command does not prove the requested layers or context can serve requests. Output can expose local file paths and device details; review it before sharing. This stage is tested with synthetic GGUF and mocked subprocesses in Linux CI, **not real Windows estimation**.

## CPU-only preview

You can preview an explicitly CPU-only candidate without a device snapshot, using `--device none --gpu-layers 0`. It does not establish CPU backend compatibility and executing it can still consume significant RAM.

## Why `--fit off` / `--fit-print on`?

At the pinned upstream `tools/fit-params/fit-params.cpp` code, `--fit-print on` prints an estimated memory breakdown; without it, the program runs a fitting algorithm and prints adjusted candidate CLI arguments. This first stage intentionally requests only a fixed-parameter estimate rather than letting the engine revise user-chosen context or offload layers.

Related documentation: [GGUF inspector](gguf-metadata-inspection.md), [Runtime capability probe](runtime-capability.md) and [13-flag Fit reference](windows-build11574-fit-reference.md).
