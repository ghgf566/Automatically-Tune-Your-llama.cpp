# MTP and speculative decoding as conditional tuning dimensions

ATYL must treat speculative decoding as an independent, conditional candidate dimension. This is a **future optimization specification**, not a presently available benchmark runner or a claim that any tested GGUF supports MTP.

## Grounded upstream evidence

- Pinned `ggml-org/llama.cpp` commit `77c4540ce4667f4577753fe034a3a8042f510b8f`: [`gguf-py/gguf/constants.py`](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/gguf-py/gguf/constants.py) and [`src/llama-arch.cpp`](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/src/llama-arch.cpp) define the architecture-specific `{arch}.nextn_predict_layers` GGUF key.
- [Pinned `docs/speculative.md`](https://github.com/ggml-org/llama.cpp/blob/77c4540ce4667f4577753fe034a3a8042f510b8f/docs/speculative.md) documents `--spec-type draft-mtp`, `--spec-draft-n-max`, `--model-draft` / `--spec-draft-model`, `--spec-draft-ngl` and `--spec-draft-device`. These are documented upstream controls, **not proof the user's installed binary offers them**.
- The main-model metadata field reports NextN layers. The lack of that field **does not prove** there is no compatible external MTP head, and the presence of the field **does not prove** that the specific engine/backend will load and execute the MTP path.

## Mode classification — never conflate them

| Mode | Prerequisite and controls |
| --- | --- |
| `off` | Always provide an explicit no-speculative baseline (`--spec-type none` when supported) |
| `embedded_mtp` | Positive `{arch}.nextn_predict_layers`; engine must advertise `draft-mtp`; use `--spec-type draft-mtp`; do not assume a separate draft path is needed |
| `external_mtp` | User-selected external MTP-head GGUF matched to the target architecture/tokenizer and engine support; `--spec-type draft-mtp` with appropriate `--model-draft`/`--spec-draft-model` only after load compatibility is confirmed |
| `external_draft_simple` | A compatible small draft model, `--spec-type draft-simple` with `--model-draft`; **not synonymous with MTP** |
| `other_speculative` | EAGLE/DFlash/DSpark or model-free n-gram approaches are **separate** families; do not silently enable them while optimizing MTP |

An external MTP head is not interchangeable with an arbitrary small draft GGUF. The exact pairing behavior can change by engine build; use actual Help + GGUF metadata + an opt-in load check. Require explicit user selection for external files and never download or guess a companion from filenames alone.

## Draft N is an independent, bounded search axis

`--spec-draft-n-max` is the **requested maximum drafted token count** per speculation step, not the number accepted or a promise that each step will produce N tokens. ATYL preserves the user's exact candidate set (example: `1, 2, 4, 8`), never assumes N equals the GGUF's `nextn_predict_layers`, and records `requested_draft_n_max` separately from the eventual `effective_draft_n_max`. Unknown or silently clamped effective values remain `null` until independently observed. Very large N is not automatically better, since validation and draft overhead may outweigh accepted tokens.

Current preview caps each requested N to **1–64** purely as an ATYL search-safety bound, *not an upstream architectural limit*; limits must later be refined by installed-engine and draft-model evidence. The preview refuses candidate combinations above `max_candidates` instead of silently truncating the search.

**Quick, no-execution preview** (after refreshing the GGUF observation so `nextn_predict_layers` can appear):

```powershell
python tools/plan_speculative_search.py `
  --search examples/speculative-search.example.json `
  --model-observation model-metadata.json `
  --runtime-snapshot server-capability.json `
  --output speculative-preview.json
```

This does not read model weights or launch `llama.exe`; use `--overwrite` only to explicitly replace an existing preview. A missing NextN field, incomplete Help flags or missing draft fingerprint leaves MTP candidates **blocked**, not labeled supported. Even good metadata/flags only yield **requires load verification**, not ready-to-run configuration.

Search and observed result shapes are defined centrally in [`speculative-search.schema.json`](../../schemas/speculative-search.schema.json) and integrated as optional fields of the Profile/Result contract. See [dependency-aware tuner roadmap](../specification/tuning-roadmap.md).

## Candidate variables (only after feasibility gating)

- `spec_mode`: `off`, `embedded_mtp`, `external_mtp` (future: other families). A categorical variable, not an unbounded numeric flag.
- `draft_file_identity`: fingerprint + original private path for an external candidate; no file means no external draft. Record architecture and known target pairing evidence. Treat multimodal `mmproj` separately.
- `draft_n_max`: a **bounded set of integer candidates**, respecting upstream/build training and context limits, and always record the effective value.
- `draft_gpu_layers` and `draft_device`: placement of an external head/model independent from target model's offload. For embedded heads, only expose placement knobs that the live backend actually supports.
- `draft_sampling_or_probability_thresholds`: optimize only if the exact engine implementation exposes them; do not change the user's target sampling/quality contract.
- `main_context`, `main_offload`, `kv_cache_format`, `concurrency`, `batching`: carry as fixed experimental controls or jointly refit. Draft/MTP overhead can change the best feasible target placement.

## Search order

1. Read the **main GGUF** metadata including `nextn_predict_layers`. Read each explicitly selected external head as a different model input; verify fingerprints and split/auxiliary completeness. A missing metadata field means `unknown`, not `false`.
2. Inspect the exact `llama.exe` SHA-bound Help; verify the speculation mode and flag aliases per build. Unobserved capability is not enabled on a guess.
3. Generate an `off` baseline under the **same** workload, context, target quantization, target layer placement and sampling contract.
4. Generate only modes whose requirements are satisfied. Estimate **total** Host/GPU memory with the actual speculative flags and selected draft file. Current `tools/fit_estimate.py` does not accept MTP flags and therefore its earlier plain estimates must **not** be reused to claim MTP fits.
5. Request explicit consent for any real load/benchmark execution. Validate target + draft initialization and memory, then compare generation throughput and end-to-end latency, including startup and draft overhead where applicable.
6. Keep accepted/drafted token counts or acceptance rate (when backend exposes it), TTFT, p50/p95 decode latency, tokens/sec, GPU/Host memory and power/CPU overhead. Distinguish candidate refusal, OOM, unknown metrics and valid speedups. If MTP is slower, keep `off` as the winner.

## Current implementation status

The read-only `tools/inspect_gguf.py` preserves `{arch}.nextn_predict_layers` when present. `tools/plan_speculative_search.py` now performs **bounded, non-executable candidate enumeration** with `draft_n_max`, mode/evidence gating and explicit baseline. It does **not** infer `mtp_supported=true`, compose runnable MTP argv, authenticate target-draft pairing, do MTP-aware fitting, load actual models or benchmark.
