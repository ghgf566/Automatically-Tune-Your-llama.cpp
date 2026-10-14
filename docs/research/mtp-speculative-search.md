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
  --profile examples/workload-profile.speculative.example.json `
  --model-observation model-metadata.json `
  --runtime-snapshot server-capability.json `
  --output speculative-preview.json
```

The CLI now accepts the **full workload Profile** instead of separate `--search`; use the research example only as a synthetic reference and replace its zero SHA256/model architecture with real verified values before meaningful use. The candidate output includes stable `candidate_id`, `profile_sha256` and `model_observation_sha256`. Model records are not automatically assumed identical: without independent complete-file evidence, enabled candidates are `blocked_missing_evidence`. When you have an earlier `fit_estimate.py --full-model-sha256` result, pass `--fit-plan fit-plan.json` for recorded digest/engine/filename/size checks. This **does not re-hash the file**, and a future executor must verify disk identity again before loading.

For recorded fit-plan identity binding, the GGUF Inspector observation must also include the new **`metadata_sha256`** field. The planner compares that prefix hash and the inspected prefix length against Fit's `metadata_sha256` / `metadata_span_bytes`; older metadata files without the hash must be regenerated. This avoids accepting an unrelated metadata JSON merely because its model filename and byte size match.

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

### Target/draft read-only pairing preflight

The first Feasibility-stage slice is `tools/verify_model_pair.py`. Given a full Profile, an artifact ID, and the explicitly selected local target and draft paths, it re-inspects both files and recomputes full-file SHA-256 against the Profile and artifact. It rejects declared multi-part GGUFs, checks architecture equality, and compares a digest over the exact raw `tokenizer.ggml.*` metadata entries. The Inspector exposes only the digest and entry count; it never outputs token strings.

Example for a Profile that declares an external draft artifact:

```powershell
python tools/verify_model_pair.py `
  --profile workload-profile.json `
  --artifact-id draft-a `
  --target-model "D:\Models\target.gguf" `
  --draft-model "D:\Models\draft.gguf" `
  --output draft-pair-preflight.json
```

A report that passes these checks is still `requires_runtime_load_verification`, has `executable: false`, and cannot establish true tokenizer behavior or engine-specific compatibility. Unknown or unequal tokenizer metadata blocks the pair. This verifier does not prove weight integrity beyond matching the supplied SHA-256, estimate combined draft+target memory, load either model, or measure peak memory.

### Opt-in load feasibility

`tools/run_feasibility.py` creates a non-executable plan, then permits one bounded local load only with an explicit `--execute`. It binds the candidate to the full Profile, current target GGUF observation, matching runtime snapshot and on-disk engine SHA. External draft candidates re-run target/draft file identity and metadata checks immediately before launch. The runner requires observed Help flags, a server/serve workload and bounded timeout; it fixes context/concurrency to the Profile, forces `--fit off`, requires explicit target GPU layer placement (and an explicit device for GPU offload), disables the Web UI, binds to loopback and uses an ephemeral port.

After launch it polls `/health` for readiness and stops the owned server process tree on success, failure or timeout. It sends no inference request and performs no benchmark. The report records load outcome and startup duration, system-wide available Host RAM, and a 500 ms sampled maximum of the owned process tree's Windows Working Set or Linux RSS. The sampled maximum can miss short spikes, process-tree coverage is reported, and summing process Working Sets can double-count shared pages. When GPU offload is requested, NVIDIA device-level free/used memory is also sampled system-wide at one-second intervals; it is neither process-attributed nor mapped to the selected llama device, and does not establish a VRAM peak for the candidate. Process-attributed VRAM peak remains unknown. Readiness is not proof of MTP output correctness, reserve compliance or speed. Automated coverage uses synthetic GGUFs and a fake local HTTP service; development did not load a real model.

The standalone `fit-params` command passes no extra model and cannot establish speculative Fit behavior. The server Fit path supplies an extra model for embedded MTP with `shares_model=true`, so it estimates the shared target weights plus the MTP context. If measuring that context fails, upstream may continue with target-only Fit. ATYL now requires a trace log showing the extra-model measurement and no fallback warning before marking this path ready. It also samples process-tree Host Working Set and system-wide device memory, and stops a run if its configured host or device reserve is crossed. On this Windows WDDM host, per-process VRAM could not be attributed; that remains explicitly unknown.

### Controlled real-model run (2026-10-15)

`tools/run_speculative_benchmark.py` ran the local `Qwen3.8-27B-UD-IQ4_XS.gguf` (14,252,845,984 bytes; SHA-256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`) with llama.cpp `0.6.0-dev`, build 11574, commit `8bb3cf0d2`. Both server modes requested `Vulkan0`, 16 GPU layers, context 4096, batch 512, ubatch 128, fit enabled with a 1024 MiB device target, concurrency 1, the same fixed prompt/sampling, one warmup and five measured 128-token completions. MTP used the embedded NextN head (`nextn_predict_layers=1`, draft maximum 4). The MTP server reached the joint Fit route without the target-only fallback warning.

The five measured completions averaged 1.223 decode tokens/s and 113.15 s latency for `off`, versus 0.821 tokens/s and 165.43 s for MTP (0.671x decode throughput); MTP draft acceptance was 400/930 = 43.0%. TTFT averaged 9.21 s and 10.60 s, respectively. This is a repeated synthetic single-prompt workload, not representative traffic. Host process-tree sampled peak was 14,561 MiB (`off`) and 15,453 MiB (MTP). System-wide NVIDIA sampled maximum used memory was 5,208 MiB and 5,731 MiB; minimum free memory was 2,853 MiB and 2,330 MiB. The 4,096 MiB Host reserve and 1,024 MiB device free-memory reserve were not crossed. GPU peaks are system-wide, not process-attributed.

That earlier result remains `incomplete_or_not_comparable`: build 11574 was run at warning verbosity, so its log did not expose actual GPU layer counts. Equal requested values alone do not prove equal placement.

### Controlled CUDA build run (2026-10-15)

The layer and joint-Fit evidence path was then verified on the official integrated llama.cpp build 11624 (`af0579c78`) with CUDA `CUDA0`, a GeForce GTX 1070 Ti (Compute Capability 6.1), and the same Qwen3.8-27B IQ4_XS file and configuration. Both modes requested 16 GPU layers; trace logs independently reported **16/66 layers** for `off` and MTP. The MTP Fit trace recorded extra-model memory measurement at context 4096, with no target-only fallback warning. The runner reads trace lines incrementally while retaining only a bounded log tail, so the MTP report can mark logs truncated without losing these observations.

With context 4096, batch 512, ubatch 128, Fit on with a 1024 MiB device target, concurrency 1, one warmup and five measured completions of 128 tokens, `off` averaged **1.310 decode tokens/s**, **103.22 s** latency and **6.27 s** TTFT. Embedded MTP at requested draft maximum 4 averaged **0.903 tokens/s**, **147.58 s** latency and **6.80 s** TTFT. MTP accepted 390 of 955 drafted tokens (**40.8%**) and achieved **0.689x** the `off` decode throughput in this workload. The comparison is marked `comparable`: build, workload and requested settings match, actual layer counts match, and the joint MTP Fit path was observed.

This is one synthetic prompt on one machine, so it does not predict other workloads. Sampled process-tree Host peaks were 13,973 MiB (`off`) and 14,881 MiB (MTP). System-wide NVIDIA used-memory maxima were 5,057 MiB and 5,583 MiB; minimum free memory was 3,004 MiB and 2,478 MiB. The 4,096 MiB Host and 1,024 MiB device free-memory reserves were not crossed. WDDM did not provide process-attributed VRAM, and those device readings include other processes. The result supports choosing `off` for this measured configuration; it does not show that all MTP settings or workloads are slower.

The effective-layer line is emitted at INFO level by llama.cpp's [model loader](https://github.com/ggml-org/llama.cpp/blob/af0579c78/src/llama-model.cpp). The extra-model Fit trace requires debug-level forwarding through the [server joint-Fit setup](https://github.com/ggml-org/llama.cpp/blob/af0579c78/common/common.cpp); the pinned [CLI verbosity definitions](https://github.com/ggml-org/llama.cpp/blob/af0579c78/common/arg.cpp) confirm level 5 enables debug output. ATYL uses verbosity 5, scans output as it arrives, and keeps the saved log tail bounded.

Plan-only example (first generate the model observation, runtime snapshot and candidate preview):

```powershell
python tools/run_feasibility.py `
  --profile workload-profile.json `
  --preview speculative-preview.json `
  --model-observation model-metadata.json `
  --runtime-snapshot server-capability.json `
  --candidate-id CANDIDATE_SHA256 `
  --target-model "D:\Models\target.gguf" `
  --output load-check.json
```

Review the plan before explicitly adding `--execute`. External draft candidates also require `--draft-model PATH`.
