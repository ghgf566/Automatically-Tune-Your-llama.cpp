# ATYL variable architecture and delivery roadmap

**Goal:** Build a dependency-aware tuner, not a growing set of ad hoc UI toggles or unvalidated llama command strings. All modes must be gated by observed engine/model capabilities and backed by reproducible measurements. This roadmap is a plan, not a claim of implemented tuning.

## One shared planning contract

Validated Research Profile (`workload-profile.schema.json`) → optional `speculative_search` (shared schema) → **full-profile-bound candidate preview with stable Candidate ID** → model/load/memory validation → controlled benchmark → optional `speculative_observation` in Run Record (the shared schema's `observation` definition) → Pareto/reporting.

The search and observation definitions both live in **one** [`speculative-search.schema.json`](../../schemas/speculative-search.schema.json), resolved locally without network. New speculative families or result metrics are added there, rather than copied into independent CLI, GUI, benchmark and reporting schemas. The research Profile/Result versions remain `1.1.0` because these fields are optional; older files continue to validate.

## Variables and dependency groups

| Group | Fixed constraints / adjustable dimension | Prerequisites | Test evidence |
| --- | --- | --- | --- |
| Main model & auxiliary models | main GGUF fingerprint, external draft fingerprint, tokenizer/pairing, optional mmproj | all files explicitly selected and identified | metadata, later model load |
| Engine & backend | binary SHA, Help flags, mode-specific available values, GPUs/CPU | fresh matching snapshot | observed then tested, never assumed |
| Core placement | main GPU layers, tensor split, device, threads | actual backend and memory capacity | estimate then load |
| Context & memory | ctx, KV cache types, micro-batch, concurrency, memory reserve | workload minimum constraints | estimate/observed peaks |
| Speculative family | off / embedded MTP / external MTP / simple draft | mode-specific GGUF evidence, engine support, compatible inputs | load then benchmark |
| Speculative search | **draft_n_max**, draft placement, later acceptance thresholds | positive draft mode, bounded requested values and compatibility checks | effective vs requested N, drafted/accepted counts |
| Quality & policy | sampler, target model, context minimum, user locks | must not silently change | quality and workload contract |
| Environment | daily/reduced background, heat/power state, seed/repetitions | capture per-run state | statistically comparable repetitions |

Do not equate `nextn_predict_layers` with the upper limit for `draft_n_max`; actual MTP runtime implementation may place a different limit. Likewise do not assume `llama-bench` measures the full speculative server path. Mark missing data **unknown** and withhold unsupported candidates.

## Progress checkpoints

| Stage | Current state | Completion criterion |
| --- | --- | --- |
| Runtime / GGUF / Fit estimates / safe output / Windows CI | Implemented, partially real-Windows verified | maintain binary-linked snapshots and regression coverage |
| Typed speculative search + requested Draft N | **Full-Profile-bound non-executable preview implemented** | enumerate bounded Draft N, preserve user locks and `off` baseline, trace SHA-bound candidate ID; unverified target hash blocks MTP |
| Target-draft pairing + multisplit/auxiliary identity | **Single-file target/draft byte and metadata preflight implemented; runtime pairing untested** | verify real target+draft bytes, tokenizer/architecture and bundle at execution; prohibit unverified load |
| Opt-in load feasibility | **Bounded target/draft server load with explicit `--execute` implemented** | fixed workload load, readiness check, cleanup and classified failures; memory remains observational |
| Speculative memory feasibility | Implemented with telemetry limits: joint server Fit path is trace-verified; target-only fallback detection; reserve checks during load and requests; sampled Host process-tree peak. WDDM did not expose process-attributed GPU memory, so device reserve uses system-wide free VRAM | add process-attributed GPU peak collection where supported; unknown must remain explicit |
| Real speculative generation/serving benchmark | Comparable CUDA build 11624 Qwen3.8-27B `off`/embedded-MTP run captured: 5 repetitions, TTFT, throughput, acceptance and memory samples; both modes observed at 16/66 GPU layers. MTP Draft N=4 reached 0.689x `off` decode throughput on the fixed synthetic prompt. Earlier build 11574 Vulkan run remains incomplete because effective layer count was not logged | retain the single-run path and its evidence checks; measurements remain workload-specific |
| Bounded Draft N and workload sweep | **Completed 9/9 comparable pairs** on Qwen3.8-27B with embedded NextN metadata, build 11624 CUDA, 66/66 GPU layers and CPU FFN placement. Draft N 1/2/4 was compared across short prompt, long prompt and code-generation workloads with Q4_0 KV. MTP decode speedup ranged from 0.66x to 1.62x; results are workload/N-specific, and effective Draft N remains unknown. Full evidence and limitations are in the [MTP sweep report](../research/mtp-speculative-search.md). Qwen3.6-35B-A3B still has no positive `nextn_predict_layers` and is not an MTP sweep target. | use actual service traces to add workloads, repeat promising cells with counterbalanced runs, and independently verify effective Draft N when upstream exposes it |
| MoE expert cache and KV placement | **Bounded runner and two real CUDA sweeps implemented** in `tools/run_moe_cache_sweep.py`. Qwen3.6-35B-A3B was tested at 16/41 GPU layers with CPU MoE experts, Q4_0 K/V and 0/1024/2048 MiB GPU expert cache. The longer randomized-order sweep used five 256-token repetitions; means were 6.08 / 6.76 / 6.70 tokens/s, with sample CVs of 3.6% / 2.2% / 6.9%. The apparent cache benefit is promising but the distributions overlap, and this is still one synthetic workload and one order. Shared-expert placement was not independently identified, and WDDM process VRAM remains unknown | repeat with multiple workloads and multiple counterbalanced orders; compare GPU-layer counts and validate shared-expert and KV placement evidence before using results for recommendations |
| Adaptive bounded optimization | Not implemented | compare off vs candidate, coarse-to-fine N search, joint placement refinement within cap, stop on regressions |
| UI / config/export | Not implemented | derive controls dynamically from plan/schema; expose source/evidence, never silently alter locked values |

## Suggested bounded search protocol

1. Create an **off** baseline under the same target model, build, workload, sampling and context constraints. Record actual backend/defaults.
2. For each eligible MTP or draft mode, probe a **small increasing set** (e.g. 1/2/4/8) of `draft_n_max`. These are requested maxima, not guaranteed accepted/generated token counts.
3. Verify whether the engine clamps or rejects the requested value; store **requested and effective** separately. If effective N is unobservable, report `null` and do not present it as exact.
4. Estimate and verify combined draft+target RAM/VRAM, then run warm-up and repeated measurements. GPU layers/draft device must be held fixed for the first comparison.
5. Refine N around measured promising values under a fixed budget. Only then study interactions with draft placement, main offload, Context and KV configuration; these are coupled variables.
6. Evaluate throughput **and** tail latency, TTFT, memory overhead, stability, sampling contract and acceptance counters (if available). Choose **off** if all speculative options lose; preserve Pareto trade-offs.
7. Only launch a real model/benchmark after explicit user consent and verified pairing, flags, limits and local input paths. Preview tools remain non-executable.

## Compatibility with earlier artifacts

The optional fields do **not** retrofit old Fit v0.3 JSON with hypothetical MTP measurements. A plain `fit-params` number cannot be taken as proof that an embedded or external MTP plan fits. Historic workload and result JSON without speculative fields continue to be valid; new runs may attach their planned spec search and observed result when those stages exist.

## Pre-Feasibility integration review

The CLI `tools/plan_speculative_search.py` now **requires** `--profile`, not an isolated `--search`: the full Profile schema/cross-field checks apply before proposal generation, including user-locked parameters and the Serve workload constraint. Speculative options and aliases are reserved from ordinary parameter dictionaries via the pinned llama-server Registry. Model GGUF observation must conform to the full GGUF Schema; SHA-linked runtime snapshot is verified structurally; reported target full digest evidence is accepted only when a previously saved Fit plan matches the profile model SHA, engine SHA, filename and file size. Since no model bytes are opened, all such checks remain *recorded evidence*, never permission to execute.

Each candidate has a deterministic SHA256 `candidate_id` derived from full relevant Profile configuration, GGUF observation digest, actual observed runtime binary SHA and its individual Draft N/mode/placement. This key is intentionally distinct from Run ID. A Run Record with `speculative_observation.measurement_status=completed` must retain its `candidate_id` to support later correlation. Changing Profile workload/locks, model observation or candidate configuration creates a new identity.

`tools/process_runner.py` now launches only ATYL-owned isolated processes and uses `process_lifecycle.py` to kill their owned process group/descendants on timeout, cancellation or Python interruption. This is a **best-effort** cleanup, not an OS-level containment boundary for detached processes or arbitrary services. Explicit user cancellation at the higher-level GUI/runner is still future work.

## Final integration audit (2026-10-15)

A final source/CI review checked command routing, bounded execution, runtime/schema evidence, GGUF inspection, Fit estimates, metadata/Fit cross-identity, full-profile MTP candidate planning, stable candidate IDs, Run Record invariants, safe file writes, Windows/Linux CI and research documentation. A **new Inspector metadata-prefix SHA-256** now binds the GGUF observation to the optional Fit-plan fingerprint. Synthetic end-to-end test coverage verifies both matching and mismatching headers. Completed speculative records must link a Candidate ID; successful `off` baselines also require recorded load and completed measurement. See [detailed final audit](../research/pre-feasibility-audit.md).

## Feasibility stage: target/draft preflight and opt-in load check

`tools/verify_model_pair.py` reopens the explicitly selected local target and draft GGUFs, verifies each current full-file SHA-256 against the workload Profile and selected artifact, rejects declared multi-part inputs, and compares architecture plus a digest of the exact `tokenizer.ggml.*` metadata bytes. A passing report is `requires_runtime_load_verification` and remains non-executable; metadata equality cannot prove engine-specific pairing or load success.

`tools/run_feasibility.py` creates a bounded server-load plan for one preview candidate. Plan creation does not launch llama. Explicit `--execute` revalidates engine/model identity (including an external draft), pins context and concurrency to the Profile, forces `--fit off`, requires explicit target GPU layer placement, disables the Web UI and binds to `127.0.0.1`. It waits for `/health` readiness, sends no inference request, then stops the owned process tree on readiness or any failure/timeout. It records system-wide available Host RAM and samples the owned process tree's current Windows Working Set or Linux RSS every 500 ms; the report calls the highest observed sample the process peak and marks process-tree coverage complete, partial or unknown. Sampling can miss short spikes and summed process Working Sets can double-count shared pages. Load success is not a memory-reserve guarantee, correctness test or performance result.

When GPU offload is requested, feasibility execution also samples NVIDIA `nvidia-smi` device-level free and used memory once per second, retaining each device UUID and name plus minimum free / maximum used observations. These values include other processes, are not mapped to a llama.cpp device ID, and are not a per-process VRAM peak. Unsupported vendors, missing tools, timeouts, or unrecognized output remain `unknown`. Windows GPU Process Memory counters are not used because Microsoft documents incorrect values on affected Windows versions; system performance counters are also intended for no more than once-per-second collection. No real model was loaded to validate this collector in development.

The standalone `fit-params` plan estimates only the target model and does not take speculative mode, draft file, or draft placement. The controlled server benchmark instead uses the live server Fit path and only marks embedded MTP Fit as ready when its trace shows the extra-model measurement and no target-only fallback warning. Its system-wide free-memory samples still do not establish per-process VRAM peak or reserve compliance for arbitrary runs.

Next: expand the workload suite from measured service needs and repeat promising cells with counterbalanced runs. Keep backend comparisons on the same build when possible; CUDA on this Pascal `sm_61` GPU is verified with build 11624. Future benchmark runners now default to a 512 MiB GPU reserve/Fit target, as requested; previous reports retain the 1024 MiB configuration they actually used. See [MTP/speculative decoding research](../research/mtp-speculative-search.md).

The local `EmbeddingGemma-2-GGUF` file has metadata architecture `gemma-embedding2`, 24 blocks and 512 embedding dimensions. It has only been inspected, not loaded; it is a candidate for a **separate embeddings throughput/latency workload**, not the MTP Draft N matrix. The first embedding workload contract and runner remain future work.
